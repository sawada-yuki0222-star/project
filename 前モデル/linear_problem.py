from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Dict, List, Optional, Tuple, Set

import os
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


Preference = List[int]
Profile = List[Preference]

ProfileKey = Tuple[Tuple[int, ...], ...]
RepresentativeKey = ProfileKey

# envy-free cut identifier: on representative only
# (rep_key, i, j, r)
EFCut = Tuple[RepresentativeKey, int, int, int]

# strategy-proof cut identifier:
# truthful original profile key, agent i, misreport tuple, rank r
SPCut = Tuple[ProfileKey, int, Tuple[int, ...], int]

# violations with score
EFViolation = Tuple[float, EFCut]
SPViolation = Tuple[float, SPCut]


# =========================================================
# Basic profile utilities
# =========================================================

def profile_to_key(profile: Profile) -> ProfileKey:
    return tuple(tuple(pref) for pref in profile)


def key_to_profile(key: ProfileKey) -> Profile:
    return [list(pref) for pref in key]


def all_preferences(n: int) -> List[Preference]:
    return [list(p) for p in permutations(range(n))]


def all_preference_profiles(n: int) -> List[Profile]:
    one_agent_prefs = all_preferences(n)
    return [[list(pref) for pref in prof] for prof in product(one_agent_prefs, repeat=n)]


def count_profiles(n: int) -> int:
    return factorial(n) ** n


# =========================================================
# Agent-permutation canonicalization
# =========================================================

def invert_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for rep_row, orig_row in enumerate(perm):
        inv[orig_row] = rep_row
    return tuple(inv)


def canonical_profile_agents_only(
    profile: Profile,
) -> Tuple[RepresentativeKey, Tuple[int, ...]]:
    best_rep: Optional[RepresentativeKey] = None
    best_perm: Optional[Tuple[int, ...]] = None

    for perm in permutations(range(len(profile))):
        transformed = tuple(tuple(profile[i]) for i in perm)
        if best_rep is None or transformed < best_rep:
            best_rep = transformed
            best_perm = tuple(perm)

    if best_rep is None or best_perm is None:
        raise RuntimeError("Failed to compute canonical profile.")

    inv_perm = invert_permutation(best_perm)
    return best_rep, inv_perm


@dataclass
class IsomorphismData:
    representatives: List[Profile]
    rep_keys: List[RepresentativeKey]
    rep_weights: Dict[RepresentativeKey, int]
    profile_to_rep: Dict[ProfileKey, RepresentativeKey]
    profile_to_inv_perm: Dict[ProfileKey, Tuple[int, ...]]
    rep_to_profiles: Dict[RepresentativeKey, List[ProfileKey]]


def build_agent_isomorphism_data(profiles: List[Profile]) -> IsomorphismData:
    rep_weights: Dict[RepresentativeKey, int] = {}
    profile_to_rep: Dict[ProfileKey, RepresentativeKey] = {}
    profile_to_inv_perm: Dict[ProfileKey, Tuple[int, ...]] = {}
    rep_to_profiles: Dict[RepresentativeKey, List[ProfileKey]] = {}

    for profile in profiles:
        key = profile_to_key(profile)
        rep_key, inv_perm = canonical_profile_agents_only(profile)

        profile_to_rep[key] = rep_key
        profile_to_inv_perm[key] = inv_perm
        rep_weights[rep_key] = rep_weights.get(rep_key, 0) + 1
        rep_to_profiles.setdefault(rep_key, []).append(key)

    rep_keys = sorted(rep_weights.keys())
    representatives = [key_to_profile(k) for k in rep_keys]

    return IsomorphismData(
        representatives=representatives,
        rep_keys=rep_keys,
        rep_weights=rep_weights,
        profile_to_rep=profile_to_rep,
        profile_to_inv_perm=profile_to_inv_perm,
        rep_to_profiles=rep_to_profiles,
    )


def map_representative_matrix_to_original(
    X_rep: np.ndarray,
    inv_perm: Tuple[int, ...],
) -> np.ndarray:
    X_orig = np.zeros_like(X_rep, dtype=float)
    for i in range(len(inv_perm)):
        X_orig[i, :] = X_rep[inv_perm[i], :]
    return X_orig


# =========================================================
# Assignment / LP helpers
# =========================================================

def _rank_map(pref_i: Preference) -> Dict[int, int]:
    return {o: r + 1 for r, o in enumerate(pref_i)}


def _check_preferences(preferences: List[Preference]) -> None:
    n = len(preferences)
    for i, pref in enumerate(preferences):
        if len(pref) != n:
            raise ValueError(f"preferences[{i}] must have length {n}")
        if sorted(pref) != list(range(n)):
            raise ValueError(f"preferences[{i}] must be a permutation of 0..{n-1}")


def _check_doubly_stochastic_matrix(X: np.ndarray, name: str = "X", tol: float = 1e-8) -> None:
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape={X.shape}")

    n = X.shape[0]

    if np.any(X < -tol):
        raise ValueError(f"{name} must satisfy {name} >= 0")

    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    if not np.allclose(row_sums, np.ones(n), atol=tol, rtol=0):
        raise ValueError(f"{name} row sums must all be 1. got {row_sums}")

    if not np.allclose(col_sums, np.ones(n), atol=tol, rtol=0):
        raise ValueError(f"{name} column sums must all be 1. got {col_sums}")


def _check_inputs(preferences: List[Preference], P: np.ndarray) -> None:
    _check_preferences(preferences)
    n = len(preferences)
    if P.shape != (n, n):
        raise ValueError(f"P must be {n}x{n}, got {P.shape}")
    _check_doubly_stochastic_matrix(P, name="P")


def vec(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).reshape(-1, order="C")


def unvec(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape((n, n), order="C")


def uniform_assignment_matrix(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n, dtype=float)


def s_matrix(preferences: List[Preference]) -> np.ndarray:
    n = len(preferences)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        rm = _rank_map(preferences[i])
        for o in range(n):
            S[i, o] = n - rm[o] + 1
    return S


def s_vector(preferences: List[Preference]) -> np.ndarray:
    return vec(s_matrix(preferences))


def a_matrix(preferences: List[Preference]) -> np.ndarray:
    n = len(preferences)
    A = np.zeros((n * n, n * n), dtype=float)

    for i in range(n):
        pref_i = preferences[i]
        for r in range(1, n + 1):
            row = i * n + (r - 1)
            top_r = pref_i[:r]
            for o in top_r:
                col = i * n + o
                A[row, col] = 1.0

    return A


def b_matrix(n: int) -> np.ndarray:
    B = np.zeros((2 * n, n * n), dtype=float)

    for i in range(n):
        for o in range(n):
            B[i, i * n + o] = 1.0

    for o in range(n):
        for i in range(n):
            B[n + o, i * n + o] = 1.0

    return B


def cumulative_prob(row: np.ndarray, pref_i: Preference, r: int) -> float:
    top_r = pref_i[:r]
    return float(np.sum(row[top_r]))


def count_envy_violations(preferences: Profile, P: np.ndarray, tol: float = 1e-8) -> int:
    count = 0
    n = len(preferences)
    for i in range(n):
        pref_i = preferences[i]
        for j in range(n):
            for r in range(1, n + 1):
                own_val = cumulative_prob(P[i], pref_i, r)
                other_val = cumulative_prob(P[j], pref_i, r)
                if own_val + tol < other_val:
                    count += 1
    return count


def count_strategy_violations_at_profile(
    profile: Profile,
    p_map_original: Dict[ProfileKey, np.ndarray],
    one_agent_prefs: List[Preference],
    tol: float = 1e-8,
) -> int:
    truthful_key = profile_to_key(profile)
    P_truth = p_map_original[truthful_key]
    count = 0

    for i in range(len(profile)):
        true_pref = profile[i]

        for misreport in one_agent_prefs:
            if misreport == true_pref:
                continue

            mis_profile = [list(pref) for pref in profile]
            mis_profile[i] = list(misreport)
            mis_key = profile_to_key(mis_profile)
            P_mis = p_map_original[mis_key]

            for r in range(1, len(profile) + 1):
                truthful_val = cumulative_prob(P_truth[i], true_pref, r)
                mis_val = cumulative_prob(P_mis[i], true_pref, r)
                if truthful_val + tol < mis_val:
                    count += 1

    return count


def check_sd_dominance(preferences: Profile, P: np.ndarray, Q: np.ndarray, tol: float = 1e-8) -> bool:
    for i in range(len(preferences)):
        for r in range(1, len(preferences) + 1):
            fq = cumulative_prob(Q[i], preferences[i], r)
            fp = cumulative_prob(P[i], preferences[i], r)
            if fq + tol < fp:
                return False
    return True


def pick_linprog_method(requested: Optional[str] = None) -> str:
    allowed = {"highs", "highs-ds", "highs-ipm", "interior-point", "revised simplex", "simplex"}
    if requested is None:
        return "highs"
    if requested not in allowed:
        raise ValueError(f"Unsupported linprog method={requested}. Choose one of {sorted(allowed)}")
    return requested


# =========================================================
# Inner problem
# =========================================================

@dataclass
class LPResult:
    Q: np.ndarray
    q: np.ndarray
    objective: float
    status: str
    solver: str
    success: bool
    message: str
    nit: int
    A: np.ndarray
    B: np.ndarray
    s: np.ndarray


def solve_v_given_P(
    preferences: List[Preference],
    P: np.ndarray,
    solver: Optional[str] = None,
) -> LPResult:
    _check_inputs(preferences, P)

    n = len(preferences)
    A = a_matrix(preferences)
    B = b_matrix(n)
    s = s_vector(preferences)
    p = vec(P)
    b = np.ones(2 * n, dtype=float)

    c = -s.copy()
    constant_shift = float(s @ p)

    A_ub = -A
    b_ub = -(A @ p)

    A_eq = B
    b_eq = b

    bounds = [(0.0, None) for _ in range(n * n)]

    method = pick_linprog_method(solver)

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )

    if not res.success or res.x is None:
        raise RuntimeError(
            f"Inner LP failed. status={res.status}, method={method}, message={res.message}"
        )

    q_opt = np.asarray(res.x, dtype=float)
    Q_opt = unvec(q_opt, n)
    objective_value = float(s @ q_opt - constant_shift)

    return LPResult(
        Q=Q_opt,
        q=q_opt,
        objective=objective_value,
        status=str(res.status),
        solver=f"scipy.optimize.linprog(method='{method}')",
        success=bool(res.success),
        message=str(res.message),
        nit=int(getattr(res, "nit", -1)),
        A=A,
        B=B,
        s=s,
    )


# =========================================================
# Parallel helpers
# =========================================================

def _set_single_thread_env() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def solve_inner_chunk_on_representatives(
    chunk: List[Tuple[RepresentativeKey, np.ndarray]],
    solver: Optional[str] = None,
) -> List[Tuple[RepresentativeKey, np.ndarray, float, str, str, int]]:
    _set_single_thread_env()

    out: List[Tuple[RepresentativeKey, np.ndarray, float, str, str, int]] = []
    for rep_key, P in chunk:
        rep_profile = key_to_profile(rep_key)
        res = solve_v_given_P(rep_profile, P, solver=solver)
        out.append(
            (
                rep_key,
                res.q,
                float(res.objective),
                res.status,
                res.message,
                int(res.nit),
            )
        )
    return out


def separate_envy_chunk(
    chunk: List[Tuple[RepresentativeKey, np.ndarray]],
    existing_ef_cuts: Set[EFCut],
    tol: float = 1e-8,
) -> List[EFViolation]:
    _set_single_thread_env()

    new_cuts: List[EFViolation] = []
    for rep_key, P in chunk:
        rep_profile = key_to_profile(rep_key)
        n = len(rep_profile)

        for i in range(n):
            pref_i = rep_profile[i]
            for j in range(n):
                for r in range(1, n + 1):
                    own_val = cumulative_prob(P[i], pref_i, r)
                    other_val = cumulative_prob(P[j], pref_i, r)
                    violation = other_val - own_val
                    if violation > tol:
                        cut = (rep_key, i, j, r)
                        if cut not in existing_ef_cuts:
                            new_cuts.append((float(violation), cut))

    return new_cuts


def separate_strategy_chunk(
    chunk_profiles: List[Profile],
    p_map_original: Dict[ProfileKey, np.ndarray],
    one_agent_prefs: List[Preference],
    existing_sp_cuts: Set[SPCut],
    tol: float = 1e-8,
) -> List[SPViolation]:
    _set_single_thread_env()

    new_cuts: List[SPViolation] = []

    for truthful_profile in chunk_profiles:
        truthful_key = profile_to_key(truthful_profile)
        P_truth = p_map_original[truthful_key]

        for i in range(len(truthful_profile)):
            true_pref = truthful_profile[i]

            for misreport in one_agent_prefs:
                if misreport == true_pref:
                    continue

                mis_profile = [list(pref) for pref in truthful_profile]
                mis_profile[i] = list(misreport)
                mis_key = profile_to_key(mis_profile)
                P_mis = p_map_original[mis_key]

                for r in range(1, len(truthful_profile) + 1):
                    truthful_val = cumulative_prob(P_truth[i], true_pref, r)
                    mis_val = cumulative_prob(P_mis[i], true_pref, r)
                    violation = mis_val - truthful_val
                    if violation > tol:
                        cut = (truthful_key, i, tuple(misreport), r)
                        if cut not in existing_sp_cuts:
                            new_cuts.append((float(violation), cut))

    return new_cuts



def _row_from_representative(
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    iso: IsomorphismData,
    profile_key: ProfileKey,
    original_agent: int,
) -> np.ndarray:
    rep_key = iso.profile_to_rep[profile_key]
    inv_perm = iso.profile_to_inv_perm[profile_key]
    rep_row = inv_perm[original_agent]
    return p_rep_map[rep_key][rep_row]


def separate_strategy_chunk_light(
    chunk_profiles: List[Profile],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    iso: IsomorphismData,
    one_agent_prefs: List[Preference],
    existing_sp_cuts: Set[SPCut],
    tol: float = 1e-8,
) -> List[SPViolation]:
    _set_single_thread_env()

    new_cuts: List[SPViolation] = []

    for truthful_profile in chunk_profiles:
        truthful_key = profile_to_key(truthful_profile)
        n = len(truthful_profile)

        for i in range(n):
            true_pref = truthful_profile[i]
            p_truth_i = _row_from_representative(
                p_rep_map=p_rep_map,
                iso=iso,
                profile_key=truthful_key,
                original_agent=i,
            )

            for misreport in one_agent_prefs:
                if misreport == true_pref:
                    continue

                mis_profile = [list(pref) for pref in truthful_profile]
                mis_profile[i] = list(misreport)
                mis_key = profile_to_key(mis_profile)

                p_mis_i = _row_from_representative(
                    p_rep_map=p_rep_map,
                    iso=iso,
                    profile_key=mis_key,
                    original_agent=i,
                )

                for r in range(1, n + 1):
                    truthful_val = cumulative_prob(p_truth_i, true_pref, r)
                    mis_val = cumulative_prob(p_mis_i, true_pref, r)
                    violation = mis_val - truthful_val

                    if violation > tol:
                        cut = (truthful_key, i, tuple(misreport), r)
                        if cut not in existing_sp_cuts:
                            new_cuts.append((float(violation), cut))

    return new_cuts


def count_strategy_violations_at_profile_light(
    profile: Profile,
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    iso: IsomorphismData,
    one_agent_prefs: List[Preference],
    tol: float = 1e-8,
) -> int:
    truthful_key = profile_to_key(profile)
    n = len(profile)
    count = 0

    for i in range(n):
        true_pref = profile[i]
        p_truth_i = _row_from_representative(
            p_rep_map=p_rep_map,
            iso=iso,
            profile_key=truthful_key,
            original_agent=i,
        )

        for misreport in one_agent_prefs:
            if misreport == true_pref:
                continue

            mis_profile = [list(pref) for pref in profile]
            mis_profile[i] = list(misreport)
            mis_key = profile_to_key(mis_profile)

            p_mis_i = _row_from_representative(
                p_rep_map=p_rep_map,
                iso=iso,
                profile_key=mis_key,
                original_agent=i,
            )

            for r in range(1, n + 1):
                truthful_val = cumulative_prob(p_truth_i, true_pref, r)
                mis_val = cumulative_prob(p_mis_i, true_pref, r)

                if truthful_val + tol < mis_val:
                    count += 1

    return count

# =========================================================
# Master problem with lazy EF/SP/Benders cuts
# Sparse implementation
# =========================================================

@dataclass
class MasterResult:
    p_rep_map: Dict[RepresentativeKey, np.ndarray]
    t_rep_map: Dict[RepresentativeKey, float]
    objective: float
    success: bool
    status: str
    message: str
    nit: int
    solver: str


def solve_master_problem_on_representatives(
    rep_profiles: List[Profile],
    iso: IsomorphismData,
    benders_cuts: Dict[RepresentativeKey, List[np.ndarray]],
    ef_cuts: Set[EFCut],
    sp_cuts: Set[SPCut],
    solver: Optional[str] = None,
) -> MasterResult:
    if not rep_profiles:
        raise ValueError("rep_profiles must be non-empty")

    n = len(rep_profiles[0])
    rep_keys = iso.rep_keys
    m = len(rep_keys)

    rep_index = {k: idx for idx, k in enumerate(rep_keys)}

    p_block_size = n * n
    num_p_vars = m * p_block_size
    num_t_vars = m
    total_vars = num_p_vars + num_t_vars

    def p_offset(rep_idx: int) -> int:
        return rep_idx * p_block_size

    def t_offset(rep_idx: int) -> int:
        return num_p_vars + rep_idx

    c = np.zeros(total_vars, dtype=float)
    for rep_idx, rep_key in enumerate(rep_keys):
        c[t_offset(rep_idx)] = float(iso.rep_weights[rep_key])

    # ---------- A_eq sparse ----------
    eq_data: List[float] = []
    eq_rows: List[int] = []
    eq_cols: List[int] = []
    b_eq_list: List[float] = []

    eq_row_counter = 0
    for rep_idx in range(m):
        base = p_offset(rep_idx)

        for i in range(n):
            for o in range(n):
                eq_rows.append(eq_row_counter)
                eq_cols.append(base + i * n + o)
                eq_data.append(1.0)
            b_eq_list.append(1.0)
            eq_row_counter += 1

        for o in range(n):
            for i in range(n):
                eq_rows.append(eq_row_counter)
                eq_cols.append(base + i * n + o)
                eq_data.append(1.0)
            b_eq_list.append(1.0)
            eq_row_counter += 1

    A_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)),
        shape=(eq_row_counter, total_vars),
        dtype=float,
    ).tocsr()
    b_eq = np.array(b_eq_list, dtype=float)

    # ---------- A_ub sparse ----------
    ub_data: List[float] = []
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    b_ub_list: List[float] = []

    ub_row_counter = 0

    # active envy-free cuts
    for rep_key, i, j, r in ef_cuts:
        rep_idx = rep_index[rep_key]
        rep_profile = key_to_profile(rep_key)
        pref_i = rep_profile[i]
        top_r = pref_i[:r]

        base = p_offset(rep_idx)
        for o in top_r:
            ub_rows.append(ub_row_counter)
            ub_cols.append(base + i * n + o)
            ub_data.append(-1.0)

            ub_rows.append(ub_row_counter)
            ub_cols.append(base + j * n + o)
            ub_data.append(1.0)

        b_ub_list.append(0.0)
        ub_row_counter += 1

    # active strategy-proof cuts
    for truthful_key, i, misreport_tuple, r in sp_cuts:
        truthful_rep_key = iso.profile_to_rep[truthful_key]
        truthful_inv = iso.profile_to_inv_perm[truthful_key]
        truthful_rep_idx = rep_index[truthful_rep_key]
        truthful_base = p_offset(truthful_rep_idx)
        truthful_profile = key_to_profile(truthful_key)
        true_pref = truthful_profile[i]
        truthful_rep_row = truthful_inv[i]

        mis_profile = [list(pref) for pref in truthful_profile]
        mis_profile[i] = list(misreport_tuple)
        mis_key = profile_to_key(mis_profile)

        mis_rep_key = iso.profile_to_rep[mis_key]
        mis_inv = iso.profile_to_inv_perm[mis_key]
        mis_rep_idx = rep_index[mis_rep_key]
        mis_base = p_offset(mis_rep_idx)
        mis_rep_row = mis_inv[i]

        top_r = true_pref[:r]
        for o in top_r:
            ub_rows.append(ub_row_counter)
            ub_cols.append(truthful_base + truthful_rep_row * n + o)
            ub_data.append(-1.0)

            ub_rows.append(ub_row_counter)
            ub_cols.append(mis_base + mis_rep_row * n + o)
            ub_data.append(1.0)

        b_ub_list.append(0.0)
        ub_row_counter += 1

    # active Benders cuts
    for rep_idx, rep_profile in enumerate(rep_profiles):
        rep_key = rep_keys[rep_idx]
        base = p_offset(rep_idx)
        t_idx = t_offset(rep_idx)
        s_rep = s_vector(rep_profile)

        for q_cut in benders_cuts[rep_key]:
            for local_col, val in enumerate(-s_rep):
                if val != 0.0:
                    ub_rows.append(ub_row_counter)
                    ub_cols.append(base + local_col)
                    ub_data.append(float(val))

            ub_rows.append(ub_row_counter)
            ub_cols.append(t_idx)
            ub_data.append(-1.0)

            rhs = -float(s_rep @ q_cut)
            b_ub_list.append(rhs)
            ub_row_counter += 1

    if ub_row_counter > 0:
        A_ub = coo_matrix(
            (ub_data, (ub_rows, ub_cols)),
            shape=(ub_row_counter, total_vars),
            dtype=float,
        ).tocsr()
        b_ub = np.array(b_ub_list, dtype=float)
    else:
        A_ub = None
        b_ub = None

    bounds = [(0.0, None) for _ in range(num_p_vars)] + [(0.0, None) for _ in range(num_t_vars)]

    method = pick_linprog_method(solver)
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )

    if not res.success or res.x is None:
        raise RuntimeError(
            f"Master LP failed. status={res.status}, method={method}, message={res.message}"
        )

    x = np.asarray(res.x, dtype=float)
    p_rep_map: Dict[RepresentativeKey, np.ndarray] = {}
    t_rep_map: Dict[RepresentativeKey, float] = {}

    for rep_idx, rep_key in enumerate(rep_keys):
        base = p_offset(rep_idx)
        p_rep_map[rep_key] = unvec(x[base: base + p_block_size], n)
        t_rep_map[rep_key] = float(x[t_offset(rep_idx)])

    return MasterResult(
        p_rep_map=p_rep_map,
        t_rep_map=t_rep_map,
        objective=float(res.fun),
        success=bool(res.success),
        status=str(res.status),
        message=str(res.message),
        nit=int(getattr(res, "nit", -1)),
        solver=f"scipy.optimize.linprog(method='{method}')",
    )