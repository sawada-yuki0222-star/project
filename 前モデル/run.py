from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, List, Set, Tuple

import numpy as np

from 前モデル.linear_problem import (
    EFCut,
    SPCut,
    EFViolation,
    SPViolation,
    Profile,
    ProfileKey,
    RepresentativeKey,
    all_preference_profiles,
    all_preferences,
    build_agent_isomorphism_data,
    check_sd_dominance,
    count_envy_violations,
    count_profiles,
    count_strategy_violations_at_profile_light,
    map_representative_matrix_to_original,
    profile_to_key,
    s_matrix,
    separate_envy_chunk,
    separate_strategy_chunk_light,
    solve_inner_chunk_on_representatives,
    solve_master_problem_on_representatives,
    uniform_assignment_matrix,
)


def matrix_to_jsonable(X: np.ndarray, digits: int = 12) -> List[List[float]]:
    return [[round(float(v), digits) for v in row] for row in X.tolist()]


def matrix_to_string(X: np.ndarray, digits: int = 12) -> str:
    return json.dumps(matrix_to_jsonable(X, digits=digits), ensure_ascii=False)


def profile_to_string(profile: Profile) -> str:
    return json.dumps(profile, ensure_ascii=False)


def print_matrix(name: str, X: np.ndarray, digits: int = 6) -> None:
    print(f"{name}:")
    for row in X:
        print("  ".join(f"{v:.{digits}f}" for v in row))
    print()


def print_profile(profile: Profile) -> None:
    print("preferences:")
    for i, pref in enumerate(profile):
        print(f"  agent {i}: {pref}")
    print()


def write_csv_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def build_original_p_map_from_representatives(
    all_profiles: List[Profile],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    iso,
) -> Dict[ProfileKey, np.ndarray]:
    p_orig_map: Dict[ProfileKey, np.ndarray] = {}

    for profile in all_profiles:
        key = profile_to_key(profile)
        rep_key = iso.profile_to_rep[key]
        inv_perm = iso.profile_to_inv_perm[key]
        P_rep = p_rep_map[rep_key]
        P_orig = map_representative_matrix_to_original(P_rep, inv_perm)
        p_orig_map[key] = P_orig

    return p_orig_map


def build_rep_chunks(
    rep_keys: List[RepresentativeKey],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    chunk_size: int,
) -> List[List[Tuple[RepresentativeKey, np.ndarray]]]:
    items = [(rep_key, p_rep_map[rep_key]) for rep_key in rep_keys]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_profile_chunks(
    all_profiles: List[Profile],
    chunk_size: int,
) -> List[List[Profile]]:
    return [all_profiles[i:i + chunk_size] for i in range(0, len(all_profiles), chunk_size)]


def solve_all_rep_inner_parallel(
    rep_keys: List[RepresentativeKey],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    max_workers: int,
    chunk_size: int,
    mp_start_method: str = "spawn",
) -> Dict[RepresentativeKey, Tuple[np.ndarray, float, str, str, int]]:
    chunks = build_rep_chunks(rep_keys, p_rep_map, chunk_size=chunk_size)
    results: Dict[RepresentativeKey, Tuple[np.ndarray, float, str, str, int]] = {}

    ctx = get_context(mp_start_method)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(solve_inner_chunk_on_representatives, chunk, None) for chunk in chunks]

        completed = 0
        total = len(futures)

        for fut in as_completed(futures):
            batch = fut.result()
            for rep_key, q_opt, obj, status, message, nit in batch:
                results[rep_key] = (q_opt, obj, status, message, nit)

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  representative inner chunks solved: {completed}/{total}")

    return results


def separate_all_ef_parallel(
    rep_keys: List[RepresentativeKey],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    existing_ef_cuts: Set[EFCut],
    max_workers: int,
    chunk_size: int,
    mp_start_method: str = "spawn",
    tol: float = 1e-8,
) -> List[EFViolation]:
    chunks = build_rep_chunks(rep_keys, p_rep_map, chunk_size=chunk_size)
    new_cuts: List[EFViolation] = []

    ctx = get_context(mp_start_method)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(separate_envy_chunk, chunk, existing_ef_cuts, tol) for chunk in chunks]

        completed = 0
        total = len(futures)

        for fut in as_completed(futures):
            batch = fut.result()
            new_cuts.extend(batch)

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  EF separation chunks solved: {completed}/{total}")

    dedup = {}
    for violation, cut in new_cuts:
        if cut not in dedup or violation > dedup[cut]:
            dedup[cut] = violation

    out = [(v, cut) for cut, v in dedup.items()]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def separate_all_sp_parallel(
    all_profiles: List[Profile],
    p_rep_map: Dict[RepresentativeKey, np.ndarray],
    iso,
    one_agent_prefs: List[List[int]],
    existing_sp_cuts: Set[SPCut],
    max_workers: int,
    chunk_size: int,
    mp_start_method: str = "spawn",
    tol: float = 1e-8,
) -> List[SPViolation]:
    chunks = build_profile_chunks(all_profiles, chunk_size=chunk_size)
    new_cuts: List[SPViolation] = []

    ctx = get_context(mp_start_method)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                separate_strategy_chunk_light,
                chunk,
                p_rep_map,
                iso,
                one_agent_prefs,
                existing_sp_cuts,
                tol,
            )
            for chunk in chunks
        ]

        completed = 0
        total = len(futures)

        for fut in as_completed(futures):
            batch = fut.result()
            new_cuts.extend(batch)

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  SP separation chunks solved: {completed}/{total}")

    dedup = {}
    for violation, cut in new_cuts:
        if cut not in dedup or violation > dedup[cut]:
            dedup[cut] = violation

    out = [(v, cut) for cut, v in dedup.items()]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def take_top_violations(
    violations,
    max_new_cuts_per_iteration: int | None,
):
    if max_new_cuts_per_iteration is None:
        return violations
    return violations[:max_new_cuts_per_iteration]


def main() -> None:
    n = 4
    tol = 1e-9
    max_iterations = 100
    output_csv = f"benders_agent_isomorphism_cutting_plane_parallel_all_sparse_capped_n{n}.csv"

    # parallel settings
    cpu_count = os.cpu_count() or 4
    max_workers = min(2, cpu_count)
    rep_chunk_size = 32
    profile_chunk_size = 64
    mp_start_method = "fork" if sys.platform != "win32" else "spawn"

    # cap settings
    max_new_ef_cuts_per_iteration = 2000
    max_new_sp_cuts_per_iteration = 5000

    all_profiles = all_preference_profiles(n)
    one_agent_prefs = all_preferences(n)
    total_profiles = count_profiles(n)

    print(f"n = {n}")
    print(f"number of original profiles = {total_profiles}")

    iso = build_agent_isomorphism_data(all_profiles)
    rep_profiles = iso.representatives
    rep_keys = iso.rep_keys

    print(f"number of representative profiles (agents only) = {len(rep_profiles)}")
    print(f"parallel workers = {max_workers}")
    print(f"rep chunk size = {rep_chunk_size}")
    print(f"profile chunk size = {profile_chunk_size}")
    print(f"max_new_ef_cuts_per_iteration = {max_new_ef_cuts_per_iteration}")
    print(f"max_new_sp_cuts_per_iteration = {max_new_sp_cuts_per_iteration}")
    print(f"CSV output file = {output_csv}")
    print()

    p_rep_current: Dict[RepresentativeKey, np.ndarray] = {
        rep_key: uniform_assignment_matrix(n).copy()
        for rep_key in rep_keys
    }

    benders_cuts: Dict[RepresentativeKey, List[np.ndarray]] = {rep_key: [] for rep_key in rep_keys}
    ef_cuts: Set[EFCut] = set()
    sp_cuts: Set[SPCut] = set()

    print("initializing Benders cuts on representatives in parallel ...")
    init_inner = solve_all_rep_inner_parallel(
        rep_keys=rep_keys,
        p_rep_map=p_rep_current,
        max_workers=max_workers,
        chunk_size=rep_chunk_size,
        mp_start_method=mp_start_method,
    )
    for rep_key, (q_opt, obj, status, message, nit) in init_inner.items():
        benders_cuts[rep_key].append(q_opt.copy())
    print("initial Benders cuts initialized.")
    print()

    master_result = None
    last_inner_results: Dict[RepresentativeKey, Tuple[np.ndarray, float, str, str, int]] = {}

    for iteration in range(1, max_iterations + 1):
        print(f"===== iteration {iteration} =====")
        print(
            f"active cuts: "
            f"Benders={sum(len(v) for v in benders_cuts.values())}, "
            f"EF={len(ef_cuts)}, SP={len(sp_cuts)}"
        )

        master_result = solve_master_problem_on_representatives(
            rep_profiles=rep_profiles,
            iso=iso,
            benders_cuts=benders_cuts,
            ef_cuts=ef_cuts,
            sp_cuts=sp_cuts,
            solver=None,
        )

        p_rep_current = master_result.p_rep_map
        t_rep_current = master_result.t_rep_map

        print(f"master success: {master_result.success}")
        print(f"master status: {master_result.status}")
        print(f"master solver: {master_result.solver}")
        print(f"master message: {master_result.message}")
        print(f"master objective = weighted sum t^rep = {master_result.objective:.12f}")


        print("solving representative inner LPs in parallel ...")
        inner_results = solve_all_rep_inner_parallel(
            rep_keys=rep_keys,
            p_rep_map=p_rep_current,
            max_workers=max_workers,
            chunk_size=rep_chunk_size,
            mp_start_method=mp_start_method,
        )
        last_inner_results = inner_results

        new_benders = 0
        max_gap = 0.0

        for rep_key, (q_opt, obj, status, message, nit) in inner_results.items():
            rhs = float(t_rep_current[rep_key])
            gap = float(obj) - rhs
            max_gap = max(max_gap, gap)

            if gap > tol:
                benders_cuts[rep_key].append(q_opt.copy())
                new_benders += 1

        print("separating EF violations in parallel ...")
        ef_violations = separate_all_ef_parallel(
            rep_keys=rep_keys,
            p_rep_map=p_rep_current,
            existing_ef_cuts=ef_cuts,
            max_workers=max_workers,
            chunk_size=rep_chunk_size,
            mp_start_method=mp_start_method,
            tol=tol,
        )
        selected_ef_violations = take_top_violations(
            ef_violations,
            max_new_ef_cuts_per_iteration,
        )
        new_ef_cuts = [cut for _, cut in selected_ef_violations]
        for cut in new_ef_cuts:
            ef_cuts.add(cut)

        print("separating SP violations in parallel ...")
        sp_violations = separate_all_sp_parallel(
            all_profiles=all_profiles,
            p_rep_map=p_rep_current,
            iso=iso,
            one_agent_prefs=one_agent_prefs,
            existing_sp_cuts=sp_cuts,
            max_workers=max_workers,
            chunk_size=profile_chunk_size,
            mp_start_method=mp_start_method,
            tol=tol,
        )
        selected_sp_violations = take_top_violations(
            sp_violations,
            max_new_sp_cuts_per_iteration,
        )
        new_sp_cuts = [cut for _, cut in selected_sp_violations]
        for cut in new_sp_cuts:
            sp_cuts.add(cut)

        print(f"iteration {iteration}: new Benders cuts = {new_benders}")
        print(f"iteration {iteration}: EF violations found = {len(ef_violations)}")
        print(f"iteration {iteration}: new EF cuts added = {len(new_ef_cuts)}")
        print(f"iteration {iteration}: SP violations found = {len(sp_violations)}")
        print(f"iteration {iteration}: new SP cuts added = {len(new_sp_cuts)}")
        print(f"iteration {iteration}: max Benders gap = {max_gap:.12e}")

        if selected_ef_violations:
            print(f"iteration {iteration}: max EF violation = {selected_ef_violations[0][0]:.12e}")
        if selected_sp_violations:
            print(f"iteration {iteration}: max SP violation = {selected_sp_violations[0][0]:.12e}")
        print()

        if new_benders == 0 and len(ef_violations) == 0 and len(sp_violations) == 0:
            print("No violated cuts found. Algorithm terminated.")
            break
    else:
        print("Reached max_iterations before convergence.")
        print()

    if master_result is None:
        raise RuntimeError("master_result is None")


    fieldnames = [
        "profile_id",
        "preferences",
        "representative_profile",
        "representative_weight",
        "vP",
        "t_rep",
        "sd_ok",
        "direct_value",
        "is_sd_inefficient",
        "envy_violation_count",
        "envy_violation_is_zero",
        "strategy_violation_count",
        "strategy_violation_is_zero",
        "P",
        "Q",
    ]

    num_sd_inefficient = 0
    num_sd_efficient = 0
    num_envy_violation_zero = 0
    num_strategy_violation_zero = 0

    best_record = None

    if not last_inner_results:
        print("final representative inner pass in parallel ...")
        last_inner_results = solve_all_rep_inner_parallel(
            rep_keys=rep_keys,
            p_rep_map=p_rep_current,
            max_workers=max_workers,
            chunk_size=rep_chunk_size,
            mp_start_method=mp_start_method,
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        write_csv_header(writer)

        for profile_id, profile in enumerate(all_profiles, start=1):
            key = profile_to_key(profile)
            rep_key = iso.profile_to_rep[key]
            rep_profile = [list(pref) for pref in rep_key]
            rep_weight = iso.rep_weights[rep_key]
            inv_perm = iso.profile_to_inv_perm[key]

            P_rep = p_rep_current[rep_key]
            P_orig = map_representative_matrix_to_original(P_rep, inv_perm)
            t_rep = master_result.t_rep_map[rep_key]

            q_opt, v_theta, status, message, nit = last_inner_results[rep_key]
            n_local = len(profile)
            Q_rep = np.asarray(q_opt, dtype=float).reshape((n_local, n_local), order="C")
            Q_orig = map_representative_matrix_to_original(Q_rep, inv_perm)

            sd_ok = check_sd_dominance(profile, P_orig, Q_orig, tol=tol)

            S = s_matrix(profile)
            direct_value = float(np.sum(S * (Q_orig - P_orig)))

            envy_violation_count = count_envy_violations(profile, P_orig, tol=tol)
            envy_violation_is_zero = (envy_violation_count == 0)

            strategy_violation_count = count_strategy_violations_at_profile_light(
                profile=profile,
                p_rep_map=p_rep_current,
                iso=iso,
                one_agent_prefs=one_agent_prefs,
                tol=tol,
            )
            strategy_violation_is_zero = (strategy_violation_count == 0)

            is_sd_inefficient = (float(v_theta) > tol)

            if is_sd_inefficient:
                num_sd_inefficient += 1
            else:
                num_sd_efficient += 1

            if envy_violation_is_zero:
                num_envy_violation_zero += 1

            if strategy_violation_is_zero:
                num_strategy_violation_zero += 1

            record = {
                "profile_id": profile_id,
                "preferences": profile_to_string(profile),
                "representative_profile": profile_to_string(rep_profile),
                "representative_weight": rep_weight,
                "vP": f"{float(v_theta):.12f}",
                "t_rep": f"{float(t_rep):.12f}",
                "sd_ok": sd_ok,
                "direct_value": f"{direct_value:.12f}",
                "is_sd_inefficient": is_sd_inefficient,
                "envy_violation_count": envy_violation_count,
                "envy_violation_is_zero": envy_violation_is_zero,
                "strategy_violation_count": strategy_violation_count,
                "strategy_violation_is_zero": strategy_violation_is_zero,
                "P": matrix_to_string(P_orig),
                "Q": matrix_to_string(Q_orig),
            }

            writer.writerow(record)

            if best_record is None or float(v_theta) > float(best_record["vP"]):
                best_record = {
                    **record,
                    "P_matrix": P_orig,
                    "Q_matrix": Q_orig,
                    "profile_obj": profile,
                }

            if profile_id % 5000 == 0 or profile_id == len(all_profiles):
                print(f"wrote {profile_id}/{len(all_profiles)} original profiles to CSV")

    print()
    print("===== summary =====")
    print(f"original profiles processed: {len(all_profiles)}")
    print(f"representatives used: {len(rep_profiles)}")
    print(f"sd-inefficient profiles (v(P) > {tol}): {num_sd_inefficient}")
    print(f"sd-efficient profiles   (v(P) <= {tol}): {num_sd_efficient}")
    print(f"envy violation count == 0 profiles: {num_envy_violation_zero}")
    print(f"strategy violation count == 0 profiles: {num_strategy_violation_zero}")
    print(f"results written to: {output_csv}")
    print()

    if best_record is not None:
        print("===== best profile (largest v(P)) =====")
        print(f"profile_id: {best_record['profile_id']}")
        print_profile(best_record["profile_obj"])
        print(f"representative_weight: {best_record['representative_weight']}")
        print(f"v(P) = {best_record['vP']}")
        print(f"t_rep = {best_record['t_rep']}")
        print(f"envy_violation_count: {best_record['envy_violation_count']}")
        print(f"envy_violation_is_zero: {best_record['envy_violation_is_zero']}")
        print(f"strategy_violation_count: {best_record['strategy_violation_count']}")
        print(f"strategy_violation_is_zero: {best_record['strategy_violation_is_zero']}")
        print()
        print_matrix("Best-profile P", best_record["P_matrix"])
        print_matrix("Best-profile Q", best_record["Q_matrix"])


if __name__ == "__main__":
    main()