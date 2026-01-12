import numpy as np
import itertools
import mip
from collections import deque

# =========================================================
# Preference orders / ranks / weights
# =========================================================
def all_orders(n: int):
    return list(itertools.permutations(range(n)))

def build_rank_w(orders):
    """
    orders[r] = tuple items best..worst
    rank[r,a] = position of item a in that order
    w[r,a]    = n - rank[r,a]  (triple-sum weight aggregation)
    """
    m = len(orders)
    n = len(orders[0])
    rank = np.zeros((m, n), dtype=int)
    w = np.zeros((m, n), dtype=float)
    for r, ord_tup in enumerate(orders):
        for pos, a in enumerate(ord_tup):
            rank[r, a] = pos
        for a in range(n):
            w[r, a] = float(n - rank[r, a])
    return rank, w

def prefix_sum_num(P_num, i: int, order_tup, t: int):
    return mip.xsum(P_num[i][order_tup[k]] for k in range(t + 1))

# =========================================================
# Profile-set management (closure under unilateral deviations)
# =========================================================
def closure_unilateral(prof_set, n: int, m: int, max_profiles: int):
    q = deque(list(prof_set))
    while q and len(prof_set) < max_profiles:
        prof = q.popleft()
        for i in range(n):
            for mis_idx in range(m):
                if mis_idx == prof[i]:
                    continue
                mis = list(prof)
                mis[i] = mis_idx
                mis = tuple(mis)
                if mis not in prof_set:
                    prof_set.add(mis)
                    q.append(mis)
                    if len(prof_set) >= max_profiles:
                        break
            if len(prof_set) >= max_profiles:
                break
    return prof_set

def random_profiles(n: int, m: int, k: int, rng: np.random.Generator):
    return [tuple(int(rng.integers(0, m)) for _ in range(n)) for _ in range(k)]

# =========================================================
# Step1: Build EF=0 & SP=0 mechanism on given profile set S (feasible MIP)
#   - integer grid P_num(profile) with DS constraints
#   - EF0 and SP0 as exact prefix inequalities
# =========================================================
def build_mechanism_EFSP_feasible(
    n: int,
    profiles,
    H: int,
    rng: np.random.Generator,
    verbose: bool = False,
):
    orders = all_orders(n)
    m = len(orders)

    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 1 if verbose else 0
    # memory-saver
    model.cuts = 0
    model.cut_passes = 0
    model.clique = 0

    # variables: P_num[profile][i][a] in {0..H}
    P_num = {}
    for prof in profiles:
        P = [[model.add_var(var_type=mip.INTEGER, lb=0, ub=H) for a in range(n)] for i in range(n)]
        P_num[prof] = P
        # DS on integer grid
        for i in range(n):
            model += mip.xsum(P[i][a] for a in range(n)) == H
        for a in range(n):
            model += mip.xsum(P[i][a] for i in range(n)) == H

    # EF=0 on each profile: prefix_i(P_i) >= prefix_i(P_j)
    for prof in profiles:
        P = P_num[prof]
        for i in range(n):
            pref_i = orders[prof[i]]
            for j in range(n):
                if i == j:
                    continue
                for t in range(n):
                    lhs = prefix_sum_num(P, i, pref_i, t)
                    rhs = prefix_sum_num(P, j, pref_i, t)
                    model += lhs - rhs >= 0

    # SP=0: for each true profile, compare with unilateral misreports (must be in profiles)
    prof_set = set(profiles)
    for true_prof in profiles:
        P_true = P_num[true_prof]
        for i in range(n):
            true_pref_i = orders[true_prof[i]]
            for mis_idx in range(m):
                if mis_idx == true_prof[i]:
                    continue
                mis_prof = list(true_prof)
                mis_prof[i] = mis_idx
                mis_prof = tuple(mis_prof)
                if mis_prof not in prof_set:
                    continue
                P_mis = P_num[mis_prof]
                for t in range(n):
                    lhs = prefix_sum_num(P_true, i, true_pref_i, t)
                    rhs = prefix_sum_num(P_mis, i, true_pref_i, t)
                    model += lhs - rhs >= 0

    # random linear objective to pick a particular feasible mechanism (still EF/SP feasible)
    obj = mip.xsum(
        float(rng.standard_normal()) * P_num[prof][i][a]
        for prof in profiles for i in range(n) for a in range(n)
    )
    model.objective = mip.maximize(obj)
    model.optimize()
    return model.status, model, P_num

# =========================================================
# Step2: "argmin(s)"  = SD-dominator polytope D(P)
#        We compute:
#          min_s = min_{P' DS} s(P',P)
#        where s(P',P)=Σ_{i,t}(|d|-d), d=prefix(P')-prefix(P)
#        Note: s(P',P)=0 <=> all d>=0 <=> SD dominance.
# =========================================================
def solve_min_s_lp(P: np.ndarray, prof, orders, verbose: bool = False):
    n = P.shape[0]
    lp = mip.Model(solver_name=mip.CBC)
    lp.verbose = 1 if verbose else 0
    lp.cuts = 0
    lp.cut_passes = 0
    lp.clique = 0

    X = [[lp.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS) for a in range(n)] for i in range(n)]
    for i in range(n):
        lp += mip.xsum(X[i][a] for a in range(n)) == 1.0
    for a in range(n):
        lp += mip.xsum(X[i][a] for i in range(n)) == 1.0

    s_terms = []
    for i in range(n):
        pref_i = orders[prof[i]]
        for t in range(n):
            d = lp.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
            uabs = lp.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
            prefix_p = float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]]))
            expr = mip.xsum(X[i][pref_i[k]] for k in range(t + 1)) - prefix_p
            lp += d == expr
            lp += uabs >= d
            lp += uabs >= -d
            s_terms.append(uabs - d)

    s_expr = mip.xsum(s_terms)
    lp.objective = mip.minimize(s_expr)
    lp.optimize()

    if lp.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        return float("inf")
    return float(lp.objective_value)

def find_one_dominator(P: np.ndarray, prof, orders, verbose: bool = False):
    """
    Find one feasible point in D(P):
      X doubly stochastic and prefix_X >= prefix_P for all i,t.
    If infeasible -> return None
    """
    n = P.shape[0]
    lp = mip.Model(solver_name=mip.CBC)
    lp.verbose = 1 if verbose else 0
    lp.cuts = 0
    lp.cut_passes = 0
    lp.clique = 0

    X = [[lp.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS) for a in range(n)] for i in range(n)]
    for i in range(n):
        lp += mip.xsum(X[i][a] for a in range(n)) == 1.0
    for a in range(n):
        lp += mip.xsum(X[i][a] for i in range(n)) == 1.0

    for i in range(n):
        pref_i = orders[prof[i]]
        for t in range(n):
            prefix_x = mip.xsum(X[i][pref_i[k]] for k in range(t + 1))
            prefix_p = float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]]))
            lp += prefix_x >= prefix_p

    # choose a "less extreme" point: minimize L1 distance to uniform U (LP)
    U = 1.0 / float(n)
    z = [[lp.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS) for a in range(n)] for i in range(n)]
    for i in range(n):
        for a in range(n):
            lp += z[i][a] >= X[i][a] - U
            lp += z[i][a] >= -(X[i][a] - U)
    lp.objective = mip.minimize(mip.xsum(z[i][a] for i in range(n) for a in range(n)))
    lp.optimize()

    if lp.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        return None
    return np.array([[float(X[i][a].x) for a in range(n)] for i in range(n)], dtype=float)

# =========================================================
# Hit-and-Run sampler on polytope:
#   x in R^{n^2}
#   Aeq x = beq  (row/col sums)
#   inequalities:
#     0 <= x_j <= 1
#     dominance: a^T x >= b
# We sample approximately uniform in the feasible region.
# =========================================================
def nullspace(A, rtol=1e-12):
    # returns basis N where columns span null(A)
    # via SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(S > rtol * S[0]) if S.size > 0 else 0
    N = Vt[rank:].T
    return N

def hit_and_run(
    x0: np.ndarray,
    Aeq: np.ndarray,
    beq: np.ndarray,
    Aineq: np.ndarray,  # a^T x >= b
    bineq: np.ndarray,
    n_samples: int = 200,
    burnin: int = 200,
    thinning: int = 5,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng(0)

    x = x0.copy()
    d = x.size

    # build nullspace basis for equality constraints
    N = nullspace(Aeq)
    if N.size == 0:
        # unique point
        return np.repeat(x[None, :], n_samples, axis=0)

    samples = []
    total_steps = burnin + n_samples * thinning

    for step in range(total_steps):
        # random direction in nullspace
        z = rng.standard_normal(N.shape[1])
        dirv = N @ z
        norm = np.linalg.norm(dirv)
        if norm < 1e-12:
            continue
        dirv /= norm

        # compute alpha interval from box constraints 0<=x+αd<=1
        alpha_min = -np.inf
        alpha_max = np.inf
        for j in range(d):
            dj = dirv[j]
            if abs(dj) < 1e-15:
                continue
            # lower bound 0
            a1 = (0.0 - x[j]) / dj
            # upper bound 1
            a2 = (1.0 - x[j]) / dj
            lo = min(a1, a2)
            hi = max(a1, a2)
            alpha_min = max(alpha_min, lo)
            alpha_max = min(alpha_max, hi)

        # dominance inequalities a^T(x+αd) >= b
        # => α bounds depending on sign of a^T d
        Ax = Aineq @ x
        Ad = Aineq @ dirv
        for r in range(Aineq.shape[0]):
            den = Ad[r]
            num = bineq[r] - Ax[r]
            if abs(den) < 1e-15:
                # constraint doesn't move along this direction
                if num > 1e-12:
                    # would violate; but should not happen if x feasible
                    alpha_min, alpha_max = 1.0, 0.0
                    break
                continue
            bound = num / den
            if den > 0:
                alpha_min = max(alpha_min, bound)
            else:
                alpha_max = min(alpha_max, bound)

        if not (alpha_min <= alpha_max):
            continue

        # sample uniformly on feasible segment
        alpha = rng.uniform(alpha_min, alpha_max)
        x = x + alpha * dirv

        # numerical cleanup
        x = np.clip(x, 0.0, 1.0)
        # project (tiny) onto equality constraints to reduce drift (least squares)
        # x <- argmin ||x-y|| s.t. Aeq x = beq
        # = y - Aeq^T (Aeq Aeq^T)^-1 (Aeq y - beq)
        r = Aeq @ x - beq
        if np.linalg.norm(r) > 1e-10:
            M = Aeq @ Aeq.T
            try:
                lam = np.linalg.solve(M, r)
                x = x - Aeq.T @ lam
                x = np.clip(x, 0.0, 1.0)
            except np.linalg.LinAlgError:
                pass

        if step >= burnin and ((step - burnin) % thinning == 0):
            samples.append(x.copy())
            if len(samples) >= n_samples:
                break

    if len(samples) == 0:
        return x0[None, :]
    return np.stack(samples, axis=0)

# =========================================================
# "Integral over argmin(s)" objective for OE (Monte Carlo)
#   If min_s > tol -> indicator=0 -> infringement 0
#   Else:
#     P' ~ approx Unif(D(P))
#     infr = E[ Σ w*(P' - P) ] + lam_pairs
# =========================================================
def oe_infr_argmin_average(
    P: np.ndarray,
    prof,
    orders,
    w,
    lam: float,
    tol: float = 1e-10,
    n_mc: int = 200,
    burnin: int = 200,
    thinning: int = 5,
    rng: np.random.Generator = None,
    verbose_lp: bool = False,
):
    if rng is None:
        rng = np.random.default_rng(0)

    n = P.shape[0]
    base_pairs = n * (n * (n + 1) / 2.0)
    lam_pairs = float(lam) * float(base_pairs)

    min_s = solve_min_s_lp(P, prof, orders, verbose=verbose_lp)
    if min_s > tol:
        return 0.0, float(min_s), 0  # indicator=0

    # Get one feasible dominator point x0 in D(P)
    P0 = find_one_dominator(P, prof, orders, verbose=verbose_lp)
    if P0 is None:
        # numerical fallback: treat as no dominator
        return 0.0, float(min_s), 0

    # Build equality constraints Aeq x = beq for DS
    # vectorize x in row-major (i,a) -> i*n + a
    d = n * n
    eqs = []
    rhs = []
    # row sums
    for i in range(n):
        row = np.zeros(d)
        for a in range(n):
            row[i * n + a] = 1.0
        eqs.append(row); rhs.append(1.0)
    # col sums (drop last to avoid redundancy)
    for a in range(n - 1):
        col = np.zeros(d)
        for i in range(n):
            col[i * n + a] = 1.0
        eqs.append(col); rhs.append(1.0)
    Aeq = np.vstack(eqs)
    beq = np.array(rhs)

    # Build dominance inequalities Aineq x >= bineq
    A_list = []
    b_list = []
    for i in range(n):
        pref_i = orders[prof[i]]
        for t in range(n):
            arow = np.zeros(d)
            for k in range(t + 1):
                a = pref_i[k]
                arow[i * n + a] += 1.0
            A_list.append(arow)
            b_list.append(float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]])))
    Aineq = np.vstack(A_list) if len(A_list) else np.zeros((0, d))
    bineq = np.array(b_list) if len(b_list) else np.zeros((0,))

    x0 = P0.reshape(-1)
    sam = hit_and_run(
        x0=x0,
        Aeq=Aeq,
        beq=beq,
        Aineq=Aineq,
        bineq=bineq,
        n_samples=n_mc,
        burnin=burnin,
        thinning=thinning,
        rng=rng,
    )

    # Compute E[ Σ w*(P' - P) ]
    const = 0.0
    for i in range(n):
        pref_idx = prof[i]
        for a in range(n):
            const += float(w[pref_idx, a]) * float(P[i, a])

    vals = []
    for svec in sam:
        X = svec.reshape(n, n)
        v = 0.0
        for i in range(n):
            pref_idx = prof[i]
            for a in range(n):
                v += float(w[pref_idx, a]) * float(X[i, a])
        vals.append(v - const)

    avg_improvement = float(np.mean(vals))
    infr = avg_improvement + lam_pairs
    return float(infr), float(min_s), 1

# =========================================================
# Step3: Randomized search (fast-ish)
#   - build EF/SP feasible mechanism on S
#   - evaluate avg OE infringement by argmin-average integral (MC)
# =========================================================
def search_n4_fast_argmin_average(
    H: int = 40,
    lam: float = 1e-6,
    seed_profiles: int = 8,
    max_profiles: int = 60,
    trials: int = 10,
    rng_seed: int = 1,
    verbose_build: bool = False,
    verbose_eval_lp: bool = False,
    # MC knobs
    n_mc: int = 120,
    burnin: int = 150,
    thinning: int = 5,
):
    n = 4
    orders = all_orders(n)
    m = len(orders)
    _, w = build_rank_w(orders)

    rng = np.random.default_rng(rng_seed)
    best = None

    for tr in range(trials):
        # Build a small closed-under-deviations set S
        prof_set = set(random_profiles(n, m, seed_profiles, rng))
        prof_set = closure_unilateral(prof_set, n=n, m=m, max_profiles=max_profiles)
        profiles = list(prof_set)

        status, _, P_num = build_mechanism_EFSP_feasible(
            n=n,
            profiles=profiles,
            H=H,
            rng=rng,
            verbose=verbose_build,
        )
        print(f"[trial {tr}] |S|={len(profiles)} status={status}")
        if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            continue

        # Extract P as float
        P_float = {}
        for prof in profiles:
            P = np.zeros((n, n), dtype=float)
            for i in range(n):
                for a in range(n):
                    P[i, a] = float(P_num[prof][i][a].x) / float(H)
            P_float[prof] = P

        infr_list = []
        dom_flag = []
        min_s_list = []
        for prof in profiles:
            infr, min_s, has = oe_infr_argmin_average(
                P_float[prof],
                prof,
                orders,
                w,
                lam=lam,
                tol=1e-10,
                n_mc=n_mc,
                burnin=burnin,
                thinning=thinning,
                rng=rng,
                verbose_lp=verbose_eval_lp,
            )
            infr_list.append(infr)
            min_s_list.append(min_s)
            dom_flag.append(has)

        avg_infr = float(np.mean(infr_list))
        dom_frac = float(np.mean(dom_flag))
        avg_min_s = float(np.mean(min_s_list))

        print(f"          avg_infr={avg_infr:.6e}  dom_frac={dom_frac:.3f}  avg_min_s={avg_min_s:.3e}")

        if best is None or avg_infr < best["avg_infr"]:
            best = {
                "trial": tr,
                "profiles": profiles,
                "H": H,
                "avg_infr": avg_infr,
                "dom_frac": dom_frac,
                "avg_min_s": avg_min_s,
                "P_float": P_float,
                "status": status,
                "mc": (n_mc, burnin, thinning),
            }

    return best

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    best = search_n4_fast_argmin_average(
        H=40,
        lam=1e-6,
        seed_profiles=8,
        max_profiles=60,
        trials=10,
        rng_seed=1,
        verbose_build=False,
        verbose_eval_lp=False,
        # MC (軽め設定)
        n_mc=120,
        burnin=150,
        thinning=5,
    )

    print("\n=== Best ===")
    if best is None:
        print("No feasible EF=0 & SP=0 mechanism found on sampled profile sets.")
    else:
        print(f"trial={best['trial']} |S|={len(best['profiles'])} H={best['H']} mc={best['mc']}")
        print(f"avg_infr={best['avg_infr']:.6e}  dom_frac={best['dom_frac']:.3f}  avg_min_s={best['avg_min_s']:.3e}")

        p0 = best["profiles"][0]
        print("\nExample profile (order indices):")
        print(p0)
        print("P(profile):")
        print(best["P_float"][p0])
