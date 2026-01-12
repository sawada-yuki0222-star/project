import numpy as np
import itertools
import mip


# =========================
# Serial Dictatorship (deterministic)
# =========================
def serial_dictatorship_allocation(prefs, priority):
    """
    prefs[i] : tuple/list of items (best..worst), length n
    priority : list/tuple of agents in dictator order, length n

    return P (n x n) deterministic assignment matrix (0/1).
    """
    n = len(prefs)
    remaining = set(range(n))
    P = np.zeros((n, n), dtype=float)

    for i in priority:
        for a in prefs[i]:
            if a in remaining:
                P[i, a] = 1.0
                remaining.remove(a)
                break
    return P


# =========================
# Weights matching triple-sum aggregation
#   w_i(a) = n - rank_i(a)  (best=n, worst=1)
# =========================
def weights_from_prefs(prefs):
    n = len(prefs)
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        rank = {a: r for r, a in enumerate(prefs[i])}  # 0 best
        for a in range(n):
            w[i, a] = float(n - rank[a])
    return w


# =========================================================
# (For indicator only) delta_star via LP:
#   max_{X in DS, X SD-dominates P} sum_{i,a} w_i(a)*(X_{i,a}-P_{i,a})
#   -> >0 なら strict dominator が存在
# =========================================================
def compute_delta_star_lp(P, prefs, w, solver_name=mip.CBC, verbose=False):
    n = P.shape[0]
    model = mip.Model(solver_name=solver_name)
    model.verbose = 1 if verbose else 0

    if solver_name == mip.CBC:
        model.cuts = 0
        model.cut_passes = 0
        model.clique = 0

    X = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS) for a in range(n)] for i in range(n)]

    # DS
    for i in range(n):
        model += mip.xsum(X[i][a] for a in range(n)) == 1.0
    for a in range(n):
        model += mip.xsum(X[i][a] for i in range(n)) == 1.0

    # SD-dominance (weak)
    for i in range(n):
        pref_i = prefs[i]
        for t in range(n):
            prefix_X = mip.xsum(X[i][pref_i[k]] for k in range(t + 1))
            prefix_P = float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]]))
            model += prefix_X >= prefix_P

    obj_X = mip.xsum(float(w[i, a]) * X[i][a] for i in range(n) for a in range(n))
    const = float(np.sum(w * P))
    model.objective = mip.maximize(obj_X - const)

    model.optimize()
    if model.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        return 0.0

    val = float(model.objective_value)
    if val < 0 and val > -1e-9:
        val = 0.0
    return val


# -------------------------
# 旧版（最大化で侵害を作るやつ）
# これは argmin(s) 上の平均/積分ではないので残しつつ不使用にします。
# -------------------------
# def infringement_from_delta(delta, lam, n, eps=1e-9):
#     base_pairs = n * (n * (n + 1) / 2.0)   # n^2(n+1)/2
#     lam_pairs = float(lam) * float(base_pairs)
#     s = 1 if delta >= eps else 0
#     infr = delta + lam_pairs * s
#     return infr, s, lam_pairs


# =========================================================
# Hit-and-Run utilities (uniform-ish sampling on polytope)
# =========================================================
def nullspace(A, rtol=1e-12):
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    if S.size == 0:
        return Vt.T
    rank = np.sum(S > rtol * S[0])
    return Vt[rank:].T


def is_feasible(x, Aeq, beq, Aineq, bineq, tol_eq=1e-9, tol_ineq=1e-10):
    if np.max(np.abs(Aeq @ x - beq)) > tol_eq:
        return False
    if x.min() < -1e-10 or x.max() > 1.0 + 1e-10:
        return False
    if Aineq.size > 0:
        if np.min(Aineq @ x - bineq) < -tol_ineq:
            return False
    return True


def hit_and_run(
    x0: np.ndarray,
    Aeq: np.ndarray,
    beq: np.ndarray,
    Aineq: np.ndarray,   # a^T x >= b
    bineq: np.ndarray,
    n_samples: int = 200,
    burnin: int = 200,
    thinning: int = 5,
    rng: np.random.Generator = None,
    tol_interval: float = 1e-12,
):
    if rng is None:
        rng = np.random.default_rng(0)

    x = x0.copy()
    d = x.size

    if not is_feasible(x, Aeq, beq, Aineq, bineq):
        raise ValueError("hit_and_run: x0 is not feasible (numerically).")

    N = nullspace(Aeq)
    if N.size == 0:
        return np.repeat(x[None, :], n_samples, axis=0)

    samples = []
    total_steps = burnin + n_samples * thinning

    for step in range(total_steps):
        z = rng.standard_normal(N.shape[1])
        dirv = N @ z
        norm = np.linalg.norm(dirv)
        if norm < 1e-14:
            continue
        dirv /= norm

        alpha_min = -np.inf
        alpha_max = np.inf

        # Box constraints: 0 <= x + alpha*dirv <= 1
        for j in range(d):
            dj = dirv[j]
            if abs(dj) < 1e-16:
                continue
            a1 = (0.0 - x[j]) / dj
            a2 = (1.0 - x[j]) / dj
            lo = min(a1, a2)
            hi = max(a1, a2)
            alpha_min = max(alpha_min, lo)
            alpha_max = min(alpha_max, hi)

        if not (np.isfinite(alpha_min) and np.isfinite(alpha_max)):
            continue
        if alpha_max < alpha_min + tol_interval:
            continue

        # Inequalities: Aineq (x + alpha d) >= bineq
        if Aineq.size:
            Ad = Aineq @ dirv
            Ax = Aineq @ x
            for r in range(Aineq.shape[0]):
                den = Ad[r]
                num = bineq[r] - Ax[r]
                if abs(den) < 1e-16:
                    if num > 1e-10:
                        alpha_min, alpha_max = 1.0, 0.0
                        break
                    continue
                bound = num / den
                if den > 0:
                    alpha_min = max(alpha_min, bound)
                else:
                    alpha_max = min(alpha_max, bound)

            if alpha_max < alpha_min + tol_interval:
                continue

        alpha = rng.uniform(alpha_min, alpha_max)
        x_new = x + alpha * dirv

        # soft cleanup
        x_new[x_new < 0.0] = 0.0
        x_new[x_new > 1.0] = 1.0

        if not is_feasible(x_new, Aeq, beq, Aineq, bineq):
            continue

        x = x_new

        if step >= burnin and ((step - burnin) % thinning == 0):
            samples.append(x.copy())
            if len(samples) >= n_samples:
                break

    if len(samples) == 0:
        return x0[None, :]
    return np.stack(samples, axis=0)


# =========================================================
# Build argmin(s)=D(P): DS + SD-dominance polytope
#   argmin(s) corresponds to s=0 set (weak SD dominance)
# =========================================================
def build_DS_equalities(n: int):
    """
    DS equalities:
      row sums = 1  (n eqs)
      col sums = 1 but drop last col to avoid redundancy (n-1 eqs)
    """
    d = n * n
    eqs = []
    rhs = []

    for i in range(n):
        row = np.zeros(d)
        for a in range(n):
            row[i * n + a] = 1.0
        eqs.append(row)
        rhs.append(1.0)

    for a in range(n - 1):
        col = np.zeros(d)
        for i in range(n):
            col[i * n + a] = 1.0
        eqs.append(col)
        rhs.append(1.0)

    Aeq = np.vstack(eqs)
    beq = np.array(rhs, dtype=float)
    return Aeq, beq


def build_SD_dominance_inequalities(P: np.ndarray, prefs):
    """
    Aineq x >= bineq, where x=vec(X) and inequalities are
      sum_{k<=t} X_{i,a_{ik}} >= sum_{k<=t} P_{i,a_{ik}}
    """
    n = P.shape[0]
    d = n * n
    A_list = []
    b_list = []

    for i in range(n):
        pref_i = prefs[i]
        for t in range(n):
            arow = np.zeros(d)
            for k in range(t + 1):
                a = pref_i[k]
                arow[i * n + a] += 1.0
            A_list.append(arow)
            b_list.append(float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]])))

    Aineq = np.vstack(A_list) if len(A_list) else np.zeros((0, d))
    bineq = np.array(b_list, dtype=float) if len(b_list) else np.zeros((0,), dtype=float)
    return Aineq, bineq


def find_center_point_in_argmin(P: np.ndarray, prefs, verbose=False):
    """
    Find one feasible X in argmin(s) (i.e., D(P)), preferring a 'center-ish' point:
      minimize L1 distance to uniform U=1/n under DS + dominance constraints.
    """
    n = P.shape[0]
    lp = mip.Model(solver_name=mip.CBC)
    lp.verbose = 1 if verbose else 0
    lp.cuts = 0
    lp.cut_passes = 0
    lp.clique = 0

    X = [[lp.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS) for a in range(n)] for i in range(n)]

    # DS
    for i in range(n):
        lp += mip.xsum(X[i][a] for a in range(n)) == 1.0
    for a in range(n):
        lp += mip.xsum(X[i][a] for i in range(n)) == 1.0

    # dominance
    for i in range(n):
        pref_i = prefs[i]
        for t in range(n):
            prefix_X = mip.xsum(X[i][pref_i[k]] for k in range(t + 1))
            prefix_P = float(np.sum(P[i, [pref_i[k] for k in range(t + 1)]]))
            lp += prefix_X >= prefix_P

    # center-ish objective: min L1 distance to uniform
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

    X0 = np.array([[float(X[i][a].x) for a in range(n)] for i in range(n)], dtype=float)
    return X0


# =========================================================
# argmin(s) average / integral infringement (MC)
# =========================================================
def argmin_average_improvement_mc(
    P: np.ndarray,
    prefs,
    w: np.ndarray,
    n_mc: int = 400,
    burnin: int = 300,
    thinning: int = 5,
    rng: np.random.Generator = None,
    verbose_center_lp: bool = False,
):
    """
    Returns:
      avg_improvement = E_{X ~ approx Unif(argmin(s))}[ sum_{i,a} w_i(a)*(X_ia - P_ia) ]
      samples_used
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = P.shape[0]
    Aeq, beq = build_DS_equalities(n)
    Aineq, bineq = build_SD_dominance_inequalities(P, prefs)

    X0 = find_center_point_in_argmin(P, prefs, verbose=verbose_center_lp)
    if X0 is None:
        # should not happen because X=P is feasible, but keep safe
        return 0.0, 0

    x0 = X0.reshape(-1)
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

    const = float(np.sum(w * P))
    vals = []
    for svec in sam:
        X = svec.reshape(n, n)
        v = float(np.sum(w * X)) - const
        # 数値誤差ガード：本来 v>=0 のはず
        if v < 0 and v > -1e-10:
            v = 0.0
        vals.append(v)

    return float(np.mean(vals)), int(len(vals))


def infringement_argmin_average(
    P: np.ndarray,
    prefs,
    lam: float,
    eps: float,
    n_mc: int,
    burnin: int,
    thinning: int,
    rng: np.random.Generator,
    verbose_lp: bool = False,
):
    """
    Image-consistent (average/integral version):
      s_ind = 1{ exists strict dominator }   (detected by delta_star > eps)
      infr  = s_ind * ( E_{P' in argmin(s)}[Σ_{i,a} w_i(a)(P' - P)] + lam * n^2(n+1)/2 )
    """
    n = P.shape[0]
    w = weights_from_prefs(prefs)

    # indicator: strict dominator exists?
    delta_star = compute_delta_star_lp(P, prefs, w, verbose=verbose_lp)
    s_ind = 1 if delta_star >= eps else 0

    base_pairs = n * (n * (n + 1) / 2.0)   # n^2(n+1)/2 = Σ_{i,t,k}1
    lam_pairs = float(lam) * float(base_pairs)

    if s_ind == 0:
        return 0.0, 0.0, 0, lam_pairs, 0  # infr, avg_impr, indicator, lam_pairs, mc_used

    avg_impr, mc_used = argmin_average_improvement_mc(
        P=P,
        prefs=prefs,
        w=w,
        n_mc=n_mc,
        burnin=burnin,
        thinning=thinning,
        rng=rng,
        verbose_center_lp=False,
    )

    infr = (avg_impr + lam_pairs) * float(s_ind)
    return float(infr), float(avg_impr), int(s_ind), float(lam_pairs), int(mc_used)


# =========================
# Sampling preference profiles and reporting
# =========================
def sample_random_profile(n, rng):
    return [tuple(rng.permutation(n)) for _ in range(n)]


def evaluate_sd_oe_infringement_argmin_average(
    n=4,
    priority=(0, 1, 2, 3),
    samples=200,
    seed=1,
    lam=0.1,
    eps=1e-9,
    n_mc=400,
    burnin=300,
    thinning=5,
    lp_verbose=False,
):
    rng = np.random.default_rng(seed)

    infrs = []
    avgs = []
    inds = []

    worst = {
        "infr": -1.0,
        "avg_impr": -1.0,
        "ind": None,
        "prefs": None,
        "P": None,
        "lam_pairs": None,
        "mc_used": None,
    }

    for _ in range(samples):
        prefs = sample_random_profile(n, rng)
        P = serial_dictatorship_allocation(prefs, priority)

        infr, avg_impr, ind, lam_pairs, mc_used = infringement_argmin_average(
            P=P,
            prefs=prefs,
            lam=lam,
            eps=eps,
            n_mc=n_mc,
            burnin=burnin,
            thinning=thinning,
            rng=rng,
            verbose_lp=lp_verbose,
        )

        infrs.append(infr)
        avgs.append(avg_impr)
        inds.append(ind)

        if infr > worst["infr"]:
            worst.update({
                "infr": infr,
                "avg_impr": avg_impr,
                "ind": ind,
                "prefs": prefs,
                "P": P,
                "lam_pairs": lam_pairs,
                "mc_used": mc_used,
            })

    infrs = np.array(infrs, dtype=float)
    avgs = np.array(avgs, dtype=float)
    inds = np.array(inds, dtype=float)

    print("=== Deterministic Serial Dictatorship: OE infringement (argmin(s) average / integral) ===")
    print(f"n={n} priority={tuple(priority)} samples={samples}")
    print(f"lam={lam} eps={eps}  MC=(n_mc={n_mc}, burnin={burnin}, thinning={thinning})")
    print(f"avg E_improvement(argmin(s)) = {avgs.mean():.6e}")
    print(f"avg infringement = {infrs.mean():.6e}")
    print(f"strict-dominator fraction = {inds.mean():.3f}")

    print("\n=== Worst sampled profile ===")
    print(f"infr = {worst['infr']:.6e}  indicator={worst['ind']}  lam_pairs={worst['lam_pairs']:.6e}  mc_used={worst['mc_used']}")
    print(f"avg_improvement(argmin(s)) = {worst['avg_impr']:.6e}")
    print("prefs (each i: best..worst):")
    for i in range(n):
        print(f"  i={i}: {worst['prefs'][i]}")
    print("P (SD outcome):")
    print(worst["P"])


if __name__ == "__main__":
    evaluate_sd_oe_infringement_argmin_average(
        n=4,
        priority=(0, 1, 2, 3),
        samples=200,
        seed=1,
        lam=0.1,
        eps=1e-9,
        n_mc=400,
        burnin=300,
        thinning=5,
        lp_verbose=False,
    )

