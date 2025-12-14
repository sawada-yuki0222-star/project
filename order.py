import numpy as np
import itertools
import mip


def make_rowcol_stochastic(mat: np.ndarray) -> np.ndarray:
    mat = np.maximum(mat, 0.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    col_sums = mat.sum(axis=0, keepdims=True)
    mat = mat / col_sums
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def compute_oe_violation(P0: np.ndarray,
                         Pp: np.ndarray,
                         order: np.ndarray) -> float:
    """
    c(P0, Pp) = Σ_i Σ_t { |Σ_{k<=t}(p'_{iak}-p0_{iak})| - Σ_{k<=t}(p'_{iak}-p0_{iak}) }
    """
    n = P0.shape[0]
    total = 0.0
    for i in range(n):
        for t in range(n):
            s = 0.0
            for k in range(t + 1):
                a = int(order[i, k])
                s += Pp[i, a] - P0[i, a]
            total += abs(s) - s
    return float(total)


def compute_ef_violation(Pp: np.ndarray,
                         order: np.ndarray) -> float:
    """
    「無羨望性の侵害」：
    Σ_i Σ_{j≠i} Σ_t { |Σ_{k<=t}(p_{iak}-p_{jak})| - Σ_{k<=t}(p_{iak}-p_{jak}) }
    """
    n = Pp.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in range(n):
                s = 0.0
                for k in range(t + 1):
                    a = int(order[i, k])
                    s += Pp[i, a] - Pp[j, a]
                total += abs(s) - s
    return float(total)


def solve_oe_min(P0: np.ndarray,
                 order: np.ndarray,
                 tol: float = 1e-8,
                 verbose: bool = False):
    """
    - 目的：順序効率性の侵害 c(P0, P_new) を最小化
    - 制約：
        (i) 画像の「無羨望性の侵害」= 0
        (ii) 画像の「耐戦略性の侵害」= 0
        (iii) 各プロフィール（真実/虚偽）で割当は二重確率（行和=列和=1）
    """

    n = P0.shape[0]
    model = mip.Model()
    if not verbose:
        model.verbose = 0

    # =========================================================
    # 真実申告プロフィールでの割当 P_new（変数）
    # =========================================================
    P_new = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
              for j in range(n)] for i in range(n)]

    for i in range(n):
        model += mip.xsum(P_new[i][j] for j in range(n)) == 1.0
    for j in range(n):
        model += mip.xsum(P_new[i][j] for i in range(n)) == 1.0

    # =========================================================
    # 順序効率性侵害 c(P0, P_new) の線形化（目的関数）
    # =========================================================
    d_oe = [[model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
             for t in range(n)] for i in range(n)]
    u_oe = [[model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
             for t in range(n)] for i in range(n)]

    for i in range(n):
        for t in range(n):
            expr = 0.0
            for k in range(t + 1):
                a = int(order[i, k])
                expr += P_new[i][a] - P0[i, a]
            model += d_oe[i][t] == expr
            model += u_oe[i][t] >= d_oe[i][t]
            model += u_oe[i][t] >= -d_oe[i][t]

    c_expr = mip.xsum(u_oe[i][t] - d_oe[i][t] for i in range(n) for t in range(n))

    # =========================================================
    # 無羨望性：「侵害」式を線形化して = 0 を課す
    # =========================================================
    # 旧（不等式で直接 EF を課していた）:
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         for t in range(n):
    #             expr = 0.0
    #             for k in range(t + 1):
    #                 a = int(order[i, k])
    #                 expr += P_new[i][a] - P_new[j][a]
    #             model += expr >= 0.0

    d_ef = {}
    u_ef = {}
    ef_terms = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in range(n):
                var_d = model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
                var_u = model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
                d_ef[(i, j, t)] = var_d
                u_ef[(i, j, t)] = var_u

                expr = 0.0
                for k in range(t + 1):
                    a = int(order[i, k])         # i の順序で prefix
                    expr += P_new[i][a] - P_new[j][a]

                model += var_d == expr
                model += var_u >= var_d
                model += var_u >= -var_d

                ef_terms.append(var_u - var_d)  # = |S| - S ≥ 0

    ef_expr = mip.xsum(ef_terms)
    model += ef_expr == 0.0   # ★画像の無羨望性侵害を 0 に固定

    # =========================================================
    # 耐戦略性：「侵害」式を線形化して = 0 を課す
    # =========================================================
    all_prefs = list(itertools.permutations(range(n)))

    mis_prefs_per_i = {}
    for i in range(n):
        true_pref = tuple(int(x) for x in order[i])
        mis_prefs_per_i[i] = [perm for perm in all_prefs if perm != true_pref]

    # i が虚偽申告 perm をしたときの割当 G^{i,perm}（変数）
    G = {}
    for i in range(n):
        for m_idx, perm in enumerate(mis_prefs_per_i[i]):
            key = (i, m_idx)
            G[key] = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
                       for j in range(n)] for r in range(n)]
            for r in range(n):
                model += mip.xsum(G[key][r][j] for j in range(n)) == 1.0
            for j in range(n):
                model += mip.xsum(G[key][r][j] for r in range(n)) == 1.0

    d_sp = {}
    u_sp = {}
    sp_terms = []

    for i in range(n):
        for m_idx, perm in enumerate(mis_prefs_per_i[i]):
            key = (i, m_idx)
            G_im = G[key]
            for t in range(n):
                var_d = model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
                var_u = model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
                d_sp[(i, m_idx, t)] = var_d
                u_sp[(i, m_idx, t)] = var_u

                expr = 0.0
                for k in range(t + 1):
                    a = int(order[i, k])
                    expr += P_new[i][a] - G_im[i][a]   # p_{iak} - p''_{iak}

                model += var_d == expr
                model += var_u >= var_d
                model += var_u >= -var_d

                sp_terms.append(var_u - var_d)  # = |S| - S ≥ 0

    sp_expr = mip.xsum(sp_terms)
    model += sp_expr == 0.0   # ★画像の耐戦略性侵害を 0 に固定

    # =========================================================
    # 目的関数：順序効率性侵害 c(P0, P_new) を最小化
    # =========================================================
    model.objective = mip.minimize(c_expr)
    model.optimize()

    if model.status not in (mip.OptimizationStatus.OPTIMAL,
                            mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError("最適解が見つかりませんでした（EF=0 と SP=0 が両立しない可能性）")

    P_new_opt = np.array([[P_new[i][j].x for j in range(n)] for i in range(n)])

    # 侵害の値（制約で 0 に固定しているので ≈0 のはず）
    ef_violation = float(sum(term.x for term in ef_terms))
    sp_violation = float(sum(term.x for term in sp_terms))

    return P_new_opt, ef_violation, sp_violation, float(c_expr.x)


if __name__ == "__main__":
    n = 4
    np.random.seed(1)

    order = np.array([np.random.permutation(n) for _ in range(n)])
    P_raw = np.random.rand(n, n)
    P0 = make_rowcol_stochastic(P_raw)

    print("=== preference order (a_1,...,a_4 for each i) ===")
    print(order)
    print("\n=== original random matching P0 ===")
    print(P0)

    P_new, ef_vio, sp_vio, c_val = solve_oe_min(P0, order, tol=1e-8, verbose=False)

    c_base = compute_oe_violation(P0, P0, order)
    c_new = compute_oe_violation(P0, P_new, order)
    ef_chk = compute_ef_violation(P_new, order)

    print("\n[Violations]")
    print(f"  ordinal-efficiency violation c(P0,P0)   = {c_base:.6e}")
    print(f"  ordinal-efficiency violation c(P0,Pnew) = {c_new:.6e}   (== objective)")
    print(f"  envy-free violation (image expr)        = {ef_vio:.6e}   (check={ef_chk:.6e})")
    print(f"  strategy-proof violation (image expr)   = {sp_vio:.6e}")

    print("\n=== P_new (EF=0 & SP=0, minimizes OE violation) ===")
    print(P_new)
