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


def compute_oe_violation(P: np.ndarray,
                         P_prime: np.ndarray,
                         order: np.ndarray) -> float:
    """
    順序効率性の侵害度合い
        c(P, P') = Σ_i Σ_t { |Σ_{k<=t}(p'_{iak}-p_{iak})|
                             - Σ_{k<=t}(p'_{iak}-p_{iak}) }
    をそのまま計算する補助関数。
    """
    N = P.shape[0]
    total = 0.0
    for i in range(N):
        for t in range(N):
            s = 0.0
            for k in range(t + 1):
                a_k = int(order[i, k])
                s += P_prime[i, a_k] - P[i, a_k]
            total += abs(s) - s
    return float(total)


def solve_oe_min(P: np.ndarray,
                 order: np.ndarray,
                 lam: float = 0.1,      # 互換のため残し（ここでは未使用）
                 tol: float = 1e-8,
                 verbose: bool = False):

    N = P.shape[0]
    model = mip.Model()
    if not verbose:
        model.verbose = 0

    # ---------- 真実申告プロフィールでの割当 P_new ----------
    P_new = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
              for j in range(N)] for i in range(N)]

    for i in range(N):
        model += mip.xsum(P_new[i][j] for j in range(N)) == 1.0
    for j in range(N):
        model += mip.xsum(P_new[i][j] for i in range(N)) == 1.0

    # ---------- 順序効率性の侵害 c(P,P_new) 用の d_oe, u_oe ----------
    d_oe = [[model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
             for t in range(N)] for i in range(N)]
    u_oe = [[model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
             for t in range(N)] for i in range(N)]

    for i in range(N):
        for t in range(N):
            expr = 0.0
            for k in range(t + 1):
                a_k = int(order[i, k])
                expr += P_new[i][a_k] - P[i, a_k]
            model += d_oe[i][t] == expr
            model += u_oe[i][t] >= d_oe[i][t]
            model += u_oe[i][t] >= -d_oe[i][t]

    c_expr = mip.xsum(u_oe[i][t] - d_oe[i][t] for i in range(N) for t in range(N))

    # ---------- 無羨望性 ----------
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for t in range(N):
                expr = 0.0
                for k in range(t + 1):
                    a_k = int(order[i, k])
                    expr += P_new[i][a_k] - P_new[j][a_k]
                model += expr >= 0.0

    # ---------- Strategy-proofness 用: プロファイル別の割当 G ----------
    all_prefs = list(itertools.permutations(range(N)))

    mis_prefs_per_i = {}
    for i in range(N):
        true_pref = tuple(int(x) for x in order[i])
        mis_prefs_per_i[i] = [perm for perm in all_prefs if perm != true_pref]

    G = {}
    for i in range(N):
        mis_list = mis_prefs_per_i[i]
        for m_idx, perm in enumerate(mis_list):
            key = (i, m_idx)
            G[key] = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
                       for j in range(N)] for r in range(N)]
            for r in range(N):
                model += mip.xsum(G[key][r][j] for j in range(N)) == 1.0
            for j in range(N):
                model += mip.xsum(G[key][r][j] for r in range(N)) == 1.0

    # ---------- 画像の「耐戦略性の侵害」式を 0 にする制約 ----------
    d_sp = {}
    u_sp = {}
    sp_terms = []

    for i in range(N):
        mis_list = mis_prefs_per_i[i]
        for m_idx, perm in enumerate(mis_list):
            key = (i, m_idx)
            G_im = G[key]
            for t in range(N):
                var_d = model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
                var_u = model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
                d_sp[(i, m_idx, t)] = var_d
                u_sp[(i, m_idx, t)] = var_u

                expr = 0.0
                for k in range(t + 1):
                    a_k = int(order[i, k])
                    expr += P_new[i][a_k] - G_im[i][a_k]   # p_{iak} - p''_{iak}

                model += var_d == expr
                model += var_u >= var_d
                model += var_u >= -var_d

                sp_terms.append(var_u - var_d)  # = |S|-S ≥ 0

    sp_expr = mip.xsum(sp_terms)
    model += sp_expr == 0.0   # 耐戦略性の侵害を 0 に

    # ---------- 目的関数: 順序効率性の侵害度合い c_expr を最小化 ----------
    model.objective = mip.minimize(c_expr)
    model.optimize()

    if model.status not in (mip.OptimizationStatus.OPTIMAL,
                            mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError("最適解が見つかりませんでした")

    P_new_opt = np.array([[P_new[i][j].x for j in range(N)] for i in range(N)])

    # Strategy-proofness violation（制約で 0 にしているので ≈0 のはず）
    sp_violation = float(sum(term.x for term in sp_terms))

    return P_new_opt, sp_violation


if __name__ == "__main__":
    N = 4
    np.random.seed(1)

    order = np.array([np.random.permutation(N) for _ in range(N)])
    P_raw = np.random.rand(N, N)
    P = make_rowcol_stochastic(P_raw)

    print("=== preference order (a_1,...,a_4 for each i) ===")
    print(order)
    print("\n=== original random matching P ===")
    print(P)

    # EF + SP（画像式）を満たしつつ順序効率性の侵害を最小化した P_new を求める
    P_new, sp_vio = solve_oe_min(P, order, lam=0.1, tol=1e-8, verbose=False)

    # 順序効率性の侵害度合いを明示的に計算
    c_base = compute_oe_violation(P, P, order)          # c(P,P) ＝ 0 のはず
    c_new = compute_oe_violation(P, P_new, order)       # c(P,P_new) ＝ 最小化された値

    print(f"\n[Ordinal efficiency violation]")
    print(f"  c(P, P)      = {c_base:.6e}   (baseline, should be 0)")
    print(f"  c(P, P_new)  = {c_new:.6e}   (EF + SP 下での最小値)")
    print(f"\nstrategy-proofness violation (画像の式) = {sp_vio:.6e}")
    print("\n=== P_new (EF + SP, ordinal-efficiency-violation minimal) ===")
    print(P_new)
