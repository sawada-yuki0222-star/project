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


def solve_oe_min(P: np.ndarray,
                 order: np.ndarray,
                 lam: float = 0.1,
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

    # ---------- c(P,P_new) 用の d, u ----------
    d = [[model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
          for t in range(N)] for i in range(N)]
    u = [[model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
          for t in range(N)] for i in range(N)]

    for i in range(N):
        for t in range(N):
            expr = 0.0
            for k in range(t + 1):
                a_k = int(order[i, k])
                expr += P_new[i][a_k] - P[i, a_k]
            model += d[i][t] == expr
            model += u[i][t] >= d[i][t]
            model += u[i][t] >= -d[i][t]

    c_expr = mip.xsum(u[i][t] - d[i][t] for i in range(N) for t in range(N))

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

    # ---------- Strategy-proofness ----------
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

    for i in range(N):
        mis_list = mis_prefs_per_i[i]
        for m_idx, perm in enumerate(mis_list):
            key = (i, m_idx)
            G_im = G[key]
            for t in range(N):
                top_objs = [int(order[i, k]) for k in range(t + 1)]
                lhs = mip.xsum(P_new[i][j] for j in top_objs)
                rhs = mip.xsum(G_im[i][j] for j in top_objs)
                model += lhs >= rhs

    # ---------- 目的関数: c(P,P_new) の最小化 ----------
    model.objective = mip.minimize(c_expr)
    model.optimize()

    if model.status not in (mip.OptimizationStatus.OPTIMAL,
                            mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError("最適解が見つかりませんでした")

    c_val = model.objective_value
    P_new_opt = np.array([[P_new[i][j].x for j in range(N)] for i in range(N)])

    # ---------- 順序効率性 violation の計算 ----------
    violation = 0.0
    for i in range(N):
        for t in range(N):
            for k in range(t + 1):
                a_k = int(order[i, k])
                violation += (P_new_opt[i, a_k] - P[i, a_k] + lam)

    return float(c_val), P_new_opt, float(violation)


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

    c_val, P_new, vio = solve_oe_min(P, order, lam=0.1, tol=1e-8, verbose=False)

    print(f"\nmin c(P,P_new) with EF + SP = {c_val:.6e}")
    print("\n=== P_new (EF & SP matching that minimizes c) ===")
    print(P_new)
    print(f"\nordinal-efficiency violation (案1) = {vio:.6f}")
