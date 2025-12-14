import numpy as np
import itertools
import mip


def equal_split(n: int) -> np.ndarray:
    """等配分メカニズム：各人が各財を 1/n ずつ受け取る"""
    return np.full((n, n), 1.0 / n, dtype=float)


def all_strict_orders(n: int):
    """厳密順序（タイなし）の全列挙：n=4なら 24 通り"""
    return list(itertools.permutations(range(n)))


def compute_c_min(P: np.ndarray, order: np.ndarray, verbose: bool = False):
    """
    画像の式：
      c(P) = min_{P'} Σ_i Σ_t ( |Σ_{k<=t}(p'_{iak}-p_{iak})| - Σ_{k<=t}(...) )
    を LP で解く（P' は二重確率行列）。
    """
    n = P.shape[0]
    model = mip.Model(sense=mip.MINIMIZE)
    if not verbose:
        model.verbose = 0

    # P' variables
    Pp = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
           for a in range(n)] for i in range(n)]
    for i in range(n):
        model += mip.xsum(Pp[i][a] for a in range(n)) == 1.0
    for a in range(n):
        model += mip.xsum(Pp[i][a] for i in range(n)) == 1.0

    # d_{i,t} and u_{i,t} = |d_{i,t}|
    d = [[model.add_var(lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS)
          for t in range(n)] for i in range(n)]
    u = [[model.add_var(lb=0.0, ub=mip.INF, var_type=mip.CONTINUOUS)
          for t in range(n)] for i in range(n)]

    for i in range(n):
        for t in range(n):
            expr = 0.0
            for k in range(t + 1):
                a = int(order[i, k])
                expr += Pp[i][a] - P[i, a]
            model += d[i][t] == expr
            model += u[i][t] >= d[i][t]
            model += u[i][t] >= -d[i][t]

    model.objective = mip.xsum(u[i][t] - d[i][t] for i in range(n) for t in range(n))
    model.optimize()

    if model.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError("c(P) の最小化が解けませんでした")

    c_val = float(model.objective_value)
    Pp_opt = np.array([[Pp[i][a].x for a in range(n)] for i in range(n)])
    return c_val, Pp_opt


def oe_violation_plan1(P: np.ndarray,
                       order: np.ndarray,
                       lam: float = 0.1,
                       tol: float = 1e-9,
                       verbose: bool = False):
    """
    案1の「順序効率性の侵害度合い」を、計算可能な形で実装：
      1) c(P) を解く
      2) c(P)=0 のとき、S = {P' | P' が P をSD支配} 上で
           max Σ_i Σ_t Σ_{k<=t}(p'_{iak}-p_{iak}+λ)
         を返す（= 最も強い支配割当がどれだけあるか）
         c(P)>0 なら 0 を返す（支配割当が存在しない扱い）
    """
    n = P.shape[0]

    # --- Step1: c(P) ---
    c_val, _ = compute_c_min(P, order, verbose=verbose)
    if c_val > tol:
        return 0.0, c_val, None  # S empty (as per your interpretation)

    # --- Step2: maximize penalty over S (SD-dominators) ---
    model = mip.Model(sense=mip.MAXIMIZE)
    if not verbose:
        model.verbose = 0

    # P' variables
    Pp = [[model.add_var(lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)
           for a in range(n)] for i in range(n)]
    for i in range(n):
        model += mip.xsum(Pp[i][a] for a in range(n)) == 1.0
    for a in range(n):
        model += mip.xsum(Pp[i][a] for i in range(n)) == 1.0

    # SD-domination constraints: for all i,t  Σ_{k<=t}(p' - p) >= 0
    for i in range(n):
        for t in range(n):
            expr = 0.0
            for k in range(t + 1):
                a = int(order[i, k])
                expr += Pp[i][a] - P[i, a]
            model += expr >= 0.0

    # Objective = Σ_i Σ_t Σ_{k<=t}(p'_{iak}-p_{iak}+λ)
    # = Σ_i Σ_k w_k*(p'_{i,aik}-p_{i,aik}+λ), where w_k = (n-k) when k is 0-indexed? -> count of t>=k is n-k
    weights = [n - k for k in range(n)]  # k=0..n-1  => n, n-1, ..., 1
    obj = 0.0
    for i in range(n):
        for k in range(n):
            a = int(order[i, k])
            obj += weights[k] * (Pp[i][a] - P[i, a] + lam)

    model.objective = mip.maximize(obj)
    model.optimize()

    if model.status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError("案1の侵害度（max問題）が解けませんでした")

    vio = float(model.objective_value)
    Pp_star = np.array([[Pp[i][a].x for a in range(n)] for i in range(n)])
    return vio, c_val, Pp_star


def all_profiles_equal_split_plan1(n: int = 4,
                                   lam: float = 0.1,
                                   max_profiles: int | None = None,
                                   verbose: bool = False):
    """
    4x4 の全プロフィール（24^4）に対して、等配分メカニズムの案1侵害度を計算。
    注意：LPを大量に解くので重い。max_profiles で打ち切り可能。
    """
    P = equal_split(n)
    perms = all_strict_orders(n)

    cnt = 0
    max_v = -1.0
    argmax_order = None

    for prof in itertools.product(perms, repeat=n):
        order = np.array(prof, dtype=int)
        v, c, _ = oe_violation_plan1(P, order, lam=lam, verbose=verbose)
        if v > max_v:
            max_v = v
            argmax_order = order.copy()

        cnt += 1
        if max_profiles is not None and cnt >= max_profiles:
            break

    return {
        "count": cnt,
        "max_violation": float(max_v),
        "argmax_order": argmax_order,
    }


if __name__ == "__main__":
    n = 4
    lam = 0.1
    P = equal_split(n)

    # 例：ランダムに1プロフィールで計算
    np.random.seed(1)
    order = np.array([np.random.permutation(n) for _ in range(n)], dtype=int)

    print("=== preference order ===")
    print(order)
    print("\n=== equal split P ===")
    print(P)

    vio, c_val, Pp_star = oe_violation_plan1(P, order, lam=lam, verbose=False)
    print("\n[Plan1]")
    print(f"c(P) = {c_val:.6e}")
    print(f"ordinal-efficiency violation (plan1, max over S) = {vio:.6e}")
    if Pp_star is not None:
        print("\n=== one strongest dominator P' (argmax in S) ===")
        print(Pp_star)

    # 全プロフィールで max だけ見たいとき（重いので max_profiles で小さく試すの推奨）
    # res = all_profiles_equal_split_plan1(n=4, lam=lam, max_profiles=500, verbose=False)
    # print("\n=== partial scan over profiles ===")
    # print(res)
