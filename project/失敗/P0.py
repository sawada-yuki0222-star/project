import numpy as np
import argparse
from pathlib import Path


def make_rowcol_stochastic(mat: np.ndarray) -> np.ndarray:
    mat = np.maximum(mat, 0.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    col_sums = mat.sum(axis=0, keepdims=True)
    mat = mat / col_sums
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def check_doubly_stochastic(P: np.ndarray) -> tuple[float, float]:
    row_err = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    col_err = float(np.max(np.abs(P.sum(axis=0) - 1.0)))
    return row_err, col_err


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    # ★ここを「P0」フォルダに変更（新規作成してそこに保存）
    p.add_argument("--outdir", type=str, default="P0")
    p.add_argument("--fmt", type=str, default="%.18e")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.n
    k = args.k
    seed = args.seed

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    max_row_err = 0.0
    max_col_err = 0.0

    # 生成して保存
    for t in range(k):
        P_raw = rng.random((n, n))
        P0 = make_rowcol_stochastic(P_raw)

        rerr, cerr = check_doubly_stochastic(P0)
        max_row_err = max(max_row_err, rerr)
        max_col_err = max(max_col_err, cerr)

        np.savetxt(outdir / f"P0_trial_{t:03d}.txt", P0, fmt=args.fmt)

    # メタ情報も同じ P0 フォルダ内に保存
    meta_path = outdir / "P0_meta.txt"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write("=== P0 candidates meta ===\n")
        f.write(f"n={n}\n")
        f.write(f"k={k}\n")
        f.write(f"seed={seed}\n")
        f.write(f"dir={outdir.as_posix()}\n")
        f.write(f"max_row_sum_error={max_row_err:.6e}\n")
        f.write(f"max_col_sum_error={max_col_err:.6e}\n")
        f.write(f"fmt={args.fmt}\n")

    print(f"Saved {k} P0 candidates to: {outdir.resolve()}")
    print(f"Saved meta to: {meta_path.resolve()}")
    print(f"Max row-sum error: {max_row_err:.3e}, Max col-sum error: {max_col_err:.3e}")
