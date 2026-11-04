#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4人4財：一様割り当てからOE侵害を最小化する。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

try:
    from matrix import build_or_load_model, solve_oe_violation
except ModuleNotFoundError as exc:
    if exc.name == "matrix":
        raise SystemExit(
            "matrix.py が見つかりません。main_uniform.py を\n"
            "GitHubリポジトリの『新モデル/matrix.py』と同じフォルダに置いて実行してください。"
        ) from exc
    raise


INITIAL = "UNIFORM"
DEFAULT_OUTPUT_DIR = "results_uniform"


@dataclass
class Objective:
    value: float
    supergradient: np.ndarray
    inner_residual: float


def evaluate(model, vector: np.ndarray) -> Objective:
    """PDFの v_theta(P) の（軌道の大きさで重みづけした）平均を厳密な内側LPで評価。"""
    value = 0.0
    grad = np.zeros(model.number_of_variables, dtype=np.float64)
    max_res = 0.0
    for i, (a, score, weight) in enumerate(zip(
        model.cumulative_matrices,
        model.score_vectors,
        model.normalized_weights,
    )):
        block = model.representative_slice(i)
        result = solve_oe_violation(
            cumulative=a,
            score=score,
            inner_equalities=model.inner_equalities,
            inner_rhs=model.inner_rhs,
            allocation_vector=vector[block],
        )
        value += float(weight) * result.value
        grad[block] = float(weight) * result.supergradient
        max_res = max(max_res, float(result.max_primal_violation))
    return Objective(value, grad, max_res)


def solve_full_sp_master(model, gradient: np.ndarray, method: str) -> np.ndarray:
    """全てのSP制約を A_ub x <= b_ub としてLPに直接渡す（カット生成ではない）。"""
    result = linprog(
        c=gradient,
        A_ub=-model.sp_matrix,                   # C x >= 0 <=> -C x <= 0
        b_ub=np.zeros(model.sp_matrix.shape[0]),
        A_eq=model.equality_matrix,
        b_eq=model.equality_rhs,
        bounds=(0.0, 1.0),
        method=method,
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"全SP制約つきLPが失敗：status={result.status}; {result.message}"
        )
    return np.asarray(result.x, dtype=np.float64)


def exact_matrix_check(matrix, rhs, x: list[Fraction], *, equality: bool) -> tuple[bool, str]:
    """整数係数の疎行列を分数で乗じ、浮動小数点を一切使わず検証する。"""
    m = matrix.tocsr()
    indptr, indices, data = m.indptr, m.indices, m.data
    if not np.array_equal(data, np.rint(data)):
        raise ValueError("制約行列に非整数係数があります。厳密検証を停止します")
    coeff = np.asarray(np.rint(data), dtype=np.int64)
    for row in range(m.shape[0]):
        total = Fraction(0, 1)
        for j in range(int(indptr[row]), int(indptr[row + 1])):
            total += int(coeff[j]) * x[int(indices[j])]
        bound = Fraction(int(round(float(rhs[row]))), 1) if rhs is not None else Fraction(0, 1)
        if equality and total != bound:
            return False, f"等式制約 {row} が不成立（残差={total - bound}）"
        if not equality and total < bound:
            return False, f"耐戦略性制約 {row} が不成立（残差={total - bound}）"
    return True, "OK"


def exact_certificate(model, vector: np.ndarray, max_den: int,
                      rational_tol: float) -> tuple[list[Fraction] | None, str]:
    """float候補 -> 小さい分母の有理数 -> 全等式＆全SPの*厳密*判定。"""
    if vector.shape != (model.number_of_variables,):
        return None, "解ベクトルの長さが不正"
    if not np.all(np.isfinite(vector)):
        return None, "非有限の確率を検出"
    rationals: list[Fraction] = []
    for j, raw in enumerate(vector):
        value = float(raw)
        if value < -rational_tol or value > 1.0 + rational_tol:
            return None, f"確率が[0,1]の外: index={j}, value={value}"
        value = min(1.0, max(0.0, value))
        frac = Fraction(value).limit_denominator(max_den)
        if abs(float(frac) - value) > rational_tol:
            return None, f"有理数復元の誤差が大きい: index={j}"
        rationals.append(frac)

    ok, message = exact_matrix_check(
        model.equality_matrix, model.equality_rhs, rationals, equality=True
    )
    if not ok:
        return None, message
    ok, message = exact_matrix_check(
        model.sp_matrix, None, rationals, equality=False
    )
    if not ok:
        return None, message
    return rationals, "厳密有理数検証に合格（SP違反=0、全等式成立）"


def from_rationals(rationals: list[Fraction]) -> np.ndarray:
    return np.fromiter((float(x) for x in rationals), dtype=np.float64,
                       count=len(rationals))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loss_for_rep(model, i: int, vector: np.ndarray) -> float:
    return solve_oe_violation(
        cumulative=model.cumulative_matrices[i],
        score=model.score_vectors[i],
        inner_equalities=model.inner_equalities,
        inner_rhs=model.inner_rhs,
        allocation_vector=vector[model.representative_slice(i)],
    ).value


def main() -> None:
    parser = argparse.ArgumentParser(description='一様割り当てからOE侵害を最小化。全SP制約＋有理数の厳密検証')
    root_dir = Path(__file__).resolve().parent.parent
    parser.add_argument("--cache", type=Path, default=root_dir / "model_cache.pkl")
    parser.add_argument("--output-dir", type=Path, default=root_dir / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="0ならベースライン検証・保存のみ")
    parser.add_argument("--relative-tol", type=float, default=1e-9)
    parser.add_argument("--max-denominator", type=int, default=1000000,
                        help="結果の有理数復元に使う最大分母")
    parser.add_argument("--rational-tol", type=float, default=5e-9,
                        help="数値解から有理数への復元で許す近似誤差（厳密検証とは別）")
    parser.add_argument("--method", default="highs-ds",
                        choices=("highs", "highs-ds", "highs-ipm"))
    parser.add_argument("--max-line-search", type=int, default=13)
    args = parser.parse_args()
    if args.max_iterations < 0 or args.max_denominator < 1 or args.max_line_search < 1:
        parser.error("max-iterations >= 0、max-denominator >= 1、max-line-search >= 1が必要")

    t0 = time.perf_counter()
    print("モデルを読み込み中...", flush=True)
    model = build_or_load_model(args.cache)
    rsd = np.asarray(model.rsd_initial, dtype=np.float64).copy()
    uniform = np.full(model.number_of_variables, 1.0 / model.n, dtype=np.float64)
    init = rsd if INITIAL == "RSD" else uniform
    print(f"代表選好={len(model.representatives):,}, 元の選好={model.number_of_profiles:,}, "
          f"変数={model.number_of_variables:,}, SP制約={model.sp_matrix.shape[0]:,}", flush=True)

    # 初期メカニズムも全制約を厳密検証。失敗時には最適化を開始しない。
    initial_fractions, why = exact_certificate(
        model, init, args.max_denominator, args.rational_tol
    )
    if initial_fractions is None:
        raise RuntimeError(f"初期メカニズムが厳密実行可能ではありません: {why}")
    current_fractions = initial_fractions
    current = from_rationals(current_fractions)
    print(f"初期値 {INITIAL}：{why}", flush=True)

    print("目的関数を計算中（各762代表選好の内側LP）...", flush=True)
    initial_eval = evaluate(model, current)
    rsd_eval = initial_eval if INITIAL == "RSD" else evaluate(model, rsd)
    uniform_eval = initial_eval if INITIAL == "UNIFORM" else evaluate(model, uniform)
    evaluation = initial_eval
    print(f"RSDの平均OE侵害度       : {rsd_eval.value:.12f}")
    print(f"一様割り当て平均OE侵害度: {uniform_eval.value:.12f}")
    print(f"初期値({INITIAL})のOE侵害度: {evaluation.value:.12f}", flush=True)

    history = [evaluation.value]
    log: list[dict] = []

    def record(iteration: int, step: float, note: str, seconds: float) -> None:
        eq_float = model.max_equality_violation(current)
        sp_float = model.max_sp_violation(current)
        log.append({
            "iteration": iteration,
            "average_oe": evaluation.value,
            "improvement_from_initial": initial_eval.value - evaluation.value,
            "improvement_over_rsd": rsd_eval.value - evaluation.value,
            "step": step,
            "floating_max_eq_violation": eq_float,
            "floating_max_sp_violation": sp_float,
            "exact_sp_violation": 0,
            "exact_equalities_hold": True,
            "max_inner_lp_residual": evaluation.inner_residual,
            "seconds": seconds,
            "note": note,
        })
        print(f"iter={iteration:02d} OE={evaluation.value:.12f} "
              f"SP(float)={sp_float:.3e} SP(exact)=0 "
              f"EQ(float)={eq_float:.3e} step={step:g} {note}", flush=True)

    record(0, 0.0, "初期解は厳密実行可能", 0.0)
    stop_reason = "最大反復回数"

    for iteration in range(1, args.max_iterations + 1):
        iteration_start = time.perf_counter()
        raw = solve_full_sp_master(model, evaluation.supergradient, args.method)
        raw_sp = model.max_sp_violation(raw)
        raw_eq = model.max_equality_violation(raw)
        print(f"  LP生解: SP={raw_sp:.3e}, EQ={raw_eq:.3e}", flush=True)

        accepted = False
        last_failure = "候補なし"
        for attempt in range(args.max_line_search):
            alpha = 2.0 ** (-attempt)
            trial = current + alpha * (raw - current)
            # 採用前に整数係数の全制約を厳密に検証し、通った点だけ評価する。
            rat, reason = exact_certificate(
                model, trial, args.max_denominator, args.rational_tol
            )
            if rat is None:
                last_failure = reason
                continue
            trial_float = from_rationals(rat)
            trial_eval = evaluate(model, trial_float)
            # 浮動小数点の評価誤差だけ許容。厳密なSP制約自体は常に成立。
            if trial_eval.value <= evaluation.value + 1e-10:
                previous = evaluation.value
                current_fractions, current, evaluation = rat, trial_float, trial_eval
                history.append(evaluation.value)
                record(iteration, alpha, "有理数の全制約を厳密検証済み",
                       time.perf_counter() - iteration_start)
                accepted = True
                gain = previous - evaluation.value
                if gain <= args.relative_tol * max(1.0, abs(previous)):
                    stop_reason = "改善幅が許容値以下（局所停止。大域最適性の保証なし）"
                break
            last_failure = (
                f"目的関数が増加: {trial_eval.value - evaluation.value:.3e}"
            )

        if not accepted:
            stop_reason = "厳密実行可能かつ改善する候補がない: " + last_failure
            print("  採用可能な候補なし。以前の厳密実行可能解を維持。", flush=True)
            break
        if gain <= args.relative_tol * max(1.0, abs(previous)):
            break

    # 最終結果を再検証：失敗時には成功ファイルを絶対に保存しない。
    final_rat, reason = exact_certificate(
        model, current, args.max_denominator, args.rational_tol
    )
    if final_rat is None:
        raise RuntimeError("最終解の厳密耐戦略性検証に失敗: " + reason)
    if final_rat != current_fractions:
        raise RuntimeError("保存対象の有理数解が最終検証で変化しました")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    num = np.asarray([v.numerator for v in current_fractions], dtype=np.int64)
    den = np.asarray([v.denominator for v in current_fractions], dtype=np.int64)
    filename = "solution_" + INITIAL.lower() + ".npz"
    np.savez_compressed(
        out / filename,
        representatives=np.asarray(model.representatives, dtype=np.int8),
        orbit_weights=model.orbit_weights,
        mechanism_vector=current,               # 参照用float近似
        mechanism_numerators=num,               # 厳密な値は num/den
        mechanism_denominators=den,
        initial=init,
        rsd_initial=rsd,
        uniform_initial=uniform,
        objective_history=np.asarray(history, dtype=np.float64),
        exact_sp_certified=np.asarray(True),
        exact_eq_certified=np.asarray(True),
    )
    save_csv(out / ("history_" + INITIAL.lower() + ".csv"), log)

    # PDFに必要な選好プロファイル別の比較も保存。
    print("代表選好プロファイルごとのRSD比較を計算中...", flush=True)
    per_profile: list[dict] = []
    better = worse = equal = 0
    for i, (rep, weight) in enumerate(zip(model.representatives, model.orbit_weights)):
        baseline_loss = loss_for_rep(model, i, rsd)
        final_loss = loss_for_rep(model, i, current)
        change = baseline_loss - final_loss
        if change > 1e-8:
            better += 1
        elif change < -1e-8:
            worse += 1
        else:
            equal += 1
        per_profile.append({
            "representative_id": i,
            "preference_profile": json.dumps(rep, ensure_ascii=False),
            "orbit_weight": int(weight),
            "rsd_oe": baseline_loss,
            "final_oe": final_loss,
            "improvement_over_rsd": change,
        })
    save_csv(out / ("profile_comparison_" + INITIAL.lower() + ".csv"), per_profile)

    summary = {
        "initial_mechanism": INITIAL,
        "initial_oe": initial_eval.value,
        "uniform_oe": uniform_eval.value,
        "rsd_oe": rsd_eval.value,
        "final_oe": evaluation.value,
        "improvement_from_initial": initial_eval.value - evaluation.value,
        "improvement_over_rsd": rsd_eval.value - evaluation.value,
        "beats_rsd": bool(evaluation.value < rsd_eval.value - 1e-8),
        "iterations_accepted": len(history) - 1,
        "stop_reason": stop_reason,
        "exact_sp_certified": True,
        "exact_sp_violation": 0,
        "exact_equalities_certified": True,
        "floating_max_sp_violation": model.max_sp_violation(current),
        "floating_max_equality_violation": model.max_equality_violation(current),
        "maximum_fraction_denominator": int(np.max(den)),
        "rational_reconstruction_tolerance": args.rational_tol,
        "inner_lp_residual": evaluation.inner_residual,
        "number_of_representatives_better_than_rsd": better,
        "number_of_representatives_worse_than_rsd": worse,
        "number_of_representatives_equal_to_rsd": equal,
        "elapsed_seconds": time.perf_counter() - t0,
        "explanation": (
            "exact SP/EQ certification applies to rational numerator/denominator "
            "data; float mechanism_vector and OE values may have rounding error"
        ),
    }
    (out / ("summary_" + INITIAL.lower() + ".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=============== 最終結果 ===============")
    print(f"初期メカニズム        : {INITIAL}")
    print(f"RSD平均OE侵害度       : {rsd_eval.value:.12f}")
    print(f"一様割り当て平均OE侵害度: {uniform_eval.value:.12f}")
    print(f"最終平均OE侵害度     : {evaluation.value:.12f}")
    print(f"初期値からの改善量   : {summary['improvement_from_initial']:+.12g}")
    print(f"RSDからの改善量      : {summary['improvement_over_rsd']:+.12g}")
    print(f"RSDより良いか？      : {summary['beats_rsd']}")
    print(f"耐戦略性の厳密違反量 : 0（有理数の全SP制約で検証済み）")
    print(f"耐戦略性のfloat残差  : {summary['floating_max_sp_violation']:.15e}")
    print(f"二重確率性等のfloat残差: {summary['floating_max_equality_violation']:.15e}")
    print(f"停止理由             : {stop_reason}")
    print(f"保存先               : {out.resolve()}")
    print("ファイル: " + ", ".join(p.name for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()

