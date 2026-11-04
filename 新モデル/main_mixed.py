#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4人4財のランダム割り当て：一様とRSDの凸結合の初期値からOE違反を最小化する。
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
from matrix import build_or_load_model, solve_oe_violation

@dataclass
class Evaluation:
    loss: float
    supergradient: np.ndarray
    max_inner_residual: float


def evaluate(model, vector: np.ndarray) -> Evaluation:
    """OE侵害度とMM上界用のsupergradient。"""
    value = 0.0
    grad = np.zeros(model.number_of_variables, dtype=np.float64)
    inner_res = 0.0
    for j, (cumulative, score, weight) in enumerate(zip(
        model.cumulative_matrices, model.score_vectors, model.normalized_weights
    )):
        block = model.representative_slice(j)
        ret = solve_oe_violation(
            cumulative=cumulative,
            score=score,
            inner_equalities=model.inner_equalities,
            inner_rhs=model.inner_rhs,
            allocation_vector=vector[block],
        )
        value += float(weight) * ret.value
        grad[block] = float(weight) * ret.supergradient
        inner_res = max(inner_res, float(ret.max_primal_violation))
    if inner_res > 1e-7:
        raise RuntimeError(f'OE内側LPの数値残差が大きい: {inner_res:.3e}')
    return Evaluation(float(value), grad, inner_res)


def exact_sparse_check(matrix, rhs, fractions: list[Fraction], *, equality: bool):
    """整数係数の疎行列で全行をFraction検査。"""
    m = matrix.tocsr()
    if not np.all(np.isfinite(m.data)) or not np.array_equal(m.data, np.rint(m.data)):
        raise RuntimeError('制約行列の係数が整数でないため厳密検証できません')
    coeffs = np.asarray(np.rint(m.data), dtype=np.int64)
    for i in range(m.shape[0]):
        value = Fraction(0, 1)
        for p in range(int(m.indptr[i]), int(m.indptr[i+1])):
            value += int(coeffs[p]) * fractions[int(m.indices[p])]
        bound = Fraction(0, 1) if rhs is None else Fraction(float(rhs[i]))
        if equality and value != bound:
            return False, f'等式 {i} に違反: residual={value-bound}'
        if not equality and value < bound:
            return False, f'SP不等式 {i} に違反: residual={value-bound}'
    return True, 'OK'


def verify_exact(model, fractions: list[Fraction]):
    if len(fractions) != model.number_of_variables:
        return False, 'ベクトル長が異なります'
    if any(q < 0 or q > 1 for q in fractions):
        return False, '[0,1] 範囲外の確率があります'
    ok, why = exact_sparse_check(model.equality_matrix, model.equality_rhs, fractions, equality=True)
    if not ok:
        return False, why
    ok, why = exact_sparse_check(model.sp_matrix, None, fractions, equality=False)
    if not ok:
        return False, why
    return True, '全ての等式・SP不等式を有理数で厳密に検証（SP違反=0）'


def rationalize_and_verify(model, x: np.ndarray, max_den: int, approximation_tol: float):
    """数値解の有理数復元を試し、全制約の厳密な成立を確認する。"""
    if x.shape != (model.number_of_variables,) or not np.all(np.isfinite(x)):
        return None, '解のサイズまたは有限性が不正'
    fractions = []
    for j, raw in enumerate(x):
        raw = float(raw)
        if raw < -approximation_tol or raw > 1 + approximation_tol:
            return None, f'確率[{j}]が範囲外: {raw}'
        clipped = min(1.0, max(0.0, raw))
        q = Fraction(clipped).limit_denominator(max_den)
        if abs(float(q)-clipped) > approximation_tol:
            return None, f'有理数復元不可: {j}'
        fractions.append(q)
    ok, why = verify_exact(model, fractions)
    return (fractions, why) if ok else (None, why)


def as_float(fractions: list[Fraction]) -> np.ndarray:
    return np.fromiter((float(v) for v in fractions), dtype=np.float64,
                       count=len(fractions))


def solve_full_sp_lp(model, costs, method: str):
    """全SP制約、二重確率性、匿名性・中立性を制約したLP。"""
    result = linprog(
        c=np.asarray(costs, dtype=np.float64),
        A_ub=-model.sp_matrix, b_ub=np.zeros(model.sp_matrix.shape[0]),
        A_eq=model.equality_matrix, b_eq=model.equality_rhs,
        bounds=(0.0, 1.0), method=method,
        options={
            'primal_feasibility_tolerance': 1e-9,
            'dual_feasibility_tolerance': 1e-9,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f'全SP制約LPが解けません: {result.message}')
    return np.asarray(result.x, dtype=np.float64)


def optimize(model, initial: list[Fraction], rsd_oe: float, args,
             label: str, output: Path):
    """検証済み有理数初期値をMMで改善。各候補も厳密に検証してから採用。"""
    ok, why = verify_exact(model, initial)
    if not ok:
        raise RuntimeError(f'{label}: 初期値が厳密実行可能でない: {why}')
    current_fractions = initial
    current = as_float(initial)
    current_ev = evaluate(model, current)
    initial_oe = current_ev.loss
    history = []
    base_time = time.monotonic()
    def record(k, alpha, remark):
        history.append({
            'label': label, 'iteration': k, 'step': alpha,
            'oe': current_ev.loss,
            'difference_vs_rsd': current_ev.loss-rsd_oe,
            'improvement_over_rsd': rsd_oe-current_ev.loss,
            'float_max_sp_violation': model.max_sp_violation(current),
            'float_max_equality_violation': model.max_equality_violation(current),
            'exact_sp_violation': 0,
            'max_inner_lp_residual': current_ev.max_inner_residual,
            'elapsed_seconds': time.monotonic()-base_time,
            'remark': remark,
        })
        print(f'[{label}] iter={k:02d} OE={current_ev.loss:.12f} '
              f'vs_RSD={current_ev.loss-rsd_oe:+.12f} '
              f'SP(exact)=0 SP(float)={model.max_sp_violation(current):.2e} '
              f'step={alpha:.4g} {remark}', flush=True)
    record(0, 0.0, 'initial')
    stop_reason = 'max_iterations_reached'
    for it in range(1, args.max_iterations+1):
        raw = solve_full_sp_lp(model, current_ev.supergradient, args.method)
        accepted = False
        issue = 'no candidate'
        for attempt in range(args.max_line_search):
            alpha = 2.0**(-attempt)
            trial = current + alpha*(raw-current)
            trial_fractions, reason = rationalize_and_verify(
                model, trial, args.max_denominator, args.rational_tolerance
            )
            if trial_fractions is None:
                issue = reason
                continue
            trial_float = as_float(trial_fractions)
            trial_ev = evaluate(model, trial_float)
            if trial_ev.loss <= current_ev.loss + 1e-10:
                change = current_ev.loss - trial_ev.loss
                current_fractions, current, current_ev = trial_fractions, trial_float, trial_ev
                record(it, alpha, 'accepted_exact_sp')
                accepted = True
                if change <= args.relative_tolerance*max(1.0, abs(initial_oe)):
                    stop_reason='no_significant_improvement'
                break
            issue = f'OE grew: {trial_ev.loss-current_ev.loss:.4e}'
        if not accepted:
            stop_reason='no_exact_feasible_improving_candidate: '+issue
            print(f'[{label}] No update: {issue}', flush=True)
            break
        if stop_reason=='no_significant_improvement':
            break
    # 保存直前の厳密証明。検証に失敗した解を成功扱いで保存しない。
    ok, why=verify_exact(model, current_fractions)
    if not ok:
        raise RuntimeError('final exact certificate failed: '+why)
    token = label.replace('.', 'p').replace('-', 'minus')
    np.savez_compressed(
        output/f'solution_{token}.npz',
        mechanism_numerators=np.asarray([q.numerator for q in current_fractions], dtype=np.int64),
        mechanism_denominators=np.asarray([q.denominator for q in current_fractions], dtype=np.int64),
        mechanism_vector=current,
        representatives=np.asarray(model.representatives, dtype=np.int8),
        orbit_weights=model.orbit_weights,
        rsd_initial=model.rsd_initial,
        initial_mechanism=as_float(initial),
        objective_history=np.asarray([r['oe'] for r in history]),
        exact_sp_certified=np.asarray(True),
        exact_equalities_certified=np.asarray(True),
    )
    result={
        'label':label, 'status':'verified', 'initial_oe':initial_oe,
        'final_oe':current_ev.loss, 'rsd_oe':rsd_oe,
        'difference_vs_rsd':current_ev.loss-rsd_oe,
        'improvement_over_rsd':rsd_oe-current_ev.loss,
        'beats_rsd':bool(current_ev.loss < rsd_oe-args.comparison_tolerance),
        'improvement_from_initial':initial_oe-current_ev.loss,
        'iterations_accepted':len(history)-1, 'stop_reason':stop_reason,
        'exact_sp_violation':0, 'exact_equalities_hold':True,
        'float_max_sp_violation':model.max_sp_violation(current),
        'float_max_equality_violation':model.max_equality_violation(current),
        'max_inner_lp_residual':current_ev.max_inner_residual,
        'maximum_fraction_denominator':max(q.denominator for q in current_fractions),
    }
    return result, history, current


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_comparison(model, rsd: np.ndarray, solution: np.ndarray,
                       path: Path):
    """最終OEが最小のケースについて選好代表別のOE差を保存。"""
    result=[]
    for i, (profile, weight) in enumerate(zip(model.representatives, model.orbit_weights)):
        def loss(x):
            return solve_oe_violation(
                cumulative=model.cumulative_matrices[i],
                score=model.score_vectors[i],
                inner_equalities=model.inner_equalities,
                inner_rhs=model.inner_rhs,
                allocation_vector=x[model.representative_slice(i)],
            ).value
        reference=loss(rsd)
        candidate=loss(solution)
        result.append({
            'representative_id':i,
            'profile':json.dumps(profile, ensure_ascii=False),
            'orbit_weight':int(weight),
            'rsd_oe':reference,
            'candidate_oe':candidate,
            'improvement_over_rsd':reference-candidate,
        })
    save_csv(path,result)


def base_parser(description: str, default_output: str):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--cache',type=Path,default=Path('model_cache.pkl'))
    parser.add_argument('--output-dir',type=Path,default=Path(default_output))
    parser.add_argument('--max-iterations',type=int,default=5)
    parser.add_argument('--max-line-search',type=int,default=13)
    parser.add_argument('--method',choices=['highs-ds','highs-ipm','highs'],default='highs-ds')
    parser.add_argument('--max-denominator',type=int,default=1_000_000)
    parser.add_argument('--rational-tolerance',type=float,default=5e-9)
    parser.add_argument('--relative-tolerance',type=float,default=1e-9)
    parser.add_argument('--comparison-tolerance',type=float,default=1e-8)
    return parser


def check_args(parser, args):
    if args.max_iterations < 0 or args.max_line_search < 1 or args.max_denominator < 1:
        parser.error('max-iterations >= 0、max-line-search >=1、max-denominator>=1 が必要')
    if args.rational_tolerance < 0 or args.comparison_tolerance < 0 or args.relative_tolerance < 0:
        parser.error('許容誤差は負にできません')


def main():
    parser=base_parser('RSD×一様割り当ての混合からOEを最小化', 'results_mixed')
    parser.add_argument('--lambdas',default='0.1,0.5,0.9',
                        help='RSD混合率。カンマ区切り (例: 0.1,0.5,0.9)')
    args=parser.parse_args()
    check_args(parser,args)
    try:
        ratios=[Fraction(v.strip()) for v in args.lambdas.split(',')]
    except Exception as exc:
        parser.error(f'--lambdas が不正です: {exc}')
    if not ratios or any(q < 0 or q > 1 for q in ratios):
        parser.error('lambdaは0以上1以下、1つ以上指定してください')
    if len(set(ratios))!=len(ratios):
        parser.error('混合比率に重複があります')

    print('モデルをロード中...', flush=True)
    model=build_or_load_model(args.cache)
    print(f'代表選好={len(model.representatives)}、SP制約数={model.sp_matrix.shape[0]}', flush=True)
    rsd_float=np.asarray(model.rsd_initial, dtype=np.float64)
    rsd_fractions, why=rationalize_and_verify(model, rsd_float, args.max_denominator,
                                               args.rational_tolerance)
    if rsd_fractions is None:
        raise RuntimeError('RSD初期値の厳密検証に失敗: '+why)
    rsd_float=as_float(rsd_fractions)
    rsd_oe=evaluate(model, rsd_float).loss
    print(f'RSD benchmark average OE: {rsd_oe:.12f}',flush=True)
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    all_summaries=[]
    all_histories=[]
    best=None
    for lam in ratios:
        label='mixed_'+str(float(lam))
        # RSDも一様割り当ても確率・SP・対称性を満たすので、有理数での凸結合は厳密に実行可能。
        initial=[lam*q+(1-lam)*Fraction(1,model.n) for q in rsd_fractions]
        result,hist,solution=optimize(model,initial,rsd_oe,args,label,output)
        result['lambda']=float(lam)
        all_summaries.append(result)
        all_histories.extend(hist)
        if best is None or result['final_oe'] < best[0]['final_oe']:
            best=(result,solution)
    save_csv(output/'comparison_mixed.csv',all_summaries)
    save_csv(output/'history_mixed.csv',all_histories)
    if best is not None:
        profile_comparison(model, rsd_float, best[1],output/'best_profile_comparison_mixed.csv')
    (output/'summary_mixed.json').write_text(
        json.dumps({'rsd_oe':rsd_oe, 'runs':all_summaries},ensure_ascii=False,indent=2),
        encoding='utf-8')
    print('\n======== MIXED vs RSD ========')
    for r in all_summaries:
        print(f"λ={r['lambda']:g}: initial OE={r['initial_oe']:.12f}, "
              f"final OE={r['final_oe']:.12f}, vs RSD={r['difference_vs_rsd']:+.12f}, "
              f"SP(exact)={r['exact_sp_violation']}\n"
              f"  beat RSD? {r['beats_rsd']}  reason={r['stop_reason']}")
    print(f'CSV / NPZ saved to {output.resolve()}')

if __name__=='__main__':
    main()
