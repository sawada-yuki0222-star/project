#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSDより改善しそうなやつを探す
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

from matrix import build_or_load_model, solve_oe_violation
from main_lp import (
    evaluate, as_float, verify_exact, rationalize_and_verify,
    profile_comparison,
)

GRIDS = (12,24,48,72,96,120,144,168,192,240,288,360,480,576,
         720,960,1152,1440,1728,2304,2880,5760,10080,20160,40320,
         80640,161280,322560)


def integer_check(model, numerators: np.ndarray, denominator: int):
    """Exact (not tolerance-based) check using signed int64 sparse products.
    For present n=4, sizes/denominators cannot overflow int64.
    """
    if denominator < 1 or numerators.dtype != np.int64:
        return False, 'invalid integer format'
    if np.min(numerators) < 0 or np.max(numerators) > denominator:
        return False, 'probabilities out of range'
    eq = model.equality_matrix.tocsr()
    sp = model.sp_matrix.tocsr()
    for M in (eq, sp):
        if not np.array_equal(M.data, np.rint(M.data)):
            raise ValueError('noninteger constraint coefficient')
    if not np.array_equal(model.equality_rhs, np.rint(model.equality_rhs)):
        raise ValueError('noninteger RHS')
    lhs = eq.astype(np.int64) @ numerators
    rhs = model.equality_rhs.astype(np.int64) * denominator
    if np.any(lhs != rhs):
        ix = int(np.flatnonzero(lhs != rhs)[0])
        return False, f'equality {ix} residual {int(lhs[ix]-rhs[ix])}/{denominator}'
    lhs_sp = sp.astype(np.int64) @ numerators
    if np.any(lhs_sp < 0):
        ix = int(np.flatnonzero(lhs_sp < 0)[0])
        return False, f'SP {ix} residual {int(lhs_sp[ix])}/{denominator}'
    return True, 'exact SP and all equalities pass'


def get_exact_initial(model, raw: np.ndarray, tolerance=5e-9):
    """Try common small-denominator rational grids, then individual fractions.
    Does not silently repair an infeasible LP point; rejects it instead.
    """
    if not np.all(np.isfinite(raw)) or raw.shape != (model.number_of_variables,):
        return None, 'not finite or wrong shape', None
    for d in GRIDS:
        nums = np.rint(np.clip(raw,0,1)*d).astype(np.int64)
        if float(np.max(np.abs(raw-nums/d)))>tolerance:
            continue
        ok, why = integer_check(model, nums, d)
        if ok:
            return [Fraction(int(n),int(d)) for n in nums],why,d
    fractions, reason = rationalize_and_verify(model,raw,1_000_000,tolerance)
    if fractions is not None:
        return fractions,reason,'per_cell'
    return None,reason,None


def targets_at_rsd(model,rsd):
    """One inner OE LP per representative. Only positive-loss reps enter target."""
    dirs=np.zeros_like(rsd)
    values=np.zeros(len(model.representatives),dtype=float)
    for k in range(len(model.representatives)):
        sl=model.representative_slice(k)
        A=model.cumulative_matrices[k]
        score=model.score_vectors[k]
        p=rsd[sl]
        res=linprog(-score,A_ub=-A,b_ub=-(A@p),
                    A_eq=model.inner_equalities,b_eq=model.inner_rhs,
                    bounds=(0.0,1.0),method='highs-ds')
        if not res.success:
            raise RuntimeError(f'inner LP failure at {k}: {res.message}')
        value=max(0.,float(-res.fun-score@p))
        values[k]=value
        if value>1e-8:
            dirs[sl]=res.x-p
    return dirs, values


def objective_modes(model,dirs,values,seed):
    rng=np.random.default_rng(seed)
    weights=np.asarray(model.orbit_weights,dtype=float)
    positives=np.flatnonzero(values>1e-8)
    ordered=positives[np.argsort(-(values[positives]*weights[positives]))]
    modes=[]
    for name,keep in [
        ('OE_target_all',positives),
        ('OE_target_top16',ordered[:16]),
        ('OE_target_top48',ordered[:48]),
        ('OE_target_top96',ordered[:96]),
    ]:
        scalar=np.zeros(len(values),dtype=float)
        scalar[keep]=weights[keep]/float(weights.max())
        modes.append((name,(-dirs.reshape(-1,16)*scalar[:,None]).ravel()))
    for j in range(3):
        sel=rng.choice(positives,size=min(48,len(positives)),replace=False)
        scalar=np.zeros(len(values),dtype=float)
        scalar[sel]=weights[sel]/float(weights.max())
        modes.append((f'OE_target_subset_{j}',(-dirs.reshape(-1,16)*scalar[:,None]).ravel()))
    # Discrete random objective coefficients produce different LP vertices and
    # are often easier to reconstruct exactly than Gaussian coefficients.
    for j in range(3):
        modes.append((f'LP_discrete_{j}',rng.integers(-2,3,size=len(dirs)).astype(float)))
    return modes


def save_csv(path, rows):
    """Heterogeneous result records: take the union of all keys, not only first."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=names,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def resume_initials(out, model):
    """Read exact initial candidates from matrices CSV, avoiding expensive LP reruns."""
    with (out/'comparison_targeted_oe.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    for rec in records:
        for field in ['initial_oe_avg','initial_oe_total','final_oe_avg','final_oe_total']:
            if rec.get(field) not in ('',None):rec[field]=float(rec[field])
            else:rec[field]=None
        if rec.get('beats_rsd') in ('True','False'):
            rec['beats_rsd']=(rec['beats_rsd']=='True')
    from collections import defaultdict
    vectors=defaultdict(list)
    pfields=[f'p_{i+1}_{a+1}' for i in range(4) for a in range(4)]
    with (out/'initial_matrices.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            vectors[row['label']].extend(Fraction(float(row[c])).limit_denominator(5760) for c in pfields)
    valid=[]
    for rec in records:
        label=rec['label']
        if label not in vectors:continue
        fracs=vectors[label]
        ok,why=verify_exact(model,fracs)
        if not ok:raise RuntimeError(f'failed to certify resumed {label}: {why}')
        valid.append((label,fracs,rec))
    return records,valid


def optimize_bounded(model, initial, baseline, args, label, out):
    """Same MM linearization/line search as main_lp.optimize, but with a per-LP
    time limit and certified checkpoints; returns best certified candidate.
    """
    ok,why=verify_exact(model,initial)
    if not ok:raise RuntimeError('Initial SP/EQ certificate failed: '+why)
    N=model.number_of_profiles
    current_frac=initial
    current=as_float(initial)
    evaluation=evaluate(model,current)
    initial_loss=evaluation.loss
    hist=[]
    stop='max_iterations_reached'
    def write_checkpoint():
        ok,why=verify_exact(model,current_frac)
        if not ok:raise RuntimeError('Final SP/EQ certificate failed: '+why)
        name=out/f'solution_{label}.npz'
        np.savez_compressed(name,
           mechanism_numerators=np.asarray([q.numerator for q in current_frac],dtype=np.int64),
           mechanism_denominators=np.asarray([q.denominator for q in current_frac],dtype=np.int64),
           mechanism_vector=current,
           initial_mechanism=as_float(initial),
           representatives=np.asarray(model.representatives,dtype=np.int8),
           orbit_weights=model.orbit_weights,rsd_initial=model.rsd_initial,
           objective_history=np.asarray([r['oe'] for r in hist],dtype=float),
           exact_sp_certified=np.asarray(True),exact_equalities_certified=np.asarray(True))
    def record(it,step,remark):
        hist.append({'label':label,'iteration':it,'step':float(step),
              'oe':evaluation.loss,'total_oe':evaluation.loss*N,
              'vs_RSD_total':(evaluation.loss-baseline)*N,
              'float_max_sp_violation':model.max_sp_violation(current),
              'float_max_eq_violation':model.max_equality_violation(current),
              'exact_sp_violation':0,'max_inner_lp_residual':evaluation.max_inner_residual,
              'remark':remark})
        print(f'[{label}] iter={it:02} total OE={evaluation.loss*N:.9f} vs RSD={(evaluation.loss-baseline)*N:+.9f} exact SP=0 step={float(step):g}',flush=True)
        write_checkpoint()
    record(0,0,'exact_initial')
    for it in range(1,args.max_iterations+1):
        res=linprog(c=np.asarray(evaluation.supergradient,dtype=float),
                A_ub=-model.sp_matrix,b_ub=np.zeros(model.sp_matrix.shape[0]),
                A_eq=model.equality_matrix,b_eq=model.equality_rhs,
                bounds=(0.0,1.0),method=args.method,
                options={'time_limit':args.lp_time_limit,
                         'primal_feasibility_tolerance':1e-9,
                         'dual_feasibility_tolerance':1e-9})
        if not res.success or res.x is None:
            stop='master_LP_timeout_or_failure: '+str(res.message)
            break
        accepted=False
        issue='no_certified_candidate'
        raw=np.asarray(res.x,dtype=float)
        for a in range(args.max_line_search):
            alpha=Fraction(1,2**a)
            trial=current+float(alpha)*(raw-current)
            rat,reason,_den=get_exact_initial(model,trial,args.rational_tolerance)
            if rat is None:issue=str(reason);continue
            float_trial=as_float(rat)
            new_ev=evaluate(model,float_trial)
            if new_ev.loss<=evaluation.loss+1e-10:
                gain=evaluation.loss-new_ev.loss
                current_frac,current,evaluation=rat,float_trial,new_ev
                accepted=True
                record(it,alpha,'accepted_exact_sp')
                if gain<=args.relative_tolerance*max(1.,abs(initial_loss)):
                    stop='no_significant_improvement'
                break
            issue=f'OE would increase by {new_ev.loss-evaluation.loss:.4e}'
        if not accepted:
            stop='no_exact_feasible_nonincreasing_candidate: '+issue
            break
        if stop=='no_significant_improvement':break
    write_checkpoint()
    return ({'initial_oe':initial_loss,'final_oe':evaluation.loss,
             'beats_rsd':bool(evaluation.loss<baseline-args.comparison_tolerance),
             'stop_reason':stop,'exact_sp_violation':0}, hist, current)


def export_initial_matrices(path,model,solutions):
    names=[f'p_{i+1}_{a+1}' for i in range(4) for a in range(4)]
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['label','representative_id','profile','orbit_weight',*names])
        for label,fracs in solutions:
            for k,p in enumerate(model.representatives):
                sl=model.representative_slice(k)
                writer.writerow([label,k,json.dumps(np.asarray(p).tolist()),
                    int(model.orbit_weights[k]),*[float(q) for q in fracs[sl]]])


def main():
    parser=argparse.ArgumentParser(description='RSDのOE侵害プロファイルを狙う、SP厳密検証つき多初期値探索')
    root=Path(__file__).resolve().parent.parent
    parser.add_argument('--cache',type=Path,default=root/'model_cache.pkl')
    parser.add_argument('--output-dir',type=Path,default=root/'results_targeted_oe')
    parser.add_argument('--seed',type=int,default=20261105)
    parser.add_argument('--max-optimizations',type=int,default=3)
    parser.add_argument('--max-iterations',type=int,default=2)
    parser.add_argument('--max-line-search',type=int,default=5)
    parser.add_argument('--method',choices=['highs','highs-ipm','highs-ds'],default='highs-ipm')
    parser.add_argument('--lp-time-limit',type=float,default=40.0)
    parser.add_argument('--comparison-tolerance',type=float,default=1e-8)
    parser.add_argument('--max-denominator',type=int,default=1_000_000)
    parser.add_argument('--rational-tolerance',type=float,default=5e-9)
    parser.add_argument('--relative-tolerance',type=float,default=1e-9)
    parser.add_argument('--skip-profile-csv',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue optimization from previously saved exact initial_matrices.csv')
    args=parser.parse_args()
    if args.max_iterations<0 or args.max_optimizations<0: parser.error('counts must be nonnegative')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    model=build_or_load_model(args.cache)
    assert len(model.representatives)==762 and model.number_of_profiles==331776
    assert model.sp_matrix.shape[0]==209963
    rsd=np.asarray(model.rsd_initial,dtype=float)
    baseline=evaluate(model,rsd).loss
    N=model.number_of_profiles
    print(f'RSD average OE={baseline:.12f}; total OE={baseline*N:.9f}',flush=True)
    records=[]; valid=[]
    def save_state():
        save_csv(out/'comparison_targeted_oe.csv',records)
        (out/'summary_targeted_oe.json').write_text(json.dumps({
            'status':'partial_or_complete','total_profiles':N,'representatives':762,
            'sp_constraints':209963,'rsd_average_oe':baseline,'rsd_total_oe':baseline*N,
            'number_of_initials_tested':len(records),
            'number_of_exact_initials':len(valid),
            'optimized':sum(r.get('status')=='optimized_verified' for r in records),
            'found_better_than_rsd':any(r.get('beats_rsd') is True for r in records),
            'notes':'All saved feasible candidates certified by exact integer/Fraction constraints. Objective values are numerical LP results, not global certificates.',
            'results':records},ensure_ascii=False,indent=2),encoding='utf-8')
    if args.resume:
        records,valid=resume_initials(out,model)
        print(f'Resumed {len(valid)} exact SP initial mechanisms',flush=True)
    else:
        dirs,values=targets_at_rsd(model,rsd)
        modes=objective_modes(model,dirs,values,args.seed)
        print(f'RSD OE-positive representatives={int(np.sum(values>1e-8))}; LP objectives={len(modes)}',flush=True)
        np.savez_compressed(out/'oe_target_vectors.npz',direction=dirs,rsd_profile_oe=values,
                            representatives=np.asarray(model.representatives,dtype=np.int8),
                            orbit_weights=model.orbit_weights)
        for j,(label,c) in enumerate(modes):
            rec={'label':label,'status':'LP_not_solved','initial_oe_avg':None,'initial_oe_total':None,
                 'final_oe_avg':None,'final_oe_total':None,'improvement_over_rsd_total':None,
                 'exact_sp_violation':None,'beats_rsd':None,'stop_reason':None}
            print(f'[{j+1}/{len(modes)}] {label} solve global SP feasible LP...',flush=True)
            try:
                raw=linprog(c=np.asarray(c,dtype=float),
                            A_ub=-model.sp_matrix,b_ub=np.zeros(model.sp_matrix.shape[0]),
                            A_eq=model.equality_matrix,b_eq=model.equality_rhs,
                            bounds=(0.0,1.0),method=args.method,
                            options={'time_limit':args.lp_time_limit,'primal_feasibility_tolerance':1e-9,
                                     'dual_feasibility_tolerance':1e-9})
                if not raw.success or raw.x is None:
                    rec['stop_reason']=str(raw.message);records.append(rec);save_state();continue
                xx=np.asarray(raw.x,dtype=float)
                rec['raw_float_sp_violation']=float(model.max_sp_violation(xx))
                rec['raw_float_eq_violation']=float(model.max_equality_violation(xx))
                fracs,why,den=get_exact_initial(model,xx,args.rational_tolerance)
                if fracs is None:
                    rec.update(status='uncertified_lp',stop_reason=why)
                    records.append(rec);save_state();continue
                exact=as_float(fracs)
                loss=evaluate(model,exact).loss
                rec.update(status='exact_initial',initial_oe_avg=float(loss),initial_oe_total=float(loss*N),
                           initial_max_difference_rsd=float(np.max(np.abs(exact-rsd))),
                           exact_sp_violation=0,exact_equalities_hold=True,
                           rational_denominator=str(den),stop_reason='not yet optimized')
                valid.append((label,fracs,rec))
                print(f'  exact SP=0, OE_total={loss*N:.9f}, diff_RSD={(loss-baseline)*N:+.9f}, den={den}',flush=True)
            except Exception as ex:
                rec.update(status='LP_or_OE_error',stop_reason=repr(ex))
            records.append(rec);save_state()
        if valid:
            export_initial_matrices(out/'initial_matrices.csv',model,[(x[0],x[1]) for x in valid])
    history=[]
    selected=sorted(valid,key=lambda item:(item[2]['initial_oe_avg'],item[0]))[:args.max_optimizations]
    best=None
    for label,fracs,rec in selected:
        print(f'Optimize {label} with the same MM as project/新モデル/main_lp.py...',flush=True)
        try:
            result,h,final=optimize_bounded(model,fracs,baseline,args,label,out)
            history.extend(h)
            rec.update(status='optimized_verified',final_oe_avg=float(result['final_oe']),
                       final_oe_total=float(result['final_oe']*N),
                       improvement_over_rsd_total=float((baseline-result['final_oe'])*N),
                       exact_sp_violation=0,
                       final_max_difference_rsd=float(np.max(np.abs(final-rsd))),
                       final_is_rsd_exact=bool(np.max(np.abs(final-rsd))<1e-12),
                       beats_rsd=bool(result['beats_rsd']),stop_reason=result['stop_reason'])
            if best is None or result['final_oe']<best[0]:best=(result['final_oe'],label,final)
            print(f'  FINAL total OE={result["final_oe"]*N:.9f}, RSD diff={(result["final_oe"]-baseline)*N:+.9f}, exact SP=0',flush=True)
        except Exception as ex:
            rec.update(status='optimization_error',stop_reason=repr(ex));print('  ERROR',repr(ex),flush=True)
        save_csv(out/'history_targeted_oe.csv',history)
        save_state()
    if best is not None and not args.skip_profile_csv:
        profile_comparison(model,rsd,best[2],out/'best_profile_comparison.csv')
    save_state()
    print('\n=== TARGETED SP+SYM SEARCH SUMMARY ===')
    for r in records:
        print(r['label'],r['status'],'initial_total',r.get('initial_oe_total'),
              'final_total',r.get('final_oe_total'),'SP exact',r.get('exact_sp_violation'))
    print('Output:',out.resolve(),'time(s)',round(time.monotonic()-t0,2),flush=True)

if __name__=='__main__':main()