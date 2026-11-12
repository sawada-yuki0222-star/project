#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SP/匿名性/中立性を厳密に満たす多数の 4x4 ランダム割当メカニズムからOEを探索。
"""
from __future__ import annotations
import argparse
import csv
import itertools
import subprocess
import sys
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from main_lp import evaluate, as_float, rationalize_and_verify, verify_exact
from scipy.optimize import linprog
from matrix import build_or_load_model, solve_oe_violation

N = 4
PERMS = np.asarray(list(itertools.permutations(range(N))), dtype=np.int8)
PRIORITY = np.repeat(PERMS, len(PERMS), axis=0)
OBJECT = np.tile(PERMS, (len(PERMS), 1))
NC = len(PRIORITY)      # 576


def rank_array(representatives):
    reps = np.asarray(representatives, dtype=np.int8)
    ranks = np.empty((len(reps), N, N), dtype=np.int8)
    for t in range(N):
        np.put_along_axis(ranks, reps[:, :, t:t+1], t, axis=2)
    return ranks


def menu_basis_counts(ranks: np.ndarray, template: tuple[int, ...]):
    """各repにおける正確な分子countsを返す。共通分母は576。

    template[t]は固定の外生的メニューを示す4bit整数(0..15)。
    ラベルをすべて置換して選び、各段階のagentはメニュー内の残存財から
    自分の真の最良財を選ぶ（なければ外生的に固定した順で割当）。
    各決定ルールは、本人にとって耐戦略的である。
    """
    if len(template) != N or any(not 0 <= x <= 15 for x in template):
        raise ValueError("template must contain four 4-bit masks")
    nr = ranks.shape[0]
    avail = np.ones((nr, NC, N), dtype=bool)
    reps_i = np.arange(nr)[:, None]
    cols = np.arange(NC)[None, :]
    counts = np.zeros((nr, N, N), dtype=np.int32)
    # The order among objects used for fallback is exogenous, so reports cannot affect it.
    fallback_rank = np.empty((NC, N), dtype=np.int8)
    for pos in range(N):
        fallback_rank[cols.ravel(), OBJECT[:, pos]] = pos
    for turn, mask in enumerate(template):
        menu = np.zeros((NC, N), dtype=bool)
        for obj in range(N):
            if (mask >> obj) & 1:
                menu[cols.ravel(), OBJECT[:, obj]] = True
        ag = PRIORITY[:, turn]
        agent_ranks = ranks[reps_i, ag[None, :], :]
        candidates = avail & menu[None, :, :]
        eligible = candidates.any(axis=2)
        top = np.where(candidates, agent_ranks, 99).argmin(axis=2)
        fallback = np.where(avail, fallback_rank[None, :, :], 99).argmin(axis=2)
        picked = np.where(eligible, top, fallback)
        indices = ((reps_i * N + ag[None, :]) * N + picked).ravel()
        counts += np.bincount(indices, minlength=nr*N*N).reshape(nr, N, N).astype(np.int32)
        avail[reps_i, cols, picked] = False
    if not np.all(counts.sum(axis=2) == NC) or not np.all(counts.sum(axis=1) == NC):
        raise AssertionError("Basis allocation is not bistochastic")
    return counts.reshape(-1)


def exact_integer_check(eq, beq, sp, numerator, denominator):
    """整数演算で全等式と全SP不等式を厳密検査。丸め誤差は一切なし。"""
    if not np.issubdtype(numerator.dtype, np.integer):
        raise TypeError("integer numerator required")
    if numerator.shape[0] != eq.shape[1] or denominator <= 0:
        raise ValueError("shape/denominator mismatch")
    if np.any(numerator < 0) or np.any(numerator > denominator):
        return False, "probability outside [0,1]"
    delta = eq @ numerator - beq*int(denominator)
    ix = np.flatnonzero(delta)
    if ix.size:
        return False, f"equality[{ix[0]}] residual={int(delta[ix[0]])}/{denominator}"
    vals = sp @ numerator
    bad = np.flatnonzero(vals < 0)
    if bad.size:
        return False, f"SP[{bad[0]}] residual={int(vals[bad[0]])}/{denominator}"
    return True, "all equalities & SP inequalities hold exactly"


def save_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if not rows: return
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def make_templates(random_bases: int, rng):
    # 0xF: unrestricted serial dictator; 0x0: ignore report and take a fixed good.
    templates = [
        (15,15,15,15), # RSD
        (0,0,0,0),     # uniform
        (15,0,0,0), (15,15,0,0), (15,15,15,0),
        (0,15,15,15), (0,0,15,15), (0,0,0,15),
        (3,3,3,3), (7,7,7,7),
        (15,3,15,0), (3,15,0,15),
        (3,12,3,12), (5,10,5,10),
        (1,3,7,15), (15,7,3,1),
        (7,3,15,0), (15,0,7,3),
        (0,15,3,7), (3,7,15,0),
    ]
    seen=set(templates)
    while len(templates) < 20+random_bases:
        tpl=tuple(int(z) for z in rng.integers(0,16,size=N))
        if tpl not in seen:
            seen.add(tpl);templates.append(tpl)
    return templates


def choose_optimization_ids(records, candidates, target):
    """OEが低いものだけでなく、異なる開始点を選び探索の偏りを減らす。"""
    ready = [i for i,r in enumerate(records) if r["status"]=="validated_initial" and r["initial_oe_avg"] is not None]
    if target<=0: return []
    # Exclude RSD benchmark as one of the optimized starts, except when only one exists.
    ready = [i for i in ready if records[i]["label"]!="RSD"]
    if not ready:return []
    sorted_best = sorted(ready,key=lambda i:(records[i]["initial_oe_avg"],i))
    sorted_worst = sorted(ready,key=lambda i:(-records[i]["initial_oe_avg"],i))
    picks=[]
    for seq in [sorted_best, sorted_worst]:
        take = max(1, target//3)
        for i in seq:
            if len([x for x in picks if x in seq[:take]])>=take:break
            if i not in picks:picks.append(i)
    # greedy spread over allocations, minimize max similarity
    rest = [i for i in ready if i not in picks]
    while len(picks)<target and rest:
        # candidate vectors have common small denominators
        best=max(rest,key=lambda j:min(float(np.max(np.abs(
            candidates[j][0]/candidates[j][1] - candidates[i][0]/candidates[i][1]
        ))) for i in picks) if picks else 0)
        picks.append(best);rest.remove(best)
    return picks[:target]


def profile_csv(model, solution, rsd, out):
    data=[]
    for k,profile in enumerate(model.representatives):
        sl=model.representative_slice(k)
        a=model.cumulative_matrices[k];s=model.score_vectors[k]
        val=[]
        for x in (rsd[sl],solution[sl]):
            z=solve_oe_violation(a,s,model.inner_equalities,model.inner_rhs,x)
            val.append(float(z.value))
        data.append(dict(representative_id=k,preference_profile=json.dumps(profile),
                         orbit_weight=int(model.orbit_weights[k]),rsd_oe=val[0],
                         final_oe=val[1],improvement_over_rsd=val[0]-val[1],
                         weighted_total_improvement=float(model.orbit_weights[k])*(val[0]-val[1])))
    save_csv(out,data)
    return data


def save_solution(outdir,label,initial_float,final_frac,model,history,rsd,verify):
    from math import gcd
    final_float=as_float(final_frac)
    safe="".join(z if z.isascii() and z.isalnum() else "_" for z in label)
    file=outdir/f"solution_{safe}.npz"
    np.savez_compressed(file,
        mechanism_numerators=np.asarray([f.numerator for f in final_frac],dtype=np.int64),
        mechanism_denominators=np.asarray([f.denominator for f in final_frac],dtype=np.int64),
        mechanism_vector=final_float,initial_mechanism=initial_float,
        representatives=np.asarray(model.representatives,dtype=np.int8),
        orbit_weights=model.orbit_weights,rsd_initial=rsd,
        objective_history=np.asarray(history,dtype=float),
        exact_sp_certified=np.array(True),exact_equalities_certified=np.array(True))
    return file


def main():
    ap=argparse.ArgumentParser(description="大量の厳密SP/匿名/中立な初期値でMM多開始点探索")
    root=Path(__file__).resolve().parent.parent
    ap.add_argument("--cache",type=Path,default=root/"model_cache.pkl")
    ap.add_argument("--output-dir",type=Path,default=root/"results_many_sp")
    ap.add_argument("--random-bases",type=int,default=12)
    ap.add_argument("--mixtures",type=int,default=32)
    ap.add_argument("--optimize",type=int,default=6,help="何個の初期値からMMを実行するか")
    ap.add_argument("--max-iterations",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20261105)
    ap.add_argument("--max-denominator",type=int,default=1_000_000)
    ap.add_argument("--rational-tolerance",type=float,default=5e-9)
    ap.add_argument("--comparison-tolerance",type=float,default=1e-8)
    ap.add_argument("--max-line-search",type=int,default=4)
    ap.add_argument("--mixture-total",type=int,default=96)
    ap.add_argument("--method",choices=["highs","highs-ds","highs-ipm"],default="highs-ipm")
    ap.add_argument("--lp-time-limit",type=float,default=45.0,help="master LP 1回あたりの秒数上限")
    ap.add_argument("--no-profiles",action="store_true",help="最良解の762代表選好CSVを省略")
    ap.add_argument("--resume",action="store_true",help=argparse.SUPPRESS)
    ap.add_argument("--screen-remaining",action="store_true",help="保存された初期値でOE未計算のものだけを評価")
    ap.add_argument("--screen-limit",type=int,default=8,help="screen-remainingで1回に処理する最大件数")
    ap.add_argument("--force-id",type=int,default=None,help=argparse.SUPPRESS)
    ap.add_argument("--export-initial-csv",action="store_true",help="全初期行列を初期値×代表選好のCSVとして保存")
    ap.add_argument("--child-timeout",type=int,default=180,help="MM 1開始点の処理時間上限（秒）")
    args=ap.parse_args()
    if min(args.random_bases,args.mixtures,args.optimize,args.max_iterations)>=0 and args.mixture_total>0:
        pass
    else:ap.error("counts must be nonnegative and mixture-total positive")
    t0=time.monotonic()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    print("[1/5] GitHub新モデルの4人4財モデルを読み込みます",flush=True)
    model=build_or_load_model(args.cache)
    assert len(model.representatives)==762 and model.number_of_profiles==331776
    assert model.sp_matrix.shape[0]==209963
    for mat in (model.equality_matrix, model.sp_matrix):
        if not np.array_equal(mat.data,np.rint(mat.data)):
            raise RuntimeError("Cannot certify SP/EQ: constraint matrix has noninteger coefficients")
    if not np.array_equal(model.equality_rhs,np.rint(model.equality_rhs)):
        raise RuntimeError("Cannot certify EQ: noninteger right-hand side")
    eq=model.equality_matrix.astype(np.int64)
    beq=np.rint(model.equality_rhs).astype(np.int64)
    sp=model.sp_matrix.astype(np.int64)
    rsd=np.asarray(model.rsd_initial,dtype=float)
    rsd_fracs=[Fraction(float(v)).limit_denominator(24) for v in rsd]
    rsd_oe=evaluate(model,rsd).loss
    if args.resume:
        with np.load(out/"initial_mechanisms.npz") as z:
            base_count=int(z["menu_basis_counts"].shape[0])
            candidates=[(z["numerators"][i].astype(np.int64),int(z["denominators"][i]))
                        for i in range(len(z["denominators"]))]
        with (out/"comparison_many_sp.csv").open(encoding="utf-8-sig",newline="") as f:
            records=list(csv.DictReader(f))
        for r in records:
            if r["initial_oe_avg"]:
                r["initial_oe_avg"]=float(r["initial_oe_avg"])
                r["initial_oe_total"]=float(r["initial_oe_total"])
            else:
                r["initial_oe_avg"]=None
                r["initial_oe_total"]=None
            if r["final_oe_avg"]:
                r["final_oe_avg"]=float(r["final_oe_avg"])
                r["final_oe_total"]=float(r["final_oe_total"])
            else:
                r["final_oe_avg"]=None
                r["final_oe_total"]=None
            if r.get("improvement_over_rsd_total"):
                r["improvement_over_rsd_total"]=float(r["improvement_over_rsd_total"])
            if r.get("beats_rsd") in ("True","False"):
                r["beats_rsd"]=(r["beats_rsd"]=="True")
            r["iterations"]=int(r.get("iterations") or 0)
            r.setdefault("initial_max_abs_diff_rsd",None)
            r.setdefault("final_max_abs_diff_rsd",None)
            r.setdefault("final_is_rsd_exact",None)
        print(f"[2/5] 別プロセスで{len(candidates)}個の厳密SP初期値を読み込んだ",flush=True)
        if args.export_initial_csv:
            p=out/"initial_matrices.csv"
            names=[f"p_{i+1}_{a+1}" for i in range(4) for a in range(4)]
            with p.open("w",newline="",encoding="utf-8-sig") as f:
                writer=csv.writer(f)
                writer.writerow(["label","representative_id","profile","orbit_weight","common_denominator",*names])
                for cand_id,(num,den) in enumerate(candidates):
                    for k,profile in enumerate(model.representatives):
                        writer.writerow([records[cand_id]["label"],k,json.dumps(profile),
                            int(model.orbit_weights[k]),den,*[int(v)/den for v in num[k*16:(k+1)*16]]])
            print(f"Exported {len(candidates)*len(model.representatives)} matrices: {p}",flush=True)
            return
        if args.screen_remaining:
            done=0
            for i,(num,den) in enumerate(candidates):
                if records[i]["initial_oe_avg"] is not None: continue
                initial=num.astype(float)/den
                try:
                    loss=evaluate(model,initial).loss
                    records[i]["initial_oe_avg"]=loss
                    records[i]["initial_oe_total"]=loss*model.number_of_profiles
                    records[i]["status"]="validated_initial"
                    print(f"  screen {i+1:03}/{len(candidates)} {records[i]['label']}: OE_total={loss*model.number_of_profiles:.9f}",flush=True)
                except Exception as ex:
                    records[i]["status"]="initial_oe_failed"
                    records[i]["stop_reason"]=repr(ex)
                save_csv(out/"comparison_many_sp.csv",records)
                done+=1
                if done>=args.screen_limit:break
            remaining=sum(r["initial_oe_avg"] is None and r["status"]=="validated_initial" for r in records)
            print(f"screened now={done}, unscored remaining={remaining}",flush=True)
            return
    else:
        rank=rank_array(model.representatives)
        print(f"  representatives={len(model.representatives)} vars={model.number_of_variables}, SP={sp.shape[0]}",flush=True)
        print(f"  RSD mean OE={rsd_oe:.12f}; RSD total OE={rsd_oe*model.number_of_profiles:.9f}",flush=True)
        print("[2/5] 報告依存しないメニューをランダム化・対称化した基底を生成",flush=True)
        templates=make_templates(args.random_bases,rng)
        bases=[];basis_meta=[];seen=set()
        for tpl in templates:
            b=menu_basis_counts(rank,tpl)
            ok,why=exact_integer_check(eq,beq,sp,b,NC)
            if not ok:raise RuntimeError(f"invalid serial-menu mechanism {tpl}: {why}")
            fingerprint=b.tobytes()
            if fingerprint in seen:continue
            seen.add(fingerprint)
            label="RSD" if tpl==(15,)*4 else "UNIFORM" if tpl==(0,)*4 else "menu_"+"_".join(map(str,tpl))
            bases.append(b)
            basis_meta.append(dict(basis_id=len(bases)-1,label=label,menus=list(tpl)))
            print(f"  basis {len(bases):02} : {label} (integer SP=0)",flush=True)
        base_arr=np.stack(bases).astype(np.int32)
        base_count=len(bases)
        candidates=[];records=[];candidate_keys=set()
        def add_candidate(label,coeffs,method):
            coeffs=np.asarray(coeffs,dtype=np.int64)
            positive=np.flatnonzero(coeffs)
            if not len(positive):return
            # express all probabilities on common integral denominator
            denom=NC*int(coeffs.sum())
            num=(coeffs[positive,None]*base_arr[positive].astype(np.int64)).sum(axis=0)
            # remove common divisor: keep numerators modest for exact checks
            d=int(np.gcd.reduce(np.append(num,denom)))
            num=(num//d).astype(np.int64);denom//=d
            key=(num.tobytes(),denom)
            if key in candidate_keys:return
            candidate_keys.add(key)
            ok,why=exact_integer_check(eq,beq,sp,num,denom)
            if not ok:raise AssertionError(f"candidate {label} violates exact constraints: {why}")
            candidates.append((num,denom))
            records.append(dict(candidate_id=len(records),label=label,source=method,
                                mix_terms=int(len(positive)),common_denominator=denom,
                                exact_sp_violation=0,exact_equalities_hold=True,
                                status="validated_initial",initial_max_abs_diff_rsd=float(np.max(np.abs(num.astype(float)/denom-rsd))),
                                final_max_abs_diff_rsd=None,final_is_rsd_exact=None,initial_oe_avg=None,
                                initial_oe_total=None,final_oe_avg=None,final_oe_total=None,
                                improvement_over_rsd_total=None,beats_rsd=None,
                                iterations=0,stop_reason="not_optimized"))
        for i,meta in enumerate(basis_meta):
            c=np.zeros(len(bases),dtype=np.int64);c[i]=1
            add_candidate(meta["label"],c,"serial_menu")
        # diverse random rational mixtures, all *exact* convex combinations of SP bases
        for j in range(args.mixtures):
            c=np.zeros(len(bases),dtype=np.int64)
            k=int(rng.integers(2,min(len(bases),9)+1))
            chosen=rng.choice(len(bases),size=k,replace=False)
            alpha=rng.dirichlet(rng.uniform(.3,2.5,size=k))
            counts=rng.multinomial(args.mixture_total,alpha)
            c[chosen]=counts
            add_candidate(f"mix_{j:04}",c,"random_convex_mix")
        print(f"[3/5] {len(candidates)}個の初期ベクトルを整数で厳密検証済み。初期OEを計算",flush=True)
        # save candidates before optimization, so a partial run keeps all tested initials
        np.savez_compressed(out/"initial_mechanisms.npz",
            numerators=np.stack([a for a,d in candidates]),
            denominators=np.array([d for a,d in candidates],dtype=np.int64),
            labels=np.array([r["label"] for r in records]),
            representatives=np.asarray(model.representatives,dtype=np.int8),
            orbit_weights=model.orbit_weights,
            reference_RSD=rsd,
            menu_basis_counts=base_arr,
            menu_templates=np.asarray([x["menus"] for x in basis_meta],dtype=np.int8))
        # Each screening chunk runs in a fresh interpreter: repeated inner LPs can
        # otherwise accumulate solver state and slow down later LP optimization.
        save_csv(out/"comparison_many_sp.csv",records)
        chunks=(len(candidates)+args.screen_limit-1)//args.screen_limit
        for j in range(chunks):
            print(f"[3/5] OE screening chunk {j+1}/{chunks}",flush=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],
                 "--resume","--screen-remaining"]
            result=subprocess.run(cmd,check=False)
            if result.returncode:
                print("screening chunk interrupted, inspect comparison_many_sp.csv and resume",flush=True)
                raise RuntimeError(f"screening failed: {result.returncode}")
        if args.optimize > 0:
            print("[4/5] MMを候補ごとに新しいPythonプロセスで実行",flush=True)
            code=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                                 *sys.argv[1:],"--resume"],check=False)
            if code.returncode:
                raise RuntimeError(f"MM child process failed: exit={code.returncode}")
            return
        subprocess.run([sys.executable,str(Path(__file__).resolve()),
                        *sys.argv[1:],"--resume"],check=True)
        return
    if args.resume and args.force_id is None and args.optimize > 0:
        # isolate each objective minimization from the other runs
        selected=choose_optimization_ids(records,candidates,args.optimize)
        print(f"[4/5] MM starts selected: {[records[i]['label'] for i in selected]}",flush=True)
        for i in selected:
            args2=[sys.executable,str(Path(__file__).resolve()),
                   *sys.argv[1:],"--force-id",str(i),"--no-profiles"]
            try:
                result=subprocess.run(args2,timeout=args.child_timeout,check=False)
                if result.returncode:
                    print(f"candidate {i} process exited with {result.returncode}",flush=True)
            except subprocess.TimeoutExpired:
                print(f"candidate {i} timed out at {args.child_timeout}s; initial remains certified",flush=True)
                with (out/"comparison_many_sp.csv").open(encoding="utf-8-sig",newline="") as f:
                    current_rows=list(csv.DictReader(f))
                current_rows[i]["status"]="optimization_timeout"
                current_rows[i]["stop_reason"]=f"per-start timeout {args.child_timeout}s"
                save_csv(out/"comparison_many_sp.csv",current_rows)
        print(f"multistart batch finished: {out}",flush=True)
        return
    selected=[args.force_id] if args.force_id is not None else choose_optimization_ids(records,candidates,args.optimize)
    print(f"[4/5] 初期値{len(selected)}件からMM（各{args.max_iterations}反復まで）",flush=True)
    histpath=out/"history_many_sp.csv"
    if histpath.exists():
        with histpath.open(encoding="utf-8-sig",newline="") as f:
            history=list(csv.DictReader(f))
    else: history=[]
    results=[];best_id=None;best_loss=float("inf")
    for old_i,old_rec in enumerate(records):
        if old_rec["status"]=="optimized_verified" and old_rec["final_oe_avg"] is not None:
            if float(old_rec["final_oe_avg"])<best_loss:
                best_id=old_i;best_loss=float(old_rec["final_oe_avg"])
    for i in selected:
        rec=records[i];label=rec["label"];num,den=candidates[i]
        initial=num.astype(float)/den
        current=initial.copy()
        # Exactly construct rational starts from common numerator and denominator
        fractions=[Fraction(int(x),int(den)) for x in num]
        ok,reason=verify_exact(model,fractions)
        if not ok:
            rec.update(status="failed_start_verification",stop_reason=reason)
            continue
        loss=rec["initial_oe_avg"]
        ev=evaluate(model,current)
        used=0;stop="max_iterations_reached"
        def record(it,alpha,remark):
            history.append(dict(label=label,iteration=it,step=alpha,average_oe=ev.loss,
                 total_oe=ev.loss*model.number_of_profiles,
                 vs_RSD_total=(ev.loss-rsd_oe)*model.number_of_profiles,
                 exact_sp_violation=0,float_max_sp_violation=model.max_sp_violation(current),
                 float_max_equality_violation=model.max_equality_violation(current),
                 remark=remark,elapsed_seconds=time.monotonic()-t0))
            print(f"  {label} iter={it:02} total OE={ev.loss*model.number_of_profiles:.9f} "
                  f"vs_RSD={(ev.loss-rsd_oe)*model.number_of_profiles:+.9f} "
                  f"SP(exact)=0 alpha={alpha}",flush=True)
            save_csv(out/"history_many_sp.csv",history)
        record(0,0.0,"exact_initial")
        for it in range(1,args.max_iterations+1):
            print(f"    {label}: solve MM master LP it={it} time limit={args.lp_time_limit}s",flush=True)
            try:
                master=linprog(
                    c=np.asarray(ev.supergradient,dtype=np.float64),
                    A_ub=-model.sp_matrix,b_ub=np.zeros(model.sp_matrix.shape[0]),
                    A_eq=model.equality_matrix,b_eq=model.equality_rhs,
                    bounds=(0.,1.),method=args.method,
                    options={"time_limit":args.lp_time_limit,
                             "primal_feasibility_tolerance":1e-9,
                             "dual_feasibility_tolerance":1e-9})
                print(f"    master finished: {master.message}",flush=True)
                if not master.success or master.x is None:
                    stop="master_LP_not_solved: "+master.message;break
                raw=np.asarray(master.x,dtype=np.float64)
            except Exception as ex:
                stop="master_LP_failed: "+str(ex);break
            accepted=False;error="no rational candidate"
            for a in range(args.max_line_search):
                alpha=Fraction(1,2**a)
                # robust strategy: rationalize raw LP first. When this fails,
                # approximate the trial and then *certify* every constraint.
                trial=current+float(alpha)*(raw-current)
                print(f"    testing alpha={alpha}",flush=True)
                trial_rat,why=rationalize_and_verify(model,trial,args.max_denominator,args.rational_tolerance)
                if trial_rat is None:
                    error=why;continue
                print("    rational verified",flush=True)
                trial_float=as_float(trial_rat)
                try: ev2=evaluate(model,trial_float)
                except Exception as ex:error=str(ex);continue
                if ev2.loss <= ev.loss + 1e-10:
                    gain=ev.loss-ev2.loss
                    fractions=trial_rat;current=trial_float;ev=ev2
                    used=it;accepted=True
                    record(it,float(alpha),"accepted_exact_sp")
                    if fractions == rsd_fracs:
                        stop="reached_exact_RSD"
                    elif gain <= 1e-9*max(1.,abs(loss)):
                        stop="no_significant_improvement"
                    break
                error=f"OE grew by {ev2.loss-ev.loss:.5e}"
            if not accepted:
                stop="no_certified_nonincreasing_candidate: "+error
                break
            if stop in ("no_significant_improvement","reached_exact_RSD"):break
        ok,reason=verify_exact(model,fractions)
        if not ok:
            rec.update(status="failed_final_certificate",stop_reason=reason)
            continue
        rec.update(status="optimized_verified",iterations=used,stop_reason=stop,
                   final_max_abs_diff_rsd=float(np.max(np.abs(current-rsd))),
                   final_is_rsd_exact=bool(fractions==rsd_fracs),
                   final_oe_avg=float(ev.loss),final_oe_total=float(ev.loss*model.number_of_profiles),
                   improvement_over_rsd_total=float((rsd_oe-ev.loss)*model.number_of_profiles),
                   beats_rsd=bool(ev.loss<rsd_oe-args.comparison_tolerance))
        file=save_solution(out,label,initial,fractions,model,[x["average_oe"] for x in history if x["label"]==label],rsd,True)
        results.append(dict(label=label,solution_path=file.name,oe_total=rec["final_oe_total"],
                            initial_oe_total=rec["initial_oe_total"],strict_sp=True))
        if best_id is None or ev.loss<best_loss-1e-12:
            best_id=i;best_loss=ev.loss
            # do not recalculate 762 profiles on every improvement, after the search
        save_csv(out/"comparison_many_sp.csv",records)
    save_csv(out/"comparison_many_sp.csv",records)
    summary={
        "description":"SP/anonymous/neutral serial-menu multistart; full 331776 profiles via 762 orbit weights",
        "n":4,"profiles":model.number_of_profiles,"representatives":len(model.representatives),
        "SP_constraints":int(model.sp_matrix.shape[0]),"variables":model.number_of_variables,
        "random_seed":args.seed,"bases":base_count,"candidate_initials":len(records),
        "optimized_count":sum(r["status"]=="optimized_verified" for r in records),
        "rsd_average_oe":rsd_oe,"rsd_total_oe":rsd_oe*model.number_of_profiles,
        "best_label":records[best_id]["label"] if best_id is not None else None,
        "best_total_oe":best_loss*model.number_of_profiles if best_id is not None else None,
        "best_improvement_over_rsd_total":(rsd_oe-best_loss)*model.number_of_profiles if best_id is not None else None,
        "found_better_than_rsd":any(r["beats_rsd"] is True for r in records),
        "all_initials_exact_sp_certified":True,
        "notes":["All menu bases are uniform symmetrizations of report-independent serial-choice rules.",
                 "Initial integer certificate checks all SP inequalities and equality rows.",
                 "Only solutions with exact Fraction SP/EQ verification are saved.",
                 "Inner OE LP and MM stopping use floating point; global optimality not proven."],
        "elapsed_seconds":time.monotonic()-t0,
        "results":records}
    (out/"summary_many_sp.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if not args.no_profiles and best_id is not None:
        if records[best_id]["status"]=="optimized_verified":
            bestlabel=records[best_id]["label"]
            path=out/("solution_"+"".join(z if z.isascii() and z.isalnum() else "_" for z in bestlabel)+".npz")
            with np.load(path) as data:
                profile_csv(model,data["mechanism_vector"],rsd,out/"best_profile_comparison.csv")
    print("\n================ MULTISTART vs RSD ================",flush=True)
    for r in records:
        if r["status"]=="optimized_verified":
            print(f"{r['label']}: initial_total={r['initial_oe_total']:.6f} "
                  f"final_total={r['final_oe_total']:.6f} "
                  f"vs_RSD={-r['improvement_over_rsd_total']:+.6f} SP(exact)=0 "
                  f"beats_RSD={r['beats_rsd']}",flush=True)
    print(f"OUTPUT: {out.resolve()}",flush=True)

if __name__=="__main__":main()