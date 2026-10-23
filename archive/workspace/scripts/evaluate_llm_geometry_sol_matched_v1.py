"""Prespecified matched Sol/Luna comparison, with frozen original references.

No API call is made. Primary comparisons are Sol minus Luna and Sol minus
G+F+C in each geometry representation, one four-test Holm family. The model
comparison was added after Luna outcomes; it is an exploratory extension.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import evaluate_llm_geometry_matched as original

ROOT=REPO/'outputs/llm_geometry_sol_matched_v1'
LUNA=REPO/'outputs/llm_geometry_matched_v1'
V6=original.V6
METHODS=('kg_full','geometry_intent_filter','kg_full_llm','kg_full_sol')
COMPARATORS=('kg_full_llm','geometry_intent_filter')
KINDS=original.KINDS
METRICS=original.METRICS
SEED=20261022
TOL=1e-12
PLAN={
 'primary_panel':'Same264 new-exemplar queries,44 training references,11roles,six conditions',
 'representations':list(KINDS),'methods':list(METHODS),
 'proposed_method':'kg_full_sol','primary_comparators':list(COMPARATORS),
 'gallery':1816,'pool_top_k':10,'pool_methods':list(original.POOL_METHODS),
 'pool_order':'sorted original gallery index',
 'reranking':'(3*frozen Full score+H)/4 within unchanged Luna pool; unchanged Full tail',
 'H':'mean yes1/no0/unknown.5 over same four clauses; invalid whole-job H.5',
 'ties':'ascending frozen gallery index',
 'grading':'Unchanged immutable v5 AI-reference grades, nominal transformation and .2mm numerical guard; all null labels retain envelopes',
 'metrics':list(METRICS),'primary_endpoint':'ndcg@5',
 'averaging':'24 query conditions per role, then11roles equally',
 'tests':'Four exact two-sided11role sign flips, one Holm family',
 'bootstrap_replicates':20000,'bootstrap_seed':SEED,
 'interval':'Unadjusted95percentile role-bootstrap intervals',
 'point_tests_require_all_primary_ndcg_identified':True,
 'model_selection':'No automatic model selection or manuscript-update gate. Report every comparison and failure.',
 'history':'Luna outcomes known before this extension; all Sol inputs and analysis frozen before first Sol response',
}


def inputs(root):
    paths=original.core_paths(LUNA)
    paths += [LUNA/n for n in ('full_rankings.npz','per_query.csv','results.json',
              'graph_support.json','evidence_completion.json','protocol_freeze.json')]
    paths += [REPO/'annotations/CURRENT_ANNOTATIONS.json',root/'study.json',
              root/'protocol.md',root/'preparation.json']
    paths += [REPO/'scripts'/n for n in (
        'prepare_llm_geometry_sol_matched_v1.py','run_llm_geometry_sol_api_v1.py',
        'complete_llm_geometry_sol_evidence_v1.py','evaluate_llm_geometry_sol_matched_v1.py',
        'replay_llm_geometry_sol_matched_v1.py')]
    return sorted(set(paths))


def prepare_freeze(root):
    assert not any((root/n).exists() for n in ('api_attempts.jsonl','api_responses.jsonl','results.json'))
    queries,_=original.panel()
    gallery=original.read(V6/'cohorts.json')['closed']
    with np.load(V6/'complete_rankings.npz',allow_pickle=False) as ar:
        pools,pairs=original.validate_pools(root,queries,gallery,ar)
    value={'status':'FROZEN_BEFORE_SOL_INFERENCE','plan':PLAN,
       'input_sha256':{original.relative(p):original.sha(p) for p in inputs(root)},
       'primary_query_ids':[q['query_id'] for q in queries],
       'pool_entries':sum(len(p) for qs in pools.values() for p in qs.values()),
       'required_unique_family_part_pairs':len(pairs),'new_sol_evidence_read':False}
    target=root/'evaluation_freeze.json'
    if target.exists():assert original.read(target)==value,'Refuse changed freeze'
    else:original.dump(target,value)
    print(json.dumps({'evaluation_freeze_sha256':original.sha(target),'jobs':len(pairs)}))


def analyze(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['geometry_kind'],row['method'],row['family_id']].append(row)
    family=[]
    for (kind,method,fid),members in sorted(groups.items()):
        assert len(members)==24
        family.append(dict(geometry_kind=kind,method=method,family_id=fid,queries=24,**original.means(members)))
    lookup={(r['geometry_kind'],r['method'],r['family_id']):r for r in family}
    families=sorted({r['family_id'] for r in family});assert len(families)==11
    summary=[]
    for kind in KINDS:
        for method in METHODS:
            rr=[lookup[kind,method,fid] for fid in families]
            summary.append(dict(geometry_kind=kind,method=method,queries=264,families=11,
              averaging='24_queries_per_family_then_equal_family',**original.means(rr)))
    identified=all(r['ndcg@5_lower']==r['ndcg@5_upper'] for r in rows)
    tests=[];differences=[];rng=np.random.default_rng(SEED)
    for kind in KINDS:
        for comp in COMPARATORS:
            local=[]
            for fid in families:
                a=lookup[kind,'kg_full_sol',fid];b=lookup[kind,comp,fid]
                d=dict(geometry_kind=kind,comparator=comp,family_id=fid)
                for metric in METRICS:
                    d[metric+'_lower']=a[metric+'_lower']-b[metric+'_upper']
                    d[metric+'_upper']=a[metric+'_upper']-b[metric+'_lower']
                local.append(d);differences.append(d)
            values=np.asarray([r['ndcg@5_lower'] for r in local]);avg=original.means(local)
            record=dict(geometry_kind=kind,proposed_method='kg_full_sol',comparator=comp,
                families=11,queries=264,point_inference_eligible=identified,
                delta_lower=avg['ndcg@5_lower'],delta_upper=avg['ndcg@5_upper'],
                strong_precision_delta_lower=avg['strong_precision@5_lower'],
                strong_precision_delta_upper=avg['strong_precision@5_upper'],
                delta=None,ci95_lower=None,ci95_upper=None,exact_sign_flip_p=None,holm_p=None,
                bootstrap_replicates=20000,bootstrap_seed=SEED,holm_family_size=4,
                family_wins=None,family_ties=None,family_losses=None,
                inference_status='IDENTIFIED_EXPLORATORY_POINT' if identified else 'WITHHELD_LABEL_UNCERTAINTY')
            if identified:
                ci=original._bootstrap(values,rng,20000)
                record.update(delta=float(values.mean()),ci95_lower=ci[0],ci95_upper=ci[1],
                  exact_sign_flip_p=original._flip_p(values),family_wins=int(sum(values>TOL)),
                  family_ties=int(sum(abs(values)<=TOL)),family_losses=int(sum(values< -TOL)))
            tests.append(record)
    if identified:
        for t,p in zip(tests,original._holm([r['exact_sign_flip_p'] for r in tests])):t['holm_p']=p
    return family,summary,tests,differences,identified


def evaluate(root):
    freeze=original.read(root/'evaluation_freeze.json');assert freeze['plan']==PLAN
    assert freeze['status']=='FROZEN_BEFORE_SOL_INFERENCE'
    for p,h in freeze['input_sha256'].items():assert original.sha(REPO/p)==h,p
    main=original.read(root/'protocol_freeze.json');completion=original.read(root/'evidence_completion.json')
    assert main['evaluation_freeze_sha256']==original.sha(root/'evaluation_freeze.json')
    assert completion['status']=='completed' and completion['protocol_freeze_sha256']==original.sha(root/'protocol_freeze.json')
    assert completion['graph_support_sha256']==original.sha(root/'graph_support.json')
    names=('per_query.csv','summary.csv','family_metrics.csv','family_differences.csv','paired_tests.csv','full_rankings.npz','results.json')
    assert not any((root/n).exists() for n in names),'Never overwrite evaluated outputs'
    stage=root/'.evaluation_inprogress';assert not stage.exists();stage.mkdir()
    queries,_=original.panel();assert [q['query_id'] for q in queries]==freeze['primary_query_ids']
    gallery=original.read(V6/'cohorts.json')['closed'];families=sorted({q['scenario_id'] for q in queries})
    support=original.load_support(root/'graph_support.json',families,set(gallery))
    annotation=original.read(REPO/'annotations/intent_feature_review_v5/judgments.json')
    assert annotation['source']=='ai_reference' and annotation['human_expert_reviewed'] is False
    records=annotation['judgments'];obs={(r['family_id'],r['part_id']):r for r in records}
    assert len(obs)==len(records)==1816*11
    raw={fid:np.asarray([-1 if obs[fid,p]['grade'] is None else obs[fid,p]['grade'] for p in gallery]) for fid in families}
    consumed=original.read(REPO/'outputs/feature_review_v5/consumed_evidence_freeze.json')['input_sha256']
    consumed={str((REPO/Path(p)).resolve()).lower():h for p,h in consumed.items()}
    measurements={};numeric_hashes={}
    for pid in gallery:
        row=obs[families[0],pid];path=REPO/'outputs/feature_review_v5/evidence'/(row['inspection_id']+'.json')
        assert original.sha(path)==consumed[str(path.resolve()).lower()]
        val=original.read(path);assert val['part_id']==pid and val['source_sha256']==row['source_sha256']
        measurements[pid]=val;numeric_hashes[original.relative(path)]=original.sha(path)
    prepared={}
    for q in queries:
        vv=[original.numeric_reference(q,measurements[p],.2) for p in gallery]
        lo,hi=original.grade_limits(raw[q['scenario_id']],vv,'nominal')
        prepared[q['query_id']]=original.prepare_limits(lo,hi,(5,))
    prior={(r['geometry_kind'],r['query_id'],r['method']):r for r in original.csv_rows(LUNA/'per_query.csv')}
    rows=[];baseline=[];missing=set();missing_entries=0
    with np.load(V6/'complete_rankings.npz',allow_pickle=False) as ar,np.load(LUNA/'full_rankings.npz',allow_pickle=False) as la,original.ArrayArchive(stage/'full_rankings.npz') as writer:
        pools,pairs=original.validate_pools(root,queries,gallery,ar)
        for kind in KINDS:
            for q in queries:
                qid=q['query_id'];fid=q['scenario_id'];pool=pools[kind][qid]
                local={m:original.archive_order(ar,kind,qid,m,1816) for m in METHODS[:2]}
                local['kg_full_llm']=original.archive_order(la,kind,qid,'kg_full_llm',1816)
                local['kg_full_sol']=original.rerank(*local['kg_full'],pool,support.get(fid,{}),gallery)
                absent=[p for p in pool if p not in support.get(fid,{})];missing.update((fid,p) for p in absent);missing_entries+=len(absent)
                for method in METHODS:
                    order,scores=local[method];bounds=original.rank_bounds(order,prepared[qid])
                    row=dict(geometry_kind=kind,query_id=qid,family_id=fid,method=method,cohort='closed',grading_scenario='nominal',reference_part_id=q['reference_part_id'],pool_size=len(pool),neutral_fallback_pool_entries=len(absent),returned=min(5,len(order)))
                    row.update({f'{m}_{end}':float(bounds[m][i]) for m in METRICS for i,end in enumerate(('lower','upper'))})
                    if method in METHODS[:3]:
                        b=prior[kind,qid,method];err=max(abs(row[f'{m}_{e}']-float(b[f'{m}_{e}'])) for m in METRICS for e in ('lower','upper'))
                        assert err<=TOL;baseline.append(err)
                    rows.append(row);key=f'{kind}__{qid}__closed__{method}'
                    writer.add(key+'__order',order.astype(np.int32));writer.add(key+'__score',scores)
            print('Completed fixed '+kind+' matrix',flush=True)
    family,summary,tests,differences,identified=analyze(rows)
    for name,data in [('per_query.csv',rows),('family_metrics.csv',family),('summary.csv',summary),('paired_tests.csv',tests),('family_differences.csv',differences)]:original.write_csv(stage/name,data)
    result=dict(status='COMPLETED_FROZEN_SOL_MATCHED_STUDY',plan=PLAN,
      main_protocol_freeze_sha256=original.sha(root/'protocol_freeze.json'),source_freeze_sha256=original.sha(root/'evaluation_freeze.json'),
      evidence_completion_sha256=original.sha(root/'evidence_completion.json'),graph_support_sha256=original.sha(root/'graph_support.json'),
      script_sha256=original.sha(Path(__file__)),primary_query_method_cases=len(rows),rankings=len(rows),
      null_raw_qrel_pairs=sum(r['grade'] is None for r in records),all_primary_ndcg_identified=identified,
      pool_entries=sum(len(p) for qs in pools.values() for p in qs.values()),required_unique_support_pairs=len(pairs),
      neutral_fallback_pool_entries=missing_entries,neutral_fallback_unique_pairs=len(missing),
      baseline_continuity={'comparisons':len(baseline),'max_absolute_difference':max(baseline),'exact':sum(x==0 for x in baseline)},
      numeric_evidence_sha256=numeric_hashes,summary=summary,paired_tests=tests,
      limitations=['Exploratory model comparison added after Luna outcomes.','AI-reference relevance and model observations are not independent engineering truth.','Same fixed pools, clauses and coverage algebra; no isolated graph-representation or relation-binding effect.','All four comparisons are reported; no automatic model selection or manuscript edit.'],
      api_calls=0,labels_revised=False,score_weights_tuned=False)
    result['artifact_sha256']={n:original.sha(stage/n) for n in names if n!='results.json'}
    original.dump(stage/'results.json',result)
    for p,h in freeze['input_sha256'].items():assert original.sha(REPO/p)==h
    for name in names:(stage/name).replace(root/name)
    stage.rmdir();print(json.dumps({'summary':summary,'paired_tests':tests},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-freeze',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    assert a.prepare_freeze != a.evaluate,'Choose exactly one operation'
    if a.prepare_freeze:prepare_freeze(ROOT)
    else:evaluate(ROOT)


if __name__=='__main__':main()
