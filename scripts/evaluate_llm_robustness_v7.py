"""Fixed three-acquisition comparison and role-only ablation; no API calls."""
from collections import defaultdict
import argparse,json
from pathlib import Path
import numpy as np
from scripts import evaluate_llm_geometry_matched as old
from scripts.complete_llm_robustness_v7 import read,sha,write

ROOT=Path('outputs/llm_robustness_v7')
KINDS=('embedding','descriptor')
CONTROLS=('kg_no_class','geometry_class_filter','geometry_intent_filter','kg_no_constraints','kg_full')
NEW=('kg_luna','kg_sol','role_only_hybrid','role_only_score','kg_luna_score','kg_sol_score')
METHODS=CONTROLS+NEW
CONTRASTS=(('kg_luna','role_only_hybrid'),('kg_sol','kg_luna'))
METRICS=('ndcg@5','strong_precision@5');SEED=20261023;TOL=1e-12
PLAN=dict(arms=['kg_luna','kg_sol','role_only_luna'],replicates=3,historical_kg_first_run_reused=True,local_order_seeds=[17,29,43],api_sampling_seed_supported=False,encoder_and_classifier='fixed validation-selected checkpoint',primary_comparisons=[{'proposed':a,'comparator':b,'representations':list(KINDS)} for a,b in CONTRASTS],primary_endpoint='ndcg@5',averaging='conditions/exemplars within role, then acquisition replicates within role, then11roles equally',tests='exact two-sided11role sign enumeration; one4test Holm family',run_sd='sample SD ddof1 over3aggregate run means',bootstrap_replicates=20000,bootstrap_seed=SEED,methods=list(METHODS),scores='hybrid(3Full+H_or_J)/4; score-onlyH_or_J within same inherited eligiblepool; unchangedFulltail',fallback='whole-job .5; noresampling',scope='API acquisition variability conditional on frozen encoder; historicalKGrep1 and fresh repeats are not3freshdataset/trainingreplications')

def freeze_inputs(root):
    paths=old.core_paths(Path('outputs/llm_geometry_matched_v1').resolve())
    paths += [Path('outputs')/d/n for d in ['llm_geometry_matched_v1','llm_geometry_sol_matched_v1'] for n in ['study.json','protocol_freeze.json','api_responses.jsonl','evidence_completion.json','results.json']]
    paths += [Path('scripts')/n for n in ['prepare_llm_robustness_v7.py','run_llm_robustness_v7_api.py','complete_llm_robustness_v7.py','evaluate_llm_robustness_v7.py','replay_llm_robustness_v7.py']]
    paths += [root/n for n in ['study.json','protocol.md','role_only_contract.json','pool_manifest.json','preparation.json']]
    return {old.relative(p):sha(p) for p in sorted(set(paths))}

def prepare_freeze():
    assert not (ROOT/'api_attempts.jsonl').exists()
    qs,_=old.panel();g=read(old.V6/'cohorts.json')['closed']
    with np.load(old.V6/'complete_rankings.npz') as ar:pools,pairs=old.validate_pools(ROOT,qs,g,ar)
    value=dict(status='FROZEN_BEFORE_REPEATED_ACQUISITION',plan=PLAN,input_sha256=freeze_inputs(ROOT),primary_query_ids=[q['query_id'] for q in qs],required_pairs=len(pairs))
    target=ROOT/'evaluation_freeze.json'
    if target.exists():assert read(target)==value
    else:write(target,value)

def score_only(order,pool,H,gallery):
    idx={p:i for i,p in enumerate(gallery)};pp=[idx[p] for p in pool];used=set(pp)
    ranked=sorted(pp,key=lambda i:(-H.get(gallery[i],.5),i))
    return np.asarray(ranked+[int(i) for i in order if int(i) not in used],dtype=np.int32)

def analyze(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['geometry_kind'],r['method'],r['replicate'],r['family_id']].append(r)
    family=[]
    for (kind,method,rep,fid),rr in sorted(groups.items()):
        assert len(rr)==24
        family.append(dict(geometry_kind=kind,method=method,replicate=rep,family_id=fid,**old.means(rr)))
    run=[];summary=[]
    for kind in KINDS:
        for method in METHODS:
            local=[]
            for rep in (1,2,3):
                rr=[r for r in family if (r['geometry_kind'],r['method'],r['replicate'])==(kind,method,rep)];assert len(rr)==11
                r=dict(geometry_kind=kind,method=method,replicate=rep,**old.means(rr));run.append(r);local.append(r)
            rec=dict(geometry_kind=kind,method=method,replicates=3,sd_ddof=1)
            for m in METRICS:
                for end in ('lower','upper'):rec[m+'_'+end+'_mean']=float(np.mean([r[m+'_'+end] for r in local]));rec[m+'_'+end+'_sd']=float(np.std([r[m+'_'+end] for r in local],ddof=1))
                rec[m+'_mean']=rec[m+'_lower_mean'];rec[m+'_sd']=rec[m+'_lower_sd']
            summary.append(rec)
    collapsed={}
    fids=sorted({r['family_id'] for r in family})
    for kind in KINDS:
        for method in METHODS:
            for fid in fids:collapsed[kind,method,fid]=old.means([r for r in family if (r['geometry_kind'],r['method'],r['family_id'])==(kind,method,fid)])
    # Eligibility of point inference concerns only the four tested hybrid arms.
    primary={'kg_luna','kg_sol','role_only_hybrid'}
    identified=all(r['ndcg@5_lower']==r['ndcg@5_upper'] for r in rows if r['method'] in primary)
    rng=np.random.default_rng(SEED);tests=[];diffs=[]
    for kind in KINDS:
        for proposed,comp in CONTRASTS:
            local=[]
            for fid in fids:
                a=collapsed[kind,proposed,fid];b=collapsed[kind,comp,fid];d=dict(geometry_kind=kind,proposed=proposed,comparator=comp,family_id=fid)
                for m in METRICS:d[m+'_lower']=a[m+'_lower']-b[m+'_upper'];d[m+'_upper']=a[m+'_upper']-b[m+'_lower']
                local.append(d);diffs.append(d)
            values=np.asarray([r['ndcg@5_lower'] for r in local]);avg=old.means(local)
            t=dict(geometry_kind=kind,proposed=proposed,comparator=comp,roles=11,replicates=3,point_identified=identified,delta_lower=avg['ndcg@5_lower'],delta_upper=avg['ndcg@5_upper'],strong_precision_delta_lower=avg['strong_precision@5_lower'],strong_precision_delta_upper=avg['strong_precision@5_upper'],delta=None,ci95_lower=None,ci95_upper=None,raw_p=None,holm_p=None,family_wins=None,family_ties=None,family_losses=None)
            if identified:
                ci=old._bootstrap(values,rng,20000);t.update(delta=float(values.mean()),ci95_lower=ci[0],ci95_upper=ci[1],raw_p=old._flip_p(values),family_wins=int(sum(values>TOL)),family_ties=int(sum(abs(values)<=TOL)),family_losses=int(sum(values< -TOL)))
            tests.append(t)
    if identified:
        for t,p in zip(tests,old._holm([r['raw_p'] for r in tests])):t['holm_p']=p
    return family,run,summary,tests,diffs,identified

def evaluate(root=ROOT):
    freeze=read(root/'evaluation_freeze.json');assert freeze['plan']==PLAN
    for p,h in freeze['input_sha256'].items():assert sha(old.REPO/p)==h,p
    main=read(root/'protocol_freeze.json');complete=read(root/'evidence_completion.json');assert main['evaluation_freeze_sha256']==sha(root/'evaluation_freeze.json');assert complete['protocol_freeze_sha256']==sha(root/'protocol_freeze.json');assert complete['graph_support_sha256']==sha(root/'graph_support.json')
    assert not (root/'results.json').exists()
    qs,_=old.panel();gallery=read(old.V6/'cohorts.json')['closed'];support=read(root/'graph_support.json')
    annotation=read('annotations/intent_feature_review_v5/judgments.json');assert annotation['source']=='ai_reference' and annotation['human_expert_reviewed'] is False
    records=annotation['judgments'];obs={(r['family_id'],r['part_id']):r for r in records};fids=sorted({q['scenario_id'] for q in qs});assert len(obs)==19976
    raw={fid:np.asarray([-1 if obs[fid,p]['grade'] is None else obs[fid,p]['grade'] for p in gallery]) for fid in fids}
    consumed=read('outputs/feature_review_v5/consumed_evidence_freeze.json')['input_sha256'];consumed={str((old.REPO/Path(p)).resolve()).lower():h for p,h in consumed.items()};measure={}
    for pid in gallery:
        row=obs[fids[0],pid];path=Path('outputs/feature_review_v5/evidence')/(row['inspection_id']+'.json');assert sha(path)==consumed[str(path.resolve()).lower()];measure[pid]=read(path)
    prepared={}
    for q in qs:
        verdicts=[old.numeric_reference(q,measure[p],.2) for p in gallery];lo,hi=old.grade_limits(raw[q['scenario_id']],verdicts,'nominal');prepared[q['query_id']]=old.prepare_limits(lo,hi,(5,))
    rows=[]
    # Keep top5 orders for an independent scalar audit; complete source arrays and deterministic algorithm reconstruct the tails.
    tops=[]
    with np.load(old.V6/'complete_rankings.npz') as ar:
        pools,pairs=old.validate_pools(root,qs,gallery,ar)
        for kind in KINDS:
            for q in qs:
                qid=q['query_id'];fid=q['scenario_id'];pool=pools[kind][qid];baseline={m:old.archive_order(ar,kind,qid,m,1816) for m in CONTROLS}
                for rep in (1,2,3):
                    hl=support['kg_luna'][str(rep)][fid];hs=support['kg_sol'][str(rep)][fid];j=support['role_only_luna'][str(rep)][fid]
                    orders={m:x[0] for m,x in baseline.items()}
                    for method,v in [('kg_luna',hl),('kg_sol',hs),('role_only_hybrid',j)]:orders[method]=old.rerank(*baseline['kg_full'],pool,v,gallery)[0]
                    for method,v in [('kg_luna_score',hl),('kg_sol_score',hs),('role_only_score',j)]:orders[method]=score_only(baseline['kg_full'][0],pool,v,gallery)
                    for method in METHODS:
                        order=orders[method];bounds=old.rank_bounds(order,prepared[qid]);r=dict(geometry_kind=kind,query_id=qid,family_id=fid,method=method,replicate=rep,cohort='closed',grading_scenario='nominal')
                        r.update({f'{m}_{end}':float(bounds[m][i]) for m in METRICS for i,end in enumerate(('lower','upper'))});rows.append(r)
                        tops.append(dict(geometry_kind=kind,query_id=qid,method=method,replicate=rep,top5_indices=[int(i) for i in order[:5]]))
            print('Evaluated'+kind,flush=True)
    family,run,summary,tests,diffs,identified=analyze(rows)
    for name,data in [('per_query.csv',rows),('family_metrics.csv',family),('run_metrics.csv',run),('summary.csv',summary),('paired_tests.csv',tests),('family_differences.csv',diffs)]:old.write_csv(root/name,data)
    write(root/'top5_rankings.json',tops)
    result=dict(status='COMPLETED_REPEATED_ACQUISITION_STUDY',plan=PLAN,protocol_sha256=sha(root/'protocol_freeze.json'),evaluation_freeze_sha256=sha(root/'evaluation_freeze.json'),evidence_completion_sha256=sha(root/'evidence_completion.json'),metric_rows=len(rows),summary=summary,paired_tests=tests,primary_ndcg_identified=identified,null_raw_pairs=sum(r['grade'] is None for r in records),new_api_calls=0)
    names=['per_query.csv','family_metrics.csv','run_metrics.csv','summary.csv','paired_tests.csv','family_differences.csv','top5_rankings.json'];result['artifact_sha256']={n:sha(root/n) for n in names};write(root/'results.json',result);print(json.dumps({'summary':summary,'paired_tests':tests},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-freeze',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args();assert a.prepare_freeze !=a.evaluate
    if a.prepare_freeze:prepare_freeze()
    else:evaluate()
