"""Repeat the unchanged retrieval matrix without imputing unreadable labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256
from cadkg.bundle import load_bundle
from cadkg.catalog import CLASSES
from cadkg.experiments.full import common_gallery,retrieval_config
from cadkg.reasoning import check_constraints
from cadkg.retrieval import Retriever,cosine
from scripts.run_verified_evaluation import METHODS,rank_components,stable_seed,evaluation_protocol,near_duplicate_gallery,diagnostic_cohorts

ROOT=Path('outputs/feature_review_v5')
HIST=Path('outputs/candidate_scale_v4/evaluation')

def main(output):
    started=time.monotonic();out=Path(output)
    if out.exists():raise ValueError('Use a fresh output folder; never overwrite a run')
    frozen=read_json(ROOT/'evaluation_freeze.json')['input_sha256']
    for p,d in frozen.items():assert sha256(p)==d,p
    cfg=read_json('configs/feature_review_v5.json');old=read_json('configs/candidate_scale_v4.json')
    allowed={'version','annotation_path','annotation_audit_path','evaluation_plan','evidence_root','output','notes'}
    assert cfg.keys()==old.keys() and all(v==old[k] for k,v in cfg.items() if k not in allowed)
    historical=read_json(HIST/'run_manifest.json')
    for p,d in historical['code_sha256'].items():assert sha256(p)==d,p
    changed={str(Path(p).resolve()) for p in ['configs/candidate_scale_v4.json',old['annotation_path'],old['annotation_audit_path'],old['evaluation_plan']]}
    for p,d in historical['input_sha256'].items():
        if str(Path(p).resolve()) not in changed:assert sha256(p)==d,p
    source=Path(cfg['source_run']);protocol=evaluation_protocol(cfg,read_json(source/'protocol.json'))
    assert protocol==read_json(HIST/'effective_protocol.json')
    records=read_json(cfg['annotation_path'])['judgments'];ids=sorted({r['part_id'] for r in records})
    assert len(ids)==1816 and len(records)==19976
    catalog,knowledge,graph=load_bundle(source/'bundle');parts={p['part_id']:p for p in catalog['parts']}
    retrievers={k:Retriever(catalog,knowledge,graph,retrieval_config(protocol['settings'],k)) for k in ['embedding','descriptor']}
    gallery,contaminated=common_gallery(retrievers,catalog)
    assert gallery==protocol['gallery'] and set(ids)==set(gallery)
    for r in records:assert parts[r['part_id']]['file_sha256']==r['source_sha256']
    prior_ids={r['part_id'] for r in read_json('annotations/intent_batch2/judgments.json')['judgments']}
    train=sorted(p for p in retrievers['descriptor'].vectors if parts[p]['split']=='train')
    dedup,dup=near_duplicate_gallery(ids,train,retrievers['descriptor'],cfg['near_duplicate_threshold'])
    obs={(r['family_id'],r['part_id']):r['grade'] for r in records}
    dup['discordant_groups']=[g for g in dup['duplicate_groups'] if any(len({obs[f['family_id'],p] for p in g})>1 for f in protocol['families'])]
    cohorts=dict(closed=ids,new_only=sorted(set(ids)-prior_ids),original_pool=sorted(prior_ids),deduplicated=dedup)
    cohorts.update(diagnostic_cohorts(cfg.get('diagnostic_cohorts_path'),ids,cohorts))
    assert cohorts==read_json(HIST/'cohorts.json')
    probs={p:np.asarray([retrievers['embedding'].probabilities(p).get(c,0) for c in CLASSES]) for p in set(gallery)|{e['part_id'] for e in protocol['exemplars']}}
    weights={fid:np.asarray([sum(r['relevance']*r['confidence'] for r in knowledge['class_functions'] if r['function_id']==fid and r['class_id']==c) for c in CLASSES]) for fid in {q['function_ids'][0] for q in protocol['queries']}}
    def components(q,kind,selected,p=probs):
        r=retrievers[kind];ref=q['reference_part_id']
        a={'intent':np.asarray([float(p[x]@weights[q['function_ids'][0]]) for x in selected])}
        if ref:
            a['geometry']=np.asarray([(cosine(r.vectors[x],r.vectors[ref])+1)/2 for x in selected])
            a['reference']=np.asarray([float(p[x]@p[ref]) for x in selected])
        return a
    archive={};cache={};feas={};primary={};top=[]
    for kind in retrievers:
        for q in protocol['queries']:
            qid=q['query_id']
            for cohort,selected in cohorts.items():
                comp=components(q,kind,selected)
                feasible=np.asarray([check_constraints(parts[p],q['constraints'])['status'] in {'pass','not_requested'} for p in selected])
                if cohort=='closed':cache[kind,qid]=comp;feas[qid]=feasible
                for m in METHODS:
                    order,scores=rank_components(comp,feasible,m,selected,cfg['rrf_constant'])
                    archive[f'{kind}__{qid}__{cohort}__{m}__order']=order.astype(np.int32)
                    archive[f'{kind}__{qid}__{cohort}__{m}__score']=scores
                    if cohort=='closed':primary[kind,qid,m]=order
                    top.extend(dict(cohort=cohort,geometry_kind=kind,query_id=qid,method=m,rank=j,part_id=selected[i],score=float(scores[j-1])) for j,i in enumerate(order[:20],1))
        print('Fresh retrieval completed: '+kind,flush=True)
    with np.load(HIST/'complete_rankings.npz') as oldarrays:
        assert list(archive)==oldarrays.files
        for k,v in archive.items():assert np.array_equal(v,oldarrays[k]),k
    robust={};cases=[];robust_rows=[]
    methods=['kg_full','geometry_class_filter','geometry_intent_filter','intent_filter','kg_no_class','rrf_filter','kg_no_geometry']
    def save(kind,q,m,name,severity,seed,order,guard=.2):
        key=f"{kind}__{q['query_id']}__{m}__{name}__{severity}__{seed}__order"
        robust[key]=order.astype(np.int32)
        cases.append(dict(geometry_kind=kind,query_id=q['query_id'],method=m,robustness=name,severity=severity,seed=seed,key=key,guard_mm=guard))
        if name!='constraint_guard':robust_rows.append(dict(geometry_kind=kind,query_id=q['query_id'],method=m,robustness=name,severity=severity,seed=seed,top20='|'.join(ids[i] for i in order[:20])))
    for kind in retrievers:
        for q in protocol['queries']:
            qid=q['query_id']
            for guard in cfg['constraint_guards_mm']:
                if guard==.2 or not q['constraints']:continue
                for m in methods:save(kind,q,m,'constraint_guard',guard,0,primary[kind,qid,m],guard)
            for seed in cfg['robustness_seeds']:
                for m in methods:
                    order,_=rank_components(cache[kind,qid],feas[qid],m,ids,cfg['rrf_constant'],stable_seed(seed,qid,'ties'))
                    save(kind,q,m,'tie_order',0,seed,order)
    allids=sorted(probs);allp=np.asarray([probs[p] for p in allids])
    for fraction in cfg['classification_corruption_fractions']:
        for seed in cfg['robustness_seeds']:
            rng=np.random.default_rng(seed);p=allp.copy();chosen=rng.random(len(p))<fraction
            for i in np.where(chosen)[0]:p[i]=p[i,rng.permutation(len(CLASSES))]
            corrupted={pid:p[i] for i,pid in enumerate(allids)}
            for kind in retrievers:
                for q in protocol['queries']:
                    comp=components(q,kind,ids,corrupted)
                    for m in methods:
                        order,_=rank_components(comp,feas[q['query_id']],m,ids,cfg['rrf_constant'])
                        save(kind,q,m,'classifier_permutation',fraction,seed,order)
            print(f'Fresh robustness retrieval: {fraction}, seed {seed}',flush=True)
    previous=read_csv(HIST/'robustness_rankings.csv')
    assert len(previous)==len(robust_rows)
    for a,b in zip(robust_rows,previous):
        assert all(str(a[k])==b[k] for k in a),(a,b)
    for p,d in frozen.items():assert sha256(p)==d,p
    out.mkdir(parents=True)
    np.savez_compressed(out/'complete_rankings.npz',**archive)
    np.savez_compressed(out/'robustness_complete_rankings.npz',**robust)
    write_csv(out/'robustness_cases.csv',cases);write_csv(out/'robustness_rankings.csv',robust_rows)
    write_csv(out/'rankings_top20.csv',top);write_json(out/'effective_protocol.json',protocol);write_json(out/'cohorts.json',cohorts);write_json(out/'near_duplicate_audit.json',dup)
    write_json(out/'ranking_validation.json',dict(status='VERIFIED',fresh_retrieval=True,complete_arrays_equal_to_v4=len(archive),robustness_top20_equal_to_v4=len(robust_rows),robustness_complete_orders=len(robust),queries=110,methods=14,representations=2,candidate_cads=1816,exact_byte_contaminated_excluded=len(contaminated),core_methods_unchanged=True,null_labels_not_imputed=True,elapsed_seconds=time.monotonic()-started,input_sha256=frozen,code_sha256=historical['code_sha256'],historical_archive_sha256=sha256(HIST/'complete_rankings.npz'),files_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print('Verified fresh retrieval and all historical ranking equalities',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'evaluation'));main(p.parse_args().output)
