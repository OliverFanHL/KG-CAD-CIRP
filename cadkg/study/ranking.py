"""Fixed candidate pools, hybrid reranking and role-balanced analysis."""
from collections import defaultdict
import numpy as np
from cadkg.experiments.verified_statistics import _bootstrap, _flip_p, _holm
from .io import read, write_csv
KINDS = ('embedding', 'descriptor')
POOL_METHODS = ('kg_full', 'geometry_intent_filter', 'kg_no_class')
CONTROLS = ('kg_no_class','geometry_class_filter','geometry_intent_filter','kg_no_constraints','kg_full')
NEW = ('kg_luna','kg_sol','role_only_hybrid','role_only_score','kg_luna_score','kg_sol_score')
METHODS = CONTROLS + NEW
CONTRASTS = (('kg_luna','role_only_hybrid'),('kg_sol','kg_luna'))
METRICS = ('ndcg@5','strong_precision@5')
SEED = 20261023
TOL = 1e-12

def need(condition, message):
    if not condition:
        raise ValueError(message)

def archive_order(archive,kind,qid,method,n):
    key=f'{kind}__{qid}__closed__{method}'
    order=np.asarray(archive[key+'__order'],dtype=np.int64)
    scores=np.asarray(archive[key+'__score'],dtype=np.float64)
    need(order.ndim==scores.ndim==1 and len(order)==len(scores),'Order/score shape mismatch')
    need(len(order)==len(np.unique(order)) and np.all((order>=0)&(order<n)),'Invalid frozen ranking indices')
    need(np.all(np.isfinite(scores)),'Nonfinite frozen score')
    return order,scores

def validate_pools(root,queries,gallery,archive):
    pools=read(root/'pool_manifest.json')
    need(set(pools)==set(KINDS),'Pool manifest needs exactly two representation mappings')
    qids={q['query_id'] for q in queries};pairs=set()
    index={pid:i for i,pid in enumerate(gallery)}
    for kind in KINDS:
        need(set(pools[kind])==qids,'Pool manifest must cover all264 primary queries exactly')
        for q in queries:
            expected=sorted(set(int(i) for method in POOL_METHODS
                 for i in archive_order(archive,kind,q['query_id'],method,len(gallery))[0][:10]))
            got=pools[kind][q['query_id']]
            need(isinstance(got,list) and all(pid in index for pid in got),'Pool entry contains unknown IDs')
            need(got==[gallery[i] for i in expected],f'Pool differs from frozen union:{kind}/{q["query_id"]}')
            pairs.update((q['scenario_id'],pid) for pid in got)
    return pools,sorted(pairs)

def rerank(base_order,base_scores,pool,H,gallery):
    score_by_index=dict(zip(map(int,base_order),map(float,base_scores)))
    idx={pid:i for i,pid in enumerate(gallery)}
    indices=np.asarray([idx[pid] for pid in pool],dtype=np.int64)
    need(len(indices)==len(np.unique(indices)) and all(i in score_by_index for i in indices),'Pool must be a unique subset of eligible baseline Full candidates')
    values=np.asarray([(3*score_by_index[int(i)]+float(H.get(gallery[int(i)],.5)))/4 for i in indices])
    need(np.all(np.isfinite(values)),'Nonfinite reranking score')
    ranked=np.lexsort((indices,-values))
    used=set(map(int,indices));tail=np.asarray([i for i in base_order if int(i) not in used],dtype=np.int64)
    order=np.concatenate([indices[ranked],tail])
    scores=np.concatenate([values[ranked],np.asarray([score_by_index[int(i)] for i in tail])])
    need(len(order)==len(base_order) and set(map(int,order))==set(map(int,base_order)),'Proposed ranking changes eligibility')
    return order,scores

def means(rows):
    return {f'{metric}_{end}':float(np.mean([r[f'{metric}_{end}'] for r in rows]))
            for metric in METRICS for end in ('lower','upper')}
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
        family.append(dict(geometry_kind=kind,method=method,replicate=rep,family_id=fid,**means(rr)))
    run=[];summary=[]
    for kind in KINDS:
        for method in METHODS:
            local=[]
            for rep in (1,2,3):
                rr=[r for r in family if (r['geometry_kind'],r['method'],r['replicate'])==(kind,method,rep)];assert len(rr)==11
                r=dict(geometry_kind=kind,method=method,replicate=rep,**means(rr));run.append(r);local.append(r)
            rec=dict(geometry_kind=kind,method=method,replicates=3,sd_ddof=1)
            for m in METRICS:
                for end in ('lower','upper'):rec[m+'_'+end+'_mean']=float(np.mean([r[m+'_'+end] for r in local]));rec[m+'_'+end+'_sd']=float(np.std([r[m+'_'+end] for r in local],ddof=1))
                rec[m+'_mean']=rec[m+'_lower_mean'];rec[m+'_sd']=rec[m+'_lower_sd']
            summary.append(rec)
    collapsed={}
    fids=sorted({r['family_id'] for r in family})
    for kind in KINDS:
        for method in METHODS:
            for fid in fids:collapsed[kind,method,fid]=means([r for r in family if (r['geometry_kind'],r['method'],r['family_id'])==(kind,method,fid)])
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
            values=np.asarray([r['ndcg@5_lower'] for r in local]);avg=means(local)
            t=dict(geometry_kind=kind,proposed=proposed,comparator=comp,roles=11,replicates=3,point_identified=identified,delta_lower=avg['ndcg@5_lower'],delta_upper=avg['ndcg@5_upper'],strong_precision_delta_lower=avg['strong_precision@5_lower'],strong_precision_delta_upper=avg['strong_precision@5_upper'],delta=None,ci95_lower=None,ci95_upper=None,raw_p=None,holm_p=None,family_wins=None,family_ties=None,family_losses=None)
            if identified:
                ci=_bootstrap(values,rng,20000);t.update(delta=float(values.mean()),ci95_lower=ci[0],ci95_upper=ci[1],raw_p=_flip_p(values),family_wins=int(sum(values>TOL)),family_ties=int(sum(abs(values)<=TOL)),family_losses=int(sum(values< -TOL)))
            tests.append(t)
    if identified:
        for t,p in zip(tests,_holm([r['raw_p'] for r in tests])):t['holm_p']=p
    return family,run,summary,tests,diffs,identified
