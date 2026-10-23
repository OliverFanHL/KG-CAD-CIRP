"""Fixed-design evaluation of verified AI CAD judgments; no label-based tuning."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from cadkg.bundle import load_bundle
from cadkg.catalog import CLASSES
from cadkg.experiments.full import common_gallery, retrieval_config
from cadkg.experiments.intent import score_grade
from cadkg.experiments.reference import numeric_reference
from cadkg.io import read_json, read_csv, write_json, write_csv, sha256, object_hash
from cadkg.reasoning import check_constraints
from cadkg.retrieval import Retriever, cosine, REFERENCE_METHODS

EXTRA=['intent_filter','category_filter','reference_filter','geometry_class_filter','geometry_intent_filter','rrf_filter']
METHODS=REFERENCE_METHODS+EXTRA
RECIPES={
 'geometry':(['geometry'],False), 'geometry_class':(['geometry','reference'],False),
 'flat_full':(['geometry','reference','intent'],True), 'kg_full':(['geometry','reference','intent'],True),
 'kg_no_geometry':(['reference','intent'],True), 'kg_no_class':(['geometry'],True),
 'kg_no_constraints':(['geometry','reference','intent'],False),
 'category_from_intent':(['intent'],False), 'intent_filter':(['intent'],True),
 'category_filter':(['intent'],True), 'reference_filter':(['reference'],True),
 'geometry_class_filter':(['geometry','reference'],True),
 'geometry_intent_filter':(['geometry','intent'],True),
 'rrf_filter':(['geometry','reference','intent'],True)}

def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)

def average_ranks(values):
    unique, inverse, counts=np.unique(-values,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts)
    return (ends-counts+(counts+1)/2)[inverse]

def rank_components(components, feasible, method, ids, rrf=60, tie_seed=None):
    names,filtered=RECIPES[method]
    active=[components[n] for n in names if n in components]
    valid=np.where(feasible if filtered else np.ones(len(ids),dtype=bool))[0]
    if not active or not len(valid):return np.array([],dtype=int),np.array([],dtype=float)
    if method=='rrf_filter':
        scores=sum(1/(rrf+average_ranks(x[valid])) for x in active)/len(active)
    else:
        scores=sum(x[valid] for x in active)/len(active)
    tie=np.arange(len(ids)) if tie_seed is None else np.random.default_rng(tie_seed).random(len(ids))
    order=np.lexsort((tie[valid],-scores))
    return valid[order],scores[order]

def grade_vector(query,ids,observations,evidence,scenario='nominal',guard=.2):
    return np.asarray([score_grade(observations[query['scenario_id'],pid]['grade'],
        numeric_reference(query,evidence[pid],guard),scenario) for pid in ids],dtype=int)

def diagnostic_cohorts(path, ids, existing):
    """Optional predeclared candidate strata; they may never widen the judged set."""
    if not path:return {}
    definitions=read_json(path)
    if not isinstance(definitions,dict):raise ValueError('Diagnostic cohorts must be a mapping')
    result={}
    for name,selected in definitions.items():
        if name in existing or not isinstance(selected,list) or not selected:
            raise ValueError('Invalid or colliding diagnostic cohort')
        if len(selected)!=len(set(selected)) or not set(selected)<=set(ids):
            raise ValueError('Diagnostic cohort must contain unique judged held-out IDs')
        result[name]=sorted(selected)
    return result

def measure(order,grades,verdicts,cutoffs):
    found=grades[order]; positives=int((grades>0).sum()); strong=int((grades==2).sum())
    ideal=np.sort(grades)[::-1]
    result={'returned':min(5,len(order)),'relevant_total':positives,'strong_total':strong}
    for k in cutoffs:
        g=found[:k];discount=1/np.log2(np.arange(k)+2)
        idcg=float(((2.**ideal[:k]-1)*discount[:min(k,len(ideal))]).sum())
        result[f'ndcg@{k}']=float(((2.**g-1)*discount[:len(g)]).sum()/idcg) if idcg else 0.
        result[f'precision@{k}']=float((g>0).sum()/k)
        result[f'strong_precision@{k}']=float((g==2).sum()/k)
        result[f'recall@{k}']=float((g>0).sum()/positives) if positives else 0.
    hits=(found>0).astype(float); precision=np.cumsum(hits)/np.arange(1,len(hits)+1)
    result['ap']=float((precision*hits).sum()/positives) if positives else 0.
    result['ap@20']=float((precision[:20]*hits[:20]).sum()/min(20,positives)) if positives else 0.
    result['dimension_failures']=int(sum(verdicts[i]=='fail' for i in order[:5]))
    result['dimension_unknowns']=int(sum(verdicts[i]=='unknown' for i in order[:5]))
    return result

def near_duplicate_gallery(ids,train_ids,retriever,tolerance):
    x=np.asarray([retriever.vectors[p] for p in ids]); t=np.asarray([retriever.vectors[p] for p in train_ids])
    parent=list(range(len(ids))); matching=set(); pairs=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    tn=(t*t).sum(1)
    for start in range(0,len(ids),100):
        block=x[start:start+100];dist=(block*block).sum(1)[:,None]+tn[None,:]-2*(block@t.T)
        for a,b in zip(*np.where(dist<=max(tolerance*tolerance*10,1e-9))):
            if np.linalg.norm(block[a]-t[b])<=tolerance:matching.add(start+int(a))
    norm=(x*x).sum(1)
    for start in range(0,len(ids),100):
        block=x[start:start+100];dist=(block*block).sum(1)[:,None]+norm[None,:]-2*(block@x.T)
        for a,b in zip(*np.where(dist<=max(tolerance*tolerance*10,1e-9))):
            a=start+int(a);b=int(b)
            if a<b and np.linalg.norm(x[a]-x[b])<=tolerance:
                ra,rb=find(a),find(b);parent[max(ra,rb)]=min(ra,rb);pairs.append((a,b))
    groups=defaultdict(list)
    for i in range(len(ids)):groups[find(i)].append(i)
    kept=[];excluded_train=set()
    for group in groups.values():
        if any(i in matching for i in group):excluded_train.update(group)
        else:kept.append(group[0])
    return [ids[i] for i in sorted(kept)],dict(train_matches=len(matching),excluded_train_group_members=len(excluded_train),
        duplicate_groups=[[ids[i] for i in v] for v in groups.values() if len(v)>1],remaining=len(kept),
        threshold=tolerance,scope='Train-normalized descriptor numerical signatures; not complete design decontamination')

def evaluation_protocol(cfg, original_protocol):
    """Explicit scoped transfer; never rewrite the annotation's original query hash."""
    if 'expanded_query_path' not in cfg:return original_protocol
    from copy import deepcopy
    from scripts.prepare_expanded_queries import build
    extension=read_json(cfg['query_extension_path']);queries=read_json(cfg['expanded_query_path'])
    if extension['original_query_sha256']!=original_protocol['query_sha256'] or extension['scope_sha256']!=original_protocol['scope_sha256']:
        raise ValueError('Query extension does not preserve the annotated family scope')
    for path,digest in {**extension['input_sha256'],**extension['file_sha256']}.items():
        if sha256(path)!=digest:raise ValueError(f'Frozen extension input changed: {path}')
    source=Path(cfg['source_run'])
    regenerated,_,_=build(original_protocol,read_json(source/'bundle/catalog.json'),read_json(source/'bundle/knowledge.json'))
    if queries!=regenerated or object_hash(queries)!=extension['query_sha256']:raise ValueError('Query selection does not reproduce')
    families={f['family_id']:f for f in original_protocol['families']};refs={e['part_id'] for e in original_protocol['exemplars']}
    for q in queries:
        if q['function_ids']!=[families[q['scenario_id']]['function_id']] or q['category_ids']:
            raise ValueError('Changed task semantics')
        if q['reference_part_id'] and q['reference_part_id'] not in refs:raise ValueError('Uninspected exemplar')
        for c in q['constraints']:
            if c['attribute']!='aabb_large_mm' or c['op']!='<=' or c['unit']!='mm' or not np.isfinite(c['value']) or c['value']<=0:
                raise ValueError('Unsupported constraint transfer')
    effective=deepcopy(original_protocol);effective.update(queries=queries,query_sha256=extension['query_sha256'],
        annotation_query_sha256=original_protocol['query_sha256'],query_extension=extension)
    return effective

def main(config_path, output_override=None):
    started=time.monotonic();cfg=read_json(config_path);out=Path(output_override or cfg['output']);out.mkdir(parents=True,exist_ok=True)
    original=Path(cfg['source_run']); protocol=read_json(original/'protocol.json')
    annotation=read_json(cfg['annotation_path']);records=annotation['judgments']
    if annotation.get('source')!='ai_reference' or annotation.get('human_expert_reviewed') is not False:raise ValueError('AI provenance required')
    if (annotation['query_sha256'],annotation['scope_sha256'])!=(protocol['query_sha256'],protocol['scope_sha256']):raise ValueError('Scope mismatch')
    original_query_ids={q['query_id'] for q in protocol['queries']}
    protocol=evaluation_protocol(cfg,protocol)
    write_json(out/'effective_protocol.json',protocol)
    observations={(r['family_id'],r['part_id']):r for r in records};ids=sorted({r['part_id'] for r in records})
    fids=sorted({q['scenario_id'] for q in protocol['queries']})
    if len(observations)!=len(records) or len(records)!=len(ids)*len(fids):raise ValueError('Incomplete or duplicate closed qrels')
    if any(r['grade'] not in (0,1,2) or r.get('review_status')!='visually_reviewed' for r in records):raise ValueError('Unreviewed labels')
    catalog,knowledge,graph=load_bundle(original/'bundle');parts={p['part_id']:p for p in catalog['parts']}
    retrievers={kind:Retriever(catalog,knowledge,graph,retrieval_config(protocol['settings'],kind)) for kind in ['embedding','descriptor']}
    gallery,contaminated=common_gallery(retrievers,catalog)
    if gallery!=protocol['gallery'] or not set(ids)<=set(gallery):raise ValueError('Frozen gallery mismatch')
    for r in records:
        if parts[r['part_id']]['file_sha256']!=r['source_sha256']:raise ValueError('Label source mismatch')
    for ref in protocol['exemplars']:
        if parts[ref['part_id']]['split']!='train' or parts[ref['part_id']]['file_sha256']!=ref['source_sha256']:raise ValueError('Bad train exemplar')
    evidence={r['part_id']:read_json(Path(cfg['evidence_root'])/f"{r['inspection_id']}.json") for r in records if r['family_id']==fids[0]}
    for pid,e in evidence.items():
        if e['part_id']!=pid or e['source_sha256']!=parts[pid]['file_sha256']:raise ValueError('Evidence binding mismatch')
    prior=read_json('annotations/intent_batch2/judgments.json'); prior_ids={r['part_id'] for r in prior['judgments']}
    train_ids=sorted(p for p in retrievers['descriptor'].vectors if parts[p]['split']=='train')
    dedup,duplicate_audit=near_duplicate_gallery(ids,train_ids,retrievers['descriptor'],cfg['near_duplicate_threshold'])
    duplicate_audit['discordant_groups']=[g for g in duplicate_audit['duplicate_groups'] if any(len({observations[f,p]['grade'] for p in g})>1 for f in fids)]
    write_json(out/'near_duplicate_audit.json',duplicate_audit)
    cohorts={'closed':ids,'new_only':sorted(set(ids)-prior_ids),'original_pool':sorted(prior_ids),'deduplicated':dedup}
    cohorts.update(diagnostic_cohorts(cfg.get('diagnostic_cohorts_path'),ids,cohorts))
    probabilities={p:np.asarray([retrievers['embedding'].probabilities(p).get(c,0) for c in CLASSES]) for p in set(gallery)|{x['part_id'] for x in protocol['exemplars']}}
    function_weights={fid:np.asarray([sum(r['relevance']*r['confidence'] for r in knowledge['class_functions'] if r['function_id']==fid and r['class_id']==c) for c in CLASSES]) for fid in {q['function_ids'][0] for q in protocol['queries']}}
    for fw in function_weights.values():
        if not (np.count_nonzero(fw)==1 and fw.max()==1):raise ValueError('Category-filter equivalence requires one unit-weight class prior')
    all_ids=sorted(probabilities); all_p=np.asarray([probabilities[p] for p in all_ids]); all_index={p:i for i,p in enumerate(all_ids)}
    components_cache={}; serving_cache={}; verdict_cache={}; grades_cache={}
    def components(query,kind,selected,probs=probabilities):
        reference=query['reference_part_id'];r=retrievers[kind]
        comp={'intent':np.asarray([float(probs[p]@function_weights[query['function_ids'][0]]) for p in selected])}
        if reference:
            comp['geometry']=np.asarray([(cosine(r.vectors[p],r.vectors[reference])+1)/2 for p in selected])
            comp['reference']=np.asarray([float(probs[p]@probs[reference]) for p in selected])
        return comp
    def assess(query,kind,cohort,method,order,scenario='nominal',guard=.2):
        selected=cohorts[cohort];key=(query['query_id'],cohort,scenario,guard)
        if key not in grades_cache:grades_cache[key]=grade_vector(query,selected,observations,evidence,scenario,guard)
        verdict_key=(query['query_id'],cohort,guard)
        if verdict_key not in verdict_cache:verdict_cache[verdict_key]=[numeric_reference(query,evidence[p],guard) for p in selected]
        return dict(cohort=cohort,grading_scenario=scenario,geometry_kind=kind,query_id=query['query_id'],family_id=query['scenario_id'],
          method=method,has_exemplar=bool(query['reference_part_id']),has_constraint=bool(query['constraints']),
          **measure(order,grades_cache[key],verdict_cache[verdict_key],cfg['cutoffs']))
    rows=[];top_rows=[];archive={};primary_orders={};full_rows=[];continuity=[];random_rows=[]
    serving_checks=[]
    legacy=defaultdict(list)
    for row in read_csv(original/'rankings.csv'):legacy[row['geometry_kind'],row['query_id'],row['method']].append(row)
    for kind in retrievers:
        for query in protocol['queries']:
            qid=query['query_id']; full_comp=components(query,kind,gallery)
            full_feasible=np.asarray([check_constraints(parts[p],query['constraints'])['status'] in {'pass','not_requested'} for p in gallery])
            for method in METHODS:
                order,scores=rank_components(full_comp,full_feasible,method,gallery,cfg['rrf_constant'])
                if method in REFERENCE_METHODS:
                    if qid in original_query_ids:
                        expected=[r['part_id'] for r in sorted(legacy[kind,qid,method],key=lambda x:int(x['rank']))]
                    else:
                        serving_start=time.monotonic()
                        serving_query=query;serving_method=method
                        if method=='category_from_intent':
                            serving_query={**query,'category_ids':sorted({r['class_id'] for r in knowledge['class_functions'] if r['function_id'] in query['function_ids']})}
                            serving_method='category'
                        actual=retrievers[kind].rank(serving_query,serving_method,k=5,candidate_ids=gallery)
                        expected=[r['part_id'] for r in actual['results']]
                        serving_checks.append(dict(geometry_kind=kind,query_id=qid,method=method,top5_matches=True,
                            elapsed_seconds=time.monotonic()-serving_start))
                    got=[gallery[i] for i in order[:5]]
                    if expected!=got:raise ValueError(f'Serving mismatch {kind}/{qid}/{method}: {expected} != {got}')
                    if qid not in original_query_ids and any(abs(float(s)-float(r['score']))>1e-12 for s,r in zip(scores[:5],actual['results'])):
                        raise ValueError('Serving score mismatch')
                g=[];unknown=0
                for i in order[:5]:
                    pid=gallery[i]
                    if pid in evidence:g.append(score_grade(observations[query['scenario_id'],pid]['grade'],numeric_reference(query,evidence[pid])))
                    else:g.append(None);unknown+=1
                discounts=1/np.log2(np.arange(5)+2);known=grade_vector(query,ids,observations,evidence)
                ideal_known=float(((2.**np.sort(known)[::-1][:5]-1)*discounts).sum())
                ideal_max=float(((2.**np.sort(np.r_[known,np.full(len(gallery)-len(ids),2)])[::-1][:5]-1)*discounts).sum())
                low=sum((2**(v or 0)-1)*discounts[j] for j,v in enumerate(g)); high=sum((2**(2 if v is None else v)-1)*discounts[j] for j,v in enumerate(g))
                full_rows.append(dict(geometry_kind=kind,query_id=qid,family_id=query['scenario_id'],method=method,has_exemplar=bool(query['reference_part_id']),
                  judged_at5=len(g)-unknown,unjudged_at5=unknown,ndcg_lower=low/ideal_max if ideal_max else 0.,ndcg_upper=min(1.,high/ideal_known) if ideal_known else 1.,
                  precision_lower=sum(v is not None and v>0 for v in g)/5,precision_upper=(sum(v is not None and v>0 for v in g)+unknown)/5))
                if method in REFERENCE_METHODS and qid in original_query_ids:
                    old={(r['family_id'],r['part_id']):r['grade'] for r in prior['judgments']}
                    old_values=[score_grade(old[query['scenario_id'],gallery[i]],numeric_reference(query,evidence[gallery[i]])) for i in order[:5]]
                    continuity.append(dict(geometry_kind=kind,query_id=qid,family_id=query['scenario_id'],method=method,
                        old_dcg=sum((2**v-1)*discounts[j] for j,v in enumerate(old_values)),corrected_dcg=low,changed_return_grades=sum(a!=b for a,b in zip(old_values,g))))
            for cohort,selected in cohorts.items():
                comp=components(query,kind,selected)
                feasible=np.asarray([check_constraints(parts[p],query['constraints'])['status'] in {'pass','not_requested'} for p in selected])
                components_cache[kind,qid,cohort]=comp;serving_cache[qid,cohort]=feasible
                for method in METHODS:
                    order,scores=rank_components(comp,feasible,method,selected,cfg['rrf_constant'])
                    if cohort=='closed':primary_orders[kind,qid,method]=order
                    archive[f'{kind}__{qid}__{cohort}__{method}__order']=order.astype(np.int32)
                    archive[f'{kind}__{qid}__{cohort}__{method}__score']=scores
                    scenarios=['nominal','conditional_zero','conditional_two'] if cohort=='closed' else ['nominal']
                    for scenario in scenarios:rows.append(assess(query,kind,cohort,method,order,scenario))
                    for rank,i in enumerate(order[:20],1):top_rows.append(dict(cohort=cohort,geometry_kind=kind,query_id=qid,method=method,rank=rank,part_id=selected[i],score=float(scores[rank-1])))
                # Uniform sampling without replacement over feasible parts: exact expected gain and hit count.
                grades=grade_vector(query,selected,observations,evidence);valid=np.where(feasible)[0];k=5;n=min(k,len(valid))
                ideal=float(((2.**np.sort(grades)[::-1][:k]-1)/np.log2(np.arange(min(k,len(grades)))+2)).sum())
                mu=float(np.mean(2.**grades[valid]-1)) if len(valid) else 0
                random_rows.append(dict(cohort=cohort,geometry_kind=kind,query_id=qid,family_id=query['scenario_id'],
                    expected_ndcg5=mu*sum(1/np.log2(np.arange(n)+2))/ideal if ideal else 0,
                    expected_precision5=float((grades[valid]>0).mean())*n/k if len(valid) else 0,
                    expected_strong_precision5=float((grades[valid]==2).mean())*n/k if len(valid) else 0))
        print(f'Primary and cohort rankings complete: {kind}',flush=True)
    # Equality facts are integrity checks, not null-hypothesis tests.
    for query in protocol['queries']:
        qid=query['query_id']
        for kind in retrievers:
            assert np.array_equal(primary_orders[kind,qid,'kg_full'],primary_orders[kind,qid,'flat_full'])
            assert np.array_equal(primary_orders[kind,qid,'intent_filter'],primary_orders[kind,qid,'category_filter'])
        assert np.array_equal(primary_orders['embedding',qid,'kg_no_geometry'],primary_orders['descriptor',qid,'kg_no_geometry'])
    write_csv(out/'per_query.csv',rows);write_csv(out/'rankings_top20.csv',top_rows)
    write_csv(out/'full_gallery_bounds.csv',full_rows);write_csv(out/'correction_impact.csv',continuity);write_csv(out/'random_expectation.csv',random_rows)
    if serving_checks:write_json(out/'new_query_serving_checks.json',dict(cases=serving_checks,matched=len(serving_checks),
        total_serving_seconds=sum(r['elapsed_seconds'] for r in serving_checks),score_tolerance=1e-12))
    write_json(out/'cohorts.json',cohorts);np.savez_compressed(out/'complete_rankings.npz',**archive)
    print('Primary artifacts saved; robustness begins',flush=True)
    robust=[];robust_ranks=[]
    robust_methods=['kg_full','geometry_class_filter','geometry_intent_filter','intent_filter','kg_no_class','rrf_filter','kg_no_geometry']
    for kind in retrievers:
        for query in protocol['queries']:
            qid=query['query_id'];comp=components_cache[kind,qid,'closed'];feasible=serving_cache[qid,'closed']
            for guard in cfg['constraint_guards_mm']:
                if guard==.2 or not query['constraints']:continue
                for method in robust_methods:
                    order=primary_orders[kind,qid,method]
                    robust.append(dict(assess(query,kind,'closed',method,order,guard=guard),robustness='constraint_guard',severity=guard,seed=0))
            for seed in cfg['robustness_seeds']:
                for method in robust_methods:
                    order,scores=rank_components(comp,feasible,method,ids,cfg['rrf_constant'],stable_seed(seed,qid,'ties'))
                    robust.append(dict(assess(query,kind,'closed',method,order),robustness='tie_order',severity=0,seed=seed))
                    robust_ranks.append(dict(geometry_kind=kind,query_id=qid,method=method,robustness='tie_order',severity=0,seed=seed,top20='|'.join(ids[i] for i in order[:20])))
    for fraction in cfg['classification_corruption_fractions']:
        for seed in cfg['robustness_seeds']:
            rng=np.random.default_rng(seed);p=all_p.copy();chosen=rng.random(len(p))<fraction
            for i in np.where(chosen)[0]:p[i]=p[i,rng.permutation(len(CLASSES))]
            corrupted={pid:p[i] for i,pid in enumerate(all_ids)}
            for kind in retrievers:
                for query in protocol['queries']:
                    qid=query['query_id'];comp=components(query,kind,ids,corrupted);feasible=serving_cache[qid,'closed']
                    for method in robust_methods:
                        order,scores=rank_components(comp,feasible,method,ids,cfg['rrf_constant'])
                        robust.append(dict(assess(query,kind,'closed',method,order),robustness='classifier_permutation',severity=fraction,seed=seed))
                        robust_ranks.append(dict(geometry_kind=kind,query_id=qid,method=method,robustness='classifier_permutation',severity=fraction,seed=seed,top20='|'.join(ids[i] for i in order[:20])))
            print(f'Classifier corruption complete: fraction={fraction}, seed={seed}',flush=True)
    write_csv(out/'robustness_per_query.csv',robust);write_csv(out/'robustness_rankings.csv',robust_ranks)
    descriptive=[]
    for cohort,selected in cohorts.items():
        for fid in fids:
            descriptive.append(dict(cohort=cohort,family_id=fid,candidates=len(selected),**dict(Counter(str(observations[fid,p]['grade']) for p in selected))))
    write_csv(out/'cohort_prevalence.csv',descriptive)
    write_csv(out/'sampling_distribution.csv',[dict(cohort=c,source_category=label,count=count) for c,sel in cohorts.items() for label,count in Counter(parts[p]['label'] for p in sel).items()])
    from cadkg.experiments.verified_statistics import analyze
    analyze(rows,cfg,out)
    inputs=[config_path,cfg['annotation_path'],cfg.get('annotation_audit_path','annotations/intent_batch2_verified/audit.json'),original/'protocol.json',original/'rankings.csv',
        original/'bundle/catalog.json',original/'bundle/graph.json',original/'bundle/knowledge.json','annotations/review_tasks.json',
        'annotations/intent_batch2/judgments.json',cfg.get('evaluation_plan','docs/VERIFIED_EVALUATION_PLAN.md')]
    if 'expanded_query_path' in cfg:inputs.extend([cfg['expanded_query_path'],cfg['query_extension_path']])
    if 'diagnostic_cohorts_path' in cfg:inputs.append(cfg['diagnostic_cohorts_path'])
    input_hashes={str(p):sha256(p) for p in inputs}
    code_hashes={str(p):sha256(p) for p in [Path(__file__),Path('cadkg/experiments/verified_statistics.py'),Path('cadkg/retrieval.py'),Path('cadkg/reasoning.py')]}
    digests={p.name:sha256(p) for p in out.glob('*.csv')}
    write_json(out/'run_manifest.json',dict(config=cfg,input_sha256=input_hashes,code_sha256=code_hashes,artifact_sha256=digests,
        numpy_version=np.__version__,python=sys.version,elapsed_seconds=time.monotonic()-started,source='ai_reference',human_expert_reviewed=False))
    write_json(out/'integrity.json',dict(unique_complete_pairs=len(records),closed_candidates=len(ids),full_gallery=len(gallery),
      train_exemplars=len(protocol['exemplars']),source_bindings_verified=True,legacy_serving_top5_matches=2*len(original_query_ids)*8,
      new_query_serving_top5_and_score_matches=len(serving_checks),queries=len(protocol['queries']),
      full_flat_exact_order_equal=True,no_geometry_representation_order_equal=True,intent_category_filter_order_equal=True,
      exact_byte_contaminated_excluded=len(contaminated),cohort_sizes={k:len(v) for k,v in cohorts.items()},
      primary_cases=len(rows),robustness_cases=len(robust),methods=METHODS,weights_tuned=False,models_retrained=False))
    print(f'Finished in {time.monotonic()-started:.1f}s',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/verified_evaluation.json');parser.add_argument('--output')
    args=parser.parse_args();main(args.config,args.output)
