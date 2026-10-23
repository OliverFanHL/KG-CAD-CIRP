"""Independent v6 design, freeze, replay, task-transform and metric checks."""
import argparse,json,math,hashlib
from collections import Counter,defaultdict,OrderedDict
from itertools import product,permutations
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,sha256,object_hash
from scripts.run_query_diversity import METHODS,CODE_DEPENDENCIES,_paths
from scripts.finalize_feature_review import notes

ROOT=Path('outputs/query_diversity_v6');ANN=Path('annotations/intent_queries440_v6');CFG='configs/query_diversity_v6.json'

def _validation_population(parts,category,testhashes):
    """Reconstruct every eligible source from the catalog, not a supplied subset."""
    result=[]
    for p in parts.values():
        m=p['measurements'].get('aabb_large_mm',{});v=m.get('value')
        if (p['split']=='val' and p['label']==category and p['file_sha256'] not in testhashes
            and m.get('status')=='computed' and v is not None and not isinstance(v,bool)
            and math.isfinite(v) and v>0):
            assert m.get('unit')=='mm','Eligible validation extent has an unexpected unit'
            result.append(dict(part_id=p['part_id'],source_sha256=p['file_sha256'],value_mm=v))
    result.sort(key=lambda r:r['part_id'])
    assert result,'No eligible validation extent distribution'
    return result


def _linear_bound(measurements,quantile):
    """Independent scalar linear interpolation and the fixed six-decimal policy."""
    quantile=float(quantile);assert math.isfinite(quantile) and 0<=quantile<=1
    values=sorted(r['value_mm'] for r in measurements);assert values
    position=(len(values)-1)*quantile;lo=math.floor(position);hi=math.ceil(position)
    return round(values[lo]+(values[hi]-values[lo])*(position-lo),6)


def _text_schedule(fid,old,measurements):
    """Replay the prespecified priority/grid schedule, skipping structured bounds."""
    used={float(x['constraints'][0]['value']) for x in old
          if x['scenario_id']==fid and x['reference_part_id'] is None and x['constraints']}
    selected=[]
    for quantile in [.05,.10,.35,.50,.65,.90]+[i/100 for i in range(1,100)]:
        value=_linear_bound(measurements,quantile)
        if value in used:continue
        selected.append((quantile,value));used.add(value)
        if len(selected)==6:break
    assert len(selected)==6,'Insufficient distinct validation-only text bounds'
    return selected

def design():
    cfg=read_json(CFG);q=read_json(cfg['queries_path']);meta={r['query_id']:r for r in read_csv(cfg['query_manifest_path'])}
    ext=read_json(ANN/'protocol_extension.json')
    for path,h in {**ext['input_sha256'],**ext['file_sha256']}.items():assert sha256(path)==h,path
    assert tuple(cfg['methods'])==METHODS and len(q)==440 and len(meta)==440
    old=read_json('annotations/intent_queries110_v3/queries.json');assert q[:110]==old
    assert Counter(x['scenario_id'] for x in q)=={x['scenario_id']:40 for x in q}
    assert Counter(r['origin'] for r in meta.values())==dict(legacy110=110,new_exemplar=264,new_text=66)
    assert len({json.dumps({k:x[k] for k in ['scenario_id','function_ids','category_ids','reference_part_id','constraints']},sort_keys=True) for x in q})==440
    protocol=read_json(Path(cfg['source_run'])/'protocol.json');families={f['family_id']:f for f in protocol['families']}
    catalog=read_json(Path(cfg['source_run'])/'bundle/catalog.json');parts={p['part_id']:p for p in catalog['parts']}
    knowledge=read_json(Path(cfg['source_run'])/'bundle/knowledge.json')
    testhashes={p['file_sha256'] for p in parts.values() if p['split']=='test'};refs=read_csv(ANN/'exemplars.csv');source=read_json(ANN/'validation_sources.json')
    assert len(refs)==44 and Counter(r['family_id'] for r in refs)=={f:4 for f in families}
    audited={}
    for path in sorted((ROOT/'exemplar_pool').glob('*_notes.txt')):
        coverage=path.with_name(path.name.replace('_notes.txt','_coverage.json'))
        if coverage.exists():
            c=read_json(coverage)
            if c.get('complete') or c.get('status')=='complete':
                assert c['notes_sha256']==sha256(path)
                audited.update(notes(path))
    pool=read_csv(ROOT/'exemplar_pool/private/candidate_manifest.csv')
    # Independently verify additive files and the complete original FPS prefix.
    expansion_path=ROOT/'exemplar_pool/expansion_freeze.json'
    if expansion_path.exists():
        expansion=read_json(expansion_path)
        for path,h in expansion['inputs_sha256'].items():assert sha256(path)==h,(path,'additive input')
        for path,h in expansion['outputs_sha256'].items():assert sha256(ROOT/'exemplar_pool'/path)==h,(path,'additive output')
        original=read_json(ROOT/'exemplar_pool/pool_freeze.json')
        for path,h in original['outputs_sha256'].items():assert sha256(ROOT/'exemplar_pool'/path)==h,(path,'original pool output')
        continuity=expansion['original_selection_continuity']
        assert sha256(continuity['original_preparation_script_snapshot'])==continuity['original_preparation_script_sha256']
        assert sha256(continuity['current_render_resume_adapter'])==continuity['current_render_resume_adapter_sha256']
        check=read_json(ROOT/'exemplar_pool/private/expansion_selection_audit.json')
        before={(r['family_id'],int(r['selection_step'])):r for r in pool}
        assert len(pool)==len(check['prefix_checks'])==132 and check['all_original_132_prefixes_identical']
        assert expansion['original_132_prefixes_identical'] and expansion['original_normalization_reused']
        assert {(r['family_id'],r['selection_step']) for r in check['prefix_checks']}==set(before)
        for r in check['prefix_checks']:
            p=before[r['family_id'],r['selection_step']]
            assert (p['part_id'],p['source_sha256'],p['review_id'])==(r['part_id'],r['source_sha256'],r['original_review_id'])
        added=read_csv(ROOT/'exemplar_pool/private/expansion_candidate_manifest.csv')
        assert len(added)==24 and {r['family_id'] for r in added}=={'support_body','tension_return'}
        for fid in {'support_body','tension_return'}:assert sorted(int(r['selection_step']) for r in added if r['family_id']==fid)==list(range(13,25))
        assert {r['review_id'] for r in added}=={f'Q{i:04}' for i in range(133,157)}
        assert {r['review_id'] for r in added}<=set(audited),'All additive source audits must precede release'
        pool+=added
        assert len(pool)==len({r['review_id'] for r in pool})==len({r['source_sha256'] for r in pool})==156
    for fid in families:
        prefix=[]
        for r in sorted([r for r in pool if r['family_id']==fid],key=lambda r:int(r['selection_step'])):
            if len(prefix)==4:break
            assert r['review_id'] in audited,'Cannot skip an unreviewed earlier FPS candidate'
            if audited[r['review_id']]['grades'].get(r['role'],0) in (1,2):prefix.append(r['review_id'])
        actual=[r['review_id'] for r in sorted([r for r in refs if r['family_id']==fid],key=lambda r:int(r['selection_step']))]
        assert actual==prefix
        links=[r for r in knowledge['class_functions'] if r['function_id']==families[fid]['function_id']]
        assert len(links)==1 and links[0]['relevance']*links[0]['confidence']==1
        category=links[0]['class_id'];expected=_validation_population(parts,category,testhashes)
        assert source[fid]['split']=='val' and source[fid]['category']==category and source[fid]['exact_test_hashes_excluded'] is True
        assert source[fid]['measurements']==expected,'Validation distribution omitted/added/reordered an eligible mapped-class source'
        for row in expected:
            p=parts[row['part_id']];assert sha256(p['absolute_path'])==row['source_sha256']
    crossed=defaultdict(list)
    for x in q:
        r=meta[x['query_id']];fid=x['scenario_id']
        assert x['function_ids']==[families[fid]['function_id']] and x['category_ids']==[]
        if r['origin']=='legacy110':continue
        if r['origin']=='new_exemplar':
            assert x['reference_part_id'] is not None
            crosskey=fid,x['reference_part_id'];crossed[crosskey].append(r['condition'])
            p=parts[x['reference_part_id']];assert p['split']=='train' and p['file_sha256'] not in testhashes
            assert sha256(p['absolute_path'])==p['file_sha256']
            ref=next(a for a in refs if a['part_id']==p['part_id']);assert int(r['reference_grade'])==int(ref['source_grade']) in (1,2)
        else:assert x['reference_part_id'] is None
        if r['origin']=='new_exemplar':
            wanted={'unconstrained':None,'val_q10':.10,'val_q25':.25,'val_q50':.50,'val_q75':.75,'val_q90':.90}
            assert r['condition'] in wanted
            expected_quantile=wanted[r['condition']]
            assert (not x['constraints'] and expected_quantile is None and r['validation_quantile']=='') or (
                bool(x['constraints']) and expected_quantile is not None and float(r['validation_quantile'])==expected_quantile)
        else:
            assert r['origin']=='new_text' and x['constraints'],'Every new text query must have its scheduled numeric bound'
            assert r['condition']==f"val_q{int(round(float(r['validation_quantile'])*100)):02}"
        assert r['query_object_sha256']==object_hash(x),'Query manifest object binding changed'
        if not x['constraints']:assert r['validation_quantile']=='';continue
        assert len(x['constraints'])==1;c=x['constraints'][0]
        assert (c['attribute'],c['op'],c['unit'])==('aabb_large_mm','<=','mm')
        # Independent linear quantile interpolation, rather than calling builder/np.quantile.
        value=_linear_bound(source[fid]['measurements'],r['validation_quantile'])
        assert abs(c['value']-value)<=1e-6 and c['value']==float(r['bound_mm'])
    assert len(crossed)==44 and all(set(v)=={'unconstrained','val_q10','val_q25','val_q50','val_q75','val_q90'} and len(v)==6 for v in crossed.values())
    for fid in families:
        actual=[meta[x['query_id']] for x in q if x['scenario_id']==fid and meta[x['query_id']]['origin']=='new_text']
        schedule=_text_schedule(fid,old,source[fid]['measurements'])
        assert len(actual)==6 and [float(r['validation_quantile']) for r in actual]==[quantile for quantile,_ in schedule], 'New-text quantile priority/duplicate-skip/grid schedule changed'
        assert all(abs(float(r['bound_mm'])-bound)<=1e-6 for r,(_,bound) in zip(actual,schedule))
    result=dict(status='VERIFIED',queries=440,methods=7,new_references=44,new_reference_queries=264,text_queries=99,constrained_exemplar_queries=278,constrained_queries=366,all_old_queries_exact=True,structured_queries_unique=True,validation_only_bound_reconstruction=True,complete_validation_source_population=True,reference_quantile_conditions_verified=True,text_priority_skip_grid_verified=True,first_four_eligible_source_order_verified=True,annotation_sha256=sha256(cfg['annotation_path']))
    write_json(ROOT/'design_validation.json',result);print(result);return result

def freeze():
    assert not (ROOT/'experiment_freeze.json').exists()
    design();cfg=read_json(CFG);paths=[Path(__file__),Path('scripts/run_query_diversity.py'),Path('scripts/analyze_query_diversity.py'),Path('scripts/prepare_query_diversity.py'),Path('docs/QUERY_DIVERSITY_V6.md'),Path(CFG),Path(cfg['annotation_path']),Path(cfg['queries_path']),Path(cfg['query_manifest_path']),Path(cfg['cohorts_path']),ROOT/'design_validation.json',ANN/'protocol_extension.json']
    paths.extend(Path(p) for p in CODE_DEPENDENCIES);paths.extend(_paths(cfg).values());paths.extend([Path('cadkg/experiments/verified_statistics.py'),ROOT/'environment.json'])
    source=Path(cfg['source_run']);paths.extend(source/p for p in ['protocol.json','bundle/catalog.json','bundle/knowledge.json','bundle/graph.json','bundle/bundle_manifest.json'])
    ext=read_json(ANN/'protocol_extension.json');paths.extend(Path(p) for p in {**ext['input_sha256'],**ext['file_sha256']})
    write_json(ROOT/'experiment_freeze.json',dict(stage='Before any v6 retrieval/metric result',input_sha256={str(p):sha256(p) for p in paths},primary_origin='new_exemplar',primary_queries=264,primary_tests=12,models_tuned=False,labels_changed=False,human_reviewed=False))
    print('Experiment frozen')

def _ratio(a,b,c,d):
    return (a/d,min(1.,b/c) if c else float(b>0)) if d else (0.,0.)


def _metric_denominators(lo,hi):
    low=list(map(int,lo));high=list(map(int,hi))
    assert len(low)==len(high) and all(0<=a<=b<=2 for a,b in zip(low,high))
    return dict(rlo=sum(g>0 for g in low),rhi=sum(g>0 for g in high),
                ideal_lo=sorted(low,reverse=True)[:20],ideal_hi=sorted(high,reverse=True)[:20])


def _metrics_prepared(order,lo,hi,denominators):
    order=np.asarray(order)
    assert order.ndim==1 and order.dtype.kind in 'iu' and len(np.unique(order))==len(order)
    assert np.all((order>=0)&(order<len(lo)))
    low=lo[order].tolist();high=hi[order].tolist();rlo=denominators['rlo'];rhi=denominators['rhi']
    ans={};ideal_lo=denominators['ideal_lo'];ideal_hi=denominators['ideal_hi']
    for k in [5,10,20]:
        dcg=lambda g:sum((2**x-1)/math.log2(i+2) for i,x in enumerate(g[:k]))
        ans[f'ndcg@{k}']=_ratio(dcg(low),dcg(high),dcg(ideal_lo),dcg(ideal_hi))
        ans[f'precision@{k}']=(sum(g>0 for g in low[:k])/k,sum(g>0 for g in high[:k])/k)
        ans[f'strong_precision@{k}']=(sum(g==2 for g in low[:k])/k,sum(g==2 for g in high[:k])/k)
        ans[f'recall@{k}']=_ratio(sum(g>0 for g in low[:k]),sum(g>0 for g in high[:k]),max(1,rlo),rhi)
    def num(g,k=None):
        count=0;total=0.
        for i,x in enumerate(g if k is None else g[:k],1):
            if x>0:count+=1;total+=count/i
        return total
    ans['ap']=_ratio(num(low),num(high),max(1,rlo),rhi)
    ans['ap@20']=_ratio(num(low,20),num(high,20),max(1,min(20,rlo)),min(20,rhi))
    return ans


def independent_metrics(order,lo,hi):
    """Original public helper interface; scalar denominators can be cached by callers."""
    lo=np.asarray(lo);hi=np.asarray(hi)
    return _metrics_prepared(order,lo,hi,_metric_denominators(lo,hi))


def independent_endpoints(query,ids,obs,evidence,scenario,guard):
    """Separate raw-set enumeration, conditional transform, then all-constraint verdict."""
    assert scenario in {'nominal','conditional_zero','conditional_two'} and math.isfinite(guard) and guard>=0
    low=[];high=[]
    for pid in ids:
        raw_grade=obs[query['scenario_id'],pid]['grade'];e=evidence[pid];allowed=[]
        assert raw_grade is None or type(raw_grade) is int and raw_grade in (0,1,2)
        status='not_requested'
        if query['constraints']:
            if e['aabb_quarantined'] or e['faces_missing_mesh']:status='unknown'
            else:
                outcomes=[]
                for c in query['constraints']:
                    assert (c['attribute'],c['op'],c['unit'])==('aabb_large_mm','<=','mm')
                    diff=e['surface_large_mm']-c['value'];assert math.isfinite(diff)
                    outcomes.append('fail' if diff>guard else 'unknown' if abs(diff)<=guard else 'pass')
                status='fail' if 'fail' in outcomes else 'unknown' if 'unknown' in outcomes else 'pass'
        for raw in ([0,1,2] if raw_grade is None else [raw_grade]):
            grade=raw
            if grade==1 and scenario!='nominal':grade=0 if scenario=='conditional_zero' else 2
            if status=='fail':grade=0
            elif status=='unknown':grade=min(grade,1)
            allowed.append(grade)
        low.append(min(allowed));high.append(max(allowed))
    return np.asarray(low,dtype=np.int8),np.asarray(high,dtype=np.int8)


def _random_metrics(feasible,lo,hi,denominators):
    """Independent scalar uniform-feasible expectation envelopes at existing k=5."""
    if not len(feasible):return {k:(0.,0.) for k in ('expected_ndcg5','expected_precision5','expected_strong_precision5')}
    low=[int(lo[i]) for i in feasible];high=[int(hi[i]) for i in feasible];count=len(feasible);returned=min(5,count)
    discounts=sum(1/math.log2(i+2) for i in range(returned))
    numerator_low=sum(2**g-1 for g in low)/count*discounts;numerator_high=sum(2**g-1 for g in high)/count*discounts
    idcg_low=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(denominators['ideal_lo'][:5]))
    idcg_high=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(denominators['ideal_hi'][:5]))
    return dict(expected_ndcg5=_ratio(numerator_low,numerator_high,idcg_low,idcg_high),
                expected_precision5=(sum(g>0 for g in low)/count*returned/5,sum(g>0 for g in high)/count*returned/5),
                expected_strong_precision5=(sum(g==2 for g in low)/count*returned/5,sum(g==2 for g in high)/count*returned/5))


def replay():
    from cadkg.reasoning import check_constraints
    cfg=read_json(CFG);out=Path(cfg['output']);rep=ROOT/'replay';a=read_json(out/'run_manifest.json');b=read_json(rep/'run_manifest.json');assert a==b
    frozen=read_json(ROOT/'experiment_freeze.json')['input_sha256']
    for p,h in frozen.items():assert sha256(p)==h,p
    for name,h in a['artifact_sha256'].items():assert sha256(out/name)==sha256(rep/name)==h,name
    arrays=0
    for name in ['complete_rankings.npz','robustness_complete_rankings.npz']:
        with np.load(out/name,allow_pickle=False) as x,np.load(rep/name,allow_pickle=False) as y:
            assert x.files==y.files
            for k in x.files:assert np.array_equal(x[k],y[k]),k
            arrays+=len(x.files)
    q={x['query_id']:x for x in read_json(cfg['queries_path'])};cohorts=read_json(cfg['cohorts_path']);ids=cohorts['closed']
    records=read_json(cfg['annotation_path'])['judgments'];obs={(r['family_id'],r['part_id']):r for r in records}
    assert len(q)==440 and len(ids)==1816 and len(obs)==len(records)==19976
    consumed=read_json(_paths(cfg)['evidence_freeze'])['input_sha256']
    for p,h in consumed.items():assert sha256(p)==h,p
    resolved_consumed={Path(p).resolve() for p in consumed};evidence={};inspection={}
    for r in records:
        pid=r['part_id'];path=Path(cfg['evidence_root'])/(r['inspection_id']+'.json')
        if pid not in evidence:
            assert path.resolve() in resolved_consumed
            evidence[pid]=read_json(path);inspection[pid]=r['inspection_id']
        assert inspection[pid]==r['inspection_id'] and evidence[pid]['part_id']==pid and evidence[pid]['source_sha256']==r['source_sha256']
    assert len(evidence)==1816
    position={pid:i for i,pid in enumerate(ids)}
    indexes={name:np.asarray([position[p] for p in selected],dtype=int) for name,selected in cohorts.items()}
    base_endpoints={};endpoints={};denominators={};transform_cases=0;cache=OrderedDict();computed_metrics=0

    def get_endpoints(query,scenario,guard,cohort):
        nonlocal transform_cases
        base=(query['scenario_id'],json.dumps(query['constraints'],sort_keys=True),scenario,float(guard))
        key=base+(cohort,)
        if base not in base_endpoints:
            base_endpoints[base]=independent_endpoints(query,ids,obs,evidence,scenario,guard)
            transform_cases+=len(ids)
        if key not in endpoints:
            lo,hi=base_endpoints[base];ix=indexes[cohort]
            endpoints[key]=(lo[ix],hi[ix]);denominators[key]=_metric_denominators(*endpoints[key])
        return key,endpoints[key]

    def check_row(row,order,scenario='nominal',guard=.2,cohort='closed'):
        nonlocal computed_metrics
        key,(lo,hi)=get_endpoints(q[row['query_id']],scenario,guard,cohort)
        raw_order=order.tobytes();checkkey=key,hashlib.sha256(raw_order).digest()
        if checkkey not in cache:
            cache[checkkey]=(raw_order,_metrics_prepared(order,lo,hi,denominators[key]));computed_metrics+=1
            if len(cache)>4096:cache.popitem(last=False)
        else:assert cache[checkkey][0]==raw_order,'Ranking cache hash collision'
        cache.move_to_end(checkkey)
        for metric,(lower,upper) in cache[checkkey][1].items():
            assert abs(float(row[metric+'_lower'])-lower)<1e-12 and abs(float(row[metric+'_upper'])-upper)<1e-12,(row['query_id'],cohort,guard,metric)

    checked=closed_checked=0;seen=set()
    expected={(kind,qid,cohort,method,scenario) for kind,qid,cohort,method in product(('embedding','descriptor'),q,cohorts,METHODS)
              for scenario in (['nominal','conditional_zero','conditional_two'] if cohort=='closed' else ['nominal'])}
    with np.load(out/'complete_rankings.npz',allow_pickle=False) as orders:
        assert set(orders.files)=={f'{kind}__{qid}__{cohort}__{method}__{suffix}' for kind,qid,cohort,method in product(('embedding','descriptor'),q,cohorts,METHODS) for suffix in ('order','score')}
        for row in read_csv(out/'per_query_bounds.csv'):
            key=tuple(row[k] for k in ('geometry_kind','query_id','cohort','method','grading_scenario'));assert key in expected and key not in seen;seen.add(key)
            kind,qid,cohort,method,scenario=key
            order=orders[f'{kind}__{qid}__{cohort}__{method}__order']
            check_row(row,order,scenario,.2,cohort);checked+=1;closed_checked+=int(cohort=='closed')
        assert seen==expected
        nominal_orders=orders
        cases=read_csv(out/'robustness_cases.csv')
        case_by_id={(r['geometry_kind'],r['query_id'],r['method'],r['robustness'],float(r['severity']),int(r['seed'])):r for r in cases}
        expected_robust=set()
        for kind,x,method in product(('embedding','descriptor'),q.values(),METHODS):
            for guard in cfg['constraint_guards_mm']:
                if guard!=.2 and x['constraints']:expected_robust.add((kind,x['query_id'],method,'constraint_guard',float(guard),0))
            for seed in cfg['robustness_seeds']:expected_robust.add((kind,x['query_id'],method,'tie_order',0.,int(seed)))
            for fraction,seed in product(cfg['classification_corruption_fractions'],cfg['robustness_seeds']):
                expected_robust.add((kind,x['query_id'],method,'classifier_permutation',float(fraction),int(seed)))
        assert set(case_by_id)==expected_robust and len(case_by_id)==len(cases)
        robust_checked=0;robust_seen=set()
        with np.load(out/'robustness_complete_rankings.npz',allow_pickle=False) as robust_orders:
            assert set(robust_orders.files)=={r[name] for r in cases for name in ('key','score_key')}
            for row in read_csv(out/'robustness_per_query_bounds.csv'):
                key=row['geometry_kind'],row['query_id'],row['method'],row['robustness'],float(row['severity']),int(row['seed'])
                assert key in case_by_id and key not in robust_seen;robust_seen.add(key);case=case_by_id[key]
                guard=float(case['guard_mm']);assert guard==float(row['guard_mm'])==(key[4] if key[3]=='constraint_guard' else .2)
                order=robust_orders[case['key']]
                if key[3]=='constraint_guard':assert np.array_equal(order,nominal_orders[f'{key[0]}__{key[1]}__closed__{key[2]}__order'])
                check_row(row,order,'nominal',guard,'closed');robust_checked+=1
            assert robust_seen==expected_robust
    parts={p['part_id']:p for p in read_json(Path(cfg['source_run'])/'bundle/catalog.json')['parts']}
    feasible_cache={};random_checked=0;random_seen=set()
    for row in read_csv(out/'random_expectation_bounds.csv'):
        kind,qid,cohort=row['geometry_kind'],row['query_id'],row['cohort'];identity=kind,qid,cohort
        assert identity not in random_seen;random_seen.add(identity);query=q[qid]
        key,(lo,hi)=get_endpoints(query,'nominal',.2,cohort)
        feasible_key=json.dumps(query['constraints'],sort_keys=True),cohort
        if feasible_key not in feasible_cache:
            feasible_cache[feasible_key]=[i for i,pid in enumerate(cohorts[cohort]) if check_constraints(parts[pid],query['constraints'])['status'] in {'pass','not_requested'}]
        for metric,(lower,upper) in _random_metrics(feasible_cache[feasible_key],lo,hi,denominators[key]).items():
            assert abs(float(row[metric+'_lower'])-lower)<1e-12 and abs(float(row[metric+'_upper'])-upper)<1e-12,(identity,metric)
        random_checked+=1
    assert random_seen==set(product(('embedding','descriptor'),q,cohorts))
    for pid in ids:assert sha256(parts[pid]['absolute_path'])==parts[pid]['file_sha256']
    for p,h in frozen.items():assert sha256(p)==h,p
    for p,h in consumed.items():assert sha256(p)==h,p
    for name,h in a['artifact_sha256'].items():assert sha256(out/name)==sha256(rep/name)==h,name
    result=dict(status='VERIFIED',run_manifest_sha256=sha256(out/'run_manifest.json'),queries=440,methods=7,representations=2,candidate_cads=1816,identical_replay_artifacts=len(a['artifact_sha256']),identical_replay_arrays=arrays,closed_metric_bound_cases_independently_checked=closed_checked,all_cohort_metric_bound_cases_independently_checked=checked,robustness_metric_bound_cases_independently_checked=robust_checked,random_expectation_cases_independently_checked=random_checked,unique_metric_orders_calculated=computed_metrics,independent_grade_endpoint_cases=transform_cases,cohort_grade_endpoint_vectors=len(endpoints),numerical_evidence_json_files_read=len(evidence),raw_test_step_hashes_verified=1816,annotation_sha256=sha256(cfg['annotation_path']),annotation_unchanged_from_v5=True,null_pairs_preserved=a['raw_null_pairs'],legacy110_continuity=a['continuity'],human_reviewed=False,verification_scope='Computational source/input/replay/metric checks; not human relevance validation')
    write_json(ROOT/'validation.json',result);print(result)

def self_test():
    """No files or real labels: independent exact completions and frozen-code parity."""
    from cadkg.experiments.reference import numeric_reference
    from scripts.supplement_partial_metrics import prepare_limits,rank_bounds,random_bounds,grade_limits
    cases=completions=random_cases=endpoint_cases=0

    def exact(order,grades):
        result={};relevant=sum(g!=0 for g in grades)
        for k in (5,10,20):
            gain=lambda g:3 if g==2 else int(g==1)
            ideal=sum(gain(g)/math.log2(i+2) for i,g in enumerate(sorted(grades,reverse=True)[:k]))
            actual=sum(gain(grades[j])/math.log2(i+2) for i,j in enumerate(order[:k]))
            hits=sum(grades[j]!=0 for j in order[:k])
            result.update({f'ndcg@{k}':actual/ideal if ideal else 0.,f'precision@{k}':hits/k,
                f'strong_precision@{k}':sum(grades[j]==2 for j in order[:k])/k,
                f'recall@{k}':hits/relevant if relevant else 0.})
        hits=0;full=short=0.
        for rank,j in enumerate(order,1):
            if grades[j]!=0:
                hits+=1;full+=hits/rank
                if rank<=20:short+=hits/rank
        result['ap']=full/relevant if relevant else 0.
        result['ap@20']=short/min(20,relevant) if relevant else 0.
        return result

    def check(sets,order):
        nonlocal cases,completions
        lo=np.asarray([min(s) for s in sets],dtype=np.int8);hi=np.asarray([max(s) for s in sets],dtype=np.int8)
        order=np.asarray(order,dtype=int);actual=independent_metrics(order,lo,hi)
        frozen=rank_bounds(order,prepare_limits(lo,hi))
        for name,bounds in actual.items():assert np.allclose(bounds,frozen[name],rtol=0,atol=1e-12),(name,bounds,frozen[name])
        for grades in product(*sets):
            for name,value in exact(order,grades).items():
                lower,upper=actual[name];assert lower-1e-12<=value<=upper+1e-12,(sets,order,name,value,actual[name])
                if all(len(s)==1 for s in sets):assert abs(lower-value)<1e-12 and abs(upper-value)<1e-12
            completions+=1
        cases+=1

    choices=((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
    for n in range(4):
        for sets in product(choices,repeat=n):
            for count in range(n+1):
                for order in permutations(range(n),count):check(sets,order)
            lo=np.asarray([min(s) for s in sets],dtype=np.int8);hi=np.asarray([max(s) for s in sets],dtype=np.int8)
            for mask in product((False,True),repeat=n):
                feasible=np.asarray([i for i,v in enumerate(mask) if v],dtype=int)
                bounds=_random_metrics(feasible,lo,hi,_metric_denominators(lo,hi))
                frozen=random_bounds(feasible,prepare_limits(lo,hi))
                for name in bounds:assert np.allclose(bounds[name],frozen[name],rtol=0,atol=1e-12)
                random_orders=list(permutations(feasible,min(5,len(feasible))))
                for grades in product(*sets):
                    values=[exact(order,grades) for order in random_orders]
                    for name,metric in [('expected_ndcg5','ndcg@5'),('expected_precision5','precision@5'),('expected_strong_precision5','strong_precision@5')]:
                        value=sum(v[metric] for v in values)/len(values);lower,upper=bounds[name]
                        assert lower-1e-12<=value<=upper+1e-12
                        if all(len(s)==1 for s in sets):assert abs(lower-value)<1e-12 and abs(upper-value)<1e-12
                random_cases+=1
    # Exercise cutoff saturation, missing returns and relevance beyond rank20.
    for baseline in (0,1,2):
        for tail in product(choices,repeat=2):
            sets=((baseline,),)*20+tail
            for order in (tuple(range(22)),tuple(reversed(range(22))),tuple(range(7)),()):check(sets,order)

    ids=['a','b','c'];evidence={p:dict(surface_large_mm=v,aabb_quarantined=False,faces_missing_mesh=False) for p,v in zip(ids,(8.,10.,12.))}
    schedules=[[],[10.],[8.],[12.],[9.8],[10.2],[11.,9.]]
    for raw in product((None,0,1,2),repeat=3):
        obs={('f',p):dict(grade=g) for p,g in zip(ids,raw)}
        for thresholds,guard,scenario,flags in product(schedules,(0.,.2,.5),('nominal','conditional_zero','conditional_two'),((False,False),(True,False),(False,True))):
            query=dict(scenario_id='f',constraints=[dict(attribute='aabb_large_mm',op='<=',unit='mm',value=v) for v in thresholds])
            evidence['b'].update(aabb_quarantined=flags[0],faces_missing_mesh=flags[1])
            actual=independent_endpoints(query,ids,obs,evidence,scenario,guard)
            expected=grade_limits(np.asarray([-1 if g is None else g for g in raw]),[numeric_reference(query,evidence[p],guard) for p in ids],scenario)
            assert all(np.array_equal(a,b) for a,b in zip(actual,expected)),(raw,thresholds,guard,scenario,flags)
            endpoint_cases+=len(ids)
    def part(pid,split='val',label='category',value=10.,status='computed',source=None):
        return dict(part_id=pid,split=split,label=label,file_sha256=source or pid,
                    measurements=dict(aabb_large_mm=dict(value=value,status=status,unit='mm')))
    parts={p['part_id']:p for p in [part('b',value=20.),part('a'),part('test',split='test'),part('train',split='train'),part('wrong',label='other'),part('overlap',source='heldout'),part('missing',value=None),part('nan',value=float('nan')),part('zero',value=0.),part('bad',status='failed'),part('boolean',value=True)]}
    assert _validation_population(parts,'category',{'heldout'})==[dict(part_id='a',source_sha256='a',value_mm=10.),dict(part_id='b',source_sha256='b',value_mm=20.)]
    values=[dict(value_mm=v) for v in range(10,101,10)]
    old=[dict(scenario_id='f',reference_part_id=None,constraints=[dict(value=_linear_bound(values,q))]) for q in (.10,.50)]
    assert [q for q,_ in _text_schedule('f',old,values)]==[.05,.35,.65,.90,.01,.02]
    assert _linear_bound(values,.25)==32.5 and _linear_bound(values,0)==10 and _linear_bound(values,1)==100
    result=dict(status='PASSED',synthetic_metric_orders=cases,exact_metric_completions=completions,
                synthetic_random_populations=random_cases,synthetic_endpoint_cases=endpoint_cases,
                complete_source_population_fixture=True,text_priority_skip_grid_fixture=True,
                files_read=0,human_validation=False)
    print(json.dumps(result,sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['design','freeze','replay','self-test']);a=p.parse_args();{'design':design,'freeze':freeze,'replay':replay,'self-test':self_test}[a.action]()
