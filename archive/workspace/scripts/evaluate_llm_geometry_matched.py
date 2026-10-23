"""Frozen, matched GPT geometric-evidence reranking study.

--dry-validate uses synthetic inputs only and does not inspect study outcomes.
--prepare-freeze seals this evaluator, existing v6 inputs, study and pool before
new evidence is evaluated. --evaluate requires the main protocol and completed
evidence receipt to bind that seal and graph_support.json. No API is called.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from cadkg.experiments.reference import numeric_reference
from cadkg.experiments.verified_statistics import _bootstrap, _flip_p, _holm
from scripts.run_query_diversity import ArrayArchive
from scripts.supplement_partial_metrics import grade_limits, prepare_limits, rank_bounds

ROOT=REPO/'outputs/llm_geometry_matched_v1'
V6=REPO/'outputs/query_diversity_v6/evaluation'
METHODS=('kg_full','geometry_intent_filter','kg_full_llm')
POOL_METHODS=('kg_full','geometry_intent_filter','kg_no_class')
KINDS=('embedding','descriptor')
METRICS=('ndcg@5','strong_precision@5')
SEED=20261022
TOL=1e-12
PLAN={'primary_panel':'264 new_exemplar queries; four references and six conditions per each of eleven families',
      'representations':list(KINDS),'methods':list(METHODS),'gallery':1816,
      'pool_top_k':10,'pool_methods':list(POOL_METHODS),
      'pool_order':'sorted original closed-gallery index',
      'reranking':'(3 * frozen baseline kg_full score + H) / 4 inside fixed pool; append baseline Full tail outside pool',
      'rerank_ties':'ascending frozen gallery index (lexicographic part ID)',
      'H_range':[0,1],'H_neutral_fallback':.5,
      'H_semantics':'fixed mean of four geometric-evidence clauses: observed yes=1, no=0, unknown=.5; no learned/test-fitted weights',
      'grading':'unchanged nominal v6 grade_limits and independent numeric_reference; guard .2mm; all null labels retained',
      'metrics':list(METRICS),'averaging':'24 matched query conditions within family, then eleven families equally',
      'paired_tests':4,'endpoint':'ndcg@5','comparators':['kg_full','geometry_intent_filter'],
      'sign_flip':'exact two-sided family enumeration; eleven authored families',
      'multiple_testing':'one Holm family over four primary comparisons',
      'bootstrap_replicates':20000,'bootstrap_seed':SEED,
      'bootstrap_interval':'95% percentile; unadjusted exploratory family intervals',
      'gate':{'comparator':'geometry_intent_filter','ndcg_delta_at_least':.02,
              'holm_p_at_most':.05,'strong_precision_delta_at_least':0.,
              'required_representations':list(KINDS),'point_tests_require_all_primary_ndcg_identified':True},
      'manuscript_updates':'none by this script; parent may consider only if full prespecified gate passes'}


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def relative(path):return str(Path(path).resolve().relative_to(REPO)).replace('\\','/')


def core_paths(root):
    paths=[Path(__file__),REPO/'configs/query_diversity_v6.json',
        REPO/'annotations/intent_queries440_v6/queries.json',
        REPO/'annotations/intent_queries440_v6/query_manifest.csv',
        REPO/'outputs/intent_batch2_matrix_reproduction/protocol.json',
        REPO/'outputs/intent_batch2_matrix_reproduction/bundle/catalog.json',
        REPO/'outputs/intent_batch2_matrix_reproduction/bundle/knowledge.json',
        REPO/'outputs/intent_batch2_matrix_reproduction/bundle/graph.json',
        REPO/'outputs/intent_batch2_matrix_reproduction/bundle/bundle_manifest.json',
        REPO/'annotations/intent_feature_review_v5/judgments.json',
        REPO/'outputs/feature_review_v5/consumed_evidence_freeze.json',
        *[V6/name for name in ['run_manifest.json','effective_protocol.json','cohorts.json',
                             'complete_rankings.npz','per_query_bounds.csv']],
        *[REPO/name for name in ['cadkg/experiments/reference.py','cadkg/experiments/intent.py',
          'cadkg/experiments/verified_statistics.py','scripts/run_query_diversity.py',
          'scripts/partial_feature_review.py','scripts/supplement_partial_metrics.py']],
        root/'pool_manifest.json']
    paths.extend(p for p in [root/'study.json',root/'study.md',root/'protocol.md'] if p.exists())
    return paths


def panel():
    queries=read(REPO/'annotations/intent_queries440_v6/queries.json')
    meta={r['query_id']:r for r in csv_rows(REPO/'annotations/intent_queries440_v6/query_manifest.csv')}
    selected=[q for q in queries if meta[q['query_id']]['origin']=='new_exemplar']
    need(len(selected)==264 and len({q['query_id'] for q in selected})==264,'Primary panel is not264 unique new-exemplar queries')
    groups=defaultdict(list)
    for q in selected:groups[q['scenario_id']].append(q)
    need(len(groups)==11,'Exactly eleven primary families required')
    for family,rows in groups.items():
        refs=defaultdict(list)
        for q in rows:refs[q['reference_part_id']].append(q)
        need(len(rows)==24 and len(refs)==4 and all(len(rr)==6 for rr in refs.values()),f'Unbalanced primary family:{family}')
    return selected,meta


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


def prepare_freeze(root):
    need(not (root/'results.json').exists() and not (root/'evidence_completion.json').exists(),
         'Evaluator freeze must precede completed evidence and outcomes')
    queries,_=panel();gallery=read(V6/'cohorts.json')['closed']
    need(len(gallery)==len(set(gallery))==1816 and gallery==sorted(gallery),'Preserve1816 sorted closed candidates')
    with np.load(V6/'complete_rankings.npz',allow_pickle=False) as archive:
        pools,pairs=validate_pools(root,queries,gallery,archive)
    value={'status':'FROZEN_BEFORE_NEW_EVIDENCE_EVALUATION','plan':PLAN,
           'input_sha256':{relative(p):sha(p) for p in core_paths(root)},
           'primary_query_ids':[q['query_id'] for q in queries],
           'pool_entries':sum(len(p) for per_kind in pools.values() for p in per_kind.values()),
           'required_unique_family_part_pairs':len(pairs),'source_outcomes_used_for_new_weight_tuning':False,
           'new_evidence_read':False,'qrels_parsed':False}
    target=root/'evaluation_freeze.json'
    if target.exists():need(read(target)==value,'Refuse to replace a differing evaluator freeze')
    else:dump(target,value)
    print(json.dumps({'evaluation_freeze':relative(target),'sha256':sha(target),'pool_entries':value['pool_entries'],
                      'required_support_pairs':len(pairs)},indent=2))


def load_support(path,families,gallery):
    support=read(path)
    need(isinstance(support,dict) and set(support)<=set(families),'Support needs family→part→H mappings')
    for family,values in support.items():
        need(isinstance(values,dict),'Family support must be an object')
        for pid,value in values.items():
            need(pid in gallery and type(value) in (int,float) and np.isfinite(value) and 0<=value<=1,
                 f'Invalid geometric support:{family}/{pid}')
    return support


def means(rows):
    return {f'{metric}_{end}':float(np.mean([r[f'{metric}_{end}'] for r in rows]))
            for metric in METRICS for end in ('lower','upper')}


def analyze(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['geometry_kind'],r['method'],r['family_id']].append(r)
    family=[]
    for (kind,method,fid),members in sorted(grouped.items()):
        need(len(members)==24,'Every primary method/family needs24 query conditions')
        family.append(dict(geometry_kind=kind,method=method,family_id=fid,queries=24,**means(members)))
    summary=[]
    for kind in KINDS:
        for method in METHODS:
            rr=[r for r in family if r['geometry_kind']==kind and r['method']==method]
            need(len(rr)==11,'Every summary must average eleven families')
            summary.append(dict(geometry_kind=kind,method=method,queries=264,families=11,averaging='24_queries_per_family_then_equal_family',**means(rr)))
    identified=all(r['ndcg@5_lower']==r['ndcg@5_upper'] for r in rows)
    lookup={(r['geometry_kind'],r['method'],r['family_id']):r for r in family}
    families=sorted({r['family_id'] for r in family});tests=[];diffs=[]
    rng=np.random.default_rng(SEED)
    for kind in KINDS:
        for comparator in METHODS[:2]:
            local=[]
            for fid in families:
                a=lookup[kind,'kg_full_llm',fid];b=lookup[kind,comparator,fid]
                rec={'geometry_kind':kind,'comparator':comparator,'family_id':fid}
                for metric in METRICS:
                    rec[metric+'_lower']=a[metric+'_lower']-b[metric+'_upper']
                    rec[metric+'_upper']=a[metric+'_upper']-b[metric+'_lower']
                local.append(rec);diffs.append(rec)
            avg=means(local);values=np.asarray([r['ndcg@5_lower'] for r in local])
            record=dict(geometry_kind=kind,proposed_method='kg_full_llm',comparator=comparator,
                families=11,queries=264,point_inference_eligible=identified,
                delta_lower=avg['ndcg@5_lower'],delta_upper=avg['ndcg@5_upper'],
                strong_precision_delta_lower=avg['strong_precision@5_lower'],
                strong_precision_delta_upper=avg['strong_precision@5_upper'],
                delta=None,ci95_lower=None,ci95_upper=None,exact_sign_flip_p=None,holm_p=None,
                bootstrap_replicates=20000,bootstrap_seed=SEED,holm_family_size=4,
                family_wins=None,family_ties=None,family_losses=None,
                inference_status='IDENTIFIED_EXPLORATORY_POINT' if identified else 'WITHHELD_LABEL_UNCERTAINTY')
            if identified:
                ci=_bootstrap(values,rng,20000)
                record.update(delta=float(np.mean(values)),ci95_lower=ci[0],ci95_upper=ci[1],
                    exact_sign_flip_p=_flip_p(values),family_wins=int(np.sum(values>TOL)),
                    family_ties=int(np.sum(abs(values)<=TOL)),family_losses=int(np.sum(values<-TOL)))
            tests.append(record)
    if identified:
        for row,p in zip(tests,_holm([r['exact_sign_flip_p'] for r in tests])):row['holm_p']=p
    conditions=[]
    for kind in KINDS:
        test=next(r for r in tests if r['geometry_kind']==kind and r['comparator']=='geometry_intent_filter')
        conditions.append({'geometry_kind':kind,'point_identified':identified,
            'ndcg_delta_at_least_0_02':identified and test['delta']>=.02,
            'holm_p_at_most_0_05':identified and test['holm_p']<=.05,
            'strong_precision_nondecrease_vs_GFC':test['strong_precision_delta_lower']>=-TOL,
            'ndcg_delta_lower':test['delta_lower'],'ndcg_delta_upper':test['delta_upper'],
            'holm_p':test['holm_p'],'strong_precision_delta_lower':test['strong_precision_delta_lower'],
            'strong_precision_delta_upper':test['strong_precision_delta_upper']})
    passed=all(all(r[k] for k in ['point_identified','ndcg_delta_at_least_0_02','holm_p_at_most_0_05',
                                 'strong_precision_nondecrease_vs_GFC']) for r in conditions)
    return family,summary,tests,diffs,{'passed':passed,'conditions':conditions,
            'rule':PLAN['gate'],'manuscript_update_permitted_for_consideration':passed,
            'interpretation':'Exploratory authored AI-graded benchmark; selection gate is not independent human/engineering validation.'}


def evaluate(root):
    freeze_path=root/'evaluation_freeze.json';freeze=read(freeze_path)
    need(freeze['status']=='FROZEN_BEFORE_NEW_EVIDENCE_EVALUATION' and freeze['plan']==PLAN,'Evaluator plan changed after freeze')
    main_path=root/'protocol_freeze.json';main=read(main_path)
    need(main.get('evaluation_freeze_sha256')==sha(freeze_path),'Main protocol does not bind this exact evaluator freeze')
    completion_path=root/'evidence_completion.json';completion=read(completion_path)
    need(completion.get('status')=='completed','Full GPT evidence is not completed/frozen')
    need(completion.get('protocol_freeze_sha256')==sha(main_path),'Evidence completion does not bind main protocol freeze')
    support_path=root/'graph_support.json'
    need(completion.get('graph_support_sha256')==sha(support_path),'Geometric support differs from completed evidence')
    for path,digest in freeze['input_sha256'].items():need(sha(REPO/path)==digest,f'Frozen input/code changed:{path}')
    outputs=['per_query.csv','summary.csv','paired_tests.csv','family_metrics.csv','family_differences.csv',
             'full_rankings.npz','results.json']
    need(not any((root/p).exists() for p in outputs),'Refuse to overwrite evaluated outcomes')
    stage=root/'.evaluation_inprogress';need(not stage.exists(),'An interrupted evaluation stage already exists')
    config=read(REPO/'configs/query_diversity_v6.json');source_manifest=read(V6/'run_manifest.json')
    for name in ['cohorts.json','complete_rankings.npz','per_query_bounds.csv','effective_protocol.json']:
        need(sha(V6/name)==source_manifest['artifact_sha256'][name],f'Original v6 artifact changed:{name}')
    queries,metadata=panel();need([q['query_id'] for q in queries]==freeze['primary_query_ids'],'Frozen primary selection changed')
    gallery=read(V6/'cohorts.json')['closed'];need(len(gallery)==1816 and gallery==sorted(gallery),'Gallery/order changed')
    families=sorted({q['scenario_id'] for q in queries})
    support=load_support(support_path,families,set(gallery))
    annotation_path=REPO/config['annotation_path']
    expected_annotation=source_manifest['input_sha256'][str(Path(config['annotation_path'])).replace('/','\\')]
    need(sha(annotation_path)==expected_annotation,'The v6 qrels changed')
    annotation=read(annotation_path)
    need(annotation['source']=='ai_reference' and annotation['human_expert_reviewed'] is False,'Unexpected judgment provenance')
    records=annotation['judgments'];obs={(r['family_id'],r['part_id']):r for r in records}
    need(len(obs)==len(records)==1816*11,'The complete qrel matrix changed')
    consumed=read(REPO/'outputs/feature_review_v5/consumed_evidence_freeze.json')['input_sha256']
    consumed_paths={str((REPO/Path(p)).resolve()).lower():h for p,h in consumed.items()}
    raw={fid:np.asarray([-1 if obs[fid,pid]['grade'] is None else obs[fid,pid]['grade'] for pid in gallery],dtype=np.int8) for fid in families}
    evidence={};evidence_hashes={}
    for pid in gallery:
        row=obs[families[0],pid]
        path=REPO/config['evidence_root']/(row['inspection_id']+'.json')
        need(str(path.resolve()).lower() in consumed_paths,'Numerical evidence absent from original freeze')
        need(sha(path)==consumed_paths[str(path.resolve()).lower()],'Original numerical evidence changed')
        value=read(path)
        need(value['part_id']==pid and value['source_sha256']==row['source_sha256'],'Numerical evidence source binding changed')
        evidence[pid]=value;evidence_hashes[relative(path)]=sha(path)
    old={}
    for r in csv_rows(V6/'per_query_bounds.csv'):
        if r['cohort']=='closed' and r['grading_scenario']=='nominal' and r['method'] in METHODS[:2]:
            old[r['geometry_kind'],r['query_id'],r['method']]=r
    prepared={}
    for q in queries:
        verdicts=[numeric_reference(q,evidence[pid],.2) for pid in gallery]
        lo,hi=grade_limits(raw[q['scenario_id']],verdicts,'nominal')
        prepared[q['query_id']]=prepare_limits(lo,hi,(5,))
    stage.mkdir();rows=[];fallback=set();fallback_entries=0;baseline_checks=[]
    with np.load(V6/'complete_rankings.npz',allow_pickle=False) as archive,ArrayArchive(stage/'full_rankings.npz') as writer:
        pools,required_pairs=validate_pools(root,queries,gallery,archive)
        for kind in KINDS:
            for q in queries:
                qid=q['query_id'];fid=q['scenario_id'];pool=pools[kind][qid]
                local={m:archive_order(archive,kind,qid,m,len(gallery)) for m in METHODS[:2]}
                local['kg_full_llm']=rerank(*local['kg_full'],pool,support.get(fid,{}),gallery)
                missing=[pid for pid in pool if pid not in support.get(fid,{})]
                fallback.update((fid,pid) for pid in missing);fallback_entries+=len(missing)
                for method in METHODS:
                    order,scores=local[method]
                    bounds=rank_bounds(order,prepared[qid])
                    row=dict(geometry_kind=kind,query_id=qid,family_id=fid,method=method,cohort='closed',
                        grading_scenario='nominal',reference_part_id=q['reference_part_id'],
                        pool_size=len(pool),neutral_fallback_pool_entries=len(missing),
                        returned=min(5,len(order)))
                    row.update({f'{m}_{end}':float(bounds[m][i]) for m in METRICS for i,end in enumerate(['lower','upper'])})
                    if method in METHODS[:2]:
                        original=old[kind,qid,method]
                        differences=[abs(row[f'{m}_{end}']-float(original[f'{m}_{end}'])) for m in METRICS for end in ['lower','upper']]
                        need(max(differences)<=TOL,f'Original v6 metric continuity failed:{kind}/{qid}/{method}')
                        baseline_checks.append({'exact':max(differences)==0,'max_difference':max(differences)})
                    rows.append(row)
                    key=f'{kind}__{qid}__closed__{method}'
                    writer.add(key+'__order',order.astype(np.int32));writer.add(key+'__score',scores)
            print(f'Completed fixed {kind} primary matrix',flush=True)
    family,summary,tests,diffs,gate=analyze(rows)
    write_csv(stage/'per_query.csv',rows);write_csv(stage/'summary.csv',summary)
    write_csv(stage/'paired_tests.csv',tests);write_csv(stage/'family_metrics.csv',family);write_csv(stage/'family_differences.csv',diffs)
    result={'status':'COMPLETED_FROZEN_MATCHED_EVIDENCE_STUDY','plan':PLAN,'gate':gate,
        'source_freeze_sha256':sha(freeze_path),'main_protocol_freeze_sha256':sha(main_path),
        'evidence_completion_sha256':sha(completion_path),'graph_support_sha256':sha(support_path),
        'script_sha256':sha(Path(__file__)),'primary_query_method_cases':len(rows),'rankings':len(rows),
        'null_raw_qrel_pairs':sum(r['grade'] is None for r in records),
        'pool_entries':sum(len(p) for kind in pools.values() for p in kind.values()),
        'required_unique_support_pairs':len(required_pairs),
        'neutral_fallback_pool_entries':fallback_entries,'neutral_fallback_unique_pairs':len(fallback),
        'neutral_fallback_pairs':[{'family_id':fid,'part_id':pid,'H':.5} for fid,pid in sorted(fallback)],
        'baseline_continuity':{'comparisons':len(baseline_checks),'exact':sum(r['exact'] for r in baseline_checks),
            'max_absolute_difference':max(r['max_difference'] for r in baseline_checks),'all_within_1e_12':True},
        'numeric_evidence_sha256':evidence_hashes,
        'score_array_semantics':'Proposed: transformed scores inside reranked pool, original baseline scores in appended tail; intentionally not a globally score-sorted full-gallery formula.',
        'limitations':['Exploratory extension after known v6 outcomes; not preregistration of the original study.',
            'AI geometric evidence and AI reference relevance do not establish independent human truth.',
            'Pool access is deliberately matched and limited; no full-gallery semantic expansion is claimed.',
            'Four exact tests use eleven authored families, not264 independent task replications.',
            'Threshold .02 is a frozen practical-selection threshold, not an engineering-validated benefit.',
            'No manuscript edit is performed even if the gate passes.'],
        'api_calls':0,'labels_revised':False,'score_weights_tuned':False,
        'summary':summary,'paired_tests':tests}
    for path,digest in freeze['input_sha256'].items():need(sha(REPO/path)==digest,f'Frozen source changed during evaluation:{path}')
    need(sha(support_path)==completion['graph_support_sha256'],'Support changed during evaluation')
    result['artifact_sha256']={name:sha(stage/name) for name in outputs if name!='results.json'}
    dump(stage/'results.json',result)
    for name in outputs:shutil.move(str(stage/name),str(root/name))
    stage.rmdir()
    print(json.dumps({'gate':gate,'summary':summary,'paired_tests':tests},indent=2))


def dry_validate():
    gallery=[f'p{i:02}' for i in range(12)]
    order=np.arange(12);scores=np.linspace(.9,.2,12);pool=gallery[:10]
    neutral_order,neutral_scores=rerank(order,scores,pool,{},gallery)
    need(np.array_equal(neutral_order,order),'Neutral support changes baseline within-pool order')
    need(np.array_equal(neutral_order[10:],order[10:]),'Baseline tail not retained')
    need(np.allclose(neutral_scores[:10],(3*scores[:10]+.5)/4),'Proposed score formula incorrect')
    tied_order,tied_scores=rerank(order,np.full(12,.5),pool,{},gallery)
    need(np.array_equal(tied_order,order),'Tie policy is not ascending gallery index')
    lo,hi=grade_limits(np.array([2,1,-1,0]),['pass','unknown','pass','fail'])
    need(lo.tolist()==[2,1,0,0] and hi.tolist()==[2,1,2,0],'Original null/size transform is not preserved')
    cache=prepare_limits(lo,hi,(5,));bounds=rank_bounds(np.array([0,1]),cache)
    need(bounds['ndcg@5'][0]<=bounds['ndcg@5'][1] and bounds['strong_precision@5']==(.2,.2),'Metric-bound fixture failed')
    need(_flip_p(np.ones(11))==2/2048,'Exact family sign-flip fixture failed')
    need(_holm([.01,.04,.03,.9])==[.04,.09,.09,.9],'Four-test Holm fixture failed')
    # Synthetic complete balanced matrix verifies means/tests/gate shape without corpus data.
    rows=[]
    for kind in KINDS:
        for fid in range(11):
            for q in range(24):
                for method,value in zip(METHODS,[.5,.6,.7]):
                    rows.append(dict(geometry_kind=kind,family_id=f'f{fid:02}',method=method,
                        **{f'{m}_{end}':value for m in METRICS for end in ['lower','upper']}))
    _,summary,tests,_,gate=analyze(rows)
    need(len(summary)==6 and len(tests)==4 and gate['passed'],'Synthetic balanced analysis/gate failed')
    rows[-1]['ndcg@5_upper']=.8
    _,_,tests,_,gate=analyze(rows)
    need(not gate['passed'] and all(r['holm_p'] is None for r in tests),'Unidentified endpoint must withhold every point test')
    print(json.dumps({'status':'SYNTHETIC_DRY_VALIDATION_PASSED','corpus_or_new_support_read':False,
        'checks':['fixed pool/tail','fixed score algebra','tie rule','null/size bounds','exact2048familyflips',
                  'four-test Holm','balanced family means','prespecified gate','global unknown-label inference gate']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-validate',action='store_true');action.add_argument('--prepare-freeze',action='store_true')
    action.add_argument('--evaluate',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args();root=args.root.resolve()
    need(root.is_relative_to(ROOT.resolve()),'Output must stay within the new study root')
    if args.dry_validate:dry_validate()
    elif args.prepare_freeze:prepare_freeze(root)
    else:evaluate(root)


if __name__=='__main__':main()
