"""Independent scalar audit of the completed, fixed Sol matched study.

Read-only toward study inputs/results. Writes only post_execution_audit.md/json.
The oracle imports no project grading, numerical-reference, metric, statistics,
evidence-completion or API runner helpers. It never reads credentials or sends API
requests. It refuses to inspect response logs before complete result receipts.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/llm_geometry_sol_matched_v1'
LUNA=REPO/'outputs/llm_geometry_matched_v1'
V6=REPO/'outputs/query_diversity_v6/evaluation'
METHODS=('kg_full','geometry_intent_filter','kg_full_llm','kg_full_sol')
KINDS=('embedding','descriptor')
COMPARATORS=('kg_full_llm','geometry_intent_filter')
FIELDS=('ndcg@5_lower','ndcg@5_upper','strong_precision@5_lower','strong_precision@5_upper')
VIEWS={'oblique','opposite','profile','axial','cut_X','cut_Y'}
TOL=1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')


def path(value):
    p=Path(value)
    return p if p.is_absolute() else REPO/p


def csv_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def lines(p):
    raw=Path(p).read_bytes()
    assert raw.endswith(b'\n'),'Refuse incomplete durable log'
    assert all(raw.splitlines()),'Refuse blank durable-log records'
    return [json.loads(s) for s in raw.splitlines()]


def structural(value,schema):
    """Independent validation of the fixed object/array/string schema subset."""
    allowed={'type','properties','required','additionalProperties','items','enum',
             'minItems','maxItems','uniqueItems','minLength','maxLength'}
    assert not set(schema)-allowed,'Unexpected schema assertion; audit needs explicit support'
    kind=schema.get('type')
    if kind=='object':
        if not isinstance(value,dict):return False
        props=schema.get('properties',{})
        if not set(schema.get('required',[]))<=set(value):return False
        if schema.get('additionalProperties') is False and not set(value)<=set(props):return False
        if not all(structural(v,props[k]) for k,v in value.items() if k in props):return False
    elif kind=='array':
        if not isinstance(value,list):return False
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',math.inf):return False
        if schema.get('uniqueItems') and len({canonical(x) for x in value})!=len(value):return False
        if not all(structural(v,schema.get('items',{})) for v in value):return False
    elif kind=='string':
        if not isinstance(value,str):return False
        if not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',math.inf):return False
    elif kind is not None:
        raise AssertionError('Unexpected schema type')
    return 'enum' not in schema or value in schema['enum']


def request_body(study,job,image):
    return canonical({'model':study['model'],'store':False,
      'max_output_tokens':study['max_output_tokens'],'reasoning':{'effort':study['reasoning_effort']},
      'instructions':study['instructions'],
      'input':[{'role':'user','content':[
        {'type':'input_text','text':canonical(job['instruction_payload']).decode('utf8')},
        {'type':'input_image','detail':'auto','image_url':'data:image/png;base64,'+base64.b64encode(image).decode('ascii')}]}],
      'text':{'format':{'type':'json_schema','name':'predicate_evidence','strict':True,'schema':job['schema']}}})


def numerical_verdict(query,measurement):
    """Missing surface evidence precedes any guard comparison, as frozen."""
    if not query['constraints']:return 'not_requested'
    if measurement['aabb_quarantined'] or measurement['faces_missing_mesh']:return 'unknown'
    vv=[]
    for c in query['constraints']:
        assert c['attribute']=='aabb_large_mm' and c['unit']=='mm' and c['op']=='<='
        delta=measurement['surface_large_mm']-c['value']
        vv.append('unknown' if abs(delta)<=.2 else 'pass' if delta<0 else 'fail')
    return 'fail' if 'fail' in vv else 'unknown' if 'unknown' in vv else 'pass'


def grade_completions(raw,verdicts):
    missing=[i for i,g in enumerate(raw) if g is None]
    for vals in itertools.product((0,1,2),repeat=len(missing)):
        grades=[0 if g is None else g for g in raw]
        for i,g in zip(missing,vals):grades[i]=g
        transformed=[0 if v=='fail' else min(g,1) if v=='unknown' else g for g,v in zip(grades,verdicts)]
        ideal=sorted(transformed,reverse=True)[:5]
        denominator=math.fsum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
        yield transformed,denominator


def scalar_metrics(order,completions):
    ndcg=[];strong=[]
    for grades,denominator in completions:
        dcg=math.fsum((2**grades[j]-1)/math.log2(i+2) for i,j in enumerate(order[:5]))
        ndcg.append(dcg/denominator if denominator else 0.)
        strong.append(sum(grades[j]==2 for j in order[:5])/5)
    return dict(zip(FIELDS,(min(ndcg),max(ndcg),min(strong),max(strong))))


def average(rows):
    return {key:math.fsum(r[key] for r in rows)/len(rows) for key in FIELDS}


def exact_p(values):
    observed=abs(math.fsum(values)/len(values))
    extreme=sum(abs(math.fsum(s*v for s,v in zip(signs,values))/len(values))>=observed-TOL
                for signs in itertools.product((-1,1),repeat=len(values)))
    return extreme/(2**len(values))


def holm(values):
    adjusted=[None]*len(values);previous=0.
    for j,i in enumerate(sorted(range(len(values)),key=lambda i:values[i])):
        previous=max(previous,min(1.,(len(values)-j)*values[i]));adjusted[i]=previous
    return adjusted


def main():
    # Crucially, inspect no active model output while collection is in progress.
    assert (ROOT/'evidence_completion.json').exists() and (ROOT/'results.json').exists(), 'Wait for explicit completed collection and evaluation'
    assert not (ROOT/'.api_runner.lock').exists(),'Do not audit an active runner'
    completion=read(ROOT/'evidence_completion.json');assert completion['status']=='completed'
    results=read(ROOT/'results.json');freeze=read(ROOT/'protocol_freeze.json')
    efreeze=read(ROOT/'evaluation_freeze.json');study=read(ROOT/'study.json')
    issues=[];warnings=[]
    def check(condition,message):
        if not condition:issues.append(message)
    seal=sha(ROOT/'protocol_freeze.json')
    check(seal==completion['protocol_freeze_sha256']==results['main_protocol_freeze_sha256'],'Main seal chain')
    check(sha(ROOT/'evaluation_freeze.json')==freeze['evaluation_freeze_sha256']==results['source_freeze_sha256'],'Evaluation seal chain')
    check(sha(ROOT/'study.json')==freeze['study_sha256'],'Study seal')
    check(sha(ROOT/'evidence_completion.json')==results['evidence_completion_sha256'],'Completion seal')
    check(sha(path(freeze['runner_path']))==freeze['runner_sha256'],'Runner seal')
    for item in freeze['required_files']:check(sha(path(item['path']))==item['sha256'],'Required source: '+item['path'])
    for name,h in efreeze['input_sha256'].items():check(sha(path(name))==h,'Evaluation source: '+name)
    for name,h in results['artifact_sha256'].items():check(sha(ROOT/name)==h,'Result artifact: '+name)
    for name in ('graph_support','graph_evidence','api_attempts','api_responses'):
        suffix='.json' if name=='graph_support' else '.jsonl'
        check(sha(ROOT/(name+suffix))==completion[name+'_sha256'],'Completion artifact: '+name)
    for name in ('pool_manifest.json','knowledge_rules.json','image_neutralization_audit.json'):
        check((ROOT/name).read_bytes()==(LUNA/name).read_bytes(),'Convenience copy: '+name)
    original_study=read(LUNA/'study.json')
    check(study['jobs']==original_study['jobs'] and study['instructions']==original_study['instructions'],'Matched Sol/Luna input continuity')

    attempts=lines(ROOT/'api_attempts.jsonl');responses=lines(ROOT/'api_responses.jsonl')
    graph=lines(ROOT/'graph_evidence.jsonl');support=read(ROOT/'graph_support.json')
    jobs={j['job_id']:j for j in study['jobs']};meta={j['job_id']:j for j in freeze['jobs']}
    attempted={a['job_id']:a for a in attempts};gm={g['job_id']:g for g in graph}
    check(len(attempts)==len(responses)==len(graph)==len(jobs)==2084,'2084 attempted/terminal/evidence jobs')
    check(len(attempted)==len({b['job_id'] for b in responses})==len({a['attempt_id'] for a in attempts})==2084,'Attempt uniqueness')
    ids=[b['response_id'] for b in responses if b.get('response_id')]
    check(len(ids)==len(set(ids)),'Response identity uniqueness')
    check(set(attempted)==set(gm)==set(jobs)=={b['job_id'] for b in responses},'Complete job coverage')
    model_counts=Counter();states=Counter();status=Counter();tiers=Counter();usage_keys=Counter()
    tokens=Counter();invalid=[];empty=Counter();word_overruns=[];latencies=[];cost=0.;max_input=max_output=0
    image_hashes={};events=[];cache_basis=Counter();prices=freeze['pricing_usd']
    check([prices[k] for k in ('uncached_input_per_million','cached_input_per_million','cache_write_input_per_million','output_per_million')]==[2.,.1,2.5,10.],'Prespecified standard rates')
    reserve=(freeze['input_token_reservation']*prices['reservation_input_per_million']+freeze['max_output_tokens']*prices['output_per_million'])/1e6
    check(abs(reserve-.06)<TOL,'USD.06 reservation')
    structural_count=0
    for b in responses:
        jid=b['job_id'];a=attempted[jid];j=jobs[jid];m=meta[jid];g=gm[jid]
        check(a['attempt_id']==b['attempt_id'],'Attempt binding: '+jid)
        check(a['protocol_sha256']==b['protocol_sha256']==seal,'Protocol binding: '+jid)
        check(a['request_sha256']==b['request_sha256']==m['request_sha256'],'Request binding: '+jid)
        check(a['image_sha256']==b['image_sha256']==j['image_sha256']==m['image_sha256'],'Image binding: '+jid)
        image=path(j['image_path']).read_bytes();image_hashes[j['part_id']]=digest(image)
        check(image_hashes[j['part_id']]==j['image_sha256'],'Actual image: '+jid)
        check(digest(request_body(study,j,image))==m['request_sha256'],'Actual request reconstruction: '+jid)
        raw=b.get('raw_response');value=b.get('output_json')
        is_structural=structural(value,j['schema'])
        check(bool(b['output_schema_valid'])==is_structural,'Schema flag: '+jid)
        structural_count+=int(is_structural)
        semantic=is_structural and b['status']=='completed' and b.get('response_status')=='completed'
        if is_structural:
            for pid,o in value.items():
                if not o['views']:
                    empty[o['status']]+=1
                    if o['status']!='unknown':semantic=False
                if len(o['observation'].split())>22:
                    word_overruns.append({'job_id':jid,'predicate_id':pid,'word_count':len(o['observation'].split())})
        if not semantic:invalid.append(jid)
        check(g['valid_response']==semantic,'Semantic validity/fallback: '+jid)
        expected=[]
        for pid in j['predicate_ids']:
            o=value[pid] if semantic else {'status':'unknown','views':[],'observation':'No valid model observation; fixed neutral fallback.'}
            d=next(p for p in j['instruction_payload']['atomic_predicates'] if p['predicate_id']==pid)
            expected.append(dict(predicate_id=pid,status=o['status'],views=o['views'],observation=o['observation'],required_triple=d['required_triple']))
            states[o['status']]+=1
        h=math.fsum({'yes':1.,'no':0.,'unknown':.5}[o['status']] for o in expected)/4
        check(g['observations']==expected and g['H']==h==support[j['family_id']][j['part_id']],'Graph states/bindings/H: '+jid)
        check(g['response_id']==b['response_id'] and g['model_resolved']==b['model_resolved'] and g['image_sha256']==j['image_sha256'],'Graph provenance: '+jid)
        u=raw.get('usage') if isinstance(raw,dict) else None
        check(u==b.get('usage'),'Raw usage binding: '+jid)
        charged=reserve;counts={};basis='reservation'
        if isinstance(u,dict) and all(k in u for k in ('input_tokens','output_tokens')):
            ni,no=u['input_tokens'],u['output_tokens'];details=u.get('input_tokens_details') or {};usage_keys.update(details.keys())
            cached=details.get('cached_tokens',0);written=details.get('cache_write_tokens',details.get('cache_creation_tokens',0))
            if all(type(v) is int and v>=0 for v in (ni,no,cached,written)) and cached+written<=ni:
                charged=((ni-cached-written)*prices['uncached_input_per_million']+cached*prices['cached_input_per_million']+written*prices['cache_write_input_per_million']+no*prices['output_per_million'])/1e6
                counts={'input_tokens':ni,'cached_input_tokens':cached,'cache_write_input_tokens':written,'output_tokens':no};basis='reported_usage'
                max_input=max(max_input,ni);max_output=max(max_output,no)
        check(abs(charged-b['cost_usd'])<TOL,'Cost: '+jid)
        check(counts==b.get('usage_tokens',{}),'Saved usage counters: '+jid)
        cost+=charged;tokens.update(counts);cache_basis[basis]+=1
        if isinstance(raw,dict):
            check(raw.get('id')==b.get('response_id') and raw.get('model')==b.get('model_resolved') and raw.get('status')==b.get('response_status'),'Raw response identity: '+jid)
            tier=raw.get('service_tier');tiers[str(tier) if tier is not None else 'not_returned']+=1
            if tier not in (None,'default','standard'):warnings.append('Returned service_tier '+str(tier)+' needs price interpretation: '+jid)
        model_counts[str(b.get('model_resolved'))]+=1;status[b['status']]+=1
        latencies.append(b.get('elapsed_seconds',0));events.extend([(b['started_utc'],1),(b['finished_utc'],-1)])
    concurrency=peak=0
    for _,step in sorted(events):concurrency+=step;peak=max(peak,concurrency)
    check(peak<=freeze['concurrency'],'Concurrency cap')
    check(abs(cost-completion['estimated_cost_usd'])<1e-9,'Total cost receipt')
    check(dict(tokens)==completion['usage_tokens'] and dict(states)==completion['predicate_status_counts'],'Usage/state completion counters')
    check(cost<=freeze['max_usd']+TOL,'USD50 ceiling')
    check(max_input<=freeze['input_token_reservation'] and max_output<=freeze['max_output_tokens'],'Token reservation/cap')

    # Full-gallery direct numerical oracle, including exhaustive raw-null values.
    gallery=read(V6/'cohorts.json')['closed'];index={p:i for i,p in enumerate(gallery)}
    check(len(gallery)==1816 and gallery==sorted(gallery),'Complete fixed gallery')
    queries={q['query_id']:q for q in read(REPO/'annotations/intent_queries440_v6/queries.json')}
    qids=efreeze['primary_query_ids'];families=sorted({queries[q]['scenario_id'] for q in qids})
    check(len(qids)==264 and len(families)==11,'Primary panel')
    rel=read(REPO/'annotations/intent_feature_review_v5/judgments.json')
    check(rel['source']=='ai_reference' and rel['human_expert_reviewed'] is False,'Reference provenance')
    judgments=rel['judgments'];gold={(r['family_id'],r['part_id']):r for r in judgments}
    check(len(gold)==len(judgments)==1816*11,'Complete 19976 qrels')
    raw_nulls=[{'family_id':r['family_id'],'part_id':r['part_id'],'inspection_id':r['inspection_id']} for r in judgments if r['grade'] is None]
    check(len(raw_nulls)==2==results['null_raw_qrel_pairs'],'Unchanged raw nulls')
    measurements={}
    for pid in gallery:
        row=gold[families[0],pid];p=REPO/'outputs/feature_review_v5/evidence'/(row['inspection_id']+'.json');m=read(p)
        check(sha(p)==results['numeric_evidence_sha256'][p.relative_to(REPO).as_posix()],'Numerical source: '+pid)
        check(m['part_id']==pid and m['source_sha256']==row['source_sha256'],'Numerical source identity: '+pid)
        measurements[pid]={k:m[k] for k in ('surface_large_mm','aabb_quarantined','faces_missing_mesh')}
    completions={};verdict_counts=Counter()
    for qid in qids:
        q=queries[qid];raw=[gold[q['scenario_id'],p]['grade'] for p in gallery]
        vv=[numerical_verdict(q,measurements[p]) for p in gallery];verdict_counts.update(vv)
        completions[qid]=list(grade_completions(raw,vv))
    reported=csv_rows(ROOT/'per_query.csv');check(len(reported)==2112,'2112 metric rows')
    check(len({(r['geometry_kind'],r['query_id'],r['method']) for r in reported})==2112,'Metric row uniqueness')
    rows=[];max_metric_error=0.;max_score_error=0.;control_arrays=0;sol_arrays=0
    pools=read(ROOT/'pool_manifest.json');pool_jobs=set()
    with np.load(ROOT/'full_rankings.npz',allow_pickle=False) as ar,np.load(LUNA/'full_rankings.npz',allow_pickle=False) as la,np.load(V6/'complete_rankings.npz',allow_pickle=False) as va:
        for kind in KINDS:
            for qid in qids:
                expected=sorted({int(i) for method in ('kg_full','geometry_intent_filter','kg_no_class') for i in va[f'{kind}__{qid}__closed__{method}__order'][:10]})
                check(pools[kind][qid]==[gallery[i] for i in expected],'Frozen union pool: '+kind+'/'+qid)
                pool_jobs.update((queries[qid]['scenario_id'],p) for p in pools[kind][qid])
        for r in reported:
            kind,qid,method=r['geometry_kind'],r['query_id'],r['method'];key=f'{kind}__{qid}__closed__{method}'
            check(r['family_id']==queries[qid]['scenario_id'],'Metric/query role identity: '+key)
            order=ar[key+'__order'].tolist();scores=ar[key+'__score'].tolist()
            check(len(order)==len(scores)==len(set(order)) and all(0<=j<1816 for j in order),'Ranking indices: '+key)
            rec={k:r[k] for k in ('geometry_kind','query_id','family_id','method')};rec.update(scalar_metrics(order,completions[qid]));rows.append(rec)
            for field in FIELDS:max_metric_error=max(max_metric_error,abs(rec[field]-float(r[field])))
            if method in METHODS[:3]:
                check(order==la[key+'__order'].tolist() and scores==la[key+'__score'].tolist(),'Frozen baseline arrays: '+key);control_arrays+=1
            else:
                check(method=='kg_full_sol','Unexpected method')
                bk=f'{kind}__{qid}__closed__kg_full';base=la[bk+'__order'].tolist();sc=dict(zip(base,la[bk+'__score'].tolist()))
                pp=[index[p] for p in pools[kind][qid]];fid=queries[qid]['scenario_id']
                val={i:(3*sc[i]+support[fid][gallery[i]])/4 for i in pp}
                expected=sorted(pp,key=lambda i:(-val[i],i))+[i for i in base if i not in set(pp)]
                check(order==expected,'Sol fixed H order: '+key)
                max_score_error=max(max_score_error,max(abs(s-(val[i] if i in val else sc[i])) for s,i in zip(scores,expected)))
                sol_arrays+=1
    check(pool_jobs=={(j['family_id'],j['part_id']) for j in jobs.values()},'Pool/job observation coverage')
    check(max_metric_error<=TOL and max_score_error<=TOL,'Independent scalar metrics or scores')
    grouped=defaultdict(list)
    for r in rows:grouped[r['geometry_kind'],r['method'],r['family_id']].append(r)
    fm={}
    for key,rr in grouped.items():check(len(rr)==24,'24 conditions per role');fm[key]=average(rr)
    family_csv=csv_rows(ROOT/'family_metrics.csv')
    check(len(family_csv)==len(fm)==88,'88 family means')
    family_error=max(abs(fm[r['geometry_kind'],r['method'],r['family_id']][f]-float(r[f])) for r in family_csv for f in FIELDS)
    summary=[]
    for kind in KINDS:
        for method in METHODS:summary.append(dict(geometry_kind=kind,method=method,**average([fm[kind,method,fid] for fid in families])))
    summary_csv=csv_rows(ROOT/'summary.csv');check(len(summary_csv)==8,'Eight aggregate means')
    summary_error=max(abs(next(s for s in summary if (s['geometry_kind'],s['method'])==(r['geometry_kind'],r['method']))[f]-float(r[f])) for r in summary_csv for f in FIELDS)
    check(max(family_error,summary_error)<=TOL,'Independent family/aggregate means')
    for saved in results['summary']:
        oracle=next(s for s in summary if (s['geometry_kind'],s['method'])==(saved['geometry_kind'],saved['method']))
        check(all(abs(oracle[f]-saved[f])<=TOL for f in FIELDS),'Results JSON aggregate')

    identified=all(r['ndcg@5_lower']==r['ndcg@5_upper'] for r in rows)
    check(identified==results['all_primary_ndcg_identified'],'Point identification')
    rng=np.random.default_rng(20261022);tests=[]
    for kind in KINDS:
        for comparator in COMPARATORS:
            low=[fm[kind,'kg_full_sol',fid]['ndcg@5_lower']-fm[kind,comparator,fid]['ndcg@5_upper'] for fid in families]
            high=[fm[kind,'kg_full_sol',fid]['ndcg@5_upper']-fm[kind,comparator,fid]['ndcg@5_lower'] for fid in families]
            strong_low=[fm[kind,'kg_full_sol',fid]['strong_precision@5_lower']-fm[kind,comparator,fid]['strong_precision@5_upper'] for fid in families]
            strong_high=[fm[kind,'kg_full_sol',fid]['strong_precision@5_upper']-fm[kind,comparator,fid]['strong_precision@5_lower'] for fid in families]
            t=dict(geometry_kind=kind,comparator=comparator,delta_lower=math.fsum(low)/11,delta_upper=math.fsum(high)/11,strong_precision_delta_lower=math.fsum(strong_low)/11,strong_precision_delta_upper=math.fsum(strong_high)/11,
              delta=None,ci95_lower=None,ci95_upper=None,exact_sign_flip_p=None,holm_p=None,family_wins=None,family_ties=None,family_losses=None)
            if identified:
                v=np.asarray(low);sample=rng.integers(0,11,size=(20000,11));ci=np.percentile(v[sample].mean(axis=1),[2.5,97.5])
                t.update(delta=math.fsum(low)/11,ci95_lower=float(ci[0]),ci95_upper=float(ci[1]),exact_sign_flip_p=exact_p(low),family_wins=sum(x>TOL for x in low),family_ties=sum(abs(x)<=TOL for x in low),family_losses=sum(x< -TOL for x in low))
            tests.append(t)
    if identified:
        for t,h in zip(tests,holm([t['exact_sign_flip_p'] for t in tests])):t['holm_p']=h
    saved_tests=csv_rows(ROOT/'paired_tests.csv');check(len(saved_tests)==4,'Exactly four primary tests')
    for t,s in zip(tests,saved_tests):
        check((t['geometry_kind'],t['comparator'])==(s['geometry_kind'],s['comparator']),'Test comparator/order')
        check(s['point_inference_eligible']==str(identified),'Point-inference flag')
        for key,value in t.items():
            if key in ('geometry_kind','comparator'):continue
            if value is None:check(s[key]=='','Withheld inference: '+key)
            else:check(abs(float(s[key])-value)<=TOL,'Independent test: '+str((t['geometry_kind'],t['comparator'],key)))
    for t,s in zip(tests,results['paired_tests']):
        for k,value in t.items():
            if isinstance(value,str) or value is None:check(s[k]==value,'Results JSON test: '+k)
            else:check(abs(s[k]-value)<=TOL,'Results JSON test: '+k)
    differences=csv_rows(ROOT/'family_differences.csv');check(len(differences)==44,'44 role differences')
    for d in differences:
        a=fm[d['geometry_kind'],'kg_full_sol',d['family_id']];b=fm[d['geometry_kind'],d['comparator'],d['family_id']]
        for metric in ('ndcg@5','strong_precision@5'):
            check(abs(float(d[metric+'_lower'])-(a[metric+'_lower']-b[metric+'_upper']))<=TOL and abs(float(d[metric+'_upper'])-(a[metric+'_upper']-b[metric+'_lower']))<=TOL,'Role effect: '+d['family_id'])

    # Same-schema model agreement is descriptive, never feature accuracy.
    luna_graph={(g['family_id'],g['part_id']):g for g in lines(LUNA/'graph_evidence.jsonl')}
    agreement=Counter();by_role=defaultdict(Counter);h_equal=0;h_diff=[]
    for g in graph:
        lg=luna_graph[g['family_id'],g['part_id']];lo={o['predicate_id']:o for o in lg['observations']}
        for o in g['observations']:
            a,b=lo[o['predicate_id']]['status'],o['status'];agreement[a+' -> '+b]+=1;by_role[g['family_id']]['clauses']+=1;by_role[g['family_id']]['equal']+=int(a==b)
        h_equal+=int(g['H']==lg['H']);h_diff.append(g['H']-lg['H'])
    total_states=sum(agreement.values());equal_states=sum(v for k,v in agreement.items() if len(set(k.split(' -> ')))==1)
    replay_path=ROOT/'independent_replay_audit.json';replay=read(replay_path) if replay_path.exists() else None
    if replay:check(replay['status']=='PASS' and all(x['exact'] for x in replay['comparisons'].values()),'Isolated replay')
    fallacies=[
      {'type':'Simpson\u2019s paradox','assessment':'Role means and both representations are retained. Equal24-query group sizes avoid weighting-induced reversal; mixed role effects do not establish a universal model advantage.','numerical_context':{t['geometry_kind']+'/'+t['comparator']:{'wins':t['family_wins'],'ties':t['family_ties'],'losses':t['family_losses']} for t in tests}},
      {'type':'Ecological fallacy','assessment':'Inference uses11authored role means. Do not infer per-component engineering correctness or effects on unseen individuals from these aggregates.'},
      {'type':'Berkson\u2019s paradox','assessment':'Gallery availability/deduplication, exemplar selection and top10union selection condition this benchmark. No unrestricted-population correlation or industrial prevalence is identified.'},
      {'type':'Collider bias','assessment':'Pool selection depends on initial ranking branches, not Sol outcomes. Matched pool comparison avoids outcome-based expansion, but results remain conditional on this selected pool.'},
      {'type':'Base rate neglect','assessment':'StrongP@5 is shortlist concentration under fixed AI-reference grades, not sensitivity, deployment positive predictive value or manufacturing-role prevalence. Full qrel counts and gallery scope remain explicit.'},
      {'type':'Regression to the mean','assessment':'Sol is a sequential candidate after known Luna outcomes; no independent dataset replication exists. Both models are tested on identical fixed inputs with unchanged controls and no selection of extreme queries.'},
      {'type':'Survivorship bias','assessment':'All2084attempted jobs, failed/invalid outputs and declared neutral fallbacks are retained. No successful-output-only analysis or resampling is allowed.','fallback_jobs':len(invalid)},
      {'type':'Look-elsewhere effect','assessment':'All four Sol−Luna/Sol−GFC tests share one Holm adjustment and are reported. Older Luna/component families are separate; no research-history-wide error guarantee is claimed.'},
      {'type':'Garden of forking paths','assessment':'Sol protocol/analysis was frozen before Sol inference but after Luna outcomes. No prompt, weight, pool or outcome-based model-selection sweep is performed; this remains exploratory.'},
      {'type':'Correlation versus causation','assessment':'This matched computational substitution measures score/ranking changes under the fixed visual module. It does not isolate graph storage or relation-binding causality, nor validate physical compatibility or external engineering truth.'},
      {'type':'Reverse causality','assessment':'Immutable qrels precede Sol calls and are excluded from requests. Sol outcomes cannot change them; however shared AI-family/visual assumptions may correlate errors without establishing independent truth.'},
    ]
    report={
      'material_passport':{'origin_skill':'ARS experiment-agent','origin_mode':'validate','origin_date':dt.datetime.now(dt.timezone.utc).date().isoformat(),'verification_status':'VERIFIED' if replay and not issues else 'ANALYZED','version_label':'sol_post_execution_audit_v1'},
      'audit_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'PASS' if not issues else 'FAIL','issues':issues,'warnings':warnings,
      'audit_identity':'Same-team independent numerical implementation, not independent human review','api_calls_by_auditor':0,
      'protocol_sha256':seal,'script_sha256':sha(Path(__file__)),'source_seals_verified':True,
      'api':{'attempts':len(attempts),'responses':len(responses),'unique_attempt_ids':len({a['attempt_id'] for a in attempts}),'unique_response_ids':len(set(ids)),
        'model_counts':dict(model_counts),'status_counts':dict(status),'structurally_valid_responses':structural_count,'semantic_fallback_jobs':invalid,'empty_view_counts':dict(empty),
        'predicate_status_counts':dict(states),'word_limit_overruns':word_overruns,'cost_usd_recomputed':cost,'usage_tokens':dict(tokens),'cost_basis_counts':dict(cache_basis),
        'service_tier_counts':dict(tiers),'input_token_detail_keys':dict(usage_keys),'price_interpretation':'Frozen standard price schedule applied to reported usage; not a billing invoice. Returned service tiers are disclosed separately.',
        'max_input_tokens':max_input,'max_output_tokens':max_output,'peak_journaled_concurrency':peak,'unique_image_hashes_verified':len(image_hashes),
        'mean_call_latency_seconds':math.fsum(latencies)/len(latencies),'wall_seconds':(dt.datetime.fromisoformat(max(b['finished_utc'] for b in responses))-dt.datetime.fromisoformat(min(a['started_utc'] for a in attempts))).total_seconds()},
      'independent_numerics':{'metric_rows':len(rows),'endpoint_values':len(rows)*4,'oracle':'Direct scalar full-gallery DCG/IDCG, exhaustive raw-null completions and reimplemented dimensional guard/caps. No project metric/numeric/statistics helpers imported.',
        'max_metric_difference':max_metric_error,'baseline_arrays_verified':control_arrays,'sol_arrays_verified':sol_arrays,'max_score_difference':max_score_error,'family_rows':len(fm),'summary_rows':len(summary),
        'max_family_difference':family_error,'max_summary_difference':summary_error,'all_primary_ndcg_identified':identified,'raw_null_qrels':raw_nulls,'raw_nulls_retained':True,
        'numeric_verdict_counts':dict(verdict_counts),'sign_patterns_each':2048,'bootstrap_replicates':20000,'bootstrap_seed':20261022},
      'summary':summary,'paired_tests':tests,
      'descriptive_sol_luna_agreement':{'clauses':total_states,'equal_states':equal_states,'state_agreement_fraction':equal_states/total_states,'state_confusion_luna_to_sol':dict(agreement),'equal_H_jobs':h_equal,'jobs':len(graph),'mean_sol_minus_luna_H':math.fsum(h_diff)/len(h_diff),'by_role':{k:dict(v) for k,v in by_role.items()},'interpretation':'Same-input model agreement, not accuracy against independent geometric truth. Whole-job neutral fallbacks, if any, are retained.'},
      'ars_fallacy_scan':{'coverage':'11/11 checked','checks':fallacies},'isolated_replay':replay,
      'sensitive_access':'No API request, credentials, other project, source edit or outcome edit. Only the audit artifacts are written.'}
    (ROOT/'post_execution_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    table='\n'.join('| '+s['geometry_kind']+' | '+s['method']+' | '+f"{s['ndcg@5_lower']:.7f}"+' | '+f"{s['strong_precision@5_lower']:.7f}"+' |' for s in summary)
    test_text='\n'.join('- '+t['geometry_kind']+' Sol−'+t['comparator']+': '+('point inference withheld' if t['delta'] is None else f"Δ={t['delta']:+.7f}; raw p={t['exact_sign_flip_p']:.8f}; Holm p={t['holm_p']:.8f}; W/T/L={t['family_wins']}/{t['family_ties']}/{t['family_losses']}.") for t in tests)
    fallacy_text='\n'.join(str(i+1)+'. **'+f['type']+'**: '+f['assessment'] for i,f in enumerate(fallacies))
    markdown=f"""# Independent completed Sol matched-study audit

## Material Passport

- Origin Skill: ARS experiment-agent
- Origin Mode: validate
- Origin Date: {report['material_passport']['origin_date']}
- Verification Status: {report['material_passport']['verification_status']}
- Version Label: sol_post_execution_audit_v1

**Status: {report['status']}.** This is a same-team computational audit using a separate scalar oracle, not independent human peer review. No API call or credential access occurred.

## Recorded execution and provenance

The audit checked {len(attempts)} durable attempts and {len(responses)} terminal responses, unique identities, request reconstruction, schema flags, view-citation semantics, graph triples/H, source seals and result-artifact hashes. Models: {dict(model_counts)}. Schema-valid responses: {structural_count}. Whole-record semantic fallbacks: {len(invalid)}. Empty view counts: {dict(empty)}. The number of descriptions exceeding the requested 22 whitespace-separated words is {len(word_overruns)}; these prose-length deviations do not alter scored states.

Reported-token usage independently gives **USD {cost:.8f}** under the frozen standard rates. Service tiers returned: {dict(tiers)}. This estimate is not a billing invoice; unreturned/alternative service tiers require the disclosed pricing interpretation. All cache-write counts and source usage values are retained in the JSON. Peak journaled concurrency is {peak}; maximal input/output usage is {max_input}/{max_output}tokens.

## Independent numerical reconstruction

All {len(rows)} metric rows ({len(rows)*4} endpoint values), {len(fm)} role means and {len(summary)} aggregate means were independently reconstructed from complete1816-grade vectors. Missing surface faces/quarantine take precedence over the0.2mm guard. Raw nulls retain all0/1/2completions; ideal DCG uses the complete transformed gallery. The maximum metric discrepancy is {max_metric_error:.3g}. Every frozen baseline array ({control_arrays}) and proposed Sol array ({sol_arrays}) was checked; maximum score discrepancy is {max_score_error:.3g}. Point-identification status is {identified}.

| Representation | Method | nDCG@5 lower | Strong P@5 lower |
|---|---|---:|---:|
{table}

Four independently enumerated2^11sign-flip tests share one Holm family. Seeded20,000role-bootstrap intervals also match:

{test_text}

The two immutable null judgments remain unchanged. Isolated saved-response replay status: {replay['status'] if replay else 'not yet supplied; numerical audit only'}. No automatic winner-selection or manuscript gate is inferred.

## Descriptive model agreement

Sol and Luna agree on {equal_states}/{total_states} clause states ({equal_states/total_states:.2%}) and on H for {h_equal}/{len(graph)}jobs. These are agreement rates on identical views and clauses, not feature accuracy or independent engineering validation. Confusion counts and role breakdowns are supplied in the JSON.

## ARS statistical fallacy scan —11/11checked

{fallacy_text}

## Discrepancies

Material issues: {issues if issues else 'none found'}.
Pricing/other warnings: {warnings if warnings else 'none found'}.
"""
    (ROOT/'post_execution_audit.md').write_text(markdown,encoding='utf8')
    print(json.dumps({'status':report['status'],'issues':issues,'warnings':len(warnings),'metric_rows':len(rows),'cost_usd':cost,'paired_tests':tests},indent=2))
    assert not issues,'Independent audit found discrepancies; see written report'


if __name__=='__main__':main()
