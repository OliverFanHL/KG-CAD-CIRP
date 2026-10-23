"""Independent scalar and provenance audit for completed v7 repeated acquisition.

Read no active observations: main refuses execution until completion/results exist
and runner lock is absent. No API, credentials, project metric/numeric/statistical
helpers or manuscript edits. Writes only v7 post_execution_audit.md/json.
"""
from __future__ import annotations
import base64
from collections import Counter,defaultdict
import csv,datetime as dt,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/llm_robustness_v7'
LUNA=REPO/'outputs/llm_geometry_matched_v1'
SOL=REPO/'outputs/llm_geometry_sol_matched_v1'
V6=REPO/'outputs/query_diversity_v6/evaluation'
CONTROLS=('kg_no_class','geometry_class_filter','geometry_intent_filter','kg_no_constraints','kg_full')
METHODS=CONTROLS+('kg_luna','kg_sol','role_only_hybrid','role_only_score','kg_luna_score','kg_sol_score')
KINDS=('embedding','descriptor')
CONTRASTS=(('kg_luna','role_only_hybrid'),('kg_sol','kg_luna'))
FIELDS=('ndcg@5_lower','ndcg@5_upper','strong_precision@5_lower','strong_precision@5_upper')
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


def structural(v,s):
    allowed={'type','properties','required','additionalProperties','items','enum'}
    assert not set(s)-allowed,'Unexpected schema assertion'
    t=s.get('type')
    if t=='object':
        if not isinstance(v,dict) or not set(s.get('required',[]))<=set(v):return False
        props=s.get('properties',{})
        if s.get('additionalProperties') is False and not set(v)<=set(props):return False
        if not all(structural(x,props[k]) for k,x in v.items() if k in props):return False
    elif t=='array':
        if not isinstance(v,list) or not all(structural(x,s.get('items',{})) for x in v):return False
    elif t=='string':
        if not isinstance(v,str):return False
    elif t=='number':
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):return False
    elif t is not None:raise AssertionError('Unexpected schema type')
    return 'enum' not in s or v in s['enum']


def body(st,j,image_b64):
    return canonical({'model':j['model'],'store':False,'max_output_tokens':st['max_output_tokens'],
      'reasoning':{'effort':st['reasoning_effort']},'instructions':j.get('instructions',st['instructions']),
      'input':[{'role':'user','content':[{'type':'input_text','text':canonical(j['instruction_payload']).decode('utf8')},
        {'type':'input_image','detail':'auto','image_url':'data:image/png;base64,'+image_b64}]}],
      'text':{'format':{'type':'json_schema','name':'predicate_evidence','strict':True,'schema':j['schema']}}})


def newcost(row,prices,reserve):
    raw=row.get('raw_response');u=raw.get('usage') if isinstance(raw,dict) else None
    if not isinstance(u,dict):return reserve,{},'reservation'
    ni,no=u.get('input_tokens'),u.get('output_tokens');detail=u.get('input_tokens_details') or {}
    ca=detail.get('cached_tokens',0);wr=detail.get('cache_write_tokens',detail.get('cache_creation_tokens',0))
    if not all(type(x) is int and x>=0 for x in (ni,no,ca,wr)) or ca+wr>ni:return reserve,{},'reservation'
    value=((ni-ca-wr)*prices['uncached_input_per_million']+ca*prices['cached_input_per_million']+wr*prices['cache_write_input_per_million']+no*prices['output_per_million'])/1e6
    return value,dict(input_tokens=ni,cached_input_tokens=ca,cache_write_input_tokens=wr,output_tokens=no),'reported_usage'


def observe(job,row):
    value=row.get('output_json');schema_ok=structural(value,job['schema'])
    valid=schema_ok and row['status']=='completed' and row.get('response_status')=='completed'
    role=job['arm']=='role_only_luna'
    if valid:valid=all((o['score']==.5 if role else o['status']=='unknown') or bool(o['views']) for o in value.values())
    observations=[]
    for pid in job['predicate_ids']:
        fallback={'score':.5,'views':[],'observation':'No valid model observation; fixed neutral fallback.'} if role else {'status':'unknown','views':[],'observation':'No valid model observation; fixed neutral fallback.'}
        o=value[pid] if valid else fallback
        obs=dict(predicate_id=pid,**o)
        defs=job['instruction_payload'].get('atomic_predicates',[])
        if defs:obs['required_triple']=next(x['required_triple'] for x in defs if x['predicate_id']==pid)
        observations.append(obs)
    h=observations[0]['score'] if role else math.fsum({'yes':1.,'no':0.,'unknown':.5}[x['status']] for x in observations)/4
    return schema_ok,valid,observations,h


def main():
    assert (ROOT/'evidence_completion.json').exists() and (ROOT/'results.json').exists(),'Wait for explicit completion'
    assert not (ROOT/'.api_runner.lock').exists(),'Refuse active run'
    complete=read(ROOT/'evidence_completion.json');assert complete['status']=='completed'
    result=read(ROOT/'results.json');freeze=read(ROOT/'protocol_freeze.json');evaluation=read(ROOT/'evaluation_freeze.json');study=read(ROOT/'study.json')
    issues=[];warnings=[]
    def ck(flag,message):
        if not flag:issues.append(message)
    seal=sha(ROOT/'protocol_freeze.json')
    ck(seal=='5de7a5b9a86efe7310b74a43bbed20cf331f76681d2490477ac8e0ac3f8fbfc4','Protocol identity')
    ck(sha(ROOT/'study.json')==freeze['study_sha256'],'Study seal')
    ck(sha(ROOT/'evaluation_freeze.json')==freeze['evaluation_freeze_sha256']==result['evaluation_freeze_sha256'],'Evaluator seal')
    ck(seal==complete['protocol_freeze_sha256']==result['protocol_sha256'],'Completion/main seal')
    ck(sha(ROOT/'evidence_completion.json')==result['evidence_completion_sha256'],'Result completion seal')
    ck(sha(path(freeze['runner_path']))==freeze['runner_sha256'],'Runner seal')
    for item in freeze['required_files']:ck(sha(path(item['path']))==item['sha256'],'Required source '+item['path'])
    for name,h in evaluation['input_sha256'].items():ck(sha(path(name))==h,'Evaluation source '+name)
    for name,h in result['artifact_sha256'].items():ck(sha(ROOT/name)==h,'Result artifact '+name)
    for name in ('graph_support','graph_evidence','api_attempts','api_responses'):
        ck(sha(ROOT/(name+('.json' if name=='graph_support' else '.jsonl')))==complete[name+'_sha256'],'Evidence artifact '+name)
    jobs={j['job_id']:j for j in study['jobs']};meta={j['job_id']:j for j in freeze['jobs']}
    attempts=lines(ROOT/'api_attempts.jsonl');responses=lines(ROOT/'api_responses.jsonl');graph=lines(ROOT/'graph_evidence.jsonl');support=read(ROOT/'graph_support.json')
    am={r['job_id']:r for r in attempts};gm={r['job_id']:r for r in graph}
    ck(len(jobs)==len(meta)==len(am)==len(attempts)==len(responses)==len({r['job_id'] for r in responses})==14588,'14588 unique new jobs')
    ck(len(graph)==len(gm)==18756,'18756 total evidence jobs')
    ck(len({r['attempt_id'] for r in attempts})==14588,'Unique new attempt IDs')
    response_ids=[r['response_id'] for r in responses if r.get('response_id')]
    ck(len(response_ids)==len(set(response_ids)),'Unique returned new response IDs')
    ck(set(jobs)==set(am)=={r['job_id'] for r in responses},'New coverage')
    cost=defaultdict(float);cost_models=defaultdict(float);tokens=Counter();status=Counter();models=Counter();tiers=Counter();validity=Counter();states=Counter();word_overruns=[];empty=Counter();events=[];latencies=[]
    totals_by_collection=Counter();schema_by_arm=Counter();fallbacks=[];hverify=0;image_b64={};requesthash={};usage_basis=Counter()
    # Actual new observation rows are reviewed only after the completed-run guard.
    def check_graph(job,row,historical):
        nonlocal hverify
        jid=job['job_id'];g=gm[jid];ok,valid,obs,h=observe(job,row)
        if not historical:ck(ok==bool(row['output_schema_valid']),'Schema validity '+jid);schema_by_arm[job['arm']]+=int(ok)
        ck(g['valid_response']==valid and g['observations']==obs and g['H']==h,'Graph H/observations '+jid)
        ck(g['historical']==historical and g['arm']==job['arm'] and g['replicate']==job['replicate'] and g['source_job_id']==job['source_job_id'],'Graph acquisition identity '+jid)
        ck(g['response_id']==row['response_id'] and g['model_resolved']==row['model_resolved'] and g['image_sha256']==job['image_sha256'],'Graph response provenance '+jid)
        ck(support[job['arm']][str(job['replicate'])][job['family_id']][job['part_id']]==h,'Saved H/J '+jid)
        ck(g['local_order_seed']==job['local_order_seed'],'Local-order seed '+jid)
        validity[job['arm']+('/valid' if valid else '/fallback')]+=1;totals_by_collection[job['arm']+'/r'+str(job['replicate'])]+=1
        if not valid:fallbacks.append(jid)
        for o in obs:
            state=str(o['score']) if job['arm']=='role_only_luna' else o['status'];states[job['arm']+'/'+state]+=1
        if ok:
            for pid,o in row['output_json'].items():
                if not o['views']:empty[job['arm']+'/'+pid]+=1
                if len(o['observation'].split())>22:word_overruns.append(dict(job_id=jid,predicate_id=pid,words=len(o['observation'].split()),historical=historical))
        hverify+=1
    for row in responses:
        jid=row['job_id'];job=jobs[jid];a=am[jid];m=meta[jid]
        ck(row['attempt_id']==a['attempt_id'] and row['protocol_sha256']==a['protocol_sha256']==seal,'New attempt binding '+jid)
        ck(row['request_sha256']==a['request_sha256']==m['request_sha256'],'New request seal '+jid)
        ck(row['model_requested']==a['model_requested']==m['model']==job['model'],'Requested model '+jid)
        ck(row['image_sha256']==a['image_sha256']==job['image_sha256']==m['image_sha256'],'Image binding '+jid)
        if job['part_id'] not in image_b64:
            data=path(job['image_path']).read_bytes();ck(digest(data)==job['image_sha256'],'Actual image '+job['part_id']);image_b64[job['part_id']]=base64.b64encode(data).decode('ascii')
        cache=(job['arm'],job['source_job_id'])
        if cache not in requesthash:requesthash[cache]=digest(body(study,job,image_b64[job['part_id']]))
        ck(requesthash[cache]==m['request_sha256'],'Independently reconstructed request '+jid)
        ck(digest(canonical(job['instruction_payload']))==m['instruction_payload_sha256'] and digest(canonical(job['schema']))==m['schema_sha256'],'Payload/schema seal '+jid)
        check_graph(job,row,False)
        charged,counts,basis=newcost(row,freeze['model_pricing_usd'][job['model']],.06)
        ck(abs(charged-row['cost_usd'])<=TOL and counts==row['usage_tokens'],'Usage/cost '+jid)
        raw=row.get('raw_response')
        if isinstance(raw,dict):
            ck(raw.get('id')==row.get('response_id') and raw.get('model')==row.get('model_resolved') and raw.get('status')==row.get('response_status') and raw.get('usage')==row.get('usage'),'Raw identity/usage '+jid)
            tier=raw.get('service_tier');tiers[job['model']+'/'+str(tier)]+=1
            if tier not in (None,'default','standard'):warnings.append('Nonstandard returned tier '+str(tier)+' '+jid)
        cost[job['arm']]+=charged;cost_models[job['model']]+=charged;tokens.update(counts);usage_basis[basis]+=1;status[row['status']]+=1;models[str(row.get('model_resolved'))]+=1
        events.extend([(row['started_utc'],1),(row['finished_utc'],-1)]);latencies.append(row['elapsed_seconds'])
    historical={};old_jobs=read(LUNA/'study.json')['jobs']
    for arm,directory in [('kg_luna',LUNA),('kg_sol',SOL)]:
        oldrows=lines(directory/'api_responses.jsonl');oldmap={r['job_id']:r for r in oldrows}
        oldfreeze=read(directory/'protocol_freeze.json');oldcomplete=read(directory/'evidence_completion.json')
        ck(sha(directory/'api_responses.jsonl')==oldcomplete['api_responses_sha256'],'Historical response seal '+arm)
        ck(sha(directory/'study.json')==oldfreeze['study_sha256'],'Historical study seal '+arm)
        for source in old_jobs:
            job=dict(source,job_id=arm+'__r1__'+source['job_id'],source_job_id=source['job_id'],arm=arm,replicate=1,local_order_seed=None)
            check_graph(job,oldmap[source['job_id']],True)
        historical[arm]=dict(jobs=len(oldrows),protocol_sha256=sha(directory/'protocol_freeze.json'),api_responses_sha256=sha(directory/'api_responses.jsonl'))
    ck(all(n==2084 for n in totals_by_collection.values()) and len(totals_by_collection)==9,'Nine complete2084-job acquisitions')
    ck(hverify==18756 and dict(validity)==complete['validity_counts'] and dict(states)==complete['predicate_states_by_arm'],'Completion H/J/state counters')
    ck(dict(status)==complete['status_counts'] and complete['new_calls']==14588 and complete['historical_calls_reused']==4168 and complete['total_observation_jobs']==18756,'Completion acquisition counters')
    ck(abs(math.fsum(cost.values())-complete['estimated_new_cost_usd'])<=1e-8,'Completion cost total')
    ck(dict(tokens)==complete['usage_tokens'],'Completion tokens')
    for arm,v in cost.items():ck(abs(v-complete['new_cost_by_arm_usd'][arm])<=1e-8,'Per-arm cost '+arm)
    ck(math.fsum(cost.values())<=75+TOL,'New USD75 ceiling')
    active=peak=0
    for _,step in sorted(events):active+=step;peak=max(peak,active)
    ck(peak<=12,'Concurrency ceiling')
    # Release heavy raw response objects before scalar ranking reconstruction.
    del responses,graph,gm,image_b64
    gallery=read(V6/'cohorts.json')['closed'];index={p:i for i,p in enumerate(gallery)};queries={q['query_id']:q for q in read(REPO/'annotations/intent_queries440_v6/queries.json')};qids=evaluation['primary_query_ids'];families=sorted({queries[q]['scenario_id'] for q in qids})
    annotation=read(REPO/'annotations/intent_feature_review_v5/judgments.json');rels=annotation['judgments'];gold={(r['family_id'],r['part_id']):r for r in rels}
    ck(annotation['source']=='ai_reference' and annotation['human_expert_reviewed'] is False,'AI reference provenance')
    ck(len(rels)==len(gold)==19976 and len(gallery)==1816 and len(qids)==264 and len(families)==11,'Corpus counts')
    nulls=[dict(family_id=r['family_id'],part_id=r['part_id'],inspection_id=r['inspection_id']) for r in rels if r['grade'] is None]
    ck(len(nulls)==2==result['null_raw_pairs'],'Two immutable nulls')
    consumed=read(REPO/'outputs/feature_review_v5/consumed_evidence_freeze.json')['input_sha256'];consumed={str(path(p).resolve()).lower():h for p,h in consumed.items()}
    measures={}
    for pid in gallery:
        rr=gold[families[0],pid];p=REPO/'outputs/feature_review_v5/evidence'/(rr['inspection_id']+'.json');ck(sha(p)==consumed[str(p.resolve()).lower()],'Numeric source '+pid);m=read(p);ck(m['part_id']==pid and m['source_sha256']==rr['source_sha256'],'Numeric binding '+pid)
        measures[pid]={k:m[k] for k in ('surface_large_mm','aabb_quarantined','faces_missing_mesh')}
    completions={}
    for qid in qids:
        q=queries[qid];raw=[gold[q['scenario_id'],p]['grade'] for p in gallery];vv=[numerical_verdict(q,measures[p]) for p in gallery];completions[qid]=list(grade_completions(raw,vv))
    tops=read(ROOT/'top5_rankings.json');topmap={(x['geometry_kind'],x['query_id'],x['method'],int(x['replicate'])):x['top5_indices'] for x in tops}
    reported=csv_rows(ROOT/'per_query.csv');ck(len(reported)==len(topmap)==len(tops)==17424,'17424 metric/top5 rows')
    repmap={(r['geometry_kind'],r['query_id'],r['method'],int(r['replicate'])):r for r in reported};ck(len(repmap)==17424,'Metric uniqueness')
    pools=read(ROOT/'pool_manifest.json');oracle=[];max_error=0.;order_checks=0;historical_topchecks=0
    with np.load(V6/'complete_rankings.npz',allow_pickle=False) as ar,np.load(LUNA/'full_rankings.npz',allow_pickle=False) as la,np.load(SOL/'full_rankings.npz',allow_pickle=False) as sa:
        for kind in KINDS:
            for qid in qids:
                fid=queries[qid]['scenario_id'];pool=pools[kind][qid];pp=[index[p] for p in pool];used=set(pp)
                union=sorted({int(i) for m in ('kg_full','geometry_intent_filter','kg_no_class') for i in ar[f'{kind}__{qid}__closed__{m}__order'][:10]});ck(pp==union,'Pool union '+kind+'/'+qid)
                base={m:ar[f'{kind}__{qid}__closed__{m}__order'].tolist() for m in CONTROLS}
                full=base['kg_full'];sc=dict(zip(full,ar[f'{kind}__{qid}__closed__kg_full__score'].tolist()));tail=[i for i in full if i not in used]
                for rep in (1,2,3):
                    hv={arm:support[arm][str(rep)][fid] for arm in ('kg_luna','kg_sol','role_only_luna')};orders=dict(base)
                    for method,arm in [('kg_luna','kg_luna'),('kg_sol','kg_sol'),('role_only_hybrid','role_only_luna')]:
                        scores={i:(3*sc[i]+hv[arm][gallery[i]])/4 for i in pp};orders[method]=sorted(pp,key=lambda i:(-scores[i],i))+tail
                    for method,arm in [('kg_luna_score','kg_luna'),('kg_sol_score','kg_sol'),('role_only_score','role_only_luna')]:orders[method]=sorted(pp,key=lambda i:(-hv[arm][gallery[i]],i))+tail
                    for method in METHODS:
                        key=(kind,qid,method,rep);o=orders[method];ck(len(o)==len(set(o)),'Eligibility uniqueness '+str(key));ck(topmap[key]==o[:5],'Independent order '+str(key));order_checks+=1
                        r=repmap[key];ck(r['family_id']==fid,'Row family '+str(key));vals=scalar_metrics(o,completions[qid]);max_error=max(max_error,max(abs(vals[f]-float(r[f])) for f in FIELDS));oracle.append(dict(geometry_kind=kind,query_id=qid,family_id=fid,method=method,replicate=rep,**vals))
                    if rep==1:
                        for method,oldarchive in [('kg_luna',la),('kg_sol',sa)]:
                            oldkey=f'{kind}__{qid}__closed__'+('kg_full_llm' if method=='kg_luna' else 'kg_full_sol')
                            ck(orders[method]==oldarchive[oldkey+'__order'].tolist(),'Historical complete order continuity '+kind+'/'+qid+'/'+method);historical_topchecks+=1
    ck(max_error<=TOL,'Independent scalar metric discrepancy')
    grouped=defaultdict(list)
    for r in oracle:grouped[r['geometry_kind'],r['method'],r['replicate'],r['family_id']].append(r)
    family={}
    for k,rr in grouped.items():ck(len(rr)==24,'24role conditions');family[k]=average(rr)
    frows=csv_rows(ROOT/'family_metrics.csv');ck(len(family)==len(frows)==726,'726family rows')
    ferr=max(abs(family[r['geometry_kind'],r['method'],int(r['replicate']),r['family_id']][f]-float(r[f])) for r in frows for f in FIELDS)
    run={};summary=[]
    for kind in KINDS:
        for method in METHODS:
            for rep in (1,2,3):run[kind,method,rep]=average([family[kind,method,rep,fid] for fid in families])
            rec=dict(geometry_kind=kind,method=method)
            for f in FIELDS:
                vals=[run[kind,method,rep][f] for rep in (1,2,3)];mean=math.fsum(vals)/3;sd=math.sqrt(math.fsum((v-mean)**2 for v in vals)/2)
                rec[f+'_mean']=mean;rec[f+'_sd']=sd
            for metric in ('ndcg@5','strong_precision@5'):rec[metric+'_mean']=rec[metric+'_lower_mean'];rec[metric+'_sd']=rec[metric+'_lower_sd']
            summary.append(rec)
    rrows=csv_rows(ROOT/'run_metrics.csv');srows=csv_rows(ROOT/'summary.csv');ck(len(rrows)==len(run)==66 and len(srows)==22,'66run/22summary rows')
    runerr=max(abs(run[r['geometry_kind'],r['method'],int(r['replicate'])][f]-float(r[f])) for r in rrows for f in FIELDS)
    sumerr=0.
    for r in srows:
        rec=next(s for s in summary if (s['geometry_kind'],s['method'])==(r['geometry_kind'],r['method']))
        for f,v in rec.items():
            if f not in ('geometry_kind','method'):sumerr=max(sumerr,abs(v-float(r[f])))
    for r in result['summary']:
        rec=next(s for s in summary if (s['geometry_kind'],s['method'])==(r['geometry_kind'],r['method']))
        for f,v in rec.items():
            if f not in ('geometry_kind','method'):sumerr=max(sumerr,abs(v-r[f]))
    ck(max(ferr,runerr,sumerr)<=TOL,'Mean/SD discrepancy')
    collapsed={(kind,method,fid):average([family[kind,method,rep,fid] for rep in (1,2,3)]) for kind in KINDS for method in METHODS for fid in families}
    identified=all(r['ndcg@5_lower']==r['ndcg@5_upper'] for r in oracle if r['method'] in {'kg_luna','kg_sol','role_only_hybrid'})
    ck(identified==result['primary_ndcg_identified'],'Primary point identification')
    rng=np.random.default_rng(20261023);tests=[]
    for kind in KINDS:
        for proposed,comp in CONTRASTS:
            vals={f:[] for f in FIELDS}
            for fid in families:
                a=collapsed[kind,proposed,fid];b=collapsed[kind,comp,fid]
                for metric in ('ndcg@5','strong_precision@5'):
                    vals[metric+'_lower'].append(a[metric+'_lower']-b[metric+'_upper']);vals[metric+'_upper'].append(a[metric+'_upper']-b[metric+'_lower'])
            t=dict(geometry_kind=kind,proposed=proposed,comparator=comp,delta_lower=math.fsum(vals['ndcg@5_lower'])/11,delta_upper=math.fsum(vals['ndcg@5_upper'])/11,strong_precision_delta_lower=math.fsum(vals['strong_precision@5_lower'])/11,strong_precision_delta_upper=math.fsum(vals['strong_precision@5_upper'])/11,delta=None,ci95_lower=None,ci95_upper=None,raw_p=None,holm_p=None,family_wins=None,family_ties=None,family_losses=None)
            if identified:
                v=np.asarray(vals['ndcg@5_lower']);sample=rng.integers(0,11,size=(20000,11));ci=np.percentile(v[sample].mean(axis=1),[2.5,97.5]);t.update(delta=math.fsum(v)/11,ci95_lower=float(ci[0]),ci95_upper=float(ci[1]),raw_p=exact_p(v),family_wins=sum(v>TOL),family_ties=sum(abs(v)<=TOL),family_losses=sum(v< -TOL))
            tests.append(t)
    if identified:
        for t,h in zip(tests,holm([t['raw_p'] for t in tests])):t['holm_p']=h
    trows=csv_rows(ROOT/'paired_tests.csv');ck(len(trows)==4,'Fourprimary comparisons')
    for t,s in zip(tests,trows):
        for f,v in t.items():
            if isinstance(v,str):ck(v==s[f],'Test identity '+f)
            elif v is None:ck(s[f]=='','Withheld test '+f)
            else:ck(abs(v-float(s[f]))<=TOL,'Independent test '+str((t['geometry_kind'],t['proposed'],t['comparator'],f)))
    for t,s in zip(tests,result['paired_tests']):
        for f,v in t.items():
            if isinstance(v,str) or v is None:ck(v==s[f],'JSONtestidentity '+f)
            else:ck(abs(v-s[f])<=TOL,'JSONtestnumber '+f)
    drows=csv_rows(ROOT/'family_differences.csv');ck(len(drows)==44,'44family contrasts')
    for d in drows:
        a=collapsed[d['geometry_kind'],d['proposed'],d['family_id']];b=collapsed[d['geometry_kind'],d['comparator'],d['family_id']]
        for metric in ('ndcg@5','strong_precision@5'):
            ck(abs(float(d[metric+'_lower'])-(a[metric+'_lower']-b[metric+'_upper']))<=TOL and abs(float(d[metric+'_upper'])-(a[metric+'_upper']-b[metric+'_lower']))<=TOL,'Familydifference '+str(d))
    deterministic_sd={s['geometry_kind']+'/'+s['method']:s['ndcg@5_sd'] for s in summary if s['method'] in CONTROLS}
    ck(all(v<=TOL for v in deterministic_sd.values()),'Deterministiccontrols have zero acquisitionSD')
    replay=read(ROOT/'independent_replay_audit.json') if (ROOT/'independent_replay_audit.json').exists() else None
    if replay:ck(replay['status']=='PASS' and all(v['exact'] for v in replay['comparisons'].values()),'Isolated replay')
    scans=[
      ('Simpson’s paradox','Allroleeffects and equal24condition grouping retained; heterogeneous role directions do not establish a universal effect.'),
      ('Ecological fallacy','Elevenauthoredrole means are the inference units; three API acquisitions are not33independent role samples or population manufacturing outcomes.'),
      ('Berkson’s paradox','Galleryavailability, exemplarselection and inherited top10union condition this benchmark; no industrial prevalence is inferred.'),
      ('Collider bias','Fixed pool was selected before new model outputs. Conditional pool evaluation is retained without outcome-based expansion.'),
      ('Base rate neglect','StrongP@5 is concentration under fixedAI-reference grades, not sensitivity or industrial deploymentPPV.'),
      ('Regression to the mean','Historical firstKG passes and sequential benchmark exposure remain explicit; repeated API acquisition is not new data or retraining replication.'),
      ('Survivorship bias','Every attempted pass/job and declared neutral fallback remains in the analysis, without retries or best-run selection.'),
      ('Look-elsewhere effect','Fourprespecified contrasts share oneHolmfamily; allreported. Older comparison families do not imply research-history-wide correction.'),
      ('Garden of forking paths','New prompts,pools,weights,analysis and queueorder were sealed before these repeated outcomes; prior result knowledge is disclosed.'),
      ('Correlation versus causation','KGpackage versus holistic comparison changes decomposition,binding instructions and coverage rule together; graphstorage or bindings alone are not isolated.'),
      ('Reverse causality','Immutable AIqrels precede new acquisitions and remain excluded from inputs. SharedAIvisual assumptions may still correlate reference and model errors.'),
    ]
    report=dict(material_passport=dict(origin_skill='ARS experiment-agent',origin_mode='validate',origin_date=dt.datetime.now(dt.timezone.utc).date().isoformat(),verification_status='VERIFIED' if replay and not issues else 'ANALYZED',version_label='llm_robustness_v7_independent_audit'),status='PASS' if not issues else 'FAIL',issues=issues,warnings=warnings,protocol_sha256=seal,script_sha256=sha(Path(__file__)),api_calls_by_auditor=0,
      new_api=dict(attempts=len(attempts),responses=14588,unique_response_ids=len(set(response_ids)),models=dict(models),status_counts=dict(status),schema_valid_by_arm=dict(schema_by_arm),cost_by_arm_usd=dict(cost),cost_by_model_usd=dict(cost_models),cost_total_usd=math.fsum(cost.values()),usage_tokens=dict(tokens),usage_basis=dict(usage_basis),service_tier_counts=dict(tiers),peak_journaled_concurrency=peak,mean_call_seconds=math.fsum(latencies)/len(latencies)),
      evidence=dict(total_jobs=hverify,historical_jobs=4168,historical_sources=historical,validity=dict(validity),fallback_job_ids=fallbacks,states=dict(states),empty_view_counts=dict(empty),word_limit_overruns=word_overruns,collection_counts=dict(totals_by_collection),model_truth='Model observations, not independently verified engineering facts'),
      independent_numerics=dict(metric_rows=len(oracle),endpoint_values=len(oracle)*4,max_metric_difference=max_error,rank_orders_checked=order_checks,historical_fullorders_checked=historical_topchecks,family_rows=726,run_rows=66,summary_rows=22,max_family_error=ferr,max_run_error=runerr,max_summary_sd_error=sumerr,primary_identified=identified,raw_nulls=nulls,fullgallery_idcg=1816,sd_ddof=1,bootstrap_replicates=20000,bootstrap_seed=20261023,sign_patterns=2048),summary=summary,paired_tests=tests,ars_fallacy_scan=dict(coverage='11/11 checked',checks=[dict(type=a,assessment=b) for a,b in scans]),isolated_replay=replay)
    # NumPy counters in exact comparisons must become plain JSON scalars.
    report=json.loads(json.dumps(report,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),ensure_ascii=False))
    (ROOT/'post_execution_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    summaries='\n'.join('| '+s['geometry_kind']+' | '+s['method']+' | '+f"{s['ndcg@5_mean']:.7f} ± {s['ndcg@5_sd']:.7f}"+' | '+f"{s['strong_precision@5_mean']:.7f} ± {s['strong_precision@5_sd']:.7f}"+' |' for s in summary)
    contrasts='\n'.join('- '+t['geometry_kind']+' '+t['proposed']+' minus '+t['comparator']+': '+('inference withheld' if t['delta'] is None else f"Δ={t['delta']:+.7f}; rawp={t['raw_p']:.8f}; Holmp={t['holm_p']:.8f}; W/T/L={t['family_wins']}/{t['family_ties']}/{t['family_losses']}.") for t in tests)
    md=f'''# Independent repeated-acquisition v7 audit

## Material Passport

- Origin Skill: ARS experiment-agent
- Origin Mode: validate
- Verification Status: {report['material_passport']['verification_status']}
- Version Label: llm_robustness_v7_independent_audit

**Status: {report['status']}.** NoAPI,keyorother-project access occurred. This is same-team computational validation, not independent human review.

## Acquisition and evidence

Checked14,588unique new attempts and terminal responses,4,168sealed historical firstKG observations,all18,756graph/source H/J records and all frozen source seals. Newreported-token cost is USD{math.fsum(cost.values()):.8f}; model breakdown is{dict(cost_models)}. Returnedtiers:{dict(tiers)}. This is an estimate under frozen prices, not a billinginvoice. Schema-valid-by-arm:{dict(schema_by_arm)}. Whole-jobfallbacks:{len(fallbacks)}. Prose22-wordoverruns:{len(word_overruns)}; all counts are archived without resampling. Historical response identities and fullKG run1orders remain unchanged.

## Independent numerical oracle

Reconstructed all17,424metric rows and69,696endpointvalues with direct scalarDCG,complete1816candidate idealDCG,original raw-null0/1/2completions and independent numericalcaps. Missing/quarantinedsurface evidence precedes0.2mmguard comparisons. Independently reconstructed all hybrid/score-only top5orders from saved H/J and originalFullarrays, including ties and eligibility. Maximum metricdifference:{max_error:.3g}. Reproduced726familymeans,66runmeans,22aggregate means/sampleSD(ddof1),44familycontrasts andfour exacttests. Maximum aggregateSDdifference:{sumerr:.3g}. Both rawnulls remain unchanged. Tested hybridendpoint identification:{identified}.

| Representation | Method | nDCG@5 mean ± runSD | StrongP@5 mean ± runSD |
|---|---|---:|---:|
{summaries}

## Four primary contrasts

{contrasts}

Intervals are unadjusted20,000role-bootstrap intervals; tests enumerate2,048signpatterns and shareoneHolmfamily. AcquisitionSD is descriptive across threepasses, not a confidenceinterval or retraining-seedrobustness claim. The historicalfirstKGpasses remain temporally separated from new interleaved passes. A nonsignificantmodel contrast does not establishequivalence. Isolated replay:{replay['status'] if replay else 'not yet supplied'}.

## ARS fallacy scan:11/11checked

'''+ '\n\n'.join(str(i+1)+'. **'+a+'**. '+b for i,(a,b) in enumerate(scans))+f'\n\nMaterialissues:{issues if issues else "none found"}. Warnings:{warnings if warnings else "none found"}.\n'
    (ROOT/'post_execution_audit.md').write_text(md,encoding='utf8')
    print(json.dumps(dict(status=report['status'],issues=issues,warnings=warnings,cost=math.fsum(cost.values()),rows=len(oracle),tests=report['paired_tests']),indent=2))
    assert not issues,'See written audit issues'


if __name__=='__main__':main()
