"""Seal repeated acquisition, including unchanged historical first KG runs."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path('outputs/llm_robustness_v7')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines()]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def main():
    assert not (ROOT/'evidence_completion.json').exists()
    freeze=read(ROOT/'protocol_freeze.json');study=read(ROOT/'study.json')
    assert sha(ROOT/'study.json')==freeze['study_sha256']
    assert sha(freeze['runner_path'])==freeze['runner_sha256']
    for x in freeze['required_files']:assert sha(x['path'])==x['sha256'],x['path']
    attempted=lines(ROOT/'api_attempts.jsonl');responses=lines(ROOT/'api_responses.jsonl')
    jobs={j['job_id']:j for j in study['jobs']};am={x['job_id']:x for x in attempted}
    meta={x['job_id']:x for x in freeze['jobs']}
    assert len(jobs)==len(am)==len(attempted)==len(responses)==14588
    assert len({r['job_id'] for r in responses})==14588
    support=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)));graph=[];states=Counter();status=Counter();costs=defaultdict(float);tokens=Counter();validations=Counter();latencies=[]
    def add(job,record,historical=False):
        value=record.get('output_json');valid=record['status']=='completed' and record['output_schema_valid'] and isinstance(value,dict) and set(value)==set(job['predicate_ids'])
        if job['arm']=='role_only_luna':
            valid=valid and all(o['score']==.5 or bool(o['views']) for o in value.values())
        else:
            valid=valid and all(o['status']=='unknown' or bool(o['views']) for o in value.values())
        observations=[]
        for pid in job['predicate_ids']:
            if job['arm']=='role_only_luna':
                o=value[pid] if valid else {'score':.5,'views':[],'observation':'No valid model observation; fixed neutral fallback.'}
                state=str(o['score']);assert o['score'] in [i/8 for i in range(9)]
            else:
                o=value[pid] if valid else {'status':'unknown','views':[],'observation':'No valid model observation; fixed neutral fallback.'}
                state=o['status'];assert state in {'yes','no','unknown'}
            obs=dict(predicate_id=pid,**o)
            defs=job['instruction_payload'].get('atomic_predicates',[])
            if defs:obs['required_triple']=next(d['required_triple'] for d in defs if d['predicate_id']==pid)
            observations.append(obs);states[job['arm']+'/'+state]+=1
        h=(observations[0]['score'] if job['arm']=='role_only_luna' else sum({'yes':1.,'no':0.,'unknown':.5}[o['status']] for o in observations)/len(observations))
        support[job['arm']][str(job['replicate'])][job['family_id']][job['part_id']]=h
        graph.append(dict(job_id=job['job_id'],source_job_id=job['source_job_id'],arm=job['arm'],replicate=job['replicate'],family_id=job['family_id'],part_id=job['part_id'],historical=historical,local_order_seed=job['local_order_seed'],model_resolved=record['model_resolved'],response_id=record['response_id'],image_sha256=job['image_sha256'],valid_response=valid,H=h,observations=observations,evidence_source='Model-observed visible evidence; not independently verified engineering facts'))
        validations[job['arm']+('/valid' if valid else '/fallback')]+=1
    for record in responses:
        jid=record['job_id'];job=jobs[jid];a=am[jid]
        assert record['protocol_sha256']==a['protocol_sha256']==sha(ROOT/'protocol_freeze.json')
        assert record['request_sha256']==a['request_sha256']==meta[jid]['request_sha256']
        assert record['attempt_id']==a['attempt_id']
        add(job,record);status[record['status']]+=1;costs[job['arm']]+=record['cost_usd'];tokens.update(record['usage_tokens']);latencies.append(record['elapsed_seconds'])
    luna=read('outputs/llm_geometry_matched_v1/study.json')
    for arm,directory in [('kg_luna','outputs/llm_geometry_matched_v1'),('kg_sol','outputs/llm_geometry_sol_matched_v1')]:
        existing={r['job_id']:r for r in lines(Path(directory)/'api_responses.jsonl')}
        assert len(existing)==2084
        for source in luna['jobs']:
            job=dict(source,job_id=arm+'__r1__'+source['job_id'],source_job_id=source['job_id'],arm=arm,replicate=1,local_order_seed=None)
            add(job,existing[source['job_id']],True)
    assert len(graph)==18756
    write(ROOT/'graph_support.json',support)
    with (ROOT/'graph_evidence.jsonl').open('w',encoding='utf8') as f:
        for r in sorted(graph,key=lambda r:r['job_id']):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),protocol_freeze_sha256=sha(ROOT/'protocol_freeze.json'),new_calls=len(responses),historical_calls_reused=4168,total_observation_jobs=18756,status_counts=dict(status),validity_counts=dict(validations),predicate_states_by_arm=dict(states),new_cost_by_arm_usd=dict(costs),estimated_new_cost_usd=sum(costs.values()),usage_tokens=dict(tokens),mean_new_call_seconds=sum(latencies)/len(latencies),api_attempts_sha256=sha(ROOT/'api_attempts.jsonl'),api_responses_sha256=sha(ROOT/'api_responses.jsonl'),graph_support_sha256=sha(ROOT/'graph_support.json'),graph_evidence_sha256=sha(ROOT/'graph_evidence.jsonl'),labels_read=False,no_resampling=True)
    write(ROOT/'evidence_completion.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
