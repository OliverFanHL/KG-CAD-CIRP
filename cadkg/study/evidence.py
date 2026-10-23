"""Aggregate saved LLM responses without contacting any API."""
from collections import Counter, defaultdict
import json
from .io import read, lines, write

def build_evidence(data, output):
    root = data / 'observations/repeated_observations'
    study = read(root / 'study.json')
    responses = lines(root / 'api_responses.jsonl')
    jobs = {job['job_id']: job for job in study['jobs']}
    if len(jobs) != 14588 or len(responses) != 14588 or len({r['job_id'] for r in responses}) != 14588:
        raise ValueError('Repeated observations require 14588 unique jobs and responses')
    support=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    graph=[]
    states=Counter()
    validations=Counter()
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
        add(jobs[record['job_id']], record)
    source_study = read(data / 'observations/directed_luna/study.json')
    for arm, directory in [('kg_luna', 'directed_luna'), ('kg_sol', 'directed_sol')]:
        existing = {r['job_id']: r for r in lines(data / 'observations' / directory / 'api_responses.jsonl')}
        if len(existing) != 2084:
            raise ValueError('Directed observations require 2084 unique responses per model')
        for source in source_study['jobs']:
            job = dict(source, job_id=arm+'__r1__'+source['job_id'], source_job_id=source['job_id'], arm=arm, replicate=1, local_order_seed=None)
            add(job, existing[source['job_id']], True)
    if len(graph) != 18756:
        raise ValueError('Expected 18756 combined observation jobs')
    write(output / 'graph_support.json', support)
    with (output / 'graph_evidence.jsonl').open('w', encoding='utf8', newline='\r\n') as stream:
        for record in sorted(graph, key=lambda r: r['job_id']):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return support
