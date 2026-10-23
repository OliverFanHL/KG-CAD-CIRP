"""Seal complete GPT observations and derive the fixed KG evidence signal.

No relevance judgments are read or modified. Failed observations are retained
and contribute the predeclared neutral signal; they are never resampled.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('outputs/llm_geometry_matched_v1')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main():
    study = read(ROOT/'study.json')
    freeze = read(ROOT/'protocol_freeze.json')
    assert sha(ROOT/'study.json') == freeze['study_sha256']
    assert sha(freeze['runner_path']) == freeze['runner_sha256']
    assert sha(ROOT/'evaluation_freeze.json') == freeze['evaluation_freeze_sha256']
    for entry in freeze['required_files']:
        assert sha(entry['path'])==entry['sha256'], entry['path']
    assert not (ROOT/'evidence_completion.json').exists(), 'Do not overwrite sealed evidence.'
    jobs = {j['job_id']:j for j in study['jobs']}
    attempted = lines(ROOT/'api_attempts.jsonl')
    records = lines(ROOT/'api_responses.jsonl')
    assert len(attempted)==len(records)==len(jobs)
    assert len({r['job_id'] for r in records})==len(records)
    assert {r['job_id'] for r in attempted}==set(jobs)=={r['job_id'] for r in records}
    protocol_sha = sha(ROOT/'protocol_freeze.json')
    metas = {j['job_id']:j for j in freeze['jobs']}
    attempts = {r['job_id']:r for r in attempted}
    support, graph = {}, []
    status, predicate_status, models = Counter(), Counter(), Counter()
    costs, tokens, elapsed = 0., Counter(), []
    for record in records:
        jid = record['job_id']; job=jobs[jid]; meta=metas[jid]
        assert record['protocol_sha256']==attempts[jid]['protocol_sha256']==protocol_sha
        assert record['request_sha256']==attempts[jid]['request_sha256']==meta['request_sha256']
        assert sha(job['image_path'])==job['image_sha256']==meta['image_sha256']
        expected = set(job['predicate_ids'])
        value = record.get('output_json')
        valid = record['status']=='completed' and record['output_schema_valid'] and isinstance(value,dict) and set(value)==expected
        valid = valid and all(o['status']=='unknown' or bool(o['views']) for o in value.values())
        observations = []
        for pid in job['predicate_ids']:
            observation = value[pid] if valid else {'status':'unknown','views':[], 'observation':'No valid model observation; fixed neutral fallback.'}
            state = observation['status']
            assert state in {'yes','no','unknown'}
            predicate_status[state] += 1
            definition = next(p for p in job['instruction_payload']['atomic_predicates'] if p['predicate_id']==pid)
            observations.append(dict(predicate_id=pid,status=state,views=observation['views'],observation=observation['observation'],required_triple=definition['required_triple']))
        h = sum({'yes':1.,'no':0.,'unknown':.5}[o['status']] for o in observations)/4
        assert len(observations)==4
        support.setdefault(job['family_id'],{})[job['part_id']]=h
        graph.append(dict(job_id=jid,part_id=job['part_id'],family_id=job['family_id'],
            evidence_source='GPT-observed geometric relations; not independent CAD verification',
            image_sha256=job['image_sha256'],response_id=record['response_id'],
            model_resolved=record['model_resolved'],valid_response=valid,H=h,observations=observations))
        status[record['status']]+=1
        models[str(record['model_resolved'])]+=1
        costs+=record['cost_usd'];tokens.update(record.get('usage_tokens') or {})
        elapsed.append(record.get('elapsed_seconds',0))
    support={fid:{pid:members[pid] for pid in sorted(members)} for fid,members in sorted(support.items())}
    write(ROOT/'graph_support.json',support)
    with (ROOT/'graph_evidence.jsonl').open('w',encoding='utf-8') as f:
        for record in sorted(graph,key=lambda r:r['job_id']):
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    write(ROOT/'evidence_completion.json',dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        protocol_freeze_sha256=protocol_sha,graph_support_sha256=sha(ROOT/'graph_support.json'),
        graph_evidence_sha256=sha(ROOT/'graph_evidence.jsonl'),api_responses_sha256=sha(ROOT/'api_responses.jsonl'),
        api_attempts_sha256=sha(ROOT/'api_attempts.jsonl'),completed_jobs=len(records),
        status_counts=dict(status),predicate_status_counts=dict(predicate_status),resolved_models=dict(models),
        estimated_cost_usd=costs,usage_tokens=dict(tokens),max_usd=study['max_usd'],
        latency_seconds=dict(mean=sum(elapsed)/len(elapsed),max=max(elapsed),sum=sum(elapsed)),
        no_qrels_read=True,no_resampling=True,source='Responses API, cached once per family-CAD pair'))
    print(json.dumps(read(ROOT/'evidence_completion.json'),indent=2))

if __name__=='__main__':main()
