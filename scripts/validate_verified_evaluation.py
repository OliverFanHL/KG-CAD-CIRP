"""Independent metric replay, source hashing and deterministic artifact checks."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from cadkg.io import read_json, read_csv, write_json, sha256
from cadkg.evaluation import metrics
from scripts.run_verified_evaluation import grade_vector

parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/verified_evaluation.json')
args=parser.parse_args()
cfg=read_json(args.config)
out=Path(cfg['output'])
run=read_json(out/'run_manifest.json');replay=read_json(out/'replay/run_manifest.json')
assert run['artifact_sha256']==replay['artifact_sha256']
archive=np.load(out/'complete_rankings.npz');copy=np.load(out/'replay/complete_rankings.npz')
assert archive.files==copy.files
assert all(np.array_equal(archive[k],copy[k]) for k in archive.files)
cohorts=read_json(out/'cohorts.json');annotation=read_json(cfg['annotation_path'])
observations={(r['family_id'],r['part_id']):r for r in annotation['judgments']}
catalog=read_json(Path(cfg['source_run'])/'bundle/catalog.json')
parts={p['part_id']:p for p in catalog['parts']}
evidence={}
for r in annotation['judgments']:
    pid=r['part_id']
    if pid in evidence:continue
    evidence[pid]=read_json(Path(cfg['evidence_root'])/f"{r['inspection_id']}.json")
    assert sha256(parts[pid]['absolute_path'])==r['source_sha256']==evidence[pid]['source_sha256']
protocol_path=out/'effective_protocol.json'
if not protocol_path.exists():protocol_path=Path(cfg['source_run'])/'protocol.json'
queries={q['query_id']:q for q in read_json(protocol_path)['queries']}
qrels={}; checked=0
for r in read_csv(out/'per_query.csv'):
    c,q,s=r['cohort'],r['query_id'],r['grading_scenario'];ids=cohorts[c]
    key=(c,q,s)
    if key not in qrels:
        values=grade_vector(queries[q],ids,observations,evidence,s)
        qrels[key]={p:{'grade':int(v)} for p,v in zip(ids,values)}
    order=archive[f"{r['geometry_kind']}__{q}__{c}__{r['method']}__order"]
    ranked=[ids[i] for i in order]
    for k in [5,10,20]:
        result=metrics(ranked,qrels[key],k)
        for name in ['ndcg','precision','recall']:
            assert abs(result[f'{name}@{k}']-float(r[f'{name}@{k}']))<1e-12
        if k==20:assert abs(result['map@20']-float(r['ap@20']))<1e-12
        checked+=1
write_json(out/'reproducibility.json',dict(status='VERIFIED',csv_tables_exactly_replayed=len(run['artifact_sha256']),
    complete_ranking_arrays_exactly_replayed=len(archive.files),metric_cases_independently_checked=checked,
    raw_step_hashes_verified=len(evidence),independent_metric_engine='cadkg.evaluation.metrics',
    scope='Computational reproduction and input integrity; not independent human relevance validation'))
print(read_json(out/'reproducibility.json'))
