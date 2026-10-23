"""No-API isolated replay of every Sol evidence and numerical artifact."""
from pathlib import Path
import json
import shutil
from scripts import complete_llm_geometry_sol_evidence_v1 as evidence
from scripts import evaluate_llm_geometry_sol_matched_v1 as evaluation

ROOT=Path('outputs/llm_geometry_sol_matched_v1')


def main():
    assert (ROOT/'results.json').exists()
    target=(ROOT/'independent_replay').resolve();assert not target.exists();target.mkdir()
    for name in ('study.json','pool_manifest.json','protocol_freeze.json','evaluation_freeze.json','api_attempts.jsonl','api_responses.jsonl'):
        shutil.copyfile(ROOT/name,target/name)
    evidence.ROOT=target;evidence.main();evaluation.evaluate(target)
    checks={}
    for name in ('graph_support.json','graph_evidence.jsonl','per_query.csv','summary.csv','paired_tests.csv','family_metrics.csv','family_differences.csv','full_rankings.npz'):
        a,b=evaluation.original.sha(ROOT/name),evaluation.original.sha(target/name)
        checks[name]={'original_sha256':a,'replay_sha256':b,'exact':a==b}
    r={'status':'PASS' if all(v['exact'] for v in checks.values()) else 'FAIL','api_calls':0,'original_outputs_modified':False,'comparisons':checks}
    evaluation.original.dump(ROOT/'independent_replay_audit.json',r)
    assert r['status']=='PASS';print(json.dumps(r,indent=2))


if __name__=='__main__':main()
