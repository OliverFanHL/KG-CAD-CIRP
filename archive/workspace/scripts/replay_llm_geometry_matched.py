"""Recompute saved GPT evidence and metrics in an isolated directory, no API.

Original sources and study outputs are read-only. The frozen completion and
evaluation implementations are reused unchanged, with an explicit isolated
output root. Numerical artifacts are then compared byte-for-byte.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from scripts import complete_llm_geometry_evidence as evidence
from scripts import evaluate_llm_geometry_matched as evaluation

ROOT=Path('outputs/llm_geometry_matched_v1')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert (ROOT/'results.json').exists(), 'Wait until the original study completes.'
    target=(ROOT/'independent_replay').resolve()
    assert target.is_relative_to(ROOT.resolve()) and not target.exists()
    target.mkdir()
    for name in ('study.json','pool_manifest.json','protocol_freeze.json','evaluation_freeze.json',
                 'api_attempts.jsonl','api_responses.jsonl'):
        shutil.copyfile(ROOT/name,target/name)
    evidence.ROOT=target
    evidence.main()
    evaluation.evaluate(target)
    comparisons={}
    for name in ('graph_support.json','graph_evidence.jsonl','per_query.csv','summary.csv',
                 'paired_tests.csv','family_metrics.csv','family_differences.csv','full_rankings.npz'):
        original,replay=sha(ROOT/name),sha(target/name)
        comparisons[name]={'original_sha256':original,'replay_sha256':replay,'exact':original==replay}
    report={'status':'PASS' if all(v['exact'] for v in comparisons.values()) else 'FAIL',
            'api_calls':0,'original_outputs_modified':False,'comparisons':comparisons,
            'script_sha256':sha(Path(__file__))}
    (ROOT/'independent_replay_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    assert report['status']=='PASS'
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
