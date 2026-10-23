"""Isolated deterministic replay of repeated evidence; never resubmit API jobs."""
from pathlib import Path
import shutil,json
from scripts import complete_llm_robustness_v7 as evidence
from scripts import evaluate_llm_robustness_v7 as evaluation
ROOT=Path('outputs/llm_robustness_v7')
def main():
    assert (ROOT/'results.json').exists();target=(ROOT/'independent_replay').resolve();assert not target.exists();target.mkdir()
    for n in ['study.json','protocol_freeze.json','evaluation_freeze.json','pool_manifest.json','api_attempts.jsonl','api_responses.jsonl']:shutil.copyfile(ROOT/n,target/n)
    evidence.ROOT=target;evidence.main();evaluation.evaluate(target)
    checks={}
    for n in ['graph_support.json','graph_evidence.jsonl','per_query.csv','family_metrics.csv','run_metrics.csv','summary.csv','paired_tests.csv','family_differences.csv','top5_rankings.json']:
        a,b=evidence.sha(ROOT/n),evidence.sha(target/n);checks[n]=dict(original_sha256=a,replay_sha256=b,exact=a==b)
    result=dict(status='PASS' if all(x['exact'] for x in checks.values()) else 'FAIL',api_calls=0,comparisons=checks);evidence.write(ROOT/'independent_replay_audit.json',result);assert result['status']=='PASS'
if __name__=='__main__':main()
