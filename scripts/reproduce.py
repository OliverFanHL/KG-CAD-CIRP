"""Reproduce the final study offline from saved observations and rankings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadkg.study.evidence import build_evidence
from cadkg.study.evaluation import evaluate, metrics_only, SUMMARY_ARTIFACTS
from cadkg.study.io import sha, verify_inputs

ARTIFACTS = ('graph_support.json', 'graph_evidence.jsonl', 'per_query.csv',
             *SUMMARY_ARTIFACTS, 'top5_rankings.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None,
                        help='Fresh destination, relative to repository root or absolute.')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Recompute role/run means, SDs and paired tests from saved per-query metrics.')
    args = parser.parse_args()
    destination = args.output or Path('results/reproduced_metrics' if args.metrics_only else 'results/reproduced')
    target = (ROOT / destination).resolve()
    if target.exists():
        parser.error('Output already exists; choose a fresh --output directory.')
    verified_inputs = 0
    if not args.metrics_only:
        verified_inputs = verify_inputs(ROOT)
    target.mkdir(parents=True)
    if args.metrics_only:
        metrics_only(ROOT / 'results/per_query.csv', target)
        names = SUMMARY_ARTIFACTS
    else:
        support = build_evidence(ROOT / 'data', target)
        evaluate(ROOT / 'data', target, support)
        names = ARTIFACTS
    comparisons = {}
    for name in names:
        source = (ROOT / 'data/observations/repeated_observations' if name.startswith('graph_') else ROOT / 'results') / name
        comparisons[name] = dict(expected_sha256=sha(source), actual_sha256=sha(target / name),
                                 exact=sha(source) == sha(target / name))
    report = dict(status='PASS' if all(item['exact'] for item in comparisons.values()) else 'FAIL',
                  mode='metrics-only' if args.metrics_only else 'full-offline-replay',
                  api_calls=0, verified_inputs=verified_inputs, comparisons=comparisons)
    (target / 'release_replay_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status=report['status'], output=str(target),
                          artifacts_compared=len(comparisons), api_calls=0), indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
