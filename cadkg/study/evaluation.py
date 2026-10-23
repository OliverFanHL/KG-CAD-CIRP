"""Evaluate the final balanced query panel from saved rankings and observations."""
from collections import defaultdict
import csv
import numpy as np
from cadkg.experiments.reference import numeric_reference
from .io import read, write, write_csv
from .metrics import grade_limits, prepare_limits, rank_bounds
from .ranking import (
    KINDS, CONTROLS, METHODS, METRICS, analyze, archive_order,
    validate_pools, rerank, score_only,
)

SUMMARY_ARTIFACTS = (
    'family_metrics.csv', 'run_metrics.csv', 'summary.csv',
    'paired_tests.csv', 'family_differences.csv',
)


def write_analysis(output, rows):
    family, run, summary, tests, differences, identified = analyze(rows)
    if not identified:
        raise ValueError('Primary metrics unexpectedly not point identified')
    for name, records in zip(SUMMARY_ARTIFACTS, (family, run, summary, tests, differences)):
        write_csv(output / name, records)


def metrics_only(source, output):
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['replicate'] = int(row['replicate'])
        for metric in METRICS:
            for endpoint in ('lower', 'upper'):
                key = metric + '_' + endpoint
                row[key] = float(row[key])
    write_analysis(output, rows)


def panel(data):
    queries = read(data / 'queries/queries.json')
    if len(queries) != 264 or len({q['query_id'] for q in queries}) != 264:
        raise ValueError('Expected 264 unique primary queries')
    families = defaultdict(lambda: defaultdict(list))
    for query in queries:
        families[query['scenario_id']][query['reference_part_id']].append(query)
    if len(families) != 11 or any(len(refs) != 4 or any(len(rows) != 6 for rows in refs.values()) for refs in families.values()):
        raise ValueError('Expected 11 roles, each with four references and six conditions')
    return queries


def evaluate(data, output, support):
    queries = panel(data)
    gallery = read(data / 'rankings/cohorts.json')['closed']
    if len(gallery) != 1816 or gallery != sorted(set(gallery)):
        raise ValueError('Expected 1816 unique sorted gallery candidates')
    records = read(data / 'annotations/judgments.json')['judgments']
    observations = {(r['family_id'], r['part_id']): r for r in records}
    families = sorted({q['scenario_id'] for q in queries})
    if len(observations) != len(records) or set(observations) != {(fid, pid) for fid in families for pid in gallery}:
        raise ValueError('Expected a complete 11 by 1816 judgment matrix')
    raw = {fid: np.asarray([-1 if observations[fid, pid]['grade'] is None else observations[fid, pid]['grade'] for pid in gallery]) for fid in families}
    measurements = {}
    for pid in gallery:
        row = observations[families[0], pid]
        measurement = read(data / 'geometry/evidence' / (row['inspection_id'] + '.json'))
        if measurement['part_id'] != pid or measurement['source_sha256'] != row['source_sha256']:
            raise ValueError('Geometry evidence source mismatch: ' + pid)
        measurements[pid] = measurement
    prepared = {}
    for query in queries:
        verdicts = [numeric_reference(query, measurements[pid], .2) for pid in gallery]
        lower, upper = grade_limits(raw[query['scenario_id']], verdicts, 'nominal')
        prepared[query['query_id']] = prepare_limits(lower, upper, (5,))
    rows, tops = [], []
    with np.load(data / 'rankings/complete_rankings.npz', allow_pickle=False) as rankings:
        pools, _ = validate_pools(data / 'observations/repeated_observations', queries, gallery, rankings)
        for kind in KINDS:
            for query in queries:
                qid, fid = query['query_id'], query['scenario_id']
                pool = pools[kind][qid]
                baseline = {method: archive_order(rankings, kind, qid, method, len(gallery)) for method in CONTROLS}
                for replicate in (1, 2, 3):
                    luna = support['kg_luna'][str(replicate)][fid]
                    sol = support['kg_sol'][str(replicate)][fid]
                    role = support['role_only_luna'][str(replicate)][fid]
                    orders = {method: values[0] for method, values in baseline.items()}
                    for method, values in [('kg_luna', luna), ('kg_sol', sol), ('role_only_hybrid', role)]:
                        orders[method] = rerank(*baseline['kg_full'], pool, values, gallery)[0]
                    for method, values in [('kg_luna_score', luna), ('kg_sol_score', sol), ('role_only_score', role)]:
                        orders[method] = score_only(baseline['kg_full'][0], pool, values, gallery)
                    for method in METHODS:
                        order = orders[method]
                        bounds = rank_bounds(order, prepared[qid])
                        row = dict(geometry_kind=kind, query_id=qid, family_id=fid, method=method,
                                   replicate=replicate, cohort='closed', grading_scenario='nominal')
                        row.update({f'{metric}_{end}': float(bounds[metric][i]) for metric in METRICS for i, end in enumerate(('lower', 'upper'))})
                        rows.append(row)
                        tops.append(dict(geometry_kind=kind, query_id=qid, method=method,
                                         replicate=replicate, top5_indices=[int(i) for i in order[:5]]))
    write_csv(output / 'per_query.csv', rows)
    write_analysis(output, rows)
    write(output / 'top5_rankings.json', tops)
