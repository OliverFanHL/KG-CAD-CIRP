"""Evaluate a separately collected observer on the fixed study panel, offline."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadkg.experiments.reference import numeric_reference
from cadkg.study.evaluation import panel
from cadkg.study.io import lines, read, sha, verify_inputs, write, write_csv
from cadkg.study.metrics import grade_limits, prepare_limits, rank_bounds
from cadkg.study.ranking import KINDS, METRICS, archive_order, means, rerank, score_only, validate_pools


def support_value(job, record, design):
    """Apply the study's whole-job neutral fallback, including schema failures."""
    value = record.get('output_json')
    valid = (record.get('status') == 'completed' and record.get('output_schema_valid') is True
             and isinstance(value, dict) and set(value) == set(job['predicate_ids']))
    observations = list(value.values()) if valid else []
    for observation in observations:
        required = {'score', 'views', 'observation'} if design == 'holistic' else {'status', 'views', 'observation'}
        if not isinstance(observation, dict) or set(observation) != required:
            valid = False
            break
        views = observation['views']
        valid = valid and isinstance(views, list) and all(view in ('oblique', 'opposite', 'profile', 'axial', 'cut_X', 'cut_Y') for view in views) and isinstance(observation['observation'], str)
        if design == 'holistic':
            score = observation['score']
            valid = valid and type(score) in (int, float) and score in [i / 8 for i in range(9)] and (score == .5 or bool(views))
        else:
            valid = valid and observation['status'] in ('yes', 'no', 'unknown') and (observation['status'] == 'unknown' or bool(views))
    if not valid:
        return .5, False
    if design == 'holistic':
        return observations[0]['score'], True
    return sum({'yes': 1., 'no': 0., 'unknown': .5}[o['status']] for o in observations) / len(observations), True


def check_plan(plan):
    from scripts.export_experiment import baseline_plan
    if plan.get('design') not in ('directed', 'holistic') or type(plan.get('collections')) is not int or plan['collections'] < 1 or not isinstance(plan.get('model'), str) or not plan['model'].strip():
        raise ValueError('Plan requires a model, directed/holistic design, and positive collection count')
    expected = baseline_plan(plan['model'], plan['design'], plan['collections'])
    if plan != expected:
        raise ValueError('Plan differs from the current immutable source tasks or data hashes')
    for job in plan['jobs']:
        if sha(ROOT / job['image_path']) != job['image_sha256']:
            raise ValueError('Neutral image differs: ' + job['part_id'])


def evaluate(plan, responses, output):
    check_plan(plan)
    jobs = {job['job_id']: job for job in plan['jobs']}
    records = {record['job_id']: record for record in responses}
    if len(records) != len(responses) or set(records) != set(jobs):
        raise ValueError('Require exactly one saved response per planned job, including failed jobs')
    support = defaultdict(lambda: defaultdict(dict))
    validity = []
    for job_id, job in jobs.items():
        value, valid = support_value(job, records[job_id], plan['design'])
        support[str(job['collection'])][job['family_id']][job['part_id']] = value
        validity.append(dict(job_id=job_id, valid_response=valid, support=value))
    data = ROOT / 'data'
    queries = panel(data)
    gallery = read(data / 'rankings/cohorts.json')['closed']
    judgments = {(r['family_id'], r['part_id']): r for r in read(data / 'annotations/judgments.json')['judgments']}
    families = sorted({q['scenario_id'] for q in queries})
    measurements = {}
    for pid in gallery:
        row = judgments[families[0], pid]
        measurement = read(data / 'geometry/evidence' / (row['inspection_id'] + '.json'))
        if measurement['source_sha256'] != row['source_sha256'] or measurement['part_id'] != pid:
            raise ValueError('Numerical evidence source mismatch: ' + pid)
        measurements[pid] = measurement
    prepared = {}
    for query in queries:
        raw = np.asarray([-1 if judgments[query['scenario_id'], pid]['grade'] is None else judgments[query['scenario_id'], pid]['grade'] for pid in gallery])
        verdicts = [numeric_reference(query, measurements[pid], .2) for pid in gallery]
        prepared[query['query_id']] = prepare_limits(*grade_limits(raw, verdicts, 'nominal'), (5,))
    rows = []
    methods = ('kg_full', 'geometry_intent_filter', 'new_hybrid', 'new_score')
    with np.load(data / 'rankings/complete_rankings.npz', allow_pickle=False) as rankings:
        pools, _ = validate_pools(data / 'observations/repeated_observations', queries, gallery, rankings)
        for kind in KINDS:
            for query in queries:
                qid, fid = query['query_id'], query['scenario_id']
                full = archive_order(rankings, kind, qid, 'kg_full', len(gallery))
                control = archive_order(rankings, kind, qid, 'geometry_intent_filter', len(gallery))[0]
                for collection in range(1, plan['collections'] + 1):
                    values = support[str(collection)][fid]
                    pool = pools[kind][qid]
                    orders = dict(kg_full=full[0], geometry_intent_filter=control,
                                  new_hybrid=rerank(*full, pool, values, gallery)[0],
                                  new_score=score_only(full[0], pool, values, gallery))
                    for method in methods:
                        bounds = rank_bounds(orders[method], prepared[qid])
                        row = dict(geometry_kind=kind, query_id=qid, family_id=fid,
                                   method=method, replicate=collection)
                        row.update({f'{m}_{end}': float(bounds[m][i]) for m in METRICS for i, end in enumerate(('lower', 'upper'))})
                        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['geometry_kind'], row['method'], row['replicate'], row['family_id']].append(row)
    family = [dict(geometry_kind=k, method=m, replicate=c, family_id=f, **means(rr)) for (k, m, c, f), rr in sorted(grouped.items())]
    run, summary, differences = [], [], []
    for kind in KINDS:
        for method in methods:
            local = []
            for collection in range(1, plan['collections'] + 1):
                rr = [r for r in family if (r['geometry_kind'], r['method'], r['replicate']) == (kind, method, collection)]
                row = dict(geometry_kind=kind, method=method, replicate=collection, **means(rr))
                run.append(row)
                local.append(row)
            row = dict(geometry_kind=kind, method=method, collections=plan['collections'])
            for metric in METRICS:
                for end in ('lower', 'upper'):
                    field = f'{metric}_{end}'
                    row[field + '_mean'] = float(np.mean([r[field] for r in local]))
                    row[field + '_sd'] = float(np.std([r[field] for r in local], ddof=1)) if len(local) > 1 else None
            summary.append(row)
        for method in ('new_hybrid', 'new_score'):
            for comparator in ('kg_full', 'geometry_intent_filter'):
                for fid in families:
                    a = means([r for r in family if (r['geometry_kind'], r['method'], r['family_id']) == (kind, method, fid)])
                    b = means([r for r in family if (r['geometry_kind'], r['method'], r['family_id']) == (kind, comparator, fid)])
                    row = dict(geometry_kind=kind, method=method, comparator=comparator, family_id=fid)
                    row.update({m + '_lower': a[m + '_lower'] - b[m + '_upper'] for m in METRICS})
                    row.update({m + '_upper': a[m + '_upper'] - b[m + '_lower'] for m in METRICS})
                    differences.append(row)
    output.mkdir(parents=True)
    for name, values in [('per_query.csv', rows), ('family_metrics.csv', family), ('run_metrics.csv', run), ('summary.csv', summary), ('family_comparisons.csv', differences)]:
        write_csv(output / name, values)
    write(output / 'support.json', support)
    write(output / 'response_validation.json', validity)
    return dict(status='COMPLETED_EXTENSION', requested_model=plan['model'], design=plan['design'],
                recorded_resolved_models=sorted({r['model_resolved'] for r in responses if isinstance(r.get('model_resolved'), str)}),
                collections=plan['collections'], api_calls=0, metric_rows=len(rows),
                fallback_jobs=sum(not r['valid_response'] for r in validity),
                comparisons='Descriptive role-level differences; no significance tests or model-selection claims')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT / 'data') or args.output.resolve() == ROOT / 'results':
        parser.error('Use a fresh extension output directory outside data/ and reported result files.')
    verify_inputs(ROOT)
    result = evaluate(read(args.plan), lines(args.responses), args.output)
    result.update(plan_sha256=sha(args.plan), responses_sha256=sha(args.responses))
    write(args.output / 'evaluation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
