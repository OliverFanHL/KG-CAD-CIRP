"""Exploratory, family-paired statistics for the verified CAD annotation study.

The family is the unit of analysis.  Neither CAD-task records nor query variants
are treated as independent replications.  Statistical uncertainty is conditional
on this authored benchmark and does not establish independent expert validity.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from ..io import write_csv, write_json


METRICS = (
    'ndcg@5', 'precision@5', 'strong_precision@5', 'ndcg@10', 'ndcg@20',
    'recall@5', 'recall@10', 'recall@20', 'ap', 'ap@20',
    'dimension_failures', 'dimension_unknowns',
)
ENDPOINT = 'ndcg@5'
SPRINGS = {'compression_return', 'tension_return'}
TOLERANCE = 1e-12


def _bool(value):
    if isinstance(value, str):
        if value.lower() not in {'true', 'false'}:
            raise ValueError('Boolean flags must be bool or True/False strings')
        return value.lower() == 'true'
    return bool(value)


def _flip_p(values, groups=None):
    """Enumerate all sign flips; no Monte Carlo approximation or +1 correction."""
    values = np.asarray(values, dtype=float)
    groups = groups if groups is not None else [[i] for i in range(len(values))]
    totals = np.asarray([values[group].sum() for group in groups])
    signs = np.asarray(list(product((-1, 1), repeat=len(groups))), dtype=float)
    null = signs @ totals / len(values)
    observed = abs(float(values.mean()))
    return float(np.mean(np.abs(null) >= observed - TOLERANCE))


def _bootstrap(values, rng, replicates, groups=None):
    values = np.asarray(values, dtype=float)
    if groups is None:
        samples = rng.integers(0, len(values), size=(replicates, len(values)))
        means = values[samples].mean(axis=1)
    else:
        # Sample task clusters, keeping the spring pair together. Ratio-of-sums
        # preserves equal family weighting rather than giving a pair one weight.
        totals = np.asarray([values[group].sum() for group in groups])
        counts = np.asarray([len(group) for group in groups])
        samples = rng.integers(0, len(groups), size=(replicates, len(groups)))
        means = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def _holm(values):
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for index, original in enumerate(order):
        running = max(running, min(1.0, (len(values) - index) * values[original]))
        adjusted[original] = running
    return adjusted.tolist()


def analyze(rows, config, output):
    """Save summaries and eleven predeclared primary paired comparisons.

    ``primary_comparators`` contains five comparator method IDs. Primary tests
    use closed-cohort nominal exemplar cases and nDCG@5 only. Other endpoints,
    grading scenarios and cohorts are descriptive. Bootstrap seeds are fixed;
    all sign-flip p-values enumerate the complete finite reference distribution.
    """
    output = Path(output)
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot analyze an empty evaluation')
    metrics = [metric for metric in METRICS if all(metric in row for row in rows)]
    if ENDPOINT not in metrics:
        raise ValueError('Primary nDCG@5 endpoint is missing')
    for row in rows:
        if any(not np.isfinite(float(row[metric])) for metric in metrics):
            raise ValueError('Evaluation metrics must be finite')
    primary_cohort = config.get('primary_cohort', 'closed')
    comparators = list(config['primary_comparators'])
    if len(comparators) != 5 or len(set(comparators)) != 5 or 'kg_full' in comparators:
        raise ValueError('Declare exactly five distinct full-model comparators')
    seed = int(config.get('seed', 17))
    replicates = 20000
    rng = np.random.default_rng(seed)

    # Build family means before any cross-family summary.
    grouped = defaultdict(list)
    for row in rows:
        variants = ('all', 'exemplar' if _bool(row['has_exemplar']) else 'text')
        for variant in variants:
            key = tuple(row[field] for field in
                        ('cohort', 'grading_scenario', 'geometry_kind', 'method'))
            grouped[(*key, variant, row['family_id'])].append(row)
    family_metrics = []
    summary_groups = defaultdict(list)
    for key, group in sorted(grouped.items()):
        cohort, scenario, kind, method, variant, family = key
        record = dict(cohort=cohort, grading_scenario=scenario, geometry_kind=kind,
                      method=method, variant=variant, family_id=family,
                      queries=len(group), constrained_queries=sum(_bool(r['has_constraint']) for r in group))
        record.update({metric: float(np.mean([float(r[metric]) for r in group])) for metric in metrics})
        family_metrics.append(record)
        summary_groups[key[:-1]].append(record)
    summary = []
    for key, group in sorted(summary_groups.items()):
        cohort, scenario, kind, method, variant = key
        record = dict(cohort=cohort, grading_scenario=scenario, geometry_kind=kind,
                      method=method, variant=variant, families=len(group),
                      queries=sum(r['queries'] for r in group), averaging='equal_family')
        record.update({metric: float(np.mean([r[metric] for r in group])) for metric in metrics})
        summary.append(record)

    primary = [r for r in rows if r['cohort'] == primary_cohort
               and r['grading_scenario'] == 'nominal' and _bool(r['has_exemplar'])]
    lookup = {}
    for row in primary:
        key = (row['geometry_kind'], row['method'], row['family_id'], row['query_id'])
        if key in lookup:
            raise ValueError('Duplicate primary query/method/representation row')
        lookup[key] = row
    definitions = [(kind, 'kg_full', kind, other)
                   for kind in ('embedding', 'descriptor') for other in comparators]
    definitions.append(('descriptor', 'kg_full', 'embedding', 'kg_full'))
    comparisons, family_differences = [], []
    for kind_a, method_a, kind_b, method_b in definitions:
        a = {(family, query): row for (kind, method, family, query), row in lookup.items()
             if kind == kind_a and method == method_a}
        b = {(family, query): row for (kind, method, family, query), row in lookup.items()
             if kind == kind_b and method == method_b}
        if not a or set(a) != set(b):
            raise ValueError(f'Primary comparison is not fully paired: {kind_a}/{method_a}, {kind_b}/{method_b}')
        families = sorted({family for family, _ in a})
        if len(families) != 11:
            raise ValueError('Primary inference requires all eleven authored task families')
        comparison = f'{kind_a}:{method_a} - {kind_b}:{method_b}'
        values = []
        local_rows = []
        for family in families:
            keys = sorted(key for key in a if key[0] == family)
            record = dict(comparison=comparison, family_id=family, paired_queries=len(keys))
            record.update({'delta_' + metric: float(np.mean([
                float(a[key][metric]) - float(b[key][metric]) for key in keys])) for metric in metrics})
            family_differences.append(record)
            local_rows.append(record)
            values.append(record['delta_' + ENDPOINT])
        values = np.asarray(values)
        spring_indices = [i for i, family in enumerate(families) if family in SPRINGS]
        if len(spring_indices) != 2:
            raise ValueError('Both spring families are required for the dependence sensitivity')
        clusters = [[i] for i in range(len(families)) if i not in spring_indices] + [spring_indices]
        ci = _bootstrap(values, rng, replicates)
        spring_ci = _bootstrap(values, rng, replicates, clusters)
        loo = [(values.sum() - value) / (len(values) - 1) for value in values]
        delta = float(values.mean())
        record = dict(comparison=comparison, cohort=primary_cohort, grading_scenario='nominal',
                      variant='exemplar', endpoint=ENDPOINT, geometry_kind_a=kind_a,
                      method_a=method_a, geometry_kind_b=kind_b, method_b=method_b,
                      families=len(families), paired_queries=len(a), delta=delta,
                      ci95_lower=ci[0], ci95_upper=ci[1],
                      exact_sign_flip_p=_flip_p(values),
                      sign_flip_assignments=2 ** len(families),
                      spring_joint_flip_p=_flip_p(values, clusters),
                      spring_joint_flip_assignments=2 ** len(clusters),
                      spring_cluster_ci95_lower=spring_ci[0], spring_cluster_ci95_upper=spring_ci[1],
                      bootstrap_replicates=replicates, bootstrap_seed=seed,
                      family_wins=int(np.sum(values > TOLERANCE)),
                      family_ties=int(np.sum(np.abs(values) <= TOLERANCE)),
                      family_losses=int(np.sum(values < -TOLERANCE)),
                      leave_one_family_out_min=float(min(loo)), leave_one_family_out_max=float(max(loo)),
                      practical_threshold=0.02, practical_improvement=delta >= 0.02,
                      practical_harm=delta <= -0.02,
                      practical_threshold_engineering_validated=False,
                      statistical_interpretation='exploratory_authored_benchmark_not_population_confirmation')
        record.update({'delta_' + metric: float(np.mean([r['delta_' + metric] for r in local_rows]))
                       for metric in metrics})
        comparisons.append(record)
    for record, adjusted in zip(comparisons, _holm([r['exact_sign_flip_p'] for r in comparisons])):
        record['holm_p'] = adjusted
        record['holm_below_0_05'] = adjusted < 0.05
        record['holm_family_size'] = len(comparisons)

    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'family_metrics.csv', family_metrics)
    write_csv(output / 'primary_comparisons.csv', comparisons)
    write_csv(output / 'primary_family_differences.csv', family_differences)
    write_json(output / 'statistical_protocol.json', {
        'primary_cohort': primary_cohort, 'primary_grading': 'nominal',
        'primary_variant': 'exemplar', 'primary_endpoint': ENDPOINT,
        'primary_comparators': comparators, 'primary_tests': len(comparisons),
        'multiple_testing': 'Holm across the eleven primary nDCG@5 comparisons',
        'averaging_unit': 'equal task family after query averaging',
        'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
        'interval': 'percentile 95%; unadjusted exploratory intervals',
        'sign_flip': 'exact two-sided enumeration, equality included',
        'spring_sensitivity': 'joint sign flip; cluster bootstrap ratio of sums retains equal-family statistic',
        'practical_threshold': 0.02, 'engineering_validated_threshold': False,
        'limitations': [
            'Authored AI-graded benchmark; uncertainty does not establish independent human or population validity.',
            'Sign-flip exchangeability is an exploratory assumption, not a randomized treatment assignment.',
            'Other endpoints and subsets are descriptive and have no confirmatory significance interpretation.',
            'The spring grouping addresses one known dependence and cannot remove all task or CAD dependence.',
        ],
    })
    return comparisons
