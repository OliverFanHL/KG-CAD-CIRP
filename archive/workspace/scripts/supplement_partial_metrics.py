"""Supplement existing fixed-cutoff diagnostics without altering the sealed study.

Self-tests use synthetic labels only. Freeze this new scoring code before
evaluating published AI-reference annotations. No relevance completion is chosen.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations, product
import json
import math
from pathlib import Path

import numpy as np

from cadkg.io import read_csv, read_json, write_csv, write_json, sha256
from scripts.partial_feature_review import (
    ROOT, KINDS, SCENARIOS, SUBSETS, ROBUST_METHODS, require, _subset,
    _verify_hashes, _frozen_path, transformed_set,
)

FREEZE = ROOT / "supplement_freeze.json"
DEFAULT_OUTPUT = ROOT / "supplemental_evaluation"
TOL = 1e-12


def metric_names(cutoffs, ap_cutoff=20):
    return tuple(f"{name}@{k}" for k in cutoffs
                 for name in ("ndcg", "precision", "strong_precision", "recall")) + ("ap", f"ap@{ap_cutoff}")


def prepare_limits(lower, upper, cutoffs=(5, 10, 20), ap_cutoff=20):
    """Cache cohort-wide denominator extrema, independent of retrieval method."""
    lo, hi = np.asarray(lower), np.asarray(upper)
    require(lo.ndim == hi.ndim == 1 and lo.shape == hi.shape, "Grade-bound vector shape mismatch")
    require(lo.dtype.kind in "iu" and hi.dtype.kind in "iu"
            and np.all((0 <= lo) & (lo <= hi) & (hi <= 2)), "Invalid grade endpoints")
    require(len(set(cutoffs)) == len(cutoffs) and all(type(k) is int and k > 0 for k in cutoffs)
            and type(ap_cutoff) is int and ap_cutoff > 0, "Invalid cutoff")
    discount = 1. / np.log2(np.arange(max(cutoffs), dtype=float) + 2.)
    ideal_lo = np.cumsum((np.exp2(np.sort(lo)[::-1][:max(cutoffs)]) - 1.) * discount[:min(max(cutoffs), len(lo))])
    ideal_hi = np.cumsum((np.exp2(np.sort(hi)[::-1][:max(cutoffs)]) - 1.) * discount[:min(max(cutoffs), len(hi))])

    def at(values, k):
        return float(values[min(k, len(values)) - 1]) if len(values) else 0.

    return dict(lower=lo.astype(np.int8), upper=hi.astype(np.int8), cutoffs=tuple(cutoffs), ap_cutoff=ap_cutoff,
                discount=discount, rlo=int(np.count_nonzero(lo)), rhi=int(np.count_nonzero(hi)),
                idcg={k: (at(ideal_lo, k), at(ideal_hi, k)) for k in cutoffs})


def _ratio(nlo, nhi, dlo, dhi, unit_interval=True):
    """Include denominator-zero metric=0; never drop a zero-relevance query."""
    if dhi == 0:
        return 0., 0.
    lower = nlo / dhi
    upper = nhi / dlo if dlo else (1. if nhi > 0 else 0.)
    return float(lower), float(min(1., upper) if unit_interval else upper)


def rank_bounds(order, prepared):
    """Vectorized monotone numerators over separately bounded denominators."""
    order = np.asarray(order)
    require(order.ndim == 1 and order.dtype.kind in "iu", "Ranking must be an integer index vector")
    require(len(order) == len(np.unique(order)) and np.all((order >= 0) & (order < len(prepared["lower"]))),
            "Duplicate or out-of-cohort ranking index")
    lo, hi = prepared["lower"][order], prepared["upper"][order]
    hit_lo, hit_hi = (lo > 0).astype(np.int64), (hi > 0).astype(np.int64)
    hits_lo, hits_hi = np.cumsum(hit_lo), np.cumsum(hit_hi)
    strong_lo, strong_hi = np.cumsum(lo == 2), np.cumsum(hi == 2)
    maximum = max(prepared["cutoffs"])
    weights = prepared["discount"][:min(maximum, len(order))]
    gains_lo = np.cumsum((np.exp2(lo[:maximum]) - 1.) * weights)
    gains_hi = np.cumsum((np.exp2(hi[:maximum]) - 1.) * weights)

    def at(values, k):
        return float(values[min(k, len(values)) - 1]) if len(values) else 0.

    bounds = {}
    for k in prepared["cutoffs"]:
        idlo, idhi = prepared["idcg"][k]
        bounds[f"ndcg@{k}"] = _ratio(at(gains_lo, k), at(gains_hi, k), idlo, idhi)
        bounds[f"precision@{k}"] = at(hits_lo, k) / k, at(hits_hi, k) / k
        bounds[f"strong_precision@{k}"] = at(strong_lo, k) / k, at(strong_hi, k) / k
        # Relevance count is integer: if it can be zero, its minimum positive
        # value is at least 1; the upper ratio is capped at 1.
        bounds[f"recall@{k}"] = _ratio(at(hits_lo, k), at(hits_hi, k),
                                       max(1, prepared["rlo"]), prepared["rhi"])
    ranks = np.arange(1, len(order) + 1, dtype=float)
    numerator_lo = np.cumsum(hit_lo * hits_lo / ranks)
    numerator_hi = np.cumsum(hit_hi * hits_hi / ranks)
    full_nlo = float(numerator_lo[-1]) if len(order) else 0.
    full_nhi = float(numerator_hi[-1]) if len(order) else 0.
    bounds["ap"] = _ratio(full_nlo, full_nhi, max(1, prepared["rlo"]), prepared["rhi"])
    apk = prepared["ap_cutoff"]
    bounds[f"ap@{apk}"] = _ratio(at(numerator_lo, apk), at(numerator_hi, apk),
                                   max(1, min(apk, prepared["rlo"])), min(apk, prepared["rhi"]))
    for name, (lower, upper) in bounds.items():
        require(-TOL <= lower <= upper + TOL and upper <= 1 + TOL, f"Invalid envelope: {name}")
    return bounds


def random_bounds(feasible, prepared, k=5):
    """Existing uniform-without-replacement expectation, not a new comparator.

    For each fixed completion expected DCG is mean feasible gain times the sum
    of the first min(k, feasible_count) discounts. IDCG uses the whole cohort.
    """
    feasible = np.asarray(feasible)
    require(feasible.ndim == 1 and feasible.dtype.kind in "iu"
            and len(feasible) == len(np.unique(feasible))
            and np.all((feasible >= 0) & (feasible < len(prepared["lower"]))), "Invalid feasible index set")
    require(k in prepared["idcg"], "Random expectation cutoff absent from cached IDCG")
    if not len(feasible):
        return {"expected_ndcg5": (0., 0.), "expected_precision5": (0., 0.), "expected_strong_precision5": (0., 0.)}
    lo, hi = prepared["lower"][feasible], prepared["upper"][feasible]
    n = min(k, len(feasible))
    scale = float(np.sum(prepared["discount"][:n]))
    # NumPy exp2(int8) yields float16: explicitly use float64 before a mean so
    # fractions such as one-third cannot be rounded outside a valid envelope.
    nlo = float(np.mean(np.exp2(lo.astype(np.float64)) - 1.)) * scale
    nhi = float(np.mean(np.exp2(hi.astype(np.float64)) - 1.)) * scale
    idlo, idhi = prepared["idcg"][k]
    return {"expected_ndcg5": _ratio(nlo, nhi, idlo, idhi),
            "expected_precision5": (float(np.mean(lo > 0)) * n / k, float(np.mean(hi > 0)) * n / k),
            "expected_strong_precision5": (float(np.mean(lo == 2)) * n / k, float(np.mean(hi == 2)) * n / k)}


def grade_limits(raw, verdicts, scenario="nominal"):
    """Endpoint image of the unchanged monotone raw/conditional/size transforms.

    A null retains all raw grades 0..2. Conditional transforms may create gaps
    such as {0,2}; endpoints suffice for monotone numerator/denominator extrema
    and do not assert that a missing intermediate grade is admissible.
    """
    require(scenario in SCENARIOS, "Unknown conditional scenario")
    raw = np.asarray(raw, dtype=int)
    require(np.all(np.isin(raw, [-1, 0, 1, 2])), "Invalid raw grade vector")
    require(len(raw) == len(verdicts) and all(v in {"pass", "not_requested", "unknown", "fail"} for v in verdicts),
            "Invalid reference verdict vector")
    lo, hi = np.where(raw < 0, 0, raw), np.where(raw < 0, 2, raw)
    if scenario != "nominal":
        value = 0 if scenario == "conditional_zero" else 2
        lo, hi = np.where(lo == 1, value, lo), np.where(hi == 1, value, hi)
    verdicts = np.asarray(verdicts)
    unknown, fail = verdicts == "unknown", verdicts == "fail"
    lo[unknown], hi[unknown] = np.minimum(lo[unknown], 1), np.minimum(hi[unknown], 1)
    lo[fail], hi[fail] = 0, 0
    return lo, hi


def _exact(order, grades, cutoffs, ap_cutoff):
    """Independent scalar test oracle, including core AP@20 denominator."""
    relevant = sum(g > 0 for g in grades)
    result = {}
    for k in cutoffs:
        ideal = sorted(grades, reverse=True)[:k]
        idcg = sum((3 if g == 2 else 1 if g == 1 else 0) / math.log(i + 2, 2) for i, g in enumerate(ideal))
        dcg = sum((3 if grades[j] == 2 else 1 if grades[j] == 1 else 0) / math.log(i + 2, 2)
                  for i, j in enumerate(order[:k]))
        hits = sum(grades[j] > 0 for j in order[:k])
        result.update({f"ndcg@{k}": dcg / idcg if idcg else 0., f"precision@{k}": hits / k,
                       f"strong_precision@{k}": sum(grades[j] == 2 for j in order[:k]) / k,
                       f"recall@{k}": hits / relevant if relevant else 0.})
    count, total, short = 0, 0., 0.
    for position, j in enumerate(order, 1):
        if grades[j] > 0:
            count += 1
            total += count / position
            if position <= ap_cutoff:
                short += count / position
    result["ap"] = total / relevant if relevant else 0.
    result[f"ap@{ap_cutoff}"] = short / min(ap_cutoff, relevant) if relevant else 0.
    return result


def self_test():
    choices = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    cases = completions = singleton = random_cases = random_completions = 0

    def check(sets, order, cutoffs=(1, 2, 5, 10, 20), apk=20):
        nonlocal cases, completions, singleton
        cache = prepare_limits(np.asarray([min(s) for s in sets], dtype=int),
                               np.asarray([max(s) for s in sets], dtype=int), cutoffs, apk)
        bounds = rank_bounds(np.asarray(order, dtype=int), cache)
        cases += 1
        for grades in product(*sets):
            exact = _exact(order, grades, cutoffs, apk)
            for metric, value in exact.items():
                lower, upper = bounds[metric]
                require(lower - TOL <= value <= upper + TOL,
                        f"Synthetic envelope failed: {sets}/{order}/{metric}/{lower}/{value}/{upper}")
                if all(len(s) == 1 for s in sets):
                    require(abs(lower - value) < TOL and abs(upper - value) < TOL, "Singleton envelope not exact")
            completions += 1
            singleton += int(all(len(s) == 1 for s in sets))

    for n in range(4):
        orders = [p for size in range(n + 1) for p in permutations(range(n), size)]
        for sets in product(choices, repeat=n):
            for order in orders:
                check(sets, order)
                check(sets, order, cutoffs=(1, 2, 5), apk=2)
            cache = prepare_limits(np.asarray([min(s) for s in sets], dtype=int),
                                   np.asarray([max(s) for s in sets], dtype=int))
            for mask in product((False, True), repeat=n):
                feasible = tuple(i for i, included in enumerate(mask) if included)
                bounds = random_bounds(np.asarray(feasible, dtype=int), cache)
                random_cases += 1
                orders_random = list(permutations(feasible, min(5, len(feasible))))
                for grades in product(*sets):
                    values = [_exact(o, grades, (5,), 20) for o in orders_random]
                    for name, metric in (("expected_ndcg5", "ndcg@5"), ("expected_precision5", "precision@5"),
                                         ("expected_strong_precision5", "strong_precision@5")):
                        expectation = sum(v[metric] for v in values) / len(values)
                        lower, upper = bounds[name]
                        require(lower - TOL <= expectation <= upper + TOL, "Random expectation envelope failed")
                        if all(len(s) == 1 for s in sets):
                            require(abs(lower - expectation) < TOL and abs(upper - expectation) < TOL,
                                    "Random singleton expectation not exact")
                    random_completions += 1
    # All admissible sets at four items, filtered/full/empty returns, and a short
    # AP cutoff exercise truncation exhaustively without enormous permutations.
    for sets in product(choices, repeat=4):
        for order in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0), ()):
            check(sets, order, cutoffs=(1, 2, 5), apk=2)
    # Actual AP@20 denominator saturation and uncertain relevance beyond rank20.
    for baseline in (0, 1, 2):
        for three in product(choices, repeat=3):
            sets = [(baseline,)] * 22
            for i, s in zip((0, 19, 21), three):
                sets[i] = s
            for order in (tuple(range(22)), tuple(reversed(range(22))), tuple(range(0, 22, 2))):
                check(sets, order, cutoffs=(5, 10, 20), apk=20)
    transforms = 0
    for raw, verdict, scenario in product((-1, 0, 1, 2), ("pass", "not_requested", "unknown", "fail"), SCENARIOS):
        lo, hi = grade_limits([raw], [verdict], scenario)
        expected = transformed_set(None if raw == -1 else raw, verdict, scenario)
        require((lo[0], hi[0]) == (min(expected), max(expected)), "Vectorized transform differs from unchanged core")
        transforms += 1
    return dict(status="VERIFIED_SYNTHETIC", envelope_cases=cases, exact_completions=completions,
                singleton_cases=singleton, random_envelope_cases=random_cases,
                random_exact_completions=random_completions, vector_transform_cases=transforms,
                oracle="Independent scalar metrics and enumerated uniform feasible permutations",
                scope="Synthetic only; AP@20, all cutoffs, short/filtered/empty rankings and zero relevance covered",
                human_validation=False)


def _ranking_state(ranking_root):
    path = Path(ranking_root) / "ranking_validation.json"
    validation = read_json(path)
    require(validation.get("status") == "VERIFIED" and validation.get("replay_verified") is True,
            "Fresh ranking replay must be verified before supplementary scoring")
    require(all(Path(name).name == name for name in validation["files_sha256"]), "Ranking hash keys must be basenames")
    hashes = {str(Path(ranking_root) / name): digest for name, digest in validation["files_sha256"].items()}
    for name in ("complete_rankings.npz", "robustness_complete_rankings.npz", "robustness_cases.csv", "effective_protocol.json", "cohorts.json"):
        _frozen_path(Path(ranking_root) / name, hashes)
    _verify_hashes(hashes)
    return validation, hashes


def freeze(config_path, ranking_root, freeze_path=FREEZE):
    """Seal new code and published input hashes without reading any metric table."""
    freeze_path = Path(freeze_path)
    require(not freeze_path.exists(), "Supplement already frozen; do not overwrite its original seal")
    initial = read_json(ROOT / "evaluation_freeze.json")["input_sha256"]
    _verify_hashes(initial)
    consumed_path = ROOT / "consumed_evidence_freeze.json"
    consumed = read_json(consumed_path)["input_sha256"]
    _verify_hashes(consumed)
    _, ranks = _ranking_state(ranking_root)
    cfg = read_json(config_path)
    paths = [Path(__file__), Path("scripts/partial_feature_review.py"), Path(config_path),
             ROOT / "evaluation_freeze.json", consumed_path, Path(ranking_root) / "ranking_validation.json",
             Path(cfg["annotation_path"]), Path(cfg["annotation_audit_path"]),
             Path(cfg["source_run"]) / "bundle/catalog.json", Path("cadkg/io.py"),
             Path("cadkg/experiments/intent.py"), Path("cadkg/experiments/reference.py"), Path("cadkg/reasoning.py")]
    hashes = {**ranks, **{str(p): sha256(p) for p in paths}}
    seal = dict(stage="Supplementary fixed-cutoff scoring code and published inputs sealed before reading metric results",
                source="ai_reference", human_expert_reviewed=False,
                code_sha256=sha256(__file__), annotation_sha256=sha256(cfg["annotation_path"]),
                initial_evaluation_freeze_sha256=sha256(ROOT / "evaluation_freeze.json"),
                input_sha256=hashes, existing_cutoffs=list(cfg["cutoffs"]),
                additions="Existing cutoff/robustness metrics and existing uniform-feasible random expectation only; no new methods, queries or conditions",
                metric_results_used_for_design=False)
    write_json(freeze_path, seal)
    return dict(status="SUPPLEMENT_FROZEN", freeze=str(freeze_path), code_sha256=seal["code_sha256"])


def _flat(bounds):
    return {f"{metric}_{edge}": float(values[i]) for metric, values in bounds.items()
            for i, edge in enumerate(("lower", "upper"))}


def summaries(rows, fields, query_by_id, names):
    """Query/seed mean within family, then equal family mean; keep zeros."""
    columns = [f"{m}_{edge}" for m in names for edge in ("lower", "upper")]
    grouped = defaultdict(list)
    for row in rows:
        for subset in SUBSETS:
            if _subset(query_by_id[row["query_id"]], subset):
                key = tuple(row[f] for f in fields) + (subset, row["family_id"])
                grouped[key].append(row)
    family_rows, macros = [], defaultdict(list)
    for key, members in sorted(grouped.items()):
        record = dict(zip(fields, key[:len(fields)]))
        record.update(subset=key[-2], family_id=key[-1], query_seed_cases=len(members),
                      unique_queries=len({r["query_id"] for r in members}),
                      **{c: float(np.mean([r[c] for r in members])) for c in columns})
        family_rows.append(record)
        macros[key[:-1]].append(record)
    macro_rows = []
    for key, members in sorted(macros.items()):
        record = dict(zip(fields, key[:len(fields)]))
        record.update(subset=key[-1], families=len(members),
                      query_seed_cases=sum(r["query_seed_cases"] for r in members),
                      unique_queries=sum(r["unique_queries"] for r in members),
                      averaging="equal_family_after_query_and_seed_mean",
                      **{c: float(np.mean([r[c] for r in members])) for c in columns})
        macro_rows.append(record)
    return family_rows, macro_rows


README = """# Supplementary scoring of existing fixed cutoffs

This addendum retains the sealed AI-reference annotations, methods, queries,
candidate cohorts, ranking orders, seeds and robustness conditions. It supplies
the existing nDCG, precision, strong precision and recall diagnostics at configured
cutoffs 5/10/20, plus AP and AP@20, and the existing nominal uniform-feasible random
expectation at 5. It introduces no new experimental condition or relevance grade.

Every null retains raw grades 0..2 before the unchanged conditional and size
transforms. The tables contain conservative lower/upper annotation envelopes.
Numerator and denominator extrema may require different completions, so endpoints
need not be jointly attainable. Zero-relevance queries keep metric value zero and
remain in query/family means. AP@20 divides by min(20, total relevance), matching
the existing evaluator. Random nDCG expectation uses mean feasible gain times the
sum of the first min(5, feasible count) discounts over whole-cohort IDCG.

Queries and seeds are averaged within each task family, then families equally.
These computational uncertainty envelopes are not sampling confidence intervals,
significance tests or independent human validation. No favorable completion is
selected. The original frozen [main report](../REPORT.md) is preserved; this
supplement has its own [code/input seal](../supplement_freeze.json).

Artifacts: per_query_bounds.csv, family_bounds.csv, summary_bounds.csv,
robustness_per_query_bounds.csv, robustness_family_bounds.csv,
robustness_summary_bounds.csv, random_expectation_bounds.csv,
random_family_bounds.csv, random_summary_bounds.csv, synthetic_validation.json,
and run_manifest.json. Independent replay validation compares every artifact hash.
"""


def evaluate(config_path, ranking_root, output, freeze_path=FREEZE):
    from cadkg.experiments.reference import numeric_reference
    from cadkg.reasoning import check_constraints
    from scripts.run_verified_evaluation import METHODS

    output, ranking_root, freeze_path = Path(output), Path(ranking_root), Path(freeze_path)
    require(not output.exists(), "Refuse to overwrite supplementary results; use a fresh folder")
    seal = read_json(freeze_path)
    _verify_hashes(seal["input_sha256"])
    require(sha256(__file__) == seal["code_sha256"], "Supplement scoring code changed after freeze")
    freeze_digest = sha256(freeze_path)
    initial = read_json(ROOT / "evaluation_freeze.json")["input_sha256"]
    _verify_hashes(initial)
    consumed = read_json(ROOT / "consumed_evidence_freeze.json")["input_sha256"]
    _verify_hashes(consumed)
    validation, ranking_hashes = _ranking_state(ranking_root)
    cfg = read_json(config_path)
    cutoffs = tuple(cfg["cutoffs"])
    require(cutoffs == (5, 10, 20) and list(cutoffs) == seal["existing_cutoffs"], "Existing cutoff matrix changed")
    annotation = read_json(cfg["annotation_path"])
    require(annotation.get("source") == "ai_reference" and annotation.get("human_expert_reviewed") is False,
            "Published AI-reference provenance required")
    records = annotation["judgments"]
    obs = {(r["family_id"], r["part_id"]): r for r in records}
    protocol, cohorts = read_json(ranking_root / "effective_protocol.json"), read_json(ranking_root / "cohorts.json")
    queries = protocol["queries"]
    query_by_id = {q["query_id"]: q for q in queries}
    families = {q["scenario_id"] for q in queries}
    ids = cohorts["closed"]
    require(len(queries) == len(query_by_id) == 110 and len(families) == 11 and len(METHODS) == 14
            and len(ids) == len(set(ids)) == 1816 and set(obs) == set(product(families, ids))
            and len(records) == len(obs), "Fixed published task/candidate/method matrix mismatch")
    require(annotation["scope_sha256"] == protocol["scope_sha256"]
            and annotation["query_sha256"] == protocol.get("annotation_query_sha256", protocol["query_sha256"]),
            "Published annotation/protocol scope mismatch")
    raw_by_family = {}
    for fid in families:
        values = []
        for pid in ids:
            row = obs[fid, pid]
            grade = row["grade"]
            require(grade is None or type(grade) is int and grade in (0, 1, 2), "Invalid published grade")
            require(row.get("source") == "ai_reference" and row.get("human_expert_reviewed") is False
                    and row.get("review_status") == "visually_reviewed", "Invalid AI review provenance")
            require(grade is not None or row.get("decision") == "cannot_assess", "Null disposition must be retained")
            values.append(-1 if grade is None else grade)
        raw_by_family[fid] = np.asarray(values, dtype=np.int8)
    evidence = {}
    resolved_consumed = {Path(p).resolve() for p in consumed}
    first_fid = sorted(families)[0]
    for pid in ids:
        row = obs[first_fid, pid]
        path = Path(cfg["evidence_root"]) / (row["inspection_id"] + ".json")
        require(path.resolve() in resolved_consumed, f"Numerical evidence absent from original freeze: {path}")
        evidence[pid] = read_json(path)
        require(evidence[pid]["part_id"] == pid and evidence[pid]["source_sha256"] == row["source_sha256"],
                "Frozen numerical evidence source mismatch")
        require(all(obs[fid, pid]["inspection_id"] == row["inspection_id"]
                    and obs[fid, pid]["source_sha256"] == row["source_sha256"] for fid in families),
                "Role-specific source binding mismatch")
    parts = {p["part_id"]: p for p in read_json(Path(cfg["source_run"]) / "bundle/catalog.json")["parts"]}
    id_index = {p: i for i, p in enumerate(ids)}
    cohort_indexes = {}
    for cohort, selected in cohorts.items():
        require(len(selected) == len(set(selected)) and set(selected) <= set(ids), "Cohort mapping mismatch")
        cohort_indexes[cohort] = np.asarray([id_index[p] for p in selected], dtype=int)
    # Global query/guard transforms are reused by both representations, all
    # methods, all seed variants and all cohort subsets.
    limits_cache, prepared_cache = {}, {}

    def prepared(query, cohort, scenario="nominal", guard=.2):
        key = query["query_id"], scenario, guard
        if key not in limits_cache:
            verdicts = [numeric_reference(query, evidence[p], guard) for p in ids]
            limits_cache[key] = grade_limits(raw_by_family[query["scenario_id"]], verdicts, scenario)
        ckey = key + (cohort,)
        if ckey not in prepared_cache:
            lo, hi = limits_cache[key]
            indexes = cohort_indexes[cohort]
            prepared_cache[ckey] = prepare_limits(lo[indexes], hi[indexes], cutoffs)
        return prepared_cache[ckey]

    synthetic = self_test()
    rows, random_rows = [], []
    expected_keys = {f"{kind}__{q['query_id']}__{cohort}__{method}__{suffix}"
                     for kind, q, cohort, method, suffix in product(KINDS, queries, cohorts, METHODS, ("order", "score"))}
    with np.load(ranking_root / "complete_rankings.npz", allow_pickle=False) as archive:
        require(set(archive.files) == expected_keys and validation.get("replay_arrays_verified") == len(expected_keys),
                "Every fixed core order/score must be replay verified")
        for query, cohort in product(queries, cohorts):
            qid = query["query_id"]
            selected = cohorts[cohort]
            feasible = np.asarray([i for i, pid in enumerate(selected)
                                   if check_constraints(parts[pid], query["constraints"])["status"] in {"pass", "not_requested"}], dtype=int)
            for kind in KINDS:
                # Intent + filter has an active component even on text queries;
                # its complete order must enumerate exactly the serving-feasible set.
                filtered = archive[f"{kind}__{qid}__{cohort}__intent_filter__order"]
                require(set(filtered.tolist()) == set(feasible.tolist()), "Uniform feasible population differs from fixed serving filter")
                random_rows.append(dict(cohort=cohort, geometry_kind=kind, query_id=qid,
                                        family_id=query["scenario_id"], feasible_count=len(feasible),
                                        grading_scenario="nominal", **_flat(random_bounds(feasible, prepared(query, cohort)))))
                for method in METHODS:
                    order = archive[f"{kind}__{qid}__{cohort}__{method}__order"]
                    for scenario in SCENARIOS if cohort == "closed" else ("nominal",):
                        cache = prepared(query, cohort, scenario)
                        rows.append(dict(cohort=cohort, geometry_kind=kind, method=method, query_id=qid,
                                         family_id=query["scenario_id"], grading_scenario=scenario,
                                         relevant_total_lower=cache["rlo"], relevant_total_upper=cache["rhi"],
                                         **_flat(rank_bounds(order, cache))))
    names = metric_names(cutoffs)
    family_rows, summary = summaries(rows, ("cohort", "geometry_kind", "method", "grading_scenario"), query_by_id, names)
    random_names = ("expected_ndcg5", "expected_precision5", "expected_strong_precision5")
    random_family, random_summary = summaries(random_rows, ("cohort", "geometry_kind", "grading_scenario"), query_by_id, random_names)

    cases = read_csv(ranking_root / "robustness_cases.csv")
    expected_cases = set()
    for kind, query, method in product(KINDS, queries, ROBUST_METHODS):
        qid = query["query_id"]
        for guard in cfg["constraint_guards_mm"]:
            if guard != .2 and query["constraints"]:
                expected_cases.add((kind, qid, method, "constraint_guard", float(guard), 0))
        for seed in cfg["robustness_seeds"]:
            expected_cases.add((kind, qid, method, "tie_order", 0., int(seed)))
        for fraction, seed in product(cfg["classification_corruption_fractions"], cfg["robustness_seeds"]):
            expected_cases.add((kind, qid, method, "classifier_permutation", float(fraction), int(seed)))
    actual = [(c["geometry_kind"], c["query_id"], c["method"], c["robustness"], float(c["severity"]), int(c["seed"])) for c in cases]
    require(set(actual) == expected_cases and len(actual) == len(expected_cases)
            and validation.get("replay_robustness_orders_verified") == len(cases), "Fixed robustness case/replay coverage mismatch")
    robust_rows = []
    with np.load(ranking_root / "robustness_complete_rankings.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {c["key"] for c in cases} and len(archive.files) == len(cases), "Robustness key mapping mismatch")
        for case in cases:
            query = query_by_id[case["query_id"]]
            guard = float(case["guard_mm"])
            require(guard == (float(case["severity"]) if case["robustness"] == "constraint_guard" else .2),
                    "Original robustness guard changed")
            cache = prepared(query, "closed", guard=guard)
            robust_rows.append(dict(cohort="closed", geometry_kind=case["geometry_kind"], method=case["method"],
                                    query_id=case["query_id"], family_id=query["scenario_id"], grading_scenario="nominal",
                                    robustness=case["robustness"], severity=float(case["severity"]), seed=int(case["seed"]),
                                    guard_mm=guard, **_flat(rank_bounds(archive[case["key"]], cache))))
    robust_family, robust_summary = summaries(robust_rows, ("geometry_kind", "method", "robustness", "severity"), query_by_id, names)
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(initial)
    _verify_hashes(consumed)
    _verify_hashes(ranking_hashes)
    require(sha256(freeze_path) == freeze_digest, "Supplement freeze changed during scoring")
    output.mkdir(parents=True)
    for filename, table in (("per_query_bounds.csv", rows), ("family_bounds.csv", family_rows), ("summary_bounds.csv", summary),
                            ("robustness_per_query_bounds.csv", robust_rows), ("robustness_family_bounds.csv", robust_family),
                            ("robustness_summary_bounds.csv", robust_summary), ("random_expectation_bounds.csv", random_rows),
                            ("random_family_bounds.csv", random_family), ("random_summary_bounds.csv", random_summary)):
        write_csv(output / filename, table)
    write_json(output / "synthetic_validation.json", synthetic)
    (output / "README.md").write_text(README, encoding="utf-8")
    manifest = dict(status="VERIFIED_CONSERVATIVE_SUPPLEMENT", source="ai_reference", human_expert_reviewed=False,
                    supplement_freeze_sha256=freeze_digest, input_sha256=seal["input_sha256"],
                    code_sha256={str(Path(__file__)): sha256(__file__)}, metrics=list(names), cutoffs=list(cutoffs),
                    core_metric_cases=len(rows), robustness_metric_cases=len(robust_rows), random_expectation_cases=len(random_rows),
                    null_labels_imputed=False, point_significance_tests=False, experimental_conditions_added=False,
                    bounds="Conservative annotation outer envelopes, not sampling confidence intervals",
                    artifact_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output / "run_manifest.json", manifest)
    return {k: manifest[k] for k in ("status", "core_metric_cases", "robustness_metric_cases", "random_expectation_cases")}


def validate(output, replay, freeze_path=FREEZE):
    output, replay = Path(output), Path(replay)
    seal = read_json(freeze_path)
    _verify_hashes(seal["input_sha256"])
    first, second = read_json(output / "run_manifest.json"), read_json(replay / "run_manifest.json")
    require(first == second, "Supplement replay manifests differ")
    for name, digest in first["artifact_sha256"].items():
        require(sha256(output / name) == digest == sha256(replay / name), f"Supplement replay differs: {name}")
    require(sha256(output / "run_manifest.json") == sha256(replay / "run_manifest.json"), "Replay manifest bytes differ")
    result = dict(status="VERIFIED", supplementary_artifacts_identical=len(first["artifact_sha256"]),
                  manifests_identical=True, supplement_freeze_sha256=sha256(freeze_path),
                  scope="Computational reproducibility only; not human annotation validation")
    write_json(output / "reproducibility.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "freeze", "evaluate", "validate"))
    parser.add_argument("--config", default="configs/feature_review_v5.json")
    parser.add_argument("--rankings", default=str(ROOT / "evaluation"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--replay", default=str(ROOT / "supplemental_replay"))
    parser.add_argument("--freeze", default=str(FREEZE))
    args = parser.parse_args()
    if args.action == "self-test":
        result = self_test()
    elif args.action == "freeze":
        result = freeze(args.config, args.rankings, args.freeze)
    elif args.action == "evaluate":
        result = evaluate(args.config, args.rankings, args.output, args.freeze)
    else:
        result = validate(args.output, args.replay, args.freeze)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
