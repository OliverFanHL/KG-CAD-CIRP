"""Conservative partial-qrel envelopes on fixed, replay-verified retrieval orders.

This is an uncertainty analysis of AI reference judgments, not a new retrieval
configuration, a label completion, a point estimate, or human validation.
``self-test`` reads no experiment data. ``evaluate`` requires a published freeze.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations, product
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from cadkg.io import read_csv, read_json, write_csv, write_json, sha256


ROOT = Path("outputs/feature_review_v5")
HISTORICAL = Path("outputs/candidate_scale_v4/evaluation")
KINDS = ("embedding", "descriptor")
SCENARIOS = ("nominal", "conditional_zero", "conditional_two")
METRICS = ("ndcg@5", "precision@5", "strong_precision@5", "recall@20", "ap")
SUBSETS = ("all", "exemplar", "text", "constrained_exemplar",
           "unconstrained_exemplar", "constrained_text")
FOCUSED = ("kg_no_constraints", "kg_no_class", "geometry_class_filter",
           "intent_filter", "rrf_filter", "geometry_intent_filter", "kg_no_geometry")
ROBUST_METHODS = ("kg_full", "geometry_class_filter", "geometry_intent_filter",
                  "intent_filter", "kg_no_class", "rrf_filter", "kg_no_geometry")
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def transformed_set(raw, verdict, scenario="nominal"):
    """Map every admissible *raw* grade, in the original transform order."""
    from cadkg.experiments.intent import score_grade
    require(scenario in SCENARIOS, "Unknown conditional scenario")
    require(verdict in {"pass", "fail", "unknown", "not_requested"}, "Unknown size verdict")
    require(raw is None or (type(raw) is int and raw in (0, 1, 2)), "Invalid raw grade")
    candidates = (0, 1, 2) if raw is None else (raw,)
    return tuple(sorted({score_grade(g, verdict, scenario) for g in candidates}))


def metric_bounds(order, possible, ndcg_k=5, precision_k=5, recall_k=20):
    """Enclose every admissible completion; generally not sharp bounds.

    DCG and IDCG are each coordinatewise monotone in the grades. AP's unnormalised
    numerator sum(hit_i * cumulative_hits_i / i) is coordinatewise monotone in
    binary hits. Bound numerator and denominator separately, then take ratios.
    Their extrema may require different completions, so these intervals can be
    wider than the attainable range. A possibly zero denominator contributes the
    evaluator's zero-relevance value 0, never a dropped query.
    """
    require(ndcg_k > 0 and precision_k > 0 and recall_k > 0, "Positive cutoffs required")
    arrays = []
    for choices in possible:
        values = tuple(sorted(set(choices)))
        require(values and all(type(g) is int and g in (0, 1, 2) for g in values),
                "Admissible sets must be nonempty subsets of integer grades 0..2")
        arrays.append(values)
    n = len(arrays)
    ranked = list(order)
    require(all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in ranked),
            "Ranking index outside cohort")
    require(len(ranked) == len(set(ranked)), "Duplicate ranking index")
    low = np.asarray([v[0] for v in arrays], dtype=int)
    high = np.asarray([v[-1] for v in arrays], dtype=int)
    idx = np.asarray(ranked, dtype=int)
    hit_low = (low > 0).astype(int)
    hit_high = (high > 0).astype(int)
    r_low, r_high = int(hit_low.sum()), int(hit_high.sum())

    def dcg(values):
        return sum((2 ** int(g) - 1) / math.log2(i + 2)
                   for i, g in enumerate(values[:ndcg_k]))

    dcg_low, dcg_high = dcg(low[idx]), dcg(high[idx])
    idcg_low = dcg(sorted(low, reverse=True))
    idcg_high = dcg(sorted(high, reverse=True))
    if idcg_high == 0:
        ndcg = (0.0, 0.0)
    else:
        ndcg = (dcg_low / idcg_high,
                min(1.0, dcg_high / idcg_low) if idcg_low else float(dcg_high > 0))

    top = idx[:precision_k]
    precision = (float(hit_low[top].sum()) / precision_k,
                 float(hit_high[top].sum()) / precision_k)
    strong = (float((low[top] == 2).sum()) / precision_k,
              float((high[top] == 2).sum()) / precision_k)
    top_recall = idx[:recall_k]
    hit_min = int(hit_low[top_recall].sum())
    hit_max = int(hit_high[top_recall].sum())

    def ratio_envelope(num_low, num_high):
        if r_high == 0:
            return 0.0, 0.0
        return num_low / r_high, min(1.0, num_high / max(1, r_low))

    recall = ratio_envelope(hit_min, hit_max)

    def ap_numerator(hits):
        count = 0
        total = 0.0
        for rank, index in enumerate(ranked, 1):
            count += int(hits[index])
            if hits[index]:
                total += count / rank
        return total

    ap = ratio_envelope(ap_numerator(hit_low), ap_numerator(hit_high))
    result = dict(zip(METRICS, (ndcg, precision, strong, recall, ap)))
    for metric, (lower, upper) in result.items():
        require(-TOL <= lower <= upper + TOL and upper <= 1 + TOL,
                f"Invalid {metric} bounds")
    return result


def _exact_metrics(order, grades, nk=5, pk=5, rk=20):
    """Independent pure-Python test oracle; never used to score real partial qrels."""
    relevant = {i for i, grade in enumerate(grades) if grade >= 1}
    numerator = 0.0
    true_seen = 0
    for position, item in enumerate(order, 1):
        if item in relevant:
            true_seen += 1
            numerator += true_seen / position
    ideal = sorted([3 if g == 2 else 1 if g == 1 else 0 for g in grades], reverse=True)
    best = sum(g / math.log(i + 2, 2) for i, g in enumerate(ideal[:nk]))
    retrieved = sum((3 if grades[item] == 2 else 1 if grades[item] == 1 else 0)
                    / math.log(i + 2, 2) for i, item in enumerate(order[:nk]))
    return {
        "ndcg@5": retrieved / best if best else 0.0,
        "precision@5": sum(i in relevant for i in order[:pk]) / pk,
        "strong_precision@5": sum(grades[i] == 2 for i in order[:pk]) / pk,
        "recall@20": sum(i in relevant for i in order[:rk]) / len(relevant) if relevant else 0.0,
        "ap": numerator / len(relevant) if relevant else 0.0,
    }


def self_test(output=None):
    """Exhaust all sets, completions and ordered subsets through three items."""
    choices = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    bounds_cases = completions = point_cases = 0

    def check(sets, order, nk=5, pk=5, rk=20):
        nonlocal bounds_cases, completions, point_cases
        bounds = metric_bounds(order, sets, nk, pk, rk)
        bounds_cases += 1
        for completion in product(*sets):
            exact = _exact_metrics(order, completion, nk, pk, rk)
            completions += 1
            for metric in METRICS:
                lo, hi = bounds[metric]
                require(lo - TOL <= exact[metric] <= hi + TOL,
                        f"Envelope failed: {sets}/{order}/{metric}/{exact[metric]}/{lo}/{hi}")
                if all(len(s) == 1 for s in sets):
                    require(abs(lo - exact[metric]) < TOL and abs(hi - exact[metric]) < TOL,
                            "Complete-qrel envelope did not collapse to the exact metric")
            if all(len(s) == 1 for s in sets):
                point_cases += 1

    for n in range(4):
        orders = [p for length in range(n + 1) for p in permutations(range(n), length)]
        for sets in product(choices, repeat=n):
            for order in orders:
                check(sets, order)
                # Actual truncation and withheld relevant items at small sizes.
                check(sets, order, nk=1, pk=2, rk=1)
    # Exhaust all grade sets and completions for four items at representative
    # full/filtered/empty rankings; no sampled label completions.
    for sets in product(choices, repeat=4):
        for order in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0), ()):
            check(sets, order, nk=2, pk=3, rk=2)
    # Exercise the real top5/top20 cutoffs and uncertain items outside the return.
    for n in (6, 22):
        for uncertain in ((0, 4, n - 1), (1, 5, n - 2)):
            for sets3 in product(choices, repeat=3):
                sets = [(i % 3,) for i in range(n)]
                for i, s in zip(uncertain, sets3):
                    sets[i] = s
                for order in (tuple(range(n)), tuple(reversed(range(n))), tuple(range(0, n, 2))):
                    check(sets, order)
    pair_cases = 0
    # Every pair of ordered subsets through two items, under every admissible
    # set/completion. Compare with the difference of independent exact metrics.
    for n in range(3):
        orders = [p for length in range(n + 1) for p in permutations(range(n), length)]
        for sets in product(choices, repeat=n):
            for order_a, order_b in product(orders, repeat=2):
                a = _flatten(metric_bounds(order_a, sets))
                b = _flatten(metric_bounds(order_b, sets))
                delta = _difference(a, b, order_a == order_b)
                for completion in product(*sets):
                    exact_a = _exact_metrics(order_a, completion)
                    exact_b = _exact_metrics(order_b, completion)
                    for metric in METRICS:
                        exact = exact_a[metric] - exact_b[metric]
                        require(delta[f"{metric}_lower"] - TOL <= exact <= delta[f"{metric}_upper"] + TOL,
                                "Paired difference envelope failed")
                    pair_cases += 1
    transforms = 0
    for raw, verdict, scenario in product((0, 1, 2, None),
                                          ("pass", "fail", "unknown", "not_requested"), SCENARIOS):
        expected = set()
        for grade in (0, 1, 2) if raw is None else (raw,):
            transformed = grade
            if transformed == 1:
                if scenario == "conditional_zero":
                    transformed = 0
                elif scenario == "conditional_two":
                    transformed = 2
            if verdict == "fail":
                transformed = 0
            elif verdict == "unknown" and transformed == 2:
                transformed = 1
            expected.add(transformed)
        require(set(transformed_set(raw, verdict, scenario)) == expected,
                "Existing task/constraint transform mismatch")
        transforms += 1
    from cadkg.experiments.reference import numeric_reference
    size_cases = 0
    query = {"constraints": [{"attribute": "aabb_large_mm", "op": "<=", "unit": "mm", "value": 10}]}
    for extent, expected in ((9.75, "pass"), (9.875, "unknown"), (10., "unknown"),
                             (10.125, "unknown"), (10.25, "fail")):
        measurement = dict(aabb_quarantined=False, faces_missing_mesh=False, surface_large_mm=extent)
        require(numeric_reference(query, measurement, .125) == expected, "Size guard transform failed")
        require(numeric_reference({"constraints": []}, measurement) == "not_requested", "No-constraint transform failed")
        size_cases += 2
        for flag in ("aabb_quarantined", "faces_missing_mesh"):
            require(numeric_reference(query, {**measurement, flag: True}) == "unknown", "Size quarantine transform failed")
            size_cases += 1
    robust_cases = _synthetic_robustness_test()
    report = dict(status="VERIFIED_SYNTHETIC", bounds_cases=bounds_cases,
                  exact_completion_cases=completions, singleton_metric_cases=point_cases,
                  paired_completion_cases=pair_cases, transform_cases=transforms,
                  size_reference_cases=size_cases,
                  synthetic_robustness_completion_cases=robust_cases,
                  scope="Exhaustive finite synthetic containment and singleton equality; no experiment inputs read",
                  human_validation=False)
    if output:
        write_json(output, report)
    return report


def _verify_hashes(mapping):
    for filename, digest in mapping.items():
        require(sha256(filename) == digest, f"Frozen input changed: {filename}")


def _frozen_path(path, mapping):
    resolved = Path(path).resolve()
    require(any(Path(p).resolve() == resolved for p in mapping),
            f"Required input absent from evaluation freeze: {path}")


def _subset(query, name):
    exemplar = bool(query["reference_part_id"])
    constrained = bool(query["constraints"])
    return {"all": True, "exemplar": exemplar, "text": not exemplar,
            "constrained_exemplar": exemplar and constrained,
            "unconstrained_exemplar": exemplar and not constrained,
            "constrained_text": not exemplar and constrained}[name]


def _flatten(bounds):
    return {f"{metric}_{edge}": float(values[i])
            for metric, values in bounds.items() for i, edge in enumerate(("lower", "upper"))}


def _mean_rows(rows):
    require(bool(rows), "Cannot average an empty query/family group")
    return {f"{metric}_{edge}": float(np.mean([r[f"{metric}_{edge}"] for r in rows]))
            for metric in METRICS for edge in ("lower", "upper")}


def _difference(a, b, same_order=False):
    return {f"{m}_{edge}": 0.0 if same_order else (
        a[f"{m}_lower"] - b[f"{m}_upper"] if edge == "lower"
        else a[f"{m}_upper"] - b[f"{m}_lower"])
        for m in METRICS for edge in ("lower", "upper")}


def _robust_identity(case):
    return (case["geometry_kind"], case["query_id"], case["method"], case["robustness"],
            float(case["severity"]), int(case["seed"]))


def robustness_bounds(ranking_root, cfg, queries, cohorts, obs, evidence, core_orders):
    """Score every fixed robustness order, with the case's original size guard."""
    from cadkg.experiments.reference import numeric_reference
    ranking_root = Path(ranking_root)
    cases = read_csv(ranking_root / "robustness_cases.csv")
    by_query = {q["query_id"]: q for q in queries}
    expected = set()
    for kind, q, method in product(KINDS, queries, ROBUST_METHODS):
        for guard in cfg["constraint_guards_mm"]:
            if guard != .2 and q["constraints"]:
                expected.add((kind, q["query_id"], method, "constraint_guard", float(guard), 0))
        for seed in cfg["robustness_seeds"]:
            expected.add((kind, q["query_id"], method, "tie_order", 0., int(seed)))
        for fraction, seed in product(cfg["classification_corruption_fractions"], cfg["robustness_seeds"]):
            expected.add((kind, q["query_id"], method, "classifier_permutation", float(fraction), int(seed)))
    actual = [_robust_identity(c) for c in cases]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "Robustness cases differ from frozen conditions, methods, seeds or queries")
    keys = [c["key"] for c in cases]
    require(len(keys) == len(set(keys)), "Duplicate robustness array key")
    possible_cache, lookup, orders, rows = {}, {}, {}, []
    ids = cohorts["closed"]
    with np.load(ranking_root / "robustness_complete_rankings.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(keys), "Robustness archive/case mapping mismatch")
        for case in cases:
            identity = _robust_identity(case)
            kind, qid, method, condition, severity, seed = identity
            query = by_query[qid]
            guard = float(case["guard_mm"])
            require(guard == (severity if condition == "constraint_guard" else .2),
                    "Robustness size guard differs from original condition")
            cache_key = qid, guard
            if cache_key not in possible_cache:
                raw = [obs[query["scenario_id"], p]["grade"] for p in ids]
                verdicts = [numeric_reference(query, evidence[p], guard) for p in ids]
                possible_cache[cache_key] = ([transformed_set(g, v) for g, v in zip(raw, verdicts)],
                                             sum(g is None for g in raw))
            possible, null_count = possible_cache[cache_key]
            order = archive[case["key"]]
            require(order.ndim == 1 and order.dtype.kind in "iu", "Invalid robustness order type")
            if condition == "constraint_guard":
                require(np.array_equal(order, core_orders[kind, qid, "closed", method]),
                        "Size-guard diagnostic changed its nominal retrieval order")
            row = dict(geometry_kind=kind, method=method, query_id=qid,
                       family_id=query["scenario_id"], cohort="closed", grading_scenario="nominal",
                       robustness=condition, severity=severity, seed=seed, guard_mm=guard,
                       raw_null_count=null_count, transformed_uncertain_count=sum(len(s) > 1 for s in possible),
                       **_flatten(metric_bounds(order, possible)))
            rows.append(row)
            lookup[identity] = row
            orders[identity] = order.copy()

    grouped = defaultdict(list)
    for row in rows:
        for subset in SUBSETS:
            if _subset(by_query[row["query_id"]], subset):
                grouped[row["geometry_kind"], row["method"], row["robustness"], row["severity"], subset].append(row)
    summary = []
    for (kind, method, condition, severity, subset), members in sorted(grouped.items()):
        family = defaultdict(list)
        for row in members:
            family[row["family_id"]].append(row)
        summary.append(dict(geometry_kind=kind, method=method, robustness=condition, severity=severity,
                            subset=subset, families=len(family), query_seed_cases=len(members),
                            unique_queries=len({r["query_id"] for r in members}),
                            seeds=len({r["seed"] for r in members}), averaging="equal_family_after_query_and_seed_mean",
                            **_mean_rows([_mean_rows(v) for v in family.values()])))
    paired_groups = defaultdict(list)
    for identity, row in lookup.items():
        kind, qid, method, condition, severity, seed = identity
        if method == "kg_full":
            continue
        full_key = kind, qid, "kg_full", condition, severity, seed
        a = lookup[full_key]
        same = np.array_equal(orders[full_key], orders[identity])
        pair = dict(family_id=row["family_id"], query_id=qid, **_difference(a, row, same))
        for subset in SUBSETS:
            if _subset(by_query[qid], subset):
                paired_groups[kind, method, condition, severity, subset].append(pair)
    comparison = []
    for (kind, control, condition, severity, subset), members in sorted(paired_groups.items()):
        family = defaultdict(list)
        for row in members:
            family[row["family_id"]].append(row)
        comparison.append(dict(geometry_kind=kind, proposed_method="kg_full", comparator=control,
                               robustness=condition, severity=severity, subset=subset,
                               families=len(family), query_seed_cases=len(members),
                               significance_test="withheld; no point p-value or confidence interval",
                               **_mean_rows([_mean_rows(v) for v in family.values()])))
    return rows, summary, comparison


def _synthetic_robustness_test():
    """Exercise the file/case/guard/aggregation adapter using only synthetic data."""
    ids = ["synthetic_a", "synthetic_b", "synthetic_c"]
    queries = [dict(query_id="synthetic_text", scenario_id="role_a", reference_part_id=None, constraints=[]),
               dict(query_id="synthetic_size", scenario_id="role_b", reference_part_id="synthetic_ref",
                    constraints=[dict(attribute="aabb_large_mm", op="<=", unit="mm", value=10)])]
    cfg = dict(constraint_guards_mm=[.125, .2, .5], robustness_seeds=[17, 31],
               classification_corruption_fractions=[.1, .3])
    evidence = {p: dict(surface_large_mm=x, aabb_quarantined=False, faces_missing_mesh=False)
                for p, x in zip(ids, [8., 10., 12.])}
    obs = {(f, p): {"grade": g} for f, grades in (("role_a", [None, 2, 0]), ("role_b", [1, None, 2]))
           for p, g in zip(ids, grades)}
    core = {(kind, q["query_id"], "closed", method): np.asarray([0, 1, 2])
            for kind, q, method in product(KINDS, queries, ROBUST_METHODS)}
    cases, arrays, rank_by_key = [], {}, {}
    for kind, q, method in product(KINDS, queries, ROBUST_METHODS):
        settings = [("constraint_guard", g, 0, g) for g in (.125, .5) if q["constraints"]]
        settings += [("tie_order", 0., s, .2) for s in cfg["robustness_seeds"]]
        settings += [("classifier_permutation", f, s, .2) for f, s in
                     product(cfg["classification_corruption_fractions"], cfg["robustness_seeds"])]
        for condition, severity, seed, guard in settings:
            key = f"synthetic_{len(cases)}__order"
            order = [0, 1, 2] if condition == "constraint_guard" else [2, 0, 1] if seed == 17 else [1, 2]
            arrays[key] = np.asarray(order, dtype=np.int32)
            rank_by_key[kind, q["query_id"], method, condition, float(severity), seed] = order
            cases.append(dict(geometry_kind=kind, query_id=q["query_id"], method=method, robustness=condition,
                              severity=severity, seed=seed, key=key, guard_mm=guard))
    checked = 0
    with tempfile.TemporaryDirectory(prefix="partial_qrel_synthetic_", dir=ROOT.resolve()) as temp:
        folder = Path(temp)
        require(folder.resolve().is_relative_to(ROOT.resolve()), "Synthetic cleanup path outside its workspace")
        write_csv(folder / "robustness_cases.csv", cases)
        np.savez_compressed(folder / "robustness_complete_rankings.npz", **arrays)
        rows, summary, comparisons = robustness_bounds(folder, cfg, queries, {"closed": ids}, obs, evidence, core)
        require(len(rows) == 196 and summary and comparisons, "Synthetic robustness coverage mismatch")
        qmap = {q["query_id"]: q for q in queries}
        for row in rows:
            q = qmap[row["query_id"]]
            raw = [obs[q["scenario_id"], p]["grade"] for p in ids]
            choices = [(0, 1, 2) if g is None else (g,) for g in raw]
            identity = (row["geometry_kind"], row["query_id"], row["method"], row["robustness"], row["severity"], row["seed"])
            for completion in product(*choices):
                grades = list(completion)
                if q["constraints"]:
                    for i, extent in enumerate((8., 10., 12.)):
                        if extent > 10 + row["guard_mm"]:
                            grades[i] = 0
                        elif 10 - row["guard_mm"] <= extent <= 10 + row["guard_mm"]:
                            grades[i] = 1 if grades[i] == 2 else grades[i]
                exact = _exact_metrics(rank_by_key[identity], grades)
                for metric in METRICS:
                    require(row[f"{metric}_lower"] - TOL <= exact[metric] <= row[f"{metric}_upper"] + TOL,
                            "Synthetic robustness adapter failed exact completion containment")
                checked += 1
    return checked


def evaluate(config_path, historical=HISTORICAL, output=None, rankings=None):
    """Read published, frozen inputs and independently rerun/replay-checked orders."""
    from cadkg.experiments.reference import numeric_reference
    from scripts.run_verified_evaluation import METHODS, evaluation_protocol

    freeze = read_json(ROOT / "evaluation_freeze.json")["input_sha256"]
    # This check occurs before opening any new annotations or historical arrays.
    _verify_hashes(freeze)
    historical = Path(historical)
    ranking_root = Path(rankings) if rankings else ROOT / "evaluation"
    historical_files = [historical / name for name in
                        ("complete_rankings.npz", "effective_protocol.json", "cohorts.json", "run_manifest.json")]
    for path in [Path(__file__), Path(config_path), Path("configs/candidate_scale_v4.json"),
                 Path("cadkg/experiments/intent.py"), Path("cadkg/experiments/reference.py"),
                 Path("scripts/rerun_feature_rankings.py"),
                 ROOT / "consumed_evidence_freeze.json", *historical_files]:
        _frozen_path(path, freeze)
    consumed = read_json(ROOT / "consumed_evidence_freeze.json")["input_sha256"]
    _verify_hashes(consumed)
    ranking_validation_path = ranking_root / "ranking_validation.json"
    ranking_validation = read_json(ranking_validation_path)
    require(ranking_validation.get("status") == "VERIFIED", "Fresh retrieval/ranking replay not verified")
    require(ranking_validation.get("replay_verified") is True, "Second ranking replay is required")
    file_hashes = ranking_validation["files_sha256"]
    require(all(Path(name).name == name for name in file_hashes), "Ranking hash keys must be basenames")
    ranking_hashes = {str(ranking_root / name): digest for name, digest in file_hashes.items()}
    for path in [ranking_root / "complete_rankings.npz", ranking_root / "robustness_complete_rankings.npz",
                 ranking_root / "robustness_cases.csv", ranking_root / "effective_protocol.json",
                 ranking_root / "cohorts.json"]:
        _frozen_path(path, ranking_hashes)
    _verify_hashes(ranking_hashes)
    require(sha256(historical / "complete_rankings.npz") == ranking_validation["historical_archive_sha256"],
            "Ranking validation references a different historical archive")
    ranking_validation_sha = sha256(ranking_validation_path)
    cfg = read_json(config_path)
    base = read_json("configs/candidate_scale_v4.json")
    allowed = {"version", "annotation_path", "annotation_audit_path", "evaluation_plan", "evidence_root", "output", "notes"}
    require(cfg.keys() == base.keys() and all(value == base[key] for key, value in cfg.items() if key not in allowed),
            "Method/model/protocol configuration differs from v4")
    _frozen_path(cfg["annotation_path"], freeze)
    _frozen_path(cfg["annotation_audit_path"], freeze)
    run = read_json(historical / "run_manifest.json")
    require(run["config"] == base, "Historical run configuration does not match v4 config")
    _verify_hashes(run["code_sha256"])
    # Check shared serving/model/protocol inputs, excluding only explicitly
    # changed version-specific release inputs. Old annotations are hashed for
    # cohort continuity, never used as v5 relevance labels.
    skipped = {Path("configs/candidate_scale_v4.json").resolve(),
               Path(base["annotation_path"]).resolve(), Path(base["annotation_audit_path"]).resolve(),
               Path(base.get("evaluation_plan", "docs/VERIFIED_EVALUATION_PLAN.md")).resolve()}
    shared = {p: d for p, d in run["input_sha256"].items() if Path(p).resolve() not in skipped}
    _verify_hashes(shared)
    protocol = read_json(historical / "effective_protocol.json")
    regenerated = evaluation_protocol(cfg, read_json(Path(cfg["source_run"]) / "protocol.json"))
    require(protocol == regenerated, "Historical query/protocol mapping differs from frozen current inputs")
    queries = protocol["queries"]
    query_by_id = {q["query_id"]: q for q in queries}
    families = {q["scenario_id"] for q in queries}
    require(len(queries) == len(query_by_id) == 110 and len(families) == 11,
            "Fixed 110-query/11-family matrix required")
    require(len(METHODS) == len(set(METHODS)) == 14, "Fixed fourteen methods required")
    require(len(cfg["primary_comparators"]) == len(set(cfg["primary_comparators"])) == 5,
            "Five fixed primary controls required")
    require(set(cfg["primary_comparators"]) <= set(METHODS) - {"kg_full"}, "Unknown primary control")
    cohorts = read_json(historical / "cohorts.json")
    require(read_json(ranking_root / "cohorts.json") == cohorts
            and read_json(ranking_root / "effective_protocol.json") == protocol,
            "Fresh rank mapping/protocol differs from historical mapping")
    annotation = read_json(cfg["annotation_path"])
    require(annotation.get("source") == "ai_reference" and annotation.get("human_expert_reviewed") is False,
            "AI reference provenance required")
    require(annotation["scope_sha256"] == protocol["scope_sha256"], "Annotation scope mismatch")
    expected_annotation_query_hash = protocol.get("annotation_query_sha256", protocol["query_sha256"])
    require(annotation["query_sha256"] == expected_annotation_query_hash, "Annotation original-query hash mismatch")
    records = annotation["judgments"]
    obs = {(r["family_id"], r["part_id"]): r for r in records}
    ids = sorted({r["part_id"] for r in records})
    require(len(ids) == 1816 and len(records) == len(obs) == 1816 * 11,
            "Complete 1816-by-11 record matrix required, including explicit nulls")
    require(set(obs) == {(f, p) for f in families for p in ids}, "Missing or extraneous role/part pair")
    require(cohorts["closed"] == ids and set(protocol["gallery"]) == set(ids), "Historical gallery mapping mismatch")
    for name, selected in cohorts.items():
        require(len(selected) == len(set(selected)) and set(selected) <= set(ids),
                f"Invalid historical cohort mapping: {name}")
    evidence = {}
    evidence_paths = {}
    for r in records:
        raw = r["grade"]
        require(raw is None or (type(raw) is int and raw in (0, 1, 2)), "Invalid annotation grade")
        require(r.get("source") == "ai_reference" and r.get("review_status") == "visually_reviewed"
                and r.get("human_expert_reviewed") is False,
                "Unreviewed or misidentified reference judgment")
        if raw is None:
            require(r.get("decision") == "cannot_assess", "Null must remain cannot-assess")
        path = Path(cfg["evidence_root"]) / (r["inspection_id"] + ".json")
        _frozen_path(path, consumed)
        if r["part_id"] not in evidence:
            evidence[r["part_id"]] = read_json(path)
            evidence_paths[r["part_id"]] = path.resolve()
        require(evidence_paths[r["part_id"]] == path.resolve(), "Inconsistent evidence path across a CAD's roles")
        e = evidence[r["part_id"]]
        require(e["part_id"] == r["part_id"] and e["source_sha256"] == r["source_sha256"],
                "Numerical evidence source mismatch")

    out = Path(output) if output else ROOT / "partial_evaluation"
    require(not out.exists(), "Refuse to overwrite an existing partial evaluation; use a fresh output path")
    synthetic = self_test()
    rows = []
    lookup = {}
    orders = {}
    expected_keys = {f"{kind}__{q['query_id']}__{cohort}__{method}__{suffix}"
                     for kind, q, cohort, method, suffix in product(KINDS, queries, cohorts, METHODS, ("order", "score"))}
    array_count = 0
    with np.load(ranking_root / "complete_rankings.npz", allow_pickle=False) as archive, \
            np.load(historical / "complete_rankings.npz", allow_pickle=False) as prior_archive:
        require(set(archive.files) == expected_keys, "Historical array matrix differs from all fixed methods/queries/cohorts")
        require(ranking_validation.get("replay_arrays_verified") == len(expected_keys),
                "Replay did not verify every core order and score array")
        require(set(prior_archive.files) == set(archive.files)
                and all(np.array_equal(archive[k], prior_archive[k]) for k in archive.files),
                "Fresh core orders/scores differ from historical v4 arrays")
        for query in queries:
            qid, fid = query["query_id"], query["scenario_id"]
            for cohort, selected in cohorts.items():
                verdicts = [numeric_reference(query, evidence[p], .2) for p in selected]
                raw = [obs[fid, p]["grade"] for p in selected]
                scenarios = SCENARIOS if cohort == "closed" else ("nominal",)
                possible_by_scenario = {scenario: [transformed_set(g, v, scenario) for g, v in zip(raw, verdicts)]
                                        for scenario in scenarios}
                for kind, method in product(KINDS, METHODS):
                    key = f"{kind}__{qid}__{cohort}__{method}"
                    order, scores = archive[key + "__order"], archive[key + "__score"]
                    require(order.ndim == scores.ndim == 1 and order.dtype.kind in "iu"
                            and len(order) == len(scores) and np.all(np.isfinite(scores)), "Invalid historical arrays")
                    orders[kind, qid, cohort, method] = order.copy()
                    array_count += 2
                    for scenario in scenarios:
                        possible = possible_by_scenario[scenario]
                        bounds = metric_bounds(order, possible)
                        row = dict(cohort=cohort, geometry_kind=kind, method=method, query_id=qid,
                                   family_id=fid, grading_scenario=scenario,
                                   has_exemplar=bool(query["reference_part_id"]), has_constraint=bool(query["constraints"]),
                                   raw_null_count=sum(g is None for g in raw),
                                   transformed_uncertain_count=sum(len(s) > 1 for s in possible),
                                   relevant_total_lower=sum(min(s) > 0 for s in possible),
                                   relevant_total_upper=sum(max(s) > 0 for s in possible),
                                   returned=min(5, len(order)), **_flatten(bounds))
                        rows.append(row)
                        lookup[kind, qid, cohort, method, scenario] = row

    robust_rows, robust_summary, robust_comparison = robustness_bounds(
        ranking_root, cfg, queries, cohorts, obs, evidence, orders)
    require(ranking_validation.get("replay_robustness_orders_verified") == len(robust_rows),
            "Replay did not verify every robustness full order")

    family_rows, summary = [], []
    for cohort, kind, method, scenario, subset in product(cohorts, KINDS, METHODS, SCENARIOS, SUBSETS):
        if cohort != "closed" and scenario != "nominal":
            continue
        selected = [r for r in rows if (r["cohort"], r["geometry_kind"], r["method"], r["grading_scenario"])
                    == (cohort, kind, method, scenario) and _subset(query_by_id[r["query_id"]], subset)]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["family_id"]].append(row)
        local = []
        for fid, members in sorted(grouped.items()):
            rec = dict(cohort=cohort, geometry_kind=kind, method=method, grading_scenario=scenario,
                       subset=subset, family_id=fid, queries=len(members), **_mean_rows(members))
            family_rows.append(rec)
            local.append(rec)
        if local:
            summary.append(dict(cohort=cohort, geometry_kind=kind, method=method, grading_scenario=scenario,
                                subset=subset, families=len(local), queries=len(selected),
                                averaging="equal_family_after_query_mean", **_mean_rows(local)))

    # Build paired envelopes on shared candidate grades. Subtraction is safe
    # despite dependence; identical rankings use the sharper exact identity 0.
    paired = []
    for cohort, kind, scenario, control in product(cohorts, KINDS, SCENARIOS, [m for m in METHODS if m != "kg_full"]):
        if cohort != "closed" and scenario != "nominal":
            continue
        for q in queries:
            qid = q["query_id"]
            a = lookup[kind, qid, cohort, "kg_full", scenario]
            b = lookup[kind, qid, cohort, control, scenario]
            same = np.array_equal(orders[kind, qid, cohort, "kg_full"], orders[kind, qid, cohort, control])
            paired.append(dict(cohort=cohort, geometry_kind=kind, comparator=control,
                               grading_scenario=scenario, query_id=qid, family_id=q["scenario_id"],
                               identical_order=same, **_difference(a, b, same)))

    pair_summary = []
    for cohort, kind, scenario, control, subset in product(cohorts, KINDS, SCENARIOS,
                                                          [m for m in METHODS if m != "kg_full"], SUBSETS):
        if cohort != "closed" and scenario != "nominal":
            continue
        selected = [r for r in paired if (r["cohort"], r["geometry_kind"], r["grading_scenario"], r["comparator"])
                    == (cohort, kind, scenario, control) and _subset(query_by_id[r["query_id"]], subset)]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["family_id"]].append(row)
        if grouped:
            means = [_mean_rows(members) for members in grouped.values()]
            pair_summary.append(dict(cohort=cohort, geometry_kind=kind, proposed_method="kg_full",
                                     comparator=control, grading_scenario=scenario, subset=subset,
                                     queries=len(selected), families=len(means), **_mean_rows(means)))

    comparisons = []
    for name, subset, controls, count in (("primary", "exemplar", cfg["primary_comparators"], 77),
                                          ("focused", "constrained_exemplar", FOCUSED, 58)):
        for kind, control in product(KINDS, controls):
            rec = next(r for r in pair_summary if (r["cohort"], r["geometry_kind"], r["comparator"],
                                                   r["grading_scenario"], r["subset"])
                       == ("closed", kind, control, "nominal", subset))
            require(rec["queries"] == count and rec["families"] == 11, "Fixed comparison query/family coverage mismatch")
            comparisons.append(dict(comparison_family=name, **rec))
    cross = defaultdict(list)
    for q in queries:
        if not q["reference_part_id"]:
            continue
        qid = q["query_id"]
        a = lookup["descriptor", qid, "closed", "kg_full", "nominal"]
        b = lookup["embedding", qid, "closed", "kg_full", "nominal"]
        same = np.array_equal(orders["descriptor", qid, "closed", "kg_full"],
                              orders["embedding", qid, "closed", "kg_full"])
        cross[q["scenario_id"]].append(_difference(a, b, same))
    require(len(cross) == 11 and sum(map(len, cross.values())) == 77, "Primary cross-representation pairing mismatch")
    comparisons.append(dict(comparison_family="primary", cohort="closed", geometry_kind="descriptor_minus_embedding",
                            proposed_method="kg_full", comparator="embedding:kg_full", grading_scenario="nominal",
                            subset="exemplar", queries=77, families=11,
                            **_mean_rows([_mean_rows(v) for v in cross.values()])))
    for r in comparisons:
        lo, hi = r["ndcg@5_lower"], r["ndcg@5_upper"]
        r.update(annotation_effect_interval="conservative_outer_envelope",
                 ndcg_difference_identified=abs(hi - lo) <= TOL,
                 ndcg_sign_for_all_admissible_completions="positive" if lo > 0 else "negative" if hi < 0 else "not_separated",
                 significance_test="withheld; no point p-value or confidence interval")
    require(sum(r["comparison_family"] == "primary" for r in comparisons) == 11
            and sum(r["comparison_family"] == "focused" for r in comparisons) == 14,
            "Comparison family size changed")

    # Recheck all seals and the source archive before committing output.
    _verify_hashes(freeze)
    _verify_hashes(consumed)
    _verify_hashes(ranking_hashes)
    require(sha256(ranking_validation_path) == ranking_validation_sha, "Ranking validation record changed")
    out.mkdir(parents=True)
    shutil.copyfile(ranking_root / "complete_rankings.npz", out / "complete_rankings.npz")
    require(sha256(out / "complete_rankings.npz") == sha256(ranking_root / "complete_rankings.npz"),
            "Fresh rankings were changed")
    with np.load(historical / "complete_rankings.npz", allow_pickle=False) as source, \
            np.load(out / "complete_rankings.npz", allow_pickle=False) as copy:
        require(set(source.files) == set(copy.files) and all(np.array_equal(source[k], copy[k]) for k in source.files),
                "Historical order/score arrays differ")
    write_csv(out / "per_query_bounds.csv", rows)
    write_csv(out / "family_bounds.csv", family_rows)
    write_csv(out / "summary_bounds.csv", summary)
    write_csv(out / "paired_query_bounds.csv", paired)
    write_csv(out / "full_minus_control_bounds.csv", pair_summary)
    write_csv(out / "comparison_bounds.csv", comparisons)
    write_csv(out / "robustness_per_query_bounds.csv", robust_rows)
    write_csv(out / "robustness_summary_bounds.csv", robust_summary)
    write_csv(out / "robustness_comparison_bounds.csv", robust_comparison)
    write_json(out / "effective_protocol.json", protocol)
    write_json(out / "cohorts.json", cohorts)
    write_json(out / "synthetic_validation.json", synthetic)
    manifest = dict(status="VERIFIED_CONSERVATIVE_PARTIAL_QREL_ENVELOPES", source="ai_reference",
                    human_expert_reviewed=False, raw_null_pairs=sum(r["grade"] is None for r in records),
                    candidate_cads=len(ids), queries=len(queries), families=len(families), methods=list(METHODS),
                    representations=list(KINDS), cohorts={k: len(v) for k, v in cohorts.items()},
                    query_metric_cases=len(rows), historical_arrays_unchanged=array_count,
                    robustness_metric_cases=len(robust_rows),
                    historical_archive_sha256=sha256(historical / "complete_rankings.npz"),
                    fresh_archive_sha256=sha256(ranking_root / "complete_rankings.npz"),
                    copied_archive_sha256=sha256(out / "complete_rankings.npz"),
                    shared_historical_model_protocol_input_hashes_verified=len(shared),
                    unchanged_core_code_hashes=run["code_sha256"],
                    input_sha256={**freeze, **consumed, **ranking_hashes,
                                  str(ranking_validation_path): ranking_validation_sha},
                    code_sha256={str(Path(__file__)): sha256(__file__)},
                    output_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
                    significance_tests_performed=False, label_completions_published=False,
                    bounds="Conservative outer envelopes; extrema need not be jointly attainable across metrics, methods, queries or families",
                    zero_relevance="Metric zero retained in denominator of every query/family mean",
                    ranking_provenance="Fresh orders/scores from separately rerun and replay-validated fixed methods; every core array equals historical v4",
                    limitations=["Annotation uncertainty envelopes are not sampling confidence intervals or human reliability.",
                                 "All admissible raw grades 0..2 retained for each null; conditional and size transforms applied per member.",
                                 "No point estimate, selected favorable completion, or p-value is reported.",
                                 "Cross-query and cross-method dependence may make averaged/subtracted envelopes loose."])
    write_json(out / "run_manifest.json", manifest)
    return {k: manifest[k] for k in ("status", "raw_null_pairs", "query_metric_cases", "historical_arrays_unchanged")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "evaluate"))
    parser.add_argument("--config", default="configs/feature_review_v5.json")
    parser.add_argument("--historical", default=str(HISTORICAL))
    parser.add_argument("--rankings", default=str(ROOT / "evaluation"), help="Fresh rerun/replay-verified ranking directory")
    parser.add_argument("--output", help="Fresh evaluation folder, or synthetic-validation JSON for self-test")
    args = parser.parse_args()
    result = self_test(args.output) if args.action == "self-test" else evaluate(
        args.config, args.historical, args.output, args.rankings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
