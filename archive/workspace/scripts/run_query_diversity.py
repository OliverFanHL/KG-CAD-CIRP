"""Frozen query-diversity experiment with seven unchanged retrieval recipes.

New query construction/model selection belongs to the separately sealed design.
This runner evaluates existing AI-reference labels using conservative partial-qrel
envelopes. It does not tune, complete labels, or perform significance tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import io
import json
from itertools import product
from pathlib import Path
import shutil
import zipfile

import numpy as np

from cadkg.bundle import load_bundle
from cadkg.catalog import CLASSES
from cadkg.experiments.full import common_gallery, retrieval_config
from cadkg.experiments.reference import numeric_reference
from cadkg.io import object_hash, read_csv, read_json, sha256, write_csv, write_json
from cadkg.query import validate_query
from cadkg.reasoning import check_constraints
from cadkg.retrieval import Retriever, cosine
from scripts.partial_feature_review import require, _verify_hashes, KINDS, SCENARIOS
from scripts.run_verified_evaluation import RECIPES, near_duplicate_gallery, rank_components, stable_seed
from scripts.supplement_partial_metrics import (
    grade_limits, metric_names, prepare_limits, rank_bounds, random_bounds,
    self_test as metric_self_test, summaries,
)


ROOT = Path("outputs/query_diversity_v6")
METHODS = ("kg_full", "geometry_class_filter", "geometry_intent_filter", "kg_no_geometry",
           "kg_no_constraints", "kg_no_class", "rrf_filter")
SERVING_METHODS = ("kg_full", "kg_no_geometry", "kg_no_constraints", "kg_no_class")
OLD = Path("outputs/feature_review_v5")
CODE_DEPENDENCIES = (
    "cadkg/io.py", "cadkg/bundle.py", "cadkg/catalog.py", "cadkg/retrieval.py",
    "cadkg/reasoning.py", "cadkg/query.py", "cadkg/experiments/full.py",
    "cadkg/experiments/intent.py", "cadkg/experiments/reference.py",
    "scripts/run_verified_evaluation.py", "scripts/partial_feature_review.py",
    "scripts/supplement_partial_metrics.py",
)


class ArrayArchive:
    """Stream deterministic NPZ members instead of retaining every ranking in RAM."""
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6)
        self.keys = set()

    def add(self, key, value):
        require(key not in self.keys, f"Duplicate array key: {key}")
        self.keys.add(key)
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(value), allow_pickle=False)
        info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        self.archive.writestr(info, buffer.getvalue(), compresslevel=6)

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _config_value(cfg, canonical, alias):
    require(canonical in cfg or alias in cfg, f"Missing configuration: {canonical}")
    if canonical in cfg and alias in cfg:
        require(cfg[canonical] == cfg[alias], f"Conflicting configuration aliases: {canonical}")
    return cfg[canonical] if canonical in cfg else cfg[alias]


def _paths(cfg):
    return dict(
        old_config=Path(cfg.get("continuity_config_path", "configs/feature_review_v5.json")),
        old_protocol=Path(cfg.get("continuity_protocol_path", OLD / "evaluation/effective_protocol.json")),
        old_cohorts=Path(cfg.get("continuity_cohorts_path", OLD / "evaluation/cohorts.json")),
        old_rankings=Path(cfg.get("continuity_rankings_path", OLD / "evaluation/complete_rankings.npz")),
        old_metrics=Path(cfg.get("continuity_metrics_path", OLD / "supplemental_evaluation/per_query_bounds.csv")),
        old_robust=Path(cfg.get("continuity_robustness_path", OLD / "evaluation/robustness_complete_rankings.npz")),
        old_robust_cases=Path(cfg.get("continuity_robustness_cases_path", OLD / "evaluation/robustness_cases.csv")),
        old_robust_metrics=Path(cfg.get("continuity_robustness_metrics_path", OLD / "supplemental_evaluation/robustness_per_query_bounds.csv")),
        evidence_freeze=Path(cfg.get("evidence_freeze_path", OLD / "consumed_evidence_freeze.json")),
        annotation_freeze=Path(cfg.get("annotation_freeze_path", OLD / "evaluation_freeze.json")),
    )


def _flat(bounds):
    return {f"{metric}_{edge}": float(values[i]) for metric, values in bounds.items()
            for i, edge in enumerate(("lower", "upper"))}


def _metadata(path, queries):
    if Path(path).suffix.lower() == ".csv":
        rows = read_csv(path)
    else:
        data = read_json(path)
        rows = data if isinstance(data, list) else data.get("queries", data.get("rows"))
        require(isinstance(rows, list), "Query design manifest needs per-query rows")
    by_id = {r["query_id"]: r for r in rows}
    require(len(by_id) == len(rows) and set(by_id) == {q["query_id"] for q in queries},
            "Query manifest must cover every query exactly once")
    query_by_id = {q["query_id"]: q for q in queries}
    reserved = {"cohort", "geometry_kind", "method", "grading_scenario", "robustness",
                "severity", "seed", "guard_mm", "relevant_total_lower", "relevant_total_upper"}
    for row in rows:
        require(not (set(row) & reserved), "Query metadata collides with evaluation columns")
        require(row.get("origin") in {"legacy110", "new_exemplar", "new_text"}, "Unknown query origin")
        if "family_id" in row:
            require(row["family_id"] == query_by_id[row["query_id"]]["scenario_id"], "Query metadata family mismatch")
    return {qid: {k: v for k, v in row.items() if k not in {"query_id", "family_id"}} for qid, row in by_id.items()}


def _metric_continuity(row, previous, names, label):
    for metric in names:
        for edge in ("lower", "upper"):
            key = f"{metric}_{edge}"
            require(row[key] == float(previous[key]), f"Metric continuity mismatch {label}/{key}")


def _collapse_rows(kind, q, cohort, local):
    groups = []
    remaining = set(METHODS)
    while remaining:
        first = next(m for m in METHODS if m in remaining)
        same = [m for m in METHODS if m in remaining and np.array_equal(local[first][0], local[m][0])]
        remaining -= set(same)
        if len(same) > 1:
            groups.append(dict(geometry_kind=kind, query_id=q["query_id"], cohort=cohort,
                               has_exemplar=bool(q["reference_part_id"]), has_constraint=bool(q["constraints"]),
                               methods="|".join(same), returned=len(local[first][0]),
                               scores_also_equal=all(np.array_equal(local[first][1], local[m][1]) for m in same),
                               interpretation="Observed identical rankings retained as all separate prespecified rows"))
    return groups


def self_test():
    report = metric_self_test()
    # Check the writer without touching an experiment/data path, using BytesIO.
    # np.save/np.load equality also covers empty filtered rankings.
    for arr in (np.asarray([], dtype=np.int32), np.asarray([2, 0, 1], dtype=np.int32), np.asarray([.7, .5])):
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        buffer.seek(0)
        require(np.array_equal(arr, np.load(buffer, allow_pickle=False)), "Array serialization changed values")
    serialized = []
    for _ in range(2):
        buffer = io.BytesIO()
        with ArrayArchive(buffer) as writer:
            writer.add("test__order", np.asarray([2, 0, 1], dtype=np.int32))
            writer.add("empty__score", np.asarray([], dtype=float))
        serialized.append(buffer.getvalue())
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as copy:
            require(np.array_equal(copy["test__order"], [2, 0, 1]) and len(copy["empty__score"]) == 0,
                    "Streaming NPZ changed an array")
    require(serialized[0] == serialized[1], "NPZ serialization is not deterministic")
    report.update(query_diversity_method_count=len(METHODS), new_retrieval_configuration=False)
    return report


def run(config_path="configs/query_diversity_v6.json", output=None):
    cfg = read_json(config_path)  # The design config itself is not a metric result.
    freeze_path = Path(cfg.get("freeze_path", ROOT / "experiment_freeze.json"))
    seal = read_json(freeze_path)
    frozen = seal["input_sha256"]
    _verify_hashes(frozen)  # Before annotations, new grades, or baseline metrics.
    frozen_paths = {Path(p).resolve() for p in frozen}

    def sealed(path):
        require(Path(path).resolve() in frozen_paths, f"Experiment input absent from freeze: {path}")

    paths = _paths(cfg)
    source = Path(cfg["source_run"])
    for path in [Path(__file__), Path(config_path), cfg["annotation_path"], cfg["queries_path"],
                 cfg["query_manifest_path"], cfg["cohorts_path"], *paths.values(),
                 source / "protocol.json", source / "bundle/catalog.json", source / "bundle/knowledge.json",
                 source / "bundle/graph.json", source / "bundle/bundle_manifest.json", *CODE_DEPENDENCIES]:
        sealed(path)
    require(tuple(cfg["methods"]) == METHODS, "Exactly the seven prespecified methods, in the fixed order, are required")
    require(tuple(cfg["cutoffs"]) == (5, 10, 20), "Preserve the existing fixed cutoffs")
    fractions = _config_value(cfg, "classification_corruption_fractions", "classifierfractions")
    guards = _config_value(cfg, "constraint_guards_mm", "guards")
    threshold = _config_value(cfg, "near_duplicate_threshold", "near_dup")
    old_cfg = read_json(paths["old_config"])
    for key, value in (("seed", cfg["seed"]), ("rrf_constant", cfg["rrf_constant"]),
                       ("cutoffs", cfg["cutoffs"]), ("robustness_seeds", cfg["robustness_seeds"]),
                       ("classification_corruption_fractions", fractions), ("constraint_guards_mm", guards),
                       ("near_duplicate_threshold", threshold)):
        require(old_cfg[key] == value, f"Prespecified existing setting changed: {key}")
    require(cfg["source_run"] == old_cfg["source_run"] and cfg["annotation_path"] == old_cfg["annotation_path"]
            and cfg["evidence_root"] == old_cfg["evidence_root"], "Model sources or reference labels changed")
    _verify_hashes(read_json(paths["annotation_freeze"])["input_sha256"])
    consumed = read_json(paths["evidence_freeze"])["input_sha256"]
    _verify_hashes(consumed)
    consumed_paths = {Path(p).resolve() for p in consumed}
    frozen_digest = sha256(freeze_path)
    out = Path(output or cfg["output"])
    stage = out.with_name(out.name + ".inprogress")
    require(out.resolve().is_relative_to(ROOT.resolve()) and stage.resolve().parent == out.resolve().parent,
            "Run output must stay inside this experiment's output root")
    require(not out.exists() and not stage.exists(), "Refuse to overwrite a completed or interrupted run")

    source_protocol = read_json(source / "protocol.json")
    previous_protocol = read_json(paths["old_protocol"])
    require(source_protocol["settings"] == previous_protocol["settings"]
            and source_protocol["families"] == previous_protocol["families"], "Previous protocol/model settings changed")
    raw_queries = read_json(cfg["queries_path"])
    queries = raw_queries if isinstance(raw_queries, list) else raw_queries["queries"]
    query_ids = {q["query_id"] for q in queries}
    require(len(query_ids) == len(queries) and all("__" not in qid for qid in query_ids), "Duplicate or ambiguous query IDs")
    metadata = _metadata(cfg["query_manifest_path"], queries)
    families = {f["family_id"]: f for f in source_protocol["families"]}
    old_queries = {q["query_id"]: q for q in previous_protocol["queries"]}
    require(len(old_queries) == 110 and set(old_queries) <= query_ids, "Preserve every original110 query")
    semantic_fields = ("query_id", "scenario_id", "reference_part_id", "function_ids", "category_ids", "constraints", "split")
    for q in queries:
        qid = q["query_id"]
        if qid in old_queries:
            require(all(q.get(k, "test" if k == "split" else None) == old_queries[qid].get(k, "test" if k == "split" else None)
                        for k in semantic_fields), f"Historical query semantics changed: {qid}")
            require(metadata[qid]["origin"] == "legacy110", "Historical query origin changed")
        else:
            require(metadata[qid]["origin"] != "legacy110", "New query mislabeled legacy")

    catalog, knowledge, graph = load_bundle(source / "bundle")
    parts = {p["part_id"]: p for p in catalog["parts"]}
    for q in queries:
        validate_query(q, knowledge)
        require(q["scenario_id"] in families and q["function_ids"] == [families[q["scenario_id"]]["function_id"]]
                and not q["category_ids"], "Query changes the existing task/intent semantics")
        for c in q["constraints"]:
            require(c["attribute"] == "aabb_large_mm" and c["op"] == "<=" and c["unit"] == "mm"
                    and not isinstance(c["value"], bool) and np.isfinite(c["value"]) and c["value"] > 0,
                    "Unsupported query constraint")
    settings = source_protocol["settings"]
    active_weights = [float(settings["weights"].get(n, 1.)) for n in ("geometry", "class_reference", "semantic")]
    require(all(w > 0 and w == active_weights[0] for w in active_weights), "Rank-components requires unchanged equal active weights")
    retrievers = {kind: Retriever(catalog, knowledge, graph, retrieval_config(settings, kind)) for kind in KINDS}
    gallery, contaminated = common_gallery(retrievers, catalog)
    cohorts = read_json(cfg["cohorts_path"])
    require(cohorts == read_json(paths["old_cohorts"]), "Candidate cohort membership/order differs from v5")
    ids = cohorts["closed"]
    require(ids == gallery == previous_protocol["gallery"] and len(ids) == len(set(ids)) == 1816,
            "Preserve the exact held-out candidate gallery/order")
    require(len(cohorts) == 6, "Preserve all six prior candidate cohorts")
    index = {p: i for i, p in enumerate(ids)}
    cohort_indexes = {}
    for name, selected in cohorts.items():
        require(len(selected) == len(set(selected)) and set(selected) <= set(ids), "Invalid prior cohort mapping")
        cohort_indexes[name] = np.asarray([index[p] for p in selected], dtype=int)
    references = sorted({q["reference_part_id"] for q in queries if q["reference_part_id"]})
    test_hashes = {parts[p]["file_sha256"] for p in ids}
    exemplar_hashes = {}
    for ref in references:
        require(ref in parts and parts[ref]["split"] == "train" and parts[ref]["file_sha256"] not in test_hashes,
                "Exemplar must be training-only without exact test source overlap")
        require(all(ref in r.vectors and np.linalg.norm(r.vectors[ref]) > 1e-12 for r in retrievers.values())
                and retrievers["embedding"].probabilities(ref) is not None, "Exemplar missing a required model representation")
        exemplar_hashes[parts[ref]["absolute_path"]] = parts[ref]["file_sha256"]
    _verify_hashes(exemplar_hashes)
    for q in queries:
        if metadata[q["query_id"]]["origin"] == "new_exemplar":
            require(bool(q["reference_part_id"]), "Exemplar-origin query lacks its reference")
        if metadata[q["query_id"]]["origin"] == "new_text":
            require(q["reference_part_id"] is None, "Text-origin query unexpectedly contains a reference")

    # Preserve v5's exact corruption draws, then extend to new reference parts
    # using an independent deterministic stream. All seven methods share noise.
    legacy_reference_ids = {e["part_id"] for e in previous_protocol["exemplars"]}
    legacy_universe = sorted(set(gallery) | legacy_reference_ids)
    probability_ids = sorted(set(legacy_universe) | set(references))
    probabilities = {p: np.asarray([retrievers["embedding"].probabilities(p).get(c, 0.) for c in CLASSES])
                     for p in probability_ids}
    function_weights = {f["function_id"]: np.asarray([sum(r["relevance"] * r["confidence"] for r in knowledge["class_functions"]
                                                        if r["function_id"] == f["function_id"] and r["class_id"] == c)
                                                       for c in CLASSES]) for f in families.values()}
    require(all(np.count_nonzero(w) == 1 and w.max() == 1 for w in function_weights.values()), "Function prior changed")
    geometry_cache, reference_cache, intent_cache = {}, {}, {}
    perturbed_reference_cache, perturbed_intent_cache = {}, {}

    def components(q, kind, probs=probabilities, caching=True):
        fid, ref = q["function_ids"][0], q["reference_part_id"]
        icache = intent_cache if caching else perturbed_intent_cache
        rcache = reference_cache if caching else perturbed_reference_cache
        if fid not in icache:
            icache[fid] = np.asarray([float(probs[p] @ function_weights[fid]) for p in ids])
        comp = {"intent": icache[fid]}
        if ref:
            gkey = kind, ref
            if gkey not in geometry_cache:
                r = retrievers[kind]
                geometry_cache[gkey] = np.asarray([(cosine(r.vectors[p], r.vectors[ref]) + 1) / 2 for p in ids])
            comp["geometry"] = geometry_cache[gkey]
            if ref not in rcache:
                rcache[ref] = np.asarray([float(probs[p] @ probs[ref]) for p in ids])
            comp["reference"] = rcache[ref]
        return comp

    feasible_cache, component_cache = {}, {}
    for q in queries:
        feasible_cache[q["query_id"]] = np.asarray([check_constraints(parts[p], q["constraints"])["status"]
                                                    in {"pass", "not_requested"} for p in ids])
        for kind in KINDS:
            component_cache[kind, q["query_id"]] = components(q, kind)
    serving_checks = []
    for kind, q in product(KINDS, [q for q in queries if q["query_id"] not in old_queries]):
        comp, feasible = component_cache[kind, q["query_id"]], feasible_cache[q["query_id"]]
        for method in SERVING_METHODS:
            order, scores = rank_components(comp, feasible, method, ids, cfg["rrf_constant"])
            actual = retrievers[kind].rank(q, method, k=5, candidate_ids=ids)["results"]
            require([ids[i] for i in order[:5]] == [r["part_id"] for r in actual]
                    and len(actual) == len(scores[:5])
                    and all(abs(float(a) - float(b["score"])) <= 1e-12 for a, b in zip(scores[:5], actual)),
                    f"Serving top5/score mismatch: {kind}/{q['query_id']}/{method}")
            serving_checks.append(dict(geometry_kind=kind, query_id=q["query_id"], method=method, top5_equal=True, scores_equal=True))
    print(f"All new-query serving checks passed: {len(serving_checks)}", flush=True)

    # Only now read grade contents, after all frozen inputs and serving checks.
    annotation = read_json(cfg["annotation_path"])
    require(annotation.get("source") == "ai_reference" and annotation.get("human_expert_reviewed") is False,
            "Reference is an AI geometric review, not human validation")
    require(annotation["scope_sha256"] == source_protocol["scope_sha256"]
            and annotation["query_sha256"] == source_protocol["query_sha256"], "Annotation task scope changed")
    records = annotation["judgments"]
    obs = {(r["family_id"], r["part_id"]): r for r in records}
    require(len(obs) == len(records) == len(ids) * len(families) and set(obs) == set(product(families, ids)),
            "Incomplete existing 1816-by-11 judgment matrix")
    raw, evidence = {}, {}
    for fid in families:
        values = []
        for pid in ids:
            r = obs[fid, pid]
            require(r["grade"] is None or type(r["grade"]) is int and r["grade"] in (0, 1, 2), "Invalid AI grade")
            require(r["source"] == "ai_reference" and r["human_expert_reviewed"] is False
                    and r["review_status"] == "visually_reviewed", "Invalid review provenance")
            require(r["source_sha256"] == parts[pid]["file_sha256"] and (r["grade"] is not None or r["decision"] == "cannot_assess"),
                    "Reference source/null disposition changed")
            values.append(-1 if r["grade"] is None else r["grade"])
            path = Path(cfg["evidence_root"]) / (r["inspection_id"] + ".json")
            require(path.resolve() in consumed_paths, "Numerical evidence absent from its original freeze")
            if pid not in evidence:
                evidence[pid] = read_json(path)
            require(evidence[pid]["part_id"] == pid and evidence[pid]["source_sha256"] == r["source_sha256"], "Evidence source mismatch")
        raw[fid] = np.asarray(values, dtype=np.int8)
    train_ids = sorted(p for p in retrievers["descriptor"].vectors if parts[p]["split"] == "train")
    dedup, duplicate_audit = near_duplicate_gallery(ids, train_ids, retrievers["descriptor"], threshold)
    require(dedup == cohorts["deduplicated"], "Prior deduplicated cohort changed")
    duplicate_audit["discordant_groups"] = [g for g in duplicate_audit["duplicate_groups"]
                                            if any(len({obs[fid, p]["grade"] for p in g}) > 1 for fid in families)]
    names = metric_names(tuple(cfg["cutoffs"]))
    limits_cache, prepared_cache, verdict_cache = {}, {}, {}
    grade_keys = {q["query_id"]: (q["scenario_id"], json.dumps(q["constraints"], sort_keys=True)) for q in queries}

    def prepared(q, cohort="closed", scenario="nominal", guard=.2):
        family_key, constraint_key = grade_keys[q["query_id"]]
        key = family_key, constraint_key, scenario, guard
        if key not in limits_cache:
            vkey = constraint_key, guard
            if vkey not in verdict_cache:
                verdict_cache[vkey] = [numeric_reference(q, evidence[p], guard) for p in ids]
            limits_cache[key] = grade_limits(raw[q["scenario_id"]], verdict_cache[vkey], scenario)
        ckey = key + (cohort,)
        if ckey not in prepared_cache:
            lo, hi = limits_cache[key]
            indexes = cohort_indexes[cohort]
            prepared_cache[ckey] = prepare_limits(lo[indexes], hi[indexes], tuple(cfg["cutoffs"]))
        return prepared_cache[ckey]

    previous_metrics = {tuple(r[k] for k in ("geometry_kind", "query_id", "cohort", "method", "grading_scenario")): r
                        for r in read_csv(paths["old_metrics"]) if r["method"] in METHODS}
    previous_robust_cases = {(r["geometry_kind"], r["query_id"], r["method"], r["robustness"], float(r["severity"]), int(r["seed"])): r
                             for r in read_csv(paths["old_robust_cases"]) if r["method"] in METHODS}
    previous_robust_metrics = {(r["geometry_kind"], r["query_id"], r["method"], r["robustness"], float(r["severity"]), int(r["seed"])): r
                              for r in read_csv(paths["old_robust_metrics"]) if r["method"] in METHODS}
    synthetic = self_test()
    rows, primary, collapses, random_rows = [], {}, [], []
    continuity_arrays = continuity_metrics = flat_checks = 0
    stage.mkdir(parents=True)
    with ArrayArchive(stage / "complete_rankings.npz") as writer, \
            np.load(paths["old_rankings"], allow_pickle=False) as old_arrays, \
            (stage / "rankings_top20.csv").open("x", encoding="utf-8", newline="") as top_stream:
        top_writer = csv.DictWriter(top_stream, fieldnames=("cohort", "geometry_kind", "query_id", "method", "rank", "part_id", "score"))
        top_writer.writeheader()
        for kind, q in product(KINDS, queries):
            qid = q["query_id"]
            for cohort, selected in cohorts.items():
                ci = cohort_indexes[cohort]
                comp = {k: values[ci] for k, values in component_cache[kind, qid].items()}
                feasible = feasible_cache[qid][ci]
                local = {}
                for method in METHODS:
                    order, scores = rank_components(comp, feasible, method, selected, cfg["rrf_constant"])
                    key = f"{kind}__{qid}__{cohort}__{method}"
                    writer.add(key + "__order", order.astype(np.int32))
                    writer.add(key + "__score", scores)
                    local[method] = order, scores
                    if cohort == "closed":
                        primary[kind, qid, method] = order.astype(np.int32), scores
                    if qid in old_queries:
                        require(np.array_equal(order, old_arrays[key + "__order"])
                                and np.array_equal(scores, old_arrays[key + "__score"]), f"Original110 rank/score continuity failed: {key}")
                        continuity_arrays += 2
                    for scenario in SCENARIOS if cohort == "closed" else ("nominal",):
                        cache = prepared(q, cohort, scenario)
                        row = dict(cohort=cohort, geometry_kind=kind, query_id=qid, family_id=q["scenario_id"],
                                   method=method, grading_scenario=scenario, **metadata[qid],
                                   relevant_total_lower=cache["rlo"], relevant_total_upper=cache["rhi"],
                                   **_flat(rank_bounds(order, cache)))
                        if qid in old_queries:
                            identity = kind, qid, cohort, method, scenario
                            require(identity in previous_metrics, f"Missing historical metric case: {identity}")
                            _metric_continuity(row, previous_metrics[identity], names, identity)
                            continuity_metrics += 1
                        rows.append(row)
                    top_writer.writerows(dict(cohort=cohort, geometry_kind=kind, query_id=qid, method=method, rank=j,
                                              part_id=selected[i], score=float(scores[j - 1])) for j, i in enumerate(order[:20], 1))
                require(RECIPES["kg_full"] == RECIPES["flat_full"], "Full/flat shared mathematical recipe changed")
                flat_order, flat_scores = rank_components(comp, feasible, "flat_full", selected, cfg["rrf_constant"])
                require(np.array_equal(local["kg_full"][0], flat_order) and np.array_equal(local["kg_full"][1], flat_scores),
                        "Full/flat shared-recipe identity failed")
                flat_checks += 1
                collapses.extend(_collapse_rows(kind, q, cohort, local))
                feasible_indexes = np.flatnonzero(feasible)
                random_rows.append(dict(cohort=cohort, geometry_kind=kind, query_id=qid, family_id=q["scenario_id"],
                                        grading_scenario="nominal", **metadata[qid],
                                        **_flat(random_bounds(feasible_indexes, prepared(q, cohort)))))
            if qid == queries[-1]["query_id"]:
                print(f"Nominal rankings and intervals completed: {kind}", flush=True)
        core_arrays = len(writer.keys)

    robust_rows, cases = [], []
    robust_continuity_orders = robust_continuity_metrics = 0
    legacy_p = np.asarray([probabilities[p] for p in legacy_universe])
    extra_ids = sorted(set(probability_ids) - set(legacy_universe))
    extra_p = np.asarray([probabilities[p] for p in extra_ids])
    with ArrayArchive(stage / "robustness_complete_rankings.npz") as writer, \
            np.load(paths["old_robust"], allow_pickle=False) as old_robust:
        def save(kind, q, method, condition, severity, seed, order, scores, guard=.2):
            nonlocal robust_continuity_orders, robust_continuity_metrics
            qid = q["query_id"]
            key = f"{kind}__{qid}__{method}__{condition}__{severity}__{seed}"
            writer.add(key + "__order", order.astype(np.int32))
            writer.add(key + "__score", scores)
            cases.append(dict(geometry_kind=kind, query_id=qid, method=method, robustness=condition,
                              severity=severity, seed=seed, key=key + "__order", score_key=key + "__score", guard_mm=guard))
            row = dict(cohort="closed", geometry_kind=kind, query_id=qid, family_id=q["scenario_id"], method=method,
                       grading_scenario="nominal", robustness=condition, severity=severity, seed=seed,
                       guard_mm=guard, **metadata[qid], **_flat(rank_bounds(order, prepared(q, guard=guard))))
            identity = kind, qid, method, condition, float(severity), int(seed)
            if qid in old_queries and identity in previous_robust_metrics:
                _metric_continuity(row, previous_robust_metrics[identity], names, identity)
                robust_continuity_metrics += 1
                # CSV integer zero and float zero spellings are normalised only
                # for lookup; preserve actual original array-key strings.
                match = previous_robust_cases.get(identity)
                require(match is not None and np.array_equal(order, old_robust[match["key"]]),
                        f"Legacy robustness full-order continuity failed: {identity}")
                robust_continuity_orders += 1
            robust_rows.append(row)

        for kind, q in product(KINDS, queries):
            qid = q["query_id"]
            for guard in guards:
                if guard == .2 or not q["constraints"]:
                    continue
                for method in METHODS:
                    order, scores = primary[kind, qid, method]
                    save(kind, q, method, "constraint_guard", guard, 0, order, scores, guard)
            for seed in cfg["robustness_seeds"]:
                for method in METHODS:
                    order, scores = rank_components(component_cache[kind, qid], feasible_cache[qid], method, ids,
                                                    cfg["rrf_constant"], stable_seed(seed, qid, "ties"))
                    save(kind, q, method, "tie_order", 0, seed, order, scores)
        for fraction, seed in product(fractions, cfg["robustness_seeds"]):
            perturbed_intent_cache.clear()
            perturbed_reference_cache.clear()
            rng = np.random.default_rng(seed)
            values = legacy_p.copy()
            chosen = rng.random(len(values)) < fraction
            for i in np.flatnonzero(chosen):
                values[i] = values[i, rng.permutation(len(CLASSES))]
            corrupted = {p: values[i] for i, p in enumerate(legacy_universe)}
            extension_rng = np.random.default_rng(stable_seed(seed, "query_diversity_v6", "new_exemplars"))
            additional = extra_p.copy()
            selected_extra = extension_rng.random(len(additional)) < fraction
            for i in np.flatnonzero(selected_extra):
                additional[i] = additional[i, extension_rng.permutation(len(CLASSES))]
            corrupted.update({p: additional[i] for i, p in enumerate(extra_ids)})
            for kind, q in product(KINDS, queries):
                comp = components(q, kind, corrupted, caching=False)
                for method in METHODS:
                    order, scores = rank_components(comp, feasible_cache[q["query_id"]], method, ids, cfg["rrf_constant"])
                    save(kind, q, method, "classifier_permutation", fraction, seed, order, scores)
            print(f"Robustness rankings and intervals completed: fraction={fraction}, seed={seed}", flush=True)
        robust_arrays = len(writer.keys)

    require(continuity_arrays == 110 * 2 * len(cohorts) * len(METHODS) * 2, "Incomplete original110 array continuity checks")
    require(continuity_metrics == len(previous_metrics), "Incomplete original110 metric continuity checks")
    require(robust_continuity_orders == robust_continuity_metrics == len(previous_robust_metrics),
            "Incomplete common-method legacy robustness continuity checks")
    by_id = {q["query_id"]: q for q in queries}
    family_rows, summary = summaries(rows, ("cohort", "geometry_kind", "method", "grading_scenario"), by_id, names)
    origin_family, origin_summary = summaries(rows, ("cohort", "geometry_kind", "method", "grading_scenario", "origin"), by_id, names)
    robust_family, robust_summary = summaries(robust_rows, ("geometry_kind", "method", "robustness", "severity"), by_id, names)
    robust_origin_family, robust_origin_summary = summaries(robust_rows, ("geometry_kind", "method", "robustness", "severity", "origin"), by_id, names)
    random_family, random_summary = summaries(random_rows, ("cohort", "geometry_kind", "grading_scenario"), by_id,
                                               ("expected_ndcg5", "expected_precision5", "expected_strong_precision5"))
    for name, data in (("per_query_bounds.csv", rows), ("family_bounds.csv", family_rows), ("summary_bounds.csv", summary),
                       ("origin_family_bounds.csv", origin_family), ("origin_summary_bounds.csv", origin_summary),
                       ("robustness_per_query_bounds.csv", robust_rows), ("robustness_cases.csv", cases),
                       ("robustness_family_bounds.csv", robust_family), ("robustness_summary_bounds.csv", robust_summary),
                       ("robustness_origin_family_bounds.csv", robust_origin_family), ("robustness_origin_summary_bounds.csv", robust_origin_summary),
                       ("serving_checks.csv", serving_checks), ("collapsed_rankings.csv", collapses),
                       ("random_expectation_bounds.csv", random_rows), ("random_family_bounds.csv", random_family), ("random_summary_bounds.csv", random_summary)):
        write_csv(stage / name, data)
    protocol = deepcopy(source_protocol)
    existing_exemplars = {e["part_id"]: e for e in previous_protocol["exemplars"]}
    for ref in references:
        existing_exemplars.setdefault(ref, dict(part_id=ref, source_sha256=parts[ref]["file_sha256"], split="train"))
    protocol.update(queries=queries, query_sha256=object_hash(queries), annotation_query_sha256=source_protocol["query_sha256"],
                    exemplars=[existing_exemplars[p] for p in sorted(existing_exemplars)],
                    query_design_manifest_sha256=sha256(cfg["query_manifest_path"]))
    write_json(stage / "effective_protocol.json", protocol)
    write_json(stage / "cohorts.json", cohorts)
    write_json(stage / "near_duplicate_audit.json", duplicate_audit)
    write_json(stage / "retrieval_transforms.json", {k: r.transform for k, r in retrievers.items()})
    write_json(stage / "synthetic_validation.json", synthetic)
    shutil.copyfile(cfg["query_manifest_path"], stage / ("query_manifest" + Path(cfg["query_manifest_path"]).suffix))
    continuity = dict(status="VERIFIED", original_queries=110, nominal_order_score_arrays_equal=continuity_arrays,
                      metric_bound_cases_exactly_equal=continuity_metrics,
                      common_robustness_orders_equal=robust_continuity_orders,
                      common_robustness_metric_cases_equal=robust_continuity_metrics,
                      full_flat_shared_recipe_identity_cases=flat_checks,
                      full_flat_interpretation="Identical algebra and shared rank function; no retrieval-quality evidence for graph advantage",
                      serving_new_query_core_cases=len(serving_checks),
                      corruption="Exact historical sorted gallery+old-exemplar RNG draws preserved; new-reference-only extension stream stable_seed(seed,'query_diversity_v6','new_exemplars'); all methods share perturbations")
    write_json(stage / "continuity_validation.json", continuity)
    _verify_hashes(frozen)
    _verify_hashes(consumed)
    _verify_hashes(exemplar_hashes)
    _verify_hashes(read_json(paths["annotation_freeze"])["input_sha256"])
    require(sha256(freeze_path) == frozen_digest, "Experiment seal changed during run")
    manifest = dict(status="VERIFIED_FROZEN_QUERY_DIVERSITY_RUN", source="ai_reference", human_expert_reviewed=False,
                    experimental_freeze_sha256=frozen_digest, input_sha256=frozen,
                    exemplar_raw_source_sha256=exemplar_hashes,
                    code_sha256={str(Path(__file__)): sha256(__file__), **{p: sha256(p) for p in CODE_DEPENDENCIES}},
                    methods=list(METHODS), representations=list(KINDS), queries=len(queries), families=len(families),
                    gallery=len(ids), cohorts={k: len(v) for k, v in cohorts.items()}, training_references=len(references),
                    core_arrays=core_arrays, robustness_arrays=robust_arrays, core_metric_cases=len(rows),
                    robustness_metric_cases=len(robust_rows), raw_null_pairs=sum(r["grade"] is None for r in records),
                    candidate_exact_contamination_excluded=len(contaminated), cutoff_metrics=list(names),
                    continuity=continuity, labels_tuned=False, methods_tuned=False, null_labels_imputed=False,
                    statistical_tests_performed=False, primary_query_selection="Defined by separately frozen design; no post-score subset selection here",
                    uncertainty="Conservative annotation outer envelopes, not sampling confidence intervals or human validation",
                    artifact_sha256={p.name: sha256(p) for p in stage.iterdir() if p.is_file()})
    write_json(stage / "run_manifest.json", manifest)
    require(not out.exists(), "Output appeared during run; refusing replacement")
    stage.rename(out)
    print(json.dumps({k: manifest[k] for k in ("status", "queries", "core_arrays", "robustness_arrays", "core_metric_cases", "robustness_metric_cases")}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "evaluate"), nargs="?", default="evaluate")
    parser.add_argument("--config", default="configs/query_diversity_v6.json")
    parser.add_argument("--output", help="Fresh run/replay folder within outputs/query_diversity_v6")
    args = parser.parse_args()
    if args.action == "self-test":
        print(json.dumps(self_test(), indent=2))
    else:
        run(args.config, args.output)


if __name__ == "__main__":
    main()
