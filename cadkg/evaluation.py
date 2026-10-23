"""Independent closed-gallery judgments; abstentions count as zero retrieval utility."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_csv, write_csv, write_json
from .query import validate_query
from .reasoning import check_constraints, clearance_screen
from .retrieval import METHODS


def classification_metrics(retriever, candidate_ids=None):
    from .catalog import CLASSES
    parts = [retriever.parts[pid] for pid in (retriever.candidate_ids() if candidate_ids is None else candidate_ids)]
    parts = [p for p in parts if p["prediction"] is not None]
    if not parts:
        return {"n": 0}
    y = np.array([CLASSES.index(p["label"]) for p in parts])
    probabilities = np.array([[p["prediction"]["probabilities"].get(c, 0) for c in CLASSES] for p in parts])
    pred = probabilities.argmax(1)
    present = np.unique(y)
    recalls, f1s = [], []
    for c in present:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        recalls.append(tp/(tp+fn))
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    result = {"n": len(parts), "accuracy": float((y == pred).mean()), "balanced_accuracy": float(np.mean(recalls)),
              "macro_f1_present_classes": float(np.mean(f1s)), "classes_present": [CLASSES[c] for c in present],
              "scope": "frozen gallery diagnostic only; labels do not enter serving"}
    if any("hard_label_only" in p["prediction"]["calibration"] for p in parts):
        result["calibration_metrics"] = "unavailable: hard labels only"
    else:
        result["negative_log_likelihood"] = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())
        result["brier"] = float(((probabilities-np.eye(len(CLASSES))[y])**2).sum(1).mean())
        confidence = probabilities.max(1)
        ece = 0.0
        for lo in np.arange(10)/10:
            mask = (confidence >= lo) & ((confidence < lo+0.1) if lo < 0.9 else (confidence <= 1))
            if mask.any():
                ece += float(mask.mean()*abs(confidence[mask].mean()-(y[mask] == pred[mask]).mean()))
        result["ece_10_bins"] = ece
    return result


def metrics(ranked_ids, judgments, k):
    grades = [judgments[pid]["grade"] for pid in ranked_ids[:k]]
    relevant = sum(r["grade"] > 0 for r in judgments.values())
    ideal = sorted([r["grade"] for r in judgments.values()], reverse=True)[:k]
    dcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(grades))
    idcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(ideal))
    hits, total = 0, 0.0
    for i, grade in enumerate(grades, 1):
        if grade > 0:
            hits += 1
            total += hits/i
    return {f"ndcg@{k}": dcg/idcg if idcg else 0.0, f"precision@{k}": hits/k,
            f"recall@{k}": hits/relevant if relevant else 0.0,
            f"map@{k}": total/min(relevant, k) if relevant else 0.0,
            "answerable": int(relevant > 0)}


def load_judgments(path, queries, retriever, synthetic=False):
    by_query = {q["query_id"]: {} for q in queries}
    sources = {"synthetic_fixture"} if synthetic else {"independent_expert"}
    query_lookup = {q["query_id"]: q for q in queries}
    for row in read_csv(path):
        qid, pid = row["query_id"], row["part_id"]
        if qid not in by_query:
            continue
        if pid not in retriever.parts or pid in by_query[qid]:
            raise ValueError(f"Unknown or duplicate judged part {qid}/{pid}")
        if row.get("source") not in sources or not row.get("reviewer", "").strip() or not row.get("evidence", "").strip():
            raise ValueError("Judgments must be independent expert assessments with reviewer/evidence; KG/label proxies forbidden")
        if row.get("split") != query_lookup[qid]["split"]:
            raise ValueError("Judgment/query split mismatch")
        grade = int(row["grade"])
        if grade not in {0, 1, 2} or row["constraint_verdict"] not in {"pass", "fail", "unknown", "not_requested"}:
            raise ValueError("Invalid relevance grade or constraint verdict")
        requested = bool(query_lookup[qid]["constraints"])
        if requested == (row["constraint_verdict"] == "not_requested"):
            raise ValueError("Constraint verdict must reflect whether the query actually requests constraints")
        by_query[qid][pid] = {**row, "grade": grade}
    gallery = retriever.candidate_ids()
    if not gallery:
        raise ValueError("Empty matched gallery")
    for query in queries:
        ref = query["reference_part_id"]
        ref_hash = retriever.parts[ref]["file_sha256"] if ref else None
        expected = {pid for pid in gallery if pid != ref and not (ref_hash and retriever.parts[pid]["file_sha256"] == ref_hash)}
        if set(by_query[query["query_id"]]) != expected:
            missing = expected-set(by_query[query["query_id"]])
            extra = set(by_query[query["query_id"]])-expected
            raise ValueError(f"Closed-gallery evaluation needs exactly one judgment per candidate for {query['query_id']}; missing={len(missing)}, extra={len(extra)}")
    return by_query


def evaluate(retriever, raw_queries, judgments_path, config, output, synthetic=False):
    output = Path(output)
    queries = [validate_query(q, retriever.knowledge) for q in raw_queries]
    if not queries or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Need nonempty unique query IDs")
    expected_split = config.get("evaluation_split", "test")
    if not synthetic and expected_split not in {"dev", "test"}:
        raise ValueError("Real evaluation split must be dev or test")
    if any(q["split"] != expected_split for q in queries):
        raise ValueError("Mixed or unexpected query split")
    if not synthetic:
        expected_gallery = ["val"] if expected_split == "dev" else ["test"]
        if config.get("gallery_splits", ["test"]) != expected_gallery:
            raise ValueError("Development queries use val gallery; test queries use test gallery")
        if any(not p.get("file_sha256") for p in retriever.catalog["parts"]):
            raise ValueError("Hash all selected STEP files before research evaluation; import verified legacy CSVs or measure them")
        for query in queries:
            ref = query["reference_part_id"]
            if ref and (ref not in retriever.parts or retriever.parts[ref]["split"] != "train"):
                raise ValueError("Evaluation exemplars must come from training split")
        duplicate_groups = retriever.catalog.get("audit", {}).get("leakage", {}).get("exact_byte_cross_split_duplicates", [])
        if config.get("require_no_exact_leak", True) and duplicate_groups:
            raise ValueError("Cross-split exact STEP duplicates found; rebuild split before evaluation")
    methods = config.get("methods", list(METHODS))
    if not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError("Invalid method matrix")
    k = config.get("k", 5)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    judgments = load_judgments(judgments_path, queries, retriever, synthetic)
    per_query, rankings, reasoning_rows = [], [], []
    result_cache = {}
    for query in queries:
        qid = query["query_id"]
        # Constraint truth comes from qrels; predictions checked on all candidates, independent of rank.
        for pid, gold in judgments[qid].items():
            status = check_constraints(retriever.parts[pid], query["constraints"])["status"]
            reasoning_rows.append({"query_id": qid, "part_id": pid, "predicted_status": status,
                                   "expert_status": gold["constraint_verdict"]})
        for method in methods:
            result = retriever.rank(query, method, k)
            result_cache[qid, method] = result
            ids = [r["part_id"] for r in result["results"]]
            scores = metrics(ids, judgments[qid], k)
            checks = [judgments[qid][pid]["constraint_verdict"] for pid in ids]
            known = [c for c in checks if c in {"pass", "fail"}]
            per_query.append({"query_id": qid, "scenario_id": query["scenario_id"], "method": method, **scores,
                              "returned": len(ids), "abstained": int(not ids), "gallery_count": result["gallery_count"],
                              "constraint_violation_rate": sum(c == "fail" for c in known)/len(known) if known else None,
                              "constraint_known_returned": len(known), "constraint_rejected": result["constraint_rejected"],
                              "unknown_returned": sum(c == "unknown" for c in checks)})
            for row in result["results"]:
                rankings.append({"query_id": qid, "method": method, "rank": row["rank"], "part_id": row["part_id"],
                                 "score": row["score"], "expert_grade": judgments[qid][row["part_id"]]["grade"],
                                 "constraint_status": row["reasoning"]["status"]})
    summaries = []
    for method in methods:
        rows = [r for r in per_query if r["method"] == method]
        summary = {"method": method, "queries": len(rows), "answered_fraction": 1-np.mean([r["abstained"] for r in rows]),
                   "answerable_queries": sum(r["answerable"] for r in rows)}
        for key in [f"ndcg@{k}", f"precision@{k}", f"recall@{k}", f"map@{k}"]:
            summary[key] = float(np.mean([r[key] for r in rows]))
        denomin = sum(r["constraint_known_returned"] for r in rows)
        summary["constraint_known_returned"] = denomin
        summary["constraint_violation_rate"] = (sum((r["constraint_violation_rate"] or 0)*r["constraint_known_returned"] for r in rows)/denomin if denomin else None)
        summaries.append(summary)
    comparisons = []
    if "kg_full" in methods:
        for method in methods:
            if method == "kg_full":
                continue
            groups = defaultdict(list)
            for query in queries:
                qid = query["query_id"]
                a = next(r for r in per_query if r["query_id"] == qid and r["method"] == "kg_full")
                b = next(r for r in per_query if r["query_id"] == qid and r["method"] == method)
                groups[query["scenario_id"]].append(a[f"ndcg@{k}"]-b[f"ndcg@{k}"])
            deltas = np.array([np.mean(v) for v in groups.values()])
            ci = None
            if len(deltas) >= 5:
                rng = np.random.default_rng(config.get("seed", 17))
                sampled = rng.choice(deltas, size=(config.get("bootstrap_samples", 1000), len(deltas)), replace=True).mean(1)
                ci = np.quantile(sampled, [0.025, 0.975]).tolist()
            comparisons.append({"comparison": f"kg_full - {method}", "metric": f"ndcg@{k}", "independent_scenarios": len(deltas),
                                "mean_scenario_delta": float(deltas.mean()), "paired_scenario_bootstrap_95_ci": ci})
    eq = None
    if "kg_full" in methods and "flat_full" in methods:
        eq = all([(r["part_id"], r["score"]) for r in result_cache[q["query_id"], "kg_full"]["results"]]
                 == [(r["part_id"], r["score"]) for r in result_cache[q["query_id"], "flat_full"]["results"]] for q in queries)
    known_reasoning = [r for r in reasoning_rows if r["expert_status"] in {"pass", "fail"}]
    answered = [r for r in known_reasoning if r["predicted_status"] in {"pass", "fail"}]
    pass_rows = [r for r in known_reasoning if r["predicted_status"] == "pass"]
    unknown_gold = [r for r in reasoning_rows if r["expert_status"] == "unknown"]
    summary = {"result_type": "synthetic_plumbing_only" if synthetic else "independent_expert_closed_gallery",
               "classification_diagnostic": classification_metrics(retriever),
               "methods": summaries, "paired_comparisons": comparisons, "flat_kg_rank_equivalent": eq,
               "reasoning": {"known_gold_cases": len(known_reasoning), "answered_cases": len(answered),
                             "unknown_gold_cases": len(unknown_gold),
                             "coverage": len(answered)/len(known_reasoning) if known_reasoning else None,
                             "selective_accuracy": sum(r["predicted_status"] == r["expert_status"] for r in answered)/len(answered) if answered else None,
                             "false_pass_rate": sum(r["expert_status"] == "fail" for r in pass_rows)/len(pass_rows) if pass_rows else None,
                             "pass_on_unknown_gold_rate": sum(r["predicted_status"] == "pass" for r in unknown_gold)/len(unknown_gold) if unknown_gold else None},
               "notes": ["Recall/MAP use complete independent judgments for the frozen gallery, not the full TMCAD corpus.",
                         "Abstentions count as zero utility. Empty-target queries also score zero; counts are reported.",
                         "Class-function priors are not relevance ground truth. Graph/table equality is expected in this bounded core.",
                         "Confidence intervals omitted below five independent scenarios; no significance claim."]}
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "rankings.csv", rankings, ["query_id", "method", "rank", "part_id", "score", "expert_grade", "constraint_status"])
    write_csv(output / "reasoning.csv", reasoning_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "explanations.json", list(result_cache.values()))
    write_json(output / "resolved_config.json", config)
    write_json(output / "representation_transform.json", retriever.transform)
    return summary


def evaluate_clearance_cases(catalog, cases, output):
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Need nonempty unique clearance case IDs")
    rows = []
    for case in cases:
        if case.get("source") != "independent_expert" or not case.get("reviewer") or not case.get("evidence"):
            raise ValueError("Clearance cases need independent expert evidence")
        if case["expected_status"] not in {"pass", "fail", "unknown"}:
            raise ValueError("Invalid expected status")
        result = clearance_screen(catalog, case["bore_part_id"], case["bore_interface_id"], case["shaft_part_id"],
                                  case["shaft_interface_id"], *case["allowed_clearance_mm"])
        rows.append({"case_id": case["case_id"], "expected": case["expected_status"], "predicted": result["status"], "result": result})
    known = [r for r in rows if r["expected"] in {"pass", "fail"}]
    answered = [r for r in known if r["predicted"] in {"pass", "fail"}]
    passes = [r for r in known if r["predicted"] == "pass"]
    unknown = [r for r in rows if r["expected"] == "unknown"]
    report = {"cases": rows, "metrics": {"n": len(rows), "known_gold": len(known),
               "exact_status_accuracy": sum(r["predicted"] == r["expected"] for r in rows)/len(rows),
               "coverage_known_gold": len(answered)/len(known) if known else None,
               "selective_accuracy": sum(r["predicted"] == r["expected"] for r in answered)/len(answered) if answered else None,
               "false_pass_rate": sum(r["expected"] == "fail" for r in passes)/len(passes) if passes else None,
               "pass_on_unknown_gold_rate": sum(r["predicted"] == "pass" for r in unknown)/len(unknown) if unknown else None},
              "scope": "independent diameter-clearance cases; not assembly compatibility"}
    write_json(output, report)
    write_csv(Path(output).with_suffix(".csv"), [{k: r[k] for k in ["case_id", "expected", "predicted"]} for r in rows])
    return rows
