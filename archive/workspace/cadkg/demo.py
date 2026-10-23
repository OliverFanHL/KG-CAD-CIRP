"""Entirely fictional numeric fixture. Never used as engineering research evidence."""
from __future__ import annotations

from pathlib import Path

from .catalog import CLASSES
from .evaluation import evaluate
from .graph import build_graph
from .io import object_hash, receipt, write_csv, write_json
from .retrieval import Retriever


def run_demo(output):
    output = Path(output)
    parts = []
    for i, (label, split, dim) in enumerate([(CLASSES[0], "train", 30), (CLASSES[2], "train", 60),
                                           (CLASSES[3], "train", 120), (CLASSES[0], "test", 40),
                                           (CLASSES[2], "test", 140), (CLASSES[3], "test", None)]):
        probabilities = {c: (0.82 if c == label else 0.02) for c in CLASSES}
        parts.append({"part_id": f"toy{i}", "label": label, "split": split, "file_sha256": f"toy-hash-{i}",
                      "relative_path": f"fictional/{i}.stp", "absolute_path": "not_a_real_file",
                      "features": {"dim_small_large_ratio": 0.15+i*0.1, "face_count": 10+i*3},
                      "embedding": [1.0, i/6, 0.5], "interfaces": [],
                      "measurements": {} if dim is None else {"aabb_large_mm": {"value": dim, "unit": "mm", "status": "computed", "source": "fictional fixture"}},
                      "prediction": {"probabilities": probabilities, "status": "predicted", "source": "fictional fixture", "model": "toy", "calibration": "synthetic"}})
    catalog = {"schema_version": 1, "dataset_type": "synthetic_fixture", "parts": parts, "classes": CLASSES, "audit": {}, "artifacts": {}}
    function = {"function_id": "toy_task", "label": "toy task", "aliases": "fictional task", "status": "curated",
                "reviewer": "fixture_author_not_an_expert", "evidence": "invented fixture only", "source": "synthetic_fixture"}
    relations = [{"function_id": "toy_task", "class_id": c, "relevance": r, "confidence": 1.0, "status": "curated",
                  "scope": "fictional testing only", "caveat": "no real engineering claim", "reviewer": "fixture_author_not_an_expert",
                  "evidence": "invented fixture only", "source": "synthetic_fixture"} for c, r in [(CLASSES[0], 1.0), (CLASSES[3], 0.6), (CLASSES[2], 0.0)]]
    knowledge = {"functions": [function], "class_functions": relations, "interfaces": [], "excluded_rows": []}
    knowledge["sha256"] = object_hash(knowledge)
    graph = build_graph(catalog, knowledge)
    queries = [{"query_id": "toy_intent", "split": "demo", "function_ids": ["toy_task"], "text": "fictional task"},
               {"query_id": "toy_envelope", "split": "demo", "function_ids": ["toy_task"], "reference_part_id": "toy0",
                "constraints": [{"attribute": "aabb_large_mm", "op": "<=", "value": 100, "unit": "mm"}]},
               {"query_id": "toy_missing_interface", "split": "demo", "function_ids": ["toy_task"],
                "constraints": [{"attribute": "bore_diameter_mm", "op": "between", "value": [19, 21], "unit": "mm"}]}]
    rows = []
    # Hand-defined fictional expected outcomes; this is solely a deterministic plumbing fixture.
    grades = {"toy_intent": [2, 0, 1], "toy_envelope": [2, 0, 0], "toy_missing_interface": [0, 0, 0]}
    verdicts = {"toy_intent": ["not_requested"]*3, "toy_envelope": ["pass", "fail", "unknown"], "toy_missing_interface": ["unknown"]*3}
    for query in queries:
        for j, pid in enumerate(["toy3", "toy4", "toy5"]):
            rows.append({"query_id": query["query_id"], "part_id": pid, "grade": grades[query["query_id"]][j],
                         "constraint_verdict": verdicts[query["query_id"]][j], "reviewer": "fictional_fixture_author",
                         "evidence": "fictional plumbing values; not engineering ground truth", "source": "synthetic_fixture", "split": "demo"})
    write_json(output / "catalog.json", catalog)
    write_json(output / "knowledge.json", knowledge)
    write_json(output / "graph.json", graph)
    write_json(output / "queries.json", queries)
    write_csv(output / "judgments.csv", rows)
    config = {"evaluation_split": "demo", "k": 3, "seed": 17, "geometry_kind": "embedding", "unknown_policy": "exclude", "matched_cohort": True}
    summary = evaluate(Retriever(catalog, knowledge, graph, config), queries, output / "judgments.csv", config, output / "evaluation", synthetic=True)
    write_json(output / "evaluation/run_manifest.json", receipt("demo", [], config))
    return summary
