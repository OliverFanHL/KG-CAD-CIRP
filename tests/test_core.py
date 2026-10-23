from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cadkg.catalog import import_descriptors, import_embeddings, inventory, sample_id
from cadkg.bundle import load_bundle, save_bundle
from cadkg.demo import run_demo
from cadkg.evaluation import evaluate, metrics
from cadkg.graph import build_graph
from cadkg.io import read_json, write_csv, write_json
from cadkg.knowledge import load_knowledge
from cadkg.query import parse_text, validate_query
from cadkg.reasoning import check_constraints, clearance_screen
from cadkg.retrieval import Retriever


@pytest.fixture
def toy(tmp_path):
    run_demo(tmp_path / "fixture")
    base = tmp_path / "fixture"
    catalog, knowledge, graph = load_bundle(base)
    config = read_json(base / "evaluation/resolved_config.json")
    return base, catalog, knowledge, graph, config


def test_no_ground_truth_label_in_serving(toy):
    _, catalog, knowledge, graph, config = toy
    query = {"function_ids": ["toy_task"], "split": "demo"}
    before = Retriever(catalog, knowledge, graph, config).rank(query)
    changed = deepcopy(catalog)
    for part in changed["parts"]:
        part["label"] = "spring"
    after = Retriever(changed, knowledge, build_graph(changed, knowledge), config).rank(query)
    assert before["results"] == after["results"]


def test_kg_paths_actually_drive_scoring(toy):
    _, catalog, knowledge, graph, config = toy
    query = {"function_ids": ["toy_task"], "split": "demo"}
    before = Retriever(catalog, knowledge, graph, config).rank(query)
    graph = deepcopy(graph)
    graph["edges"] = [e for e in graph["edges"] if e["predicate"] != "potential_function"]
    after = Retriever(catalog, knowledge, graph, config).rank(query)
    assert before["status"] == "ranked"
    assert after["status"] == "abstained"


def test_flat_kg_equality_does_not_invent_graph_gain(toy):
    base, _, _, _, _ = toy
    result = read_json(base / "evaluation/summary.json")
    assert result["flat_kg_rank_equivalent"]
    assert result["result_type"] == "synthetic_plumbing_only"


def test_unknown_is_not_pass_and_unverified_units_fail_closed():
    part = {"interfaces": [], "measurements": {"aabb_large_mm": {"value": 10, "unit": "unknown", "status": "imported_unverified"}}}
    constraint = {"attribute": "aabb_large_mm", "op": "<=", "value": 20, "unit": "mm"}
    assert check_constraints(part, [constraint])["status"] == "unknown"
    part["measurements"]["aabb_large_mm"].update(unit="mm", status="computed", source="test")
    assert check_constraints(part, [constraint])["status"] == "pass"
    constraint["value"] = 5
    assert check_constraints(part, [constraint])["status"] == "fail"


def test_cylinder_candidate_is_not_bore():
    part = {"interfaces": [], "measurements": {}, "cylinder_candidates": [{"diameter_mm": 20}]}
    assert check_constraints(part, [{"attribute": "bore_diameter_mm", "op": "<=", "value": 21, "unit": "mm"}])["status"] == "unknown"


def test_ambiguous_interfaces_abstain():
    part = {"measurements": {}, "interfaces": [{"interface_id": "a", "role": "bore", "diameter_mm": 20, "source": "test"},
                                              {"interface_id": "b", "role": "bore", "diameter_mm": 40, "source": "test"}]}
    constraint = {"attribute": "bore_diameter_mm", "op": "<=", "value": 21, "unit": "mm"}
    assert check_constraints(part, [constraint])["status"] == "unknown"
    constraint["interface_id"] = "a"
    assert check_constraints(part, [constraint])["status"] == "pass"


def test_tolerance_reasoning_is_interval_and_keeps_assembly_unknown():
    catalog = {"parts": [{"part_id": "b", "interfaces": [{"interface_id": "i", "role": "bore", "diameter_mm": 20,
                                                         "min_mm": 20, "max_mm": 20.1, "source": "bore drawing"}]},
                         {"part_id": "s", "interfaces": [{"interface_id": "i", "role": "shaft", "diameter_mm": 19.9,
                                                         "min_mm": 19.8, "max_mm": 19.9, "source": "shaft drawing"}]}]}
    result = clearance_screen(catalog, "b", "i", "s", "i", 0.09, 0.31)
    assert result["status"] == "pass"
    assert result["assembly_compatibility"] == "unknown"
    assert clearance_screen(catalog, "b", "i", "s", "i", 0.15, 0.25)["status"] == "unknown"
    assert clearance_screen(catalog, "b", "i", "s", "i", 0.4, 0.5)["status"] == "fail"
    catalog["parts"][0]["interfaces"][0]["min_mm"] = None
    assert clearance_screen(catalog, "b", "i", "s", "i", 0.09, 0.31)["status"] == "unknown"


def test_query_rejects_unparsed_numbers_negation_and_units(toy):
    _, _, knowledge, _, _ = toy
    assert parse_text("toy task aabb_large_mm <= 100 mm", knowledge)["constraints"][0]["value"] == 100
    assert parse_text("bolt_screw", knowledge)["category_ids"] == ["bolt-screw"]
    with pytest.raises(ValueError, match="Unparsed"):
        parse_text("toy task under 100 mm", knowledge)
    with pytest.raises(ValueError, match="Negated"):
        parse_text("not toy task", knowledge)
    with pytest.raises(ValueError, match="wrong/missing unit"):
        validate_query({"constraints": [{"attribute": "area_mm2", "op": "<=", "value": 100, "unit": "mm"}]}, knowledge)


def test_metrics_use_complete_relevance_and_abstention():
    gold = {"a": {"grade": 2}, "b": {"grade": 0}, "c": {"grade": 1}}
    result = metrics(["a", "b"], gold, 2)
    assert result["recall@2"] == 0.5
    assert result["precision@2"] == 0.5
    assert result["map@2"] == 0.5
    assert metrics([], gold, 2)["ndcg@2"] == 0
    assert result["ndcg@2"] == pytest.approx(3/(3+1/np.log2(3)))


def test_evaluation_rejects_proxy_judgments_and_incomplete_gallery(toy, tmp_path):
    base, catalog, knowledge, graph, config = toy
    from cadkg.io import read_csv
    rows = read_csv(base / "judgments.csv")
    rows[0]["source"] = "kg_generated"
    write_csv(tmp_path / "bad.csv", rows)
    retriever = Retriever(catalog, knowledge, graph, config)
    with pytest.raises(ValueError, match="independent"):
        evaluate(retriever, read_json(base / "queries.json"), tmp_path / "bad.csv", config, tmp_path / "eval", synthetic=True)
    rows[0]["source"] = "synthetic_fixture"
    write_csv(tmp_path / "bad.csv", rows[:-1])
    with pytest.raises(ValueError, match="exactly one judgment"):
        evaluate(retriever, read_json(base / "queries.json"), tmp_path / "bad.csv", config, tmp_path / "eval", synthetic=True)


def test_train_only_descriptor_transform(toy):
    _, catalog, knowledge, graph, config = toy
    config = {**config, "geometry_kind": "descriptor"}
    before = Retriever(catalog, knowledge, graph, config).transform
    catalog = deepcopy(catalog)
    for part in catalog["parts"]:
        if part["split"] == "test":
            part["features"]["face_count"] = 1e8
    after = Retriever(catalog, knowledge, build_graph(catalog, knowledge), config).transform
    assert before == after


def test_cache_join_requires_same_step_bytes(tmp_path):
    root, legacy = tmp_path / "data", tmp_path / "old"
    (root / "bearing").mkdir(parents=True)
    legacy.mkdir()
    (root / "bearing/0.stp").write_text("local bytes")
    old = legacy / "0.stp"
    old.write_text("different bytes")
    catalog = inventory(root)
    write_csv(tmp_path / "descriptors.csv", [{"sample_id": sample_id("bearing/0.stp"), "relative_path": "bearing/0.stp",
                                           "absolute_path": str(old), "label": "bearing", "face_count": "2"}])
    with pytest.raises(ValueError, match="bytes differ"):
        import_descriptors(catalog, tmp_path / "descriptors.csv")


def test_synthetic_cad_jepa_ids_cannot_match_tmcad(tmp_path):
    np.savez(tmp_path / "embeddings.npz", ids=np.array(["box-00001"]), embeddings=np.ones((1, 4)))
    with pytest.raises(ValueError, match="No embedding IDs match"):
        import_embeddings({"parts": [{"part_id": "tmcad_id"}]}, tmp_path / "embeddings.npz")


def test_draft_annotations_do_not_become_facts(toy, tmp_path):
    _, catalog, _, _, _ = toy
    write_csv(tmp_path / "functions.csv", [{"function_id": "draft", "label": "draft task", "aliases": "", "reviewer": "", "evidence": "", "status": "draft"}])
    knowledge = load_knowledge(tmp_path, catalog)
    assert knowledge["functions"] == []
    assert len(knowledge["excluded_rows"]) == 1


def test_bundle_tampering_requires_rebuild(toy, tmp_path):
    _, catalog, _, _, _ = toy
    save_bundle(catalog, tmp_path / "empty", tmp_path / "bundle")
    graph = read_json(tmp_path / "bundle/graph.json")
    graph["nodes"] = []
    write_json(tmp_path / "bundle/graph.json", graph)
    with pytest.raises(ValueError, match="modified"):
        load_bundle(tmp_path / "bundle")


def test_optional_probe_selection_never_reads_test_labels_or_features(tmp_path):
    from cadkg.catalog import CLASSES
    from cadkg.classifier import fit_probe, predict_probe
    parts = [{"part_id": f"{split}_{i}", "label": label, "split": split,
              "features": {"face_count": float(i+1), "dim_small_large_ratio": 0.05*(i+1)}}
             for split in ["train", "val", "test"] for i, label in enumerate(CLASSES)]
    catalog = {"parts": parts}
    before = fit_probe(catalog, tmp_path / "before.json")
    changed = deepcopy(catalog)
    for part in changed["parts"]:
        if part["split"] == "test":
            part["label"] = "spring"
            part["features"]["face_count"] = 1e9
    after = fit_probe(changed, tmp_path / "after.json")
    assert before == after
    predict_probe(catalog, before, "test_model")
    assert all(sum(p["prediction"]["probabilities"].values()) == pytest.approx(1) for p in catalog["parts"])


def test_independent_clearance_evaluation_reports_unknown_gold(tmp_path):
    from cadkg.evaluation import evaluate_clearance_cases
    catalog = {"parts": [{"part_id": "b", "interfaces": []}, {"part_id": "s", "interfaces": []}]}
    cases = [{"case_id": "missing_interface_fixture", "bore_part_id": "b", "bore_interface_id": "i",
              "shaft_part_id": "s", "shaft_interface_id": "i", "allowed_clearance_mm": [0.1, 0.2],
              "expected_status": "unknown", "source": "independent_expert", "reviewer": "unit_test_fixture_not_real_expert",
              "evidence": "synthetic test fixture; absent interfaces cannot establish clearance"}]
    evaluate_clearance_cases(catalog, cases, tmp_path / "result.json")
    result = read_json(tmp_path / "result.json")
    assert result["metrics"]["exact_status_accuracy"] == 1
    assert result["metrics"]["coverage_known_gold"] is None
    assert result["metrics"]["pass_on_unknown_gold_rate"] == 0
    assert (tmp_path / "result.csv").exists()
