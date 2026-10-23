from __future__ import annotations

import pytest

from cadkg.catalog import CLASSES
from cadkg.io import read_csv, read_json, write_csv
from cadkg.knowledge import load_knowledge
from cadkg.reasoning import check_constraints
from cadkg.experiments.technical import prepare_review_pack, record_bounds_crosscheck


def source_rows(tmp_path):
    write_csv(tmp_path / "functions.csv", [{"function_id": "f", "label": "test function", "aliases": "", "reviewer": "",
             "evidence": "https://manufacturer.example/source", "status": "source_supported", "source_id": "test_source"}])
    write_csv(tmp_path / "class_functions.csv", [{"class_id": "bearing", "function_id": "f", "relevance": 1, "confidence": 1,
             "scope": "generic source", "caveat": "not reviewed on individual parts", "reviewer": "", "evidence": "https://manufacturer.example/source",
             "status": "source_supported", "source_id": "test_source"}])


def test_source_priors_are_opt_in_and_keep_their_status(tmp_path):
    source_rows(tmp_path)
    catalog = {"parts": []}
    assert load_knowledge(tmp_path, catalog)["class_functions"] == []
    admitted = load_knowledge(tmp_path, catalog, "include_source_supported")
    assert admitted["class_functions"][0]["status"] == "source_supported"
    assert admitted["class_functions"][0]["reviewer"] == ""


def test_source_policy_cannot_admit_unreviewed_interfaces(tmp_path):
    source_rows(tmp_path)
    write_csv(tmp_path / "interfaces.csv", [{"part_id": "p", "interface_id": "i", "role": "bore", "diameter_mm": 20,
              "min_mm": "", "max_mm": "", "unit": "mm", "reviewer": "", "evidence": "https://manufacturer.example/source",
              "status": "source_supported", "source_id": "test_source"}])
    catalog = {"parts": [{"part_id": "p", "interfaces": []}]}
    assert load_knowledge(tmp_path, catalog, "include_source_supported")["interfaces"] == []


def test_numeric_roundoff_is_not_a_fit_tolerance():
    part = {"interfaces": [], "measurements": {"aabb_large_mm": {"value": 80.00000000000001, "unit": "mm", "status": "computed", "source": "fixture"}}}
    constraint = {"attribute": "aabb_large_mm", "op": "<=", "value": 80, "unit": "mm"}
    assert check_constraints(part, [constraint])["status"] == "pass"
    part["measurements"]["aabb_large_mm"]["value"] = 80.000001
    assert check_constraints(part, [constraint])["status"] == "fail"


def test_review_pack_has_blank_grades_and_preserves_expert_work(tmp_path):
    source_rows(tmp_path / "sources")
    parts = [{"part_id": f"p{i}", "split": split, "relative_path": f"bearing/{i}.stp", "label": "bearing",
              "file_sha256": f"hash{i}", "cylinder_candidates": []} for i, split in enumerate(["train", "val", "test"])]
    queries = [{"query_id": "dev_q", "split": "dev"}, {"query_id": "test_q", "split": "test"}]
    prepare_review_pack({"parts": parts}, queries, tmp_path / "sources", tmp_path / "review")
    rows = read_csv(tmp_path / "review/grading/test/judgments.csv")
    assert rows[0]["grade"] == rows[0]["reviewer"] == rows[0]["source"] == ""
    grid = read_csv(tmp_path / "review/knowledge/class_functions.csv")
    assert len(grid) == len(CLASSES)
    assert all(row["relevance"] == "" and row["confidence"] == "" and row["status"] == "draft" for row in grid)
    rows[0]["grade"] = "1"
    rows[0]["reviewer"] = "actual_expert"
    write_csv(tmp_path / "review/grading/test/judgments.csv", rows)
    prepare_review_pack({"parts": parts}, queries, tmp_path / "sources", tmp_path / "review")
    assert read_csv(tmp_path / "review/grading/test/judgments.csv")[0]["reviewer"] == "actual_expert"


def test_function_object_is_not_a_requested_part_category():
    from cadkg.query import parse_text
    knowledge = {"functions": [{"function_id": "support", "label": "support a rotating shaft", "aliases": "rotating shaft support"}]}
    parsed = parse_text("support a rotating shaft aabb_large_mm <= 100 mm", knowledge)
    assert parsed["function_ids"] == ["support"]
    assert parsed["category_ids"] == []
    explicit = parse_text("bearing to support a rotating shaft", knowledge)
    assert explicit["category_ids"] == ["bearing"]


def test_nonlargest_axis_disagreement_quarantines_numeric_evidence():
    part = {"geometry_audit": {}, "interfaces": [], "measurements": {
        f"aabb_{axis}_mm": {"value": value, "unit": "mm", "status": "computed", "source": "fixture"}
        for axis, value in zip(["x", "y", "z", "large"], [10, 20, 30, 30])}}
    bounds = {f"aabb_{axis}_mm": value for axis, value in zip(["x", "y", "z", "large"], [11, 20, 30, 30])}
    agreed, delta, axis_delta = record_bounds_crosscheck(part, bounds, .05)
    assert not agreed and delta == 0 and axis_delta == 1
    constraint = {"attribute": "aabb_large_mm", "op": "<=", "value": 50, "unit": "mm"}
    assert check_constraints(part, [constraint])["status"] == "unknown"
