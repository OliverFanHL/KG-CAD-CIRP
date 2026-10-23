from types import SimpleNamespace

import pytest

from cadkg.evaluation import load_judgments
from cadkg.io import write_csv
from cadkg.experiments.reference import numeric_reference, reference_grade


def test_provisional_numeric_reference_preserves_uncertainty():
    query = {"constraints": [{"attribute": "aabb_large_mm", "op": "<=", "value": 80, "unit": "mm"}]}
    measurement = {"surface_large_mm": 79.9, "aabb_quarantined": False, "faces_missing_mesh": 0}
    assert numeric_reference(query, measurement) == "unknown"
    measurement["surface_large_mm"] = 75
    assert numeric_reference(query, measurement) == "pass"
    measurement["surface_large_mm"] = 85
    assert numeric_reference(query, measurement) == "fail"
    measurement["aabb_quarantined"] = True
    assert numeric_reference(query, measurement) == "unknown"


def test_known_dimension_failure_cannot_become_a_positive_reference():
    observation = {"inspection_id": "T17", "nominal": {"belt": 2}}
    query = {"query_id": "belt_envelope", "function_ids": ["participate_belt_drive"]}
    assert reference_grade(observation, query, "nominal", "fail") == 0
    assert reference_grade(observation, query, "nominal", "unknown") == 1
    assert reference_grade(observation, query, "nominal", "pass") == 2


def test_ai_review_cannot_pass_the_independent_expert_gate(tmp_path):
    path = tmp_path / "ai.csv"
    write_csv(path, [{"query_id": "q", "part_id": "p", "grade": 2, "constraint_verdict": "not_requested",
                     "reviewer": "Codex AI assistant", "evidence": "Inspected fixture surface", "source": "ai_reference", "split": "test"}])
    query = {"query_id": "q", "split": "test", "constraints": [], "reference_part_id": None}
    retriever = SimpleNamespace(parts={"p": {"file_sha256": "fixture_hash"}}, candidate_ids=lambda: ["p"])
    with pytest.raises(ValueError, match="independent expert"):
        load_judgments(path, [query], retriever)


def test_reference_profiles_differ_even_when_core_vocabulary_does_not():
    observation = {"inspection_id": "T17", "nominal": {"belt": 2}}
    query = {"function_ids": ["participate_belt_drive"]}
    assert reference_grade(observation, {**query, "query_id": "belt_v_profile_gap"}, "nominal", "pass") == 2
    assert reference_grade(observation, {**query, "query_id": "belt_timing_profile_gap"}, "nominal", "pass") == 0
