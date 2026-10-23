"""Extension observer scoring and annotation packets retain study safeguards."""
import json
from pathlib import Path
import pytest
from scripts.evaluate_baseline import support_value
from scripts import export_experiment


def test_directed_observer_means_and_whole_job_fallback():
    job = {'predicate_ids': ['one', 'two', 'three', 'four']}
    observations = {pid: dict(status=status, views=['axial'] if status != 'unknown' else [], observation='visible')
                    for pid, status in zip(job['predicate_ids'], ['yes', 'no', 'unknown', 'yes'])}
    record = dict(status='completed', output_schema_valid=True, output_json=observations)
    assert support_value(job, record, 'directed') == (.625, True)
    observations['one']['views'] = []
    assert support_value(job, record, 'directed') == (.5, False)


@pytest.mark.parametrize('score,views,expected', [(.875, ['profile'], (.875, True)),
                                               (.5, [], (.5, True)),
                                               (.875, [], (.5, False)),
                                               (.9, ['axial'], (.5, False))])
def test_holistic_observer_validity(score, views, expected):
    job = {'predicate_ids': ['role_support']}
    record = dict(status='completed', output_schema_valid=True,
                  output_json={'role_support': dict(score=score, views=views, observation='visible')})
    assert support_value(job, record, 'holistic') == expected


def test_annotation_packet_does_not_copy_grades_or_claim_authorship(tmp_path, monkeypatch):
    def save(path, value):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    save('data/annotations/judgments.json', {'judgments': [dict(family_id='role', part_id='p', grade=2, inspection_id='Z1', reviewer='original')]})
    save('data/rankings/cohorts.json', {'closed': ['p']})
    rubric = tmp_path / 'data/study/relevance_rubric.md'
    rubric.parent.mkdir(parents=True)
    rubric.write_text('rubric')
    catalog = tmp_path / 'data/catalog/parts.csv'
    catalog.parent.mkdir(parents=True)
    catalog.write_text('part_id,file_sha256,relative_path\np,geometryhash,category/1.stp\n')
    monkeypatch.setattr(export_experiment, 'ROOT', tmp_path)
    packet = export_experiment.annotation_packet([], 1)
    row = packet['judgments'][0]
    assert row['grade'] is None and row['reviewer'] is None and row['human_expert_reviewed'] is None
    assert row['inspection_id'] == 'Z1' and row['source_sha256'] == 'geometryhash'
    assert json.loads((tmp_path / 'data/annotations/judgments.json').read_text())['judgments'][0]['grade'] == 2
