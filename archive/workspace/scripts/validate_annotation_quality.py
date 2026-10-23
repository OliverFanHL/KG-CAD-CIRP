"""Verify the sampled correction release against actual source files and parent."""
from pathlib import Path
from cadkg.io import read_json,read_csv,write_json,sha256

root=Path('outputs/annotation_quality_audit_v2')
parent=read_json('annotations/intent_batch2_verified/judgments.json')
child=read_json('annotations/intent_batch2_audited_v2/judgments.json')
old={(r['family_id'],r['part_id']):r for r in parent['judgments']}
new={(r['family_id'],r['part_id']):r for r in child['judgments']}
assert old.keys()==new.keys() and len(old)==len(parent['judgments'])==len(child['judgments'])==14850
manifest=read_csv(root/'sample_manifest.csv');sample={r['part_id'] for r in manifest}
assert len(sample)==len(manifest)==166
actual={(f,p) for f,p in old if old[f,p]['grade']!=new[f,p]['grade']}
logged={(r['family_id'],r['part_id']) for r in read_csv('annotations/intent_batch2_audited_v2/corrections.csv')}
assert actual==logged and len(actual)==36
assert all(old[key]==new[key] for key in old if key[1] not in sample)
parts={r['part_id']:r for r in read_json('outputs/intent_batch2_matrix_reproduction/bundle/catalog.json')['parts']}
for item in manifest:
    assert sha256(parts[item['part_id']]['absolute_path'])==item['source_sha256']
    assert Path(item['geometry_evidence']).is_file()
assert child['parent_annotation_sha256']==sha256('annotations/intent_batch2_verified/judgments.json')
gate=read_json(root/'quality_gate.json')
assert gate['final_annotation_sha256']==sha256('annotations/intent_batch2_audited_v2/judgments.json')
result=dict(status='VERIFIED_SOURCE_BINDINGS_AND_CORRECTION_PACKAGING',sampled_step_files_hash_checked=166,
    unique_records=14850,changed_grades=36,unsampled_records_preserved=True,parent_annotation_preserved=True,
    review_notes_sha256=sha256(root/'review_notes.txt'),sampling_plan_sha256=sha256(root/'sampling_plan.json'),
    revised_annotation_sha256=gate['final_annotation_sha256'],independent_human_validity=False)
write_json(root/'validation.json',result)
print(result)
