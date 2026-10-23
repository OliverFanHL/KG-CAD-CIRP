"""Audit source binding and publish the reviewed annotations; no retrieval runs."""
from collections import Counter
from pathlib import Path
from cadkg.io import read_json, read_csv, write_json, write_csv, sha256

root=Path('annotations/intent_batch2_verified')
data=read_json(root/'judgments.json')
rows=data['judgments']
old=read_json('annotations/intent_batch2/judgments.json')['judgments']
manifest=read_csv('outputs/intent_batch2_expanded_2x/pool_manifest.csv')
catalog={p['part_id']:p for p in read_json('outputs/full_scale/bundle/catalog.json')['parts']}
keys={(r['family_id'],r['part_id']) for r in rows}
assert len(keys)==len(rows)==14850
assert len({r['part_id'] for r in rows})==1350
assert all((r['family_id'],r['part_id']) in keys for r in old)
assert all(r['grade'] in (0,1,2) and r['rationale'] and r['observation'] and Path(r['geometry_evidence']).exists() for r in rows)
for item in manifest:
    source=catalog[item['part_id']]
    assert source['split']=='test'
    assert sha256(source['absolute_path'])==item['source_sha256']==source['file_sha256']
    evidence=read_json(f"outputs/intent_batch2_expanded_2x/evidence/{item['inspection_id']}.json")
    assert evidence['source_sha256']==item['source_sha256']
    assert evidence['part_id']==item['part_id']
prior_ids={r['part_id'] for r in old}
summary=dict(records=len(rows),cads=len(manifest),task_families=len({r['family_id'] for r in rows}),
    prior_records_reverified=len(old),prior_cads_reverified=len(prior_ids),
    added_cads=len(manifest)-len(prior_ids),new_task_part_assessments=len(rows)-len(old),
    corrected_prior_grades=len(read_json(root/'corrections.json')),
    source_files_hash_verified=len(manifest),missing_evidence=0,duplicate_pairs=0,
    unreviewed=0,cannot_assess=sum(r['grade'] is None for r in rows),
    decisions=dict(Counter(r['decision'] for r in rows)),
    by_task={fid:dict(Counter(r['decision'] for r in rows if r['family_id']==fid)) for fid in sorted({r['family_id'] for r in rows})},
    author_type='AI visual reviewer',human_expert_reviewed=False,
    experiment_matrix_changed=False,experiments_run=False,
    judgments_sha256=sha256(root/'judgments.json'),
    rubric_sha256=sha256('annotations/review_tasks.json'),
    limitations=['Three rendered views support shape-based preliminary suitability, not hidden geometry, engineering fit, load capacity or material certification.',
      'Conditional grades retain unresolved interfaces; absence of cannot-assess grades does not imply all candidates have complete engineering evidence.',
      'All labels are AI-authored. There is no independent human adjudication or inter-rater reliability estimate.',
      'Task-part records share CADs and task families; 14,850 records are not 14,850 independent samples.',
      'The cohort includes prior retrieval-pooled candidates; it must not be described as an independent random benchmark.'])
write_json(root/'audit.json',summary)
write_csv(root/'judgments.csv',rows)
write_json('annotations/intent_batch2_expanded_2x/judgments.json',data)
write_json('annotations/intent_batch2_expanded_2x/verified_annotation_pointer.json',dict(canonical='annotations/intent_batch2_verified/judgments.json',audit='annotations/intent_batch2_verified/audit.json',sha256=summary['judgments_sha256']))
print(summary)
