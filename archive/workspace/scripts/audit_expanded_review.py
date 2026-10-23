"""Withdraw unsupported proxy grades and prepare a clean matrix reproduction."""
from pathlib import Path
import shutil

from cadkg.io import read_json, write_json


def main():
    prior = read_json('annotations/intent_batch2/judgments.json')
    originals = {(r['family_id'],r['part_id']):r for r in prior['judgments']}
    path = Path('annotations/intent_batch2_expanded/judgments.json')
    data = read_json(path)
    archive = path.with_name('invalid_category_proxy_judgments.json')
    if not archive.exists():
        shutil.copyfile(path,archive)
    for row in data['judgments']:
        original = originals.get((row['family_id'],row['part_id']))
        if original:
            row.update(original,review_status='preserved_prior_ai_inspection')
        else:
            row.update(grade=None,decision=None,observation='',rationale='',
                       review_status='unreviewed',
                       inspection_origin='Unsupported category-derived grade withdrawn; source-view assessment pending')
    data.update(source='review_queue_pending_visual_inspection',
                reviewer='Unassigned for new pairs',ungraded_records=6751,
                disclosure='The former 6751 added grades were assigned by catalog category, not by individual CAD inspection. They are withdrawn and excluded from relevance evaluation.',
                invalid_prior_artifact=str(archive))
    write_json(path,data)
    write_json(archive.with_suffix('.INVALID.json'),{
        'valid_for_relevance_evaluation':False,
        'reason':'Grades were derived from source_label_for_sampling_only membership in ALIGNMENT. The associated visual observations were not supported by individual inspection.',
        'original_records_preserved':674,'unsupported_added_grades':6751})
    out = Path('outputs/intent_batch2_expanded_2x')
    out.mkdir(exist_ok=True)
    write_json(out/'coverage.json',{
        'target_task_part_assessments':14850,'target_unique_cads':1350,
        'completed_prior_ai_assessments':674,'pending_visual_assessments':14176,
        'valid_for_expanded_relevance_evaluation':False,
        'original_frozen_gallery_parts':1816,
        'excluded_from_prior675_as_outside_frozen_gallery':28,
        'class_labels_used_for_sampling_only':True})
    print('Withdrawn 6751 unsupported labels; original 674 preserved.',flush=True)


if __name__ == '__main__':
    main()
