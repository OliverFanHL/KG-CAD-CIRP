"""Export extension plans or blank annotation packets; never contact a model API."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadkg.study.io import read, sha, write


def baseline_plan(model, design, collections):
    directory = 'directed_luna' if design == 'directed' else 'repeated_observations'
    source = ROOT / 'data/observations' / directory / 'study.json'
    study = read(source)
    templates = study['jobs'] if design == 'directed' else [job for job in study['jobs'] if job['arm'] == 'role_only_luna' and job['replicate'] == 1]
    if len(templates) != 2084:
        raise ValueError('Expected 2084 unique role-part observer tasks')
    if design == 'holistic':
        contract = read(ROOT / 'data/study/role_only_contract.json')
        if any(job.get('instructions') != contract['instructions'] or job['schema'] != contract['schema'] for job in templates):
            raise ValueError('Holistic tasks must retain the exact whole-role observer contract')
    jobs = []
    for collection in range(1, collections + 1):
        for template in templates:
            job = {key: copy.deepcopy(template[key]) for key in (
                'family_id', 'part_id', 'image_path', 'image_sha256',
                'predicate_ids', 'schema', 'instruction_payload')}
            job['instructions'] = template.get('instructions', study['instructions'])
            job['collection'] = collection
            job['job_id'] = f"baseline__c{collection}__{job['family_id']}__{job['part_id']}"
            jobs.append(job)
    return dict(format=1, model=model, design=design, collections=collections,
                api_calls=0, jobs=jobs,
                source_study=str(source.relative_to(ROOT)).replace('\\', '/'),
                source_study_sha256=sha(source),
                input_sha256={path: sha(ROOT / path) for path in (
                    'data/annotations/judgments.json', 'data/queries/queries.json',
                    'data/rankings/cohorts.json', 'data/rankings/complete_rankings.npz',
                    'data/observations/repeated_observations/pool_manifest.json')})


def annotation_packet(part_ids, limit):
    with (ROOT / 'data/catalog/parts.csv').open(encoding='utf-8-sig', newline='') as stream:
        catalog = {row['part_id']: row for row in csv.DictReader(stream)}
    judgments = read(ROOT / 'data/annotations/judgments.json')['judgments']
    existing = {(row['family_id'], row['part_id']): row for row in judgments}
    families = sorted({row['family_id'] for row in judgments})
    selected = part_ids or sorted(read(ROOT / 'data/rankings/cohorts.json')['closed'])[:limit]
    if len(selected) != len(set(selected)) or any(pid not in catalog for pid in selected):
        raise ValueError('Part IDs must be unique and present in data/catalog/parts.csv')
    records = []
    for pid in selected:
        image = ROOT / 'data/observations/images' / (pid + '.png')
        for family in families:
            prior = existing.get((family, pid), {})
            records.append(dict(family_id=family, part_id=pid,
                                source_sha256=catalog[pid]['file_sha256'],
                                inspection_id=prior.get('inspection_id'),
                                source_step=catalog[pid]['relative_path'],
                                image_path=str(image.relative_to(ROOT)).replace('\\', '/') if image.exists() else None,
                                grade=None, decision=None, observation=None,
                                feature_evidence=[], rationale=None,
                                reviewer=None, review_date=None, review_method=None,
                                human_expert_reviewed=None, adjudication=None))
    return dict(format=1, status='BLANK_REVIEW_PACKET',
                parent_annotation_sha256=sha(ROOT / 'data/annotations/judgments.json'),
                rubric='data/study/relevance_rubric.md',
                rubric_sha256=sha(ROOT / 'data/study/relevance_rubric.md'),
                source_catalog_sha256=sha(ROOT / 'data/catalog/parts.csv'),
                judgments=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    baseline = commands.add_parser('export-baseline')
    baseline.add_argument('--model', required=True)
    baseline.add_argument('--design', choices=['directed', 'holistic'], default='directed')
    baseline.add_argument('--collections', type=int, default=3)
    baseline.add_argument('--output', type=Path, required=True)
    annotation = commands.add_parser('export-annotations')
    annotation.add_argument('--part-id', action='append', default=[])
    annotation.add_argument('--limit', type=int, default=10)
    annotation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output file; existing work will not be replaced.')
    if args.command == 'export-baseline':
        if args.collections < 1:
            parser.error('--collections must be positive')
        result = baseline_plan(args.model, args.design, args.collections)
    else:
        if args.limit < 1:
            parser.error('--limit must be positive')
        result = annotation_packet(args.part_id, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps(dict(output=str(args.output), records=len(result.get('jobs', result.get('judgments'))), api_calls=0)))


if __name__ == '__main__':
    main()
