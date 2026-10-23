"""Additive training-source FPS positions 13-24 for two depleted audit pools.

Preserves the original 132-source files.  Reconstructs every original selection
using the identical saved training normalization and stable FPS ordering, checks
the entire prefix, and only then writes 24 separately shuffled neutral sources.
No audit decisions, test grades, retrieval ranks or metric files are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from cadkg.catalog import FEATURE_KEYS
from cadkg.io import read_json, read_csv, write_json, write_csv, sha256
from scripts.prepare_query_exemplar_pool import eligibility, finite_vector, farthest_points, stable, render_one
from scripts.prepare_feature_review import contact

POOL = WORKSPACE / 'outputs/query_diversity_v6/exemplar_pool'
TARGETS = {'support_body', 'tension_return'}


def original_integrity(out):
    freeze = read_json(out / 'pool_freeze.json')
    outputs = {k.replace('\\', '/'): v for k, v in freeze['outputs_sha256'].items()}
    for name, expected in outputs.items():
        if sha256(out / name) != expected:
            raise ValueError(f'Original frozen pool output changed: {name}')
    driver = WORKSPACE / 'scripts/prepare_query_exemplar_pool.py'
    snapshot = out / 'private/preparation_script_snapshot.py'
    expected = freeze['inputs_sha256'][str(driver)]
    if sha256(snapshot) != expected:
        raise ValueError('Exact original preparation-code snapshot does not match freeze')
    # This documented render-only adapter differs from the exact selection
    # snapshot; do not assert literal current-path equality for that one input.
    for name, expected_input in freeze['inputs_sha256'].items():
        if Path(name).resolve() != driver.resolve() and sha256(name) != expected_input:
            raise ValueError(f'Original selection input changed: {name}')
    return freeze, dict(original_preparation_script_snapshot=str(snapshot),
        original_preparation_script_sha256=expected, current_render_resume_adapter=str(driver),
        current_render_resume_adapter_sha256=sha256(driver),
        preexisting_difference='Original freeze predates Windows manifest-key and render-resume integrity adapter fixes; exact preparation snapshot is preserved.',
        original_output_sha256=outputs)


def prepare(out=POOL):
    if (out / 'expansion_freeze.json').exists():
        raise FileExistsError('Expansion already frozen; use render without reselection')
    if (out.parent / 'experiment_freeze.json').exists():
        raise ValueError('Additive sources must be prepared before experiment freeze and scores')
    freeze, continuity = original_integrity(out)
    baseline = read_csv(out / 'private/candidate_manifest.csv')
    if len(baseline) != 132 or freeze['per_family'] != 12:
        raise ValueError('Expected original frozen 132-source, twelve-position design')
    catalog_path = WORKSPACE / 'outputs/intent_batch2_matrix_reproduction/bundle/catalog.json'
    knowledge_path = catalog_path.with_name('knowledge.json')
    annotations = WORKSPACE / 'annotations/intent_batch2'
    catalog = read_json(catalog_path)
    links = read_json(knowledge_path)['class_functions']
    families = read_json(annotations / 'families.json')
    refs = [(q['scenario_id'], q.get('reference_part_id')) for q in read_json(annotations / 'queries.json')]
    old = read_json(annotations / 'exemplars.json')
    old_ids = {p['part_id'] for p in old}
    old_hashes = {p['source_sha256'] for p in old}
    by_id = {p['part_id']: p for p in catalog['parts']}
    heldout = [p for p in catalog['parts'] if p['split'] in {'val', 'test'}]
    heldout_hashes = set()
    for i, p in enumerate(heldout, 1):
        actual = sha256(p['absolute_path'])
        if actual != p['file_sha256']:
            raise ValueError(f'Held-out source bytes changed: {p["part_id"]}')
        heldout_hashes.add(actual)
        if i % 500 == 0:
            print(f'Held-out source bytes verified: {i}/{len(heldout)}', flush=True)
    seed, dimension = freeze['seed'], freeze['embedding_dimension']
    eligible, seen, exclusions = [], set(), Counter()
    train = sorted((p for p in catalog['parts'] if p['split'] == 'train'),
                   key=lambda p: stable(seed, 'deduplicate', p['part_id']))
    for p in train:
        reason = eligibility(p, dimension)
        if not reason and p['file_sha256'] in heldout_hashes:
            reason = 'exact_heldout_source_overlap'
        if not reason and (p['part_id'] in old_ids or p['file_sha256'] in old_hashes):
            reason = 'fixed_exemplar_continuity_only'
        if not reason and p['file_sha256'] in seen:
            reason = 'duplicate_training_source_bytes'
        if reason:
            exclusions[reason] += 1
        else:
            eligible.append(p)
            seen.add(p['file_sha256'])
    transform = read_json(out / 'private/descriptor_transform.json')
    mean, scale = np.asarray(transform['mean']), np.asarray(transform['scale'])
    raw = np.asarray([[p['features'][k] for k in FEATURE_KEYS] for p in eligible], dtype=float)
    reproduced_scale = raw.std(0)
    reproduced_scale[reproduced_scale < 1e-8] = 1.0
    if (len(eligible) != transform['fit_count'] or list(FEATURE_KEYS) != transform['keys']
            or not np.array_equal(raw.mean(0), mean) or not np.array_equal(reproduced_scale, scale)):
        raise ValueError('Original training normalization population or transform changed')
    vectors = {p['part_id']: (x - mean) / scale for p, x in zip(eligible, raw)}
    old_lookup = {(r['family_id'], int(r['selection_step'])): r for r in baseline}
    all_old_hashes = {r['source_sha256'] for r in baseline}
    used, extra, checks = set(), [], []
    for family in families:
        fid = family['family_id']
        mapping = [r for r in links if r['function_id'] == family['function_id']]
        if len(mapping) != 1 or mapping[0]['relevance'] * mapping[0]['confidence'] != 1:
            raise ValueError(f'Unexpected development mapping: {fid}')
        label = mapping[0]['class_id']
        candidates = [p for p in eligible if p['label'] == label and p['file_sha256'] not in used]
        anchors_ids = sorted({pid for family_id, pid in refs if family_id == fid and pid in old_ids})
        anchors_ids = [pid for pid in anchors_ids if by_id[pid]['split'] == 'train'
                      and by_id[pid]['file_sha256'] not in heldout_hashes
                      and eligibility(by_id[pid], dimension) is None]
        anchors = np.asarray([([by_id[pid]['features'][k] for k in FEATURE_KEYS] - mean) / scale
                              for pid in anchors_ids], dtype=float)
        n = 24 if fid in TARGETS else 12
        chosen = farthest_points(candidates, vectors, anchors, n, seed, fid)
        for p, step, distance, reason in chosen:
            used.add(p['file_sha256'])
            if step <= 12:
                original = old_lookup[fid, step]
                if (p['part_id'], p['file_sha256']) != (original['part_id'], original['source_sha256']):
                    raise ValueError(f'Original FPS prefix changed at {fid}/{step}')
                checks.append(dict(family_id=fid, selection_step=step, part_id=p['part_id'],
                                   source_sha256=p['file_sha256'], original_review_id=original['review_id']))
            else:
                if p['file_sha256'] in all_old_hashes or p['file_sha256'] in heldout_hashes | old_hashes:
                    raise ValueError('Expansion source overlaps excluded bytes')
                if sha256(p['absolute_path']) != p['file_sha256']:
                    raise ValueError(f'Expansion source hash mismatch: {p["part_id"]}')
                extra.append(dict(old_lookup[fid, 12], part_id=p['part_id'], source_sha256=p['file_sha256'],
                    source_path=p['absolute_path'], selection_step=step, selection_reason=reason,
                    min_descriptor_distance_squared=distance, candidate_id=f'{fid}_candidate{step:02d}',
                    audit_tier='expansion', suitability='pending_individual_source_audit'))
    if len(checks) != 132 or len(extra) != 24 or len({r['source_sha256'] for r in extra}) != 24:
        raise ValueError('Wrong expansion size or duplicate source bytes')
    extra.sort(key=lambda r: stable(seed, 'additive_public_blind_order', r['source_sha256']))
    for i, row in enumerate(extra, 133):
        row['review_id'] = f'Q{i:04d}'
        row['image'] = str((out / 'views' / f'Q{i:04d}.png').resolve())
    write_csv(out / 'private/expansion_candidate_manifest.csv', extra)
    blind = [dict(review_id=r['review_id'], image=r['image']) for r in extra]
    write_csv(out / 'expansion_blind_manifest.csv', blind)
    write_csv(out / 'coordinator_expansion_manifest.csv', blind[:12])
    write_csv(out / 'reviewer_expansion_manifest.csv', blind[12:])
    write_json(out / 'private/expansion_selection_audit.json', dict(
        target_families=sorted(TARGETS), positions=list(range(13, 25)),
        prefix_checks=checks, all_original_132_prefixes_identical=True,
        normalization_reproduced_exactly=True, source_byte_exclusions=dict(exclusions),
        continuity=continuity))
    inputs = [Path(__file__), WORKSPACE / 'scripts/prepare_query_exemplar_pool.py',
        out / 'private/preparation_script_snapshot.py', WORKSPACE / 'scripts/prepare_feature_review.py',
        catalog_path, knowledge_path,
        annotations / 'families.json', annotations / 'queries.json', annotations / 'exemplars.json',
        out / 'pool_freeze.json', out / 'private/candidate_manifest.csv',
        out / 'private/descriptor_transform.json', out.parent / 'initial_source_audit_freeze.json']
    outputs = [out / 'private/expansion_candidate_manifest.csv', out / 'expansion_blind_manifest.csv',
               out / 'coordinator_expansion_manifest.csv', out / 'reviewer_expansion_manifest.csv',
               out / 'private/expansion_selection_audit.json']
    write_json(out / 'expansion_freeze.json', dict(version='query-exemplar-additive-pool-v6',
        stage='Before additive source audit and before any v6 retrieval score', seed=seed,
        candidates=24, target_family_count=2, positions=list(range(13, 25)),
        original_132_prefixes_identical=True, original_normalization_reused=True,
        source_bytes_unique=24, total_pool_source_bytes_unique=156,
        exact_val_test_overlap=0, old_fixed_exemplar_overlap=0, original_pool_overlap=0,
        heldout_sources_byte_verified=len(heldout), test_grades_read=False, v6_scores_read=False,
        original_selection_continuity=continuity,
        selection='Continue original seeded descriptor FPS to positions 13-24; first-four eligible in complete order remains fixed.',
        inputs_sha256={str(p):sha256(p) for p in inputs},
        outputs_sha256={p.relative_to(out).as_posix():sha256(p) for p in outputs}))
    print('Frozen 24 additive sources; every original 132-source prefix is identical.', flush=True)


def render(out=POOL, workers=2):
    freeze = read_json(out / 'expansion_freeze.json')
    for name, expected in freeze['inputs_sha256'].items():
        if sha256(name) != expected:
            raise ValueError(f'Expansion input changed after freeze: {name}')
    for name, expected in freeze['outputs_sha256'].items():
        if sha256(out / name) != expected:
            raise ValueError(f'Expansion output changed after freeze: {name}')
    rows = read_csv(out / 'private/expansion_candidate_manifest.csv')
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(render_one, row): row['review_id'] for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(review_id=futures[future], error=f'{type(exc).__name__}: {exc}'))
            print(f'Additive source views: {i}/{len(rows)}', flush=True)
    bindings = []
    for row in rows:
        image = Path(row['image'])
        if image.exists() and image.with_suffix('.json').exists():
            binding = read_json(image.with_suffix('.json'))
            if (binding.get('public_id') == row['review_id'] and binding['image_sha256'] == sha256(image)
                    and binding['source_sha256'] == row['source_sha256']):
                bindings.append(dict(review_id=row['review_id'], image_sha256=sha256(image),
                    source_sha256=row['source_sha256'], binding_sha256=sha256(image.with_suffix('.json'))))
    write_json(out / 'expansion_render_audit.json', dict(complete=len(bindings)==24 and not errors,
        expected=24, rendered=len(bindings), bindings=bindings, errors=errors,
        renderer_sha256=sha256(WORKSPACE / 'scripts/prepare_feature_review.py')))
    if errors or len(bindings) != 24:
        raise RuntimeError('Additive renders incomplete; see expansion_render_audit.json')
    for prefix in ['coordinator_expansion', 'reviewer_expansion']:
        contact(read_csv(out / f'{prefix}_manifest.csv'), out / f'{prefix}_pages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'render', 'all'])
    parser.add_argument('--output', type=Path, default=POOL)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.action in {'prepare', 'all'}:
        prepare(args.output.resolve())
    if args.action in {'render', 'all'}:
        render(args.output.resolve(), args.workers)
