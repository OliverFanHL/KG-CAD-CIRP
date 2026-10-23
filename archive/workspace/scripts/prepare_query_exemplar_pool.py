"""Prepare training-only, rank-independent geometry candidates for source audit.

This creates an audit pool, not approved exemplars or query variants.  Public
images and manifests contain neutral shuffled Q IDs.  Family/class associations
and source paths live only in private/.  No judgment or result file is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from cadkg.catalog import CLASSES, FEATURE_KEYS
from cadkg.io import read_json, read_csv, write_json, write_csv, sha256
from scripts.prepare_feature_review import render as source_render, contact

DEFAULT_BUNDLE = WORKSPACE / 'outputs/intent_batch2_matrix_reproduction/bundle'
DEFAULT_OUT = WORKSPACE / 'outputs/query_diversity_v6/exemplar_pool'
DEFAULT_ANNOTATIONS = WORKSPACE / 'annotations/intent_batch2'
ROLE_CODES = dict(support_body='S', shaft_connector='C', belt_wheel='B',
                 threaded_clamp='T', nut_retention='N', bracket_mount='M',
                 flanged_passage='F', gear_mesh='G', shaft_drive='D',
                 compression_return='P', tension_return='E')
SEED = 20261021


def stable(seed, *values):
    return hashlib.sha256(':'.join(map(str, (seed, *values))).encode()).hexdigest()


def finite_vector(value, size=None):
    try:
        a = np.asarray(value, dtype=float)
        return a.ndim == 1 and a.size > 0 and (size is None or a.size == size) and bool(np.isfinite(a).all())
    except (TypeError, ValueError):
        return False


def eligibility(part, embedding_dimension):
    if not part.get('file_sha256') or len(part['file_sha256']) != 64:
        return 'missing_source_hash'
    descriptor = [part.get('features', {}).get(k) for k in FEATURE_KEYS]
    if any(v is None for v in descriptor) or not finite_vector(descriptor, len(FEATURE_KEYS)):
        return 'invalid_descriptor'
    embedding = part.get('embedding')
    if not finite_vector(embedding, embedding_dimension) or np.linalg.norm(embedding) <= 1e-12:
        return 'invalid_embedding'
    prediction = part.get('prediction') or {}
    probs = prediction.get('probabilities', {})
    if prediction.get('status') != 'predicted' or set(probs) != set(CLASSES):
        return 'invalid_prediction'
    values = [probs[c] for c in CLASSES]
    if not finite_vector(values, len(CLASSES)):
        return 'invalid_prediction'
    values = np.asarray(values, dtype=float)
    if np.any(values < 0) or np.any(values > 1) or abs(float(values.sum()) - 1) > 1e-6:
        return 'invalid_prediction'
    return None


def farthest_points(rows, vectors, anchors, n, seed, family):
    """Nearest-to-fixed-continuity start, then greedy maximum minimum distance.

    Stable SHA ordering breaks equal-distance ties without part filename order.
    Fixed references act only as geometric anchors, never accepted pool items.
    """
    rows = sorted(rows, key=lambda p: stable(seed, family, p['part_id']))
    x = np.asarray([vectors[p['part_id']] for p in rows])
    if len(rows) < n:
        raise ValueError(f'{family}: only {len(rows)} unused unique candidates for {n} slots')
    if len(anchors):
        anchor_d2 = ((x[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
        minimum = anchor_d2.min(axis=1)
        first = int(np.argmin(minimum))
        first_reason = 'nearest_to_fixed_training_reference'
    else:
        minimum = np.full(len(rows), np.inf)
        first = 0
        first_reason = 'seeded_hash_order_no_eligible_fixed_reference'
    chosen = []
    for step in range(n):
        index = first if step == 0 else int(np.argmax(minimum))
        distance = float(minimum[index]) if np.isfinite(minimum[index]) else None
        chosen.append((rows[index], step + 1, distance, first_reason if step == 0 else 'descriptor_farthest_point'))
        minimum = np.minimum(minimum, ((x - x[index]) ** 2).sum(axis=1))
        for p, *_ in chosen:
            minimum[next(i for i, row in enumerate(rows) if row['part_id'] == p['part_id'])] = -np.inf
    return chosen


def prepare(bundle, annotations, out, per_family=12, seed=SEED, initial_per_family=6):
    freeze_path = out / 'pool_freeze.json'
    if freeze_path.exists():
        raise FileExistsError(f'Pool already prepared at {out}; use render/contact to resume without reselection')
    input_paths = [bundle / 'catalog.json', bundle / 'knowledge.json',
                   annotations / 'families.json', annotations / 'queries.json',
                   annotations / 'exemplars.json', Path(__file__),
                   WORKSPACE / 'scripts/prepare_feature_review.py']
    catalog = read_json(input_paths[0])
    # Only class-function links are used from knowledge; no instance assertions.
    class_functions = read_json(input_paths[1])['class_functions']
    families = read_json(input_paths[2])
    # Original reference IDs are continuity anchors only; no outcomes are loaded.
    references = [(q['scenario_id'], q.get('reference_part_id')) for q in read_json(input_paths[3])]
    old = read_json(input_paths[4])
    if len(families) != 11 or {f['family_id'] for f in families} != set(ROLE_CODES):
        raise ValueError('Expected the eleven frozen task families')
    by_id = {p['part_id']: p for p in catalog['parts']}
    if len(by_id) != len(catalog['parts']):
        raise ValueError('Duplicate catalog part IDs')
    old_ids = {e['part_id'] for e in old}
    old_hashes = {e['source_sha256'] for e in old}
    if any(pid not in by_id for pid in old_ids):
        raise ValueError('Fixed exemplar absent from catalog')
    if any(by_id[e['part_id']]['file_sha256'] != e['source_sha256'] for e in old):
        raise ValueError('Fixed exemplar hash does not match catalog')
    # Recheck held-out bytes, not just stale catalog hashes.  No labels or
    # measurements from val/test enter the selection or normalization.
    heldout = [p for p in catalog['parts'] if p['split'] in {'val', 'test'}]
    heldout_hashes = set()
    for index, p in enumerate(heldout, 1):
        actual = sha256(p['absolute_path'])
        if actual != p['file_sha256']:
            raise ValueError(f'Held-out source hash mismatch: {p["part_id"]}')
        heldout_hashes.add(actual)
        if index % 500 == 0:
            print(f'Verified held-out source bytes: {index}/{len(heldout)}', flush=True)
    dimensions = Counter(len(p['embedding']) for p in catalog['parts']
                         if p['split'] == 'train' and finite_vector(p.get('embedding')))
    if len(dimensions) != 1:
        raise ValueError(f'Expected one consistent training embedding dimension: {dict(dimensions)}')
    dimension = next(iter(dimensions))
    exclusions = Counter()
    eligible, seen_hashes = [], set()
    train = sorted((p for p in catalog['parts'] if p['split'] == 'train'),
                   key=lambda p: stable(seed, 'deduplicate', p['part_id']))
    for p in train:
        reason = eligibility(p, dimension)
        if not reason and p['file_sha256'] in heldout_hashes:
            reason = 'exact_heldout_source_overlap'
        if not reason and (p['part_id'] in old_ids or p['file_sha256'] in old_hashes):
            reason = 'fixed_exemplar_continuity_only'
        if not reason and p['file_sha256'] in seen_hashes:
            reason = 'duplicate_training_source_bytes'
        if reason:
            exclusions[reason] += 1
        else:
            eligible.append(p)
            seen_hashes.add(p['file_sha256'])
    if not eligible:
        raise ValueError('No eligible training sources')
    raw = np.asarray([[p['features'][k] for k in FEATURE_KEYS] for p in eligible], dtype=float)
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-8] = 1.0
    vectors = {p['part_id']: (x - mean) / scale for p, x in zip(eligible, raw)}
    transform = dict(kind='descriptor', fit_split='train', fit_count=len(eligible),
                     keys=FEATURE_KEYS, mean=mean.tolist(), scale=scale.tolist(),
                     missing_values='complete finite descriptors required; no imputation',
                     distance='squared Euclidean after training-only standardization')
    selected, used_hashes, family_audit = [], set(), []
    for family in families:
        fid, function = family['family_id'], family['function_id']
        links = [r for r in class_functions if r['function_id'] == function]
        if len(links) != 1 or links[0]['relevance'] * links[0]['confidence'] != 1:
            raise ValueError(f'Unexpected class-function mapping for {fid}')
        label = links[0]['class_id']
        candidates = [p for p in eligible if p['label'] == label and p['file_sha256'] not in used_hashes]
        anchor_ids = sorted({pid for family_id, pid in references if family_id == fid and pid in old_ids})
        anchor_ids = [pid for pid in anchor_ids if by_id[pid]['split'] == 'train'
                      and by_id[pid]['file_sha256'] not in heldout_hashes
                      and eligibility(by_id[pid], dimension) is None]
        for pid in anchor_ids:
            if sha256(by_id[pid]['absolute_path']) != by_id[pid]['file_sha256']:
                raise ValueError(f'Fixed training source hash mismatch: {pid}')
        anchors = np.asarray([([by_id[pid]['features'][k] for k in FEATURE_KEYS] - mean) / scale
                              for pid in anchor_ids], dtype=float)
        picked = farthest_points(candidates, vectors, anchors, per_family, seed, fid)
        family_audit.append(dict(family_id=fid, function_id=function, development_class=label,
                                 eligible_unused_unique_sources=len(candidates),
                                 fixed_training_anchor_ids=anchor_ids, requested=per_family, selected=len(picked)))
        for p, step, distance, reason in picked:
            if sha256(p['absolute_path']) != p['file_sha256']:
                raise ValueError(f'Selected training source hash mismatch: {p["part_id"]}')
            used_hashes.add(p['file_sha256'])
            selected.append(dict(family_id=fid, role=ROLE_CODES[fid], function_id=function,
                                 development_class=label, part_id=p['part_id'], split='train',
                                 source_sha256=p['file_sha256'], source_path=p['absolute_path'],
                                 selection_step=step, selection_reason=reason,
                                 min_descriptor_distance_squared=distance,
                                 candidate_id=f'{fid}_candidate{step:02d}',
                                 audit_tier='initial' if step <= initial_per_family else 'reserve',
                                 suitability='pending_individual_source_audit'))
    # The displayed order cannot reveal consecutive family blocks.
    selected.sort(key=lambda r: stable(seed, 'public_blind_order', r['source_sha256']))
    for i, row in enumerate(selected, 1):
        row['review_id'] = f'Q{i:04d}'
        row['image'] = str((out / 'views' / f'Q{i:04d}.png').resolve())
    assert len(selected) == 11 * per_family and len(used_hashes) == len(selected)
    assert not used_hashes & heldout_hashes and not used_hashes & old_hashes
    write_csv(out / 'private/candidate_manifest.csv', selected)
    for tier, name in [('initial', 'blind_manifest.csv'), ('reserve', 'reserve_blind_manifest.csv')]:
        write_csv(out / name, [dict(review_id=r['review_id'], image=r['image']) for r in selected if r['audit_tier'] == tier])
    write_json(out / 'private/descriptor_transform.json', transform)
    write_json(out / 'private/selection_audit.json', dict(families=family_audit, exclusions=dict(exclusions)))
    write_json(freeze_path, dict(version='query-exemplar-pool-v6', seed=seed,
        candidates=len(selected), per_family=per_family, families=11, split='train',
        initial_per_family=initial_per_family, initial_candidates=11 * initial_per_family,
        reserve_candidates=11 * (per_family - initial_per_family),
        reserve_policy='FPS positions 7-12 frozen before audit; render on request without reselection.',
        selection='Fixed-training-reference nearest start, then descriptor farthest-point diversity; SHA-seeded tie breaks.',
        source_bytes_unique=len(used_hashes), exact_val_test_overlap=0, old_fixed_exemplar_overlap=0,
        heldout_sources_byte_verified=len(heldout), embedding_dimension=dimension,
        valid_descriptor_and_embedding_and_prediction_required=True,
        prediction_probabilities_used_for_selection=False,
        catalog_labels_used_for_development_pool_only=True,
        test_grades_read=False, retrieval_rankings_read=False, query_variants_frozen=False,
        suitability='All candidates require individual frozen-rubric source audit before selection.',
        inputs_sha256={str(p): sha256(p) for p in input_paths},
        outputs_sha256={p.relative_to(out).as_posix(): sha256(p) for p in [
            out / 'private/candidate_manifest.csv', out / 'blind_manifest.csv', out / 'reserve_blind_manifest.csv',
            out / 'private/descriptor_transform.json', out / 'private/selection_audit.json']}))
    print(json.dumps(dict(candidates=len(selected), unique_source_bytes=len(used_hashes), exclusions=dict(exclusions))), flush=True)


def render_one(row):
    if sha256(row['source_path']) != row['source_sha256']:
        raise ValueError(f'Source bytes changed since pool freeze: {row["review_id"]}')
    target = Path(row['image'])
    sidecar = target.with_suffix('.json')
    if target.exists() and sidecar.exists():
        binding = read_json(sidecar)
        if (binding.get('render_version') == 2 and binding.get('public_id') == row['review_id']
                and binding.get('source_sha256') == row['source_sha256']
                and binding.get('image_sha256') == sha256(target)):
            return row['review_id']
        raise ValueError(f'Existing render binding mismatch: {row["review_id"]}')
    # Use the identical source renderer.  The neutral ID also occupies its
    # temporary part_id field; sanitize the resulting public binding explicitly.
    item = {k: row[k] for k in ['review_id', 'source_sha256', 'source_path', 'image']}
    item['part_id'] = row['review_id']
    source_render(item)
    binding = read_json(sidecar)
    binding.pop('part_id', None)
    binding['public_id'] = row['review_id']
    write_json(sidecar, binding)
    return row['review_id']


def render_pool(out, workers=2, reserve=False):
    freeze = read_json(out / 'pool_freeze.json')
    frozen_outputs = {key.replace('\\', '/'): value for key, value in freeze['outputs_sha256'].items()}
    for name in ['private/candidate_manifest.csv', 'blind_manifest.csv', 'reserve_blind_manifest.csv']:
        if sha256(out / name) != frozen_outputs[name]:
            raise ValueError(f'Frozen pool manifest modified: {name}')
    helper = WORKSPACE / 'scripts/prepare_feature_review.py'
    if sha256(helper) != freeze['inputs_sha256'][str(helper)]:
        raise ValueError('Source renderer changed since pool preparation')
    tier = 'reserve' if reserve else 'initial'
    rows = [r for r in read_csv(out / 'private/candidate_manifest.csv') if r['audit_tier'] == tier]
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(render_one, row): row['review_id'] for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(review_id=code, error=f'{type(exc).__name__}: {exc}'))
            print(f'Source views: {i}/{len(rows)} ({code})', flush=True)
    bindings = []
    for row in rows:
        target = Path(row['image'])
        if target.exists() and target.with_suffix('.json').exists():
            b = read_json(target.with_suffix('.json'))
            if b.get('public_id') == row['review_id'] and b.get('image_sha256') == sha256(target):
                bindings.append(dict(review_id=row['review_id'], image_sha256=b['image_sha256'],
                                     source_sha256=b['source_sha256'], binding_sha256=sha256(target.with_suffix('.json'))))
    write_json(out / ('reserve_render_audit.json' if reserve else 'render_audit.json'), dict(expected=len(rows), rendered=len(bindings),
        complete=len(bindings) == len(rows) and not errors, errors=errors, bindings=bindings,
        renderer_sha256=sha256(WORKSPACE / 'scripts/prepare_feature_review.py'),
        rendering_driver_sha256=sha256(Path(__file__)),
        rendering='Identical six-view VTK renderer: hidden edges; uncapped X/Y clips; neutral Q IDs; no family labels.'))
    if errors or len(bindings) != len(rows):
        raise RuntimeError('Some sources could not be rendered; see render_audit.json and resume render')
    contact(read_csv(out / ('reserve_blind_manifest.csv' if reserve else 'blind_manifest.csv')),
            out / ('reserve_pages' if reserve else 'pages'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'render', 'render-reserve', 'contact', 'all'])
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-family', type=int, default=12)
    parser.add_argument('--initial-per-family', type=int, default=6)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.initial_per_family < 1 or args.per_family < args.initial_per_family or args.workers < 1:
        parser.error('positive counts required with initial-per-family <= per-family')
    if args.action in {'prepare', 'all'}:
        prepare(args.bundle.resolve(), args.annotations.resolve(), args.output.resolve(), args.per_family,
                args.seed, args.initial_per_family)
    if args.action in {'render', 'render-reserve', 'all'}:
        render_pool(args.output.resolve(), args.workers, reserve=args.action == 'render-reserve')
    if args.action == 'contact':
        contact(read_csv(args.output / 'blind_manifest.csv'), args.output / 'pages')


if __name__ == '__main__':
    main()
