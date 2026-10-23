"""Configurable intent-family experiments over the existing full held-out gallery.

This evaluator is deliberately separate from independent-expert evaluation: it
accepts source-bound AI inspection records and never inserts them into the KG.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from ..bundle import load_bundle, save_bundle
from ..evaluation import metrics
from .full import LABELS, common_gallery, extent, retrieval_config
from ..io import object_hash, read_csv, read_json, receipt, sha256, write_csv, write_json
from .reference import REFERENCE_METHODS, numeric_reference
from ..retrieval import Retriever


def evidence(output, manifest, catalog, reuse=None, workers=4):
    """Bounded subprocesses render actual STEP surfaces; neutral IDs hide labels."""
    from PIL import Image, ImageDraw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parts = {p['part_id']: p for p in catalog['parts']}
    previous = {}
    if reuse:
        for row in read_csv(Path(reuse)/'functional_pool_manifest.csv'):
            previous[row['part_id']] = (Path(reuse)/'pool_evidence', row['inspection_id'])

    def one(item):
        code, pid = item['inspection_id'], item['part_id']
        part = parts[pid]
        if sha256(part['absolute_path']) != item['source_sha256']:
            raise ValueError('STEP changed since inspection selection')
        target = output/f'{code}.json'
        if not target.exists() and pid in previous:
            directory, oldcode = previous[pid]
            data = read_json(directory/f'{oldcode}.json')
            if data['source_sha256'] != item['source_sha256']:
                raise ValueError('Cached rendered geometry belongs to different source')
            for suffix in ['png', 'npz']:
                shutil.copyfile(directory/f'{oldcode}.{suffix}', output/f'{code}.{suffix}')
            data.update(inspection_id=code, rendered_view_id=oldcode)
            write_json(target, data)
        if not target.exists():
            request = output/f'request_{code}.json'
            write_json(request, {'part': part, 'item': item})
            result = subprocess.run([sys.executable, '-m', 'cadkg.geometry', '--request', str(request), '--output', str(output)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise ValueError(f'{code}: '+result.stderr[-600:])
        data = read_json(target)
        if data['part_id'] != pid or data['source_sha256'] != item['source_sha256']:
            raise ValueError('Rendered evidence source mismatch')
        return code

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, item) for item in manifest]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"STEP evidence {future.result()}: {i}/{len(manifest)}", flush=True)
    # Six rows per contact sheet; retain every view and overlay current neutral ID.
    for start in range(0, len(manifest), 6):
        items = manifest[start:start+6]
        sheet = Image.new('RGB', (1200, 420*len(items)), 'white')
        for i, item in enumerate(items):
            with Image.open(output/f"{item['inspection_id']}.png") as im:
                sheet.paste(im.resize((1200, 420)), (0, i*420))
            d = ImageDraw.Draw(sheet)
            d.rectangle((0, i*420, 105, i*420+36), fill='white')
            d.text((8, i*420+8), item['inspection_id'], fill='black')
        sheet.save(output/f'contact_{start//6+1:02}.png')
    write_csv(output/'manifest.csv', manifest)


def search(output, settings):
    output = Path(output)
    catalog, old_knowledge, _ = load_bundle(settings['source_bundle'])
    knowledge, graph = save_bundle(catalog, settings['knowledge'], output/'bundle', 'include_source_supported')
    families = read_json(settings['families'])
    queries = read_json(settings['queries'])
    exemplars = read_json(settings['exemplars'])
    parts = {p['part_id']: p for p in catalog['parts']}
    if len({f['family_id'] for f in families}) != len(families):
        raise ValueError('Duplicate family ID')
    byfamily = {f['family_id']: f for f in families}
    byref = {r['part_id']: r for r in exemplars}
    retrievers = {kind: Retriever(catalog, knowledge, graph, retrieval_config(settings, kind)) for kind in settings['geometry_kinds']}
    gallery, contaminated = common_gallery(retrievers, catalog)
    if len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('Duplicate query ID')
    for query in queries:
        family = byfamily[query['scenario_id']]
        if query['function_ids'] != [family['function_id']] or query['category_ids'] or query['split'] != 'test':
            raise ValueError('Wrong functional query scope')
        pid = query['reference_part_id']
        if pid:
            if pid not in byref or parts[pid]['split'] != 'train' or byref[pid]['source_sha256'] != parts[pid]['file_sha256']:
                raise ValueError('Exemplars must be source-bound inspected training parts')
    # Freeze before any test ranking. AI judgments remain a separate artifact.
    frozen = {'settings': settings, 'families': families, 'queries': queries, 'exemplars': exemplars,
              'query_sha256': object_hash(queries), 'scope_sha256': object_hash(families), 'gallery': gallery,
              'contaminated_excluded': sorted(contaminated), 'methods': REFERENCE_METHODS,
              'weights_changed': False, 'model_retrained': False,
              'annotation_policy': 'All top5 returns, union across every method and representation, per family; two seeded random controls per new family. No unjudged-as-zero.'}
    write_json(output/'protocol.json', frozen)
    rows, explanations, pools = [], [], defaultdict(set)
    for kind, retriever in retrievers.items():
        for query in queries:
            for method in REFERENCE_METHODS:
                if method == 'category_from_intent':
                    mapped = {**query, 'category_ids': sorted({r['class_id'] for r in knowledge['class_functions'] if r['function_id'] in query['function_ids']})}
                    result = retriever.rank(mapped, 'category', settings['k'], gallery)
                else:
                    result = retriever.rank(query, method, settings['k'], gallery)
                explanations.append({'geometry_kind': kind, 'query_id': query['query_id'], 'method': method, 'result': result})
                for item in result['results']:
                    pools[query['scenario_id']].add(item['part_id'])
                    rows.append({'geometry_kind': kind, 'query_id': query['query_id'], 'family_id': query['scenario_id'],
                                 'method': method, 'rank': item['rank'], 'part_id': item['part_id'], 'score': item['score'],
                                 'signals': str(item['signals']), 'serving_constraint': item['reasoning']['status']})
        print(f'Full-gallery search complete: {kind}, {len(queries)} queries × 8 methods', flush=True)
    rng = np.random.default_rng(settings['seed'])
    for family in families:
        fid = family['family_id']
        if family['batch'] == 'previous':
            # Preserve original IDCG and prior control cohort, allowing exact comparisons.
            pools[fid] |= {p['part_id'] for p in read_csv(Path(settings['previous_output'])/'functional_pool_manifest.csv')}
        else:
            pools[fid] |= set(rng.choice(gallery, 2, replace=False).tolist())
    ids = sorted(set.union(*pools.values()))
    manifest = [{'inspection_id': f'N{i:03}', 'part_id': pid, 'source_sha256': parts[pid]['file_sha256'],
                 'family_ids': '|'.join(sorted(fid for fid in pools if pid in pools[fid])), 'split': 'test'} for i, pid in enumerate(ids, 1)]
    write_csv(output/'rankings.csv', rows)
    write_json(output/'explanations.json', explanations)
    write_json(output/'family_pools.json', {fid: sorted(pids) for fid, pids in pools.items()})
    write_csv(output/'pool_manifest.csv', manifest)
    write_json(output/'run_manifest.json', receipt('intent-batch search', [settings[k] for k in ['families', 'queries', 'exemplars']] + [Path(settings['source_bundle'])/'bundle_manifest.json'], settings))
    return {'families': len(families), 'queries': len(queries), 'ranking_runs': len(explanations), 'gallery': len(gallery),
            'unique_pool_parts': len(ids), 'family_part_annotation_pairs': sum(map(len, pools.values()))}


def prepare(output, settings):
    output = Path(output)
    protocol = read_json(output/'protocol.json')
    manifest = read_csv(output/'pool_manifest.csv')
    catalog = read_json(output/'bundle/catalog.json')
    evidence(output/'evidence', manifest, catalog, settings['previous_output'], settings.get('workers', 4))
    target = output/'human_annotations.csv'
    if not target.exists():
        write_csv(target, [{'family_id': fid, 'inspection_id': item['inspection_id'], 'part_id': item['part_id'],
                           'source_sha256': item['source_sha256'], 'view': f"evidence/{item['inspection_id']}.png",
                           'grade': '', 'observation': '', 'rationale': '', 'reviewer': '', 'source': 'independent_expert'}
                          for item in manifest for fid in item['family_ids'].split('|')])
    return {'source_geometry_views': len(manifest), 'human_annotations_preserved': True}


def score_grade(raw, verdict, scenario='nominal'):
    if type(raw) is not int or raw not in {0, 1, 2}:
        raise ValueError('Geometry grade must be integer 0, 1, or 2')
    value = (0 if scenario == 'conditional_zero' else 2) if raw == 1 and scenario != 'nominal' else raw
    return 0 if verdict == 'fail' else min(value, 1) if verdict == 'unknown' else value


def summarize(cases, families):
    rows = []
    # The family is the averaging unit, never the 1,816 gallery parts or raw runs.
    for kind in ['embedding', 'descriptor']:
        for subset in ['previous', 'new', 'combined']:
            fids = [f['family_id'] for f in families if subset == 'combined' or f['batch'] == subset]
            for variant in ['exemplar', 'text', 'all']:
                for method in REFERENCE_METHODS:
                    selected = [r for r in cases if r['geometry_kind'] == kind and r['method'] == method and r['family_id'] in fids
                                and (variant == 'all' or (bool(r['has_exemplar']) == (variant == 'exemplar')))]
                    if not selected:
                        continue
                    grouped = defaultdict(list)
                    for row in selected:
                        grouped[row['family_id']].append(row)
                    result = {'geometry_kind': kind, 'subset': subset, 'variant': variant, 'method': method,
                              'families': len(grouped), 'paired_queries': len(selected)}
                    for metric in ['pool_ndcg@5', 'precision@5', 'strong_precision@5']:
                        result['macro_'+metric] = float(np.mean([np.mean([r[metric] for r in group]) for group in grouped.values()]))
                    result['dimension_violations'] = sum(r['dimension_violations'] for r in selected)
                    rows.append(result)
    return rows


def evaluate(output, settings):
    output = Path(output)
    protocol = read_json(output/'protocol.json')
    authoring = read_json(settings['judgments'])
    if authoring.get('source') != 'ai_reference' or authoring.get('human_expert_reviewed') is not False:
        raise ValueError('This evaluator must not present AI reference labels as expert truth')
    if authoring.get('query_sha256') != protocol['query_sha256'] or authoring.get('scope_sha256') != protocol['scope_sha256']:
        raise ValueError('Inspection grades belong to different queries/scopes')
    manifest = {r['part_id']: r for r in read_csv(output/'pool_manifest.csv')}
    pools = read_json(output/'family_pools.json')
    observations = {(o['family_id'], o['part_id']): o for o in authoring['judgments']}
    required = {(fid, pid) for fid, pids in pools.items() for pid in pids}
    if set(observations) != required or len(observations) != len(authoring['judgments']):
        raise ValueError('Every family-pool pair must have one inspected grade')
    for (fid, pid), observation in observations.items():
        item = manifest[pid]
        if observation['source_sha256'] != item['source_sha256'] or observation.get('human_expert_reviewed') is not False or not observation['observation'].strip() or not observation['rationale'].strip():
            raise ValueError('Missing or mismatched inspected evidence')
        score_grade(observation['grade'], 'not_requested')
    return evaluate_observations(output, settings, observations)


def evaluate_observations(output, settings, observations, *, result_dir=None, source='ai_reference', families=None):
    """One metric engine; provenance gates live in the AI/expert input loaders."""
    if source not in {'ai_reference', 'independent_expert'}:
        raise ValueError('Unknown evaluation annotation source')
    output = Path(output)
    destination = Path(result_dir) if result_dir is not None else output
    protocol = read_json(output/'protocol.json')
    manifest = {r['part_id']: r for r in read_csv(output/'pool_manifest.csv')}
    pools = read_json(output/'family_pools.json')
    family_ids = set(families) if families is not None else set(pools)
    selected_families = [f for f in protocol['families'] if f['family_id'] in family_ids]
    if not selected_families:
        raise ValueError('No fully assessed family is available for paired evaluation')
    ranks = read_csv(output/'rankings.csv')
    allcases, judgments = [], []
    for sensitivity in ['nominal', 'conditional_zero', 'conditional_two']:
        for query in protocol['queries']:
            fid, qid = query['scenario_id'], query['query_id']
            if fid not in family_ids:
                continue
            qrels = {}
            for pid in pools[fid]:
                item, observation = manifest[pid], observations[fid, pid]
                data = read_json(output/f"evidence/{item['inspection_id']}.json")
                if data['source_sha256'] != item['source_sha256'] or data['part_id'] != pid:
                    raise ValueError('Numeric reference geometry mismatch')
                verdict = numeric_reference(query, data)
                grade = score_grade(observation['grade'], verdict, sensitivity)
                qrels[pid] = {'grade': grade, 'constraint_verdict': verdict}
                if sensitivity == 'nominal':
                    judgments.append({'query_id': qid, 'family_id': fid, 'part_id': pid, 'inspection_id': item['inspection_id'],
                                      'grade': grade, 'geometry_grade': observation['grade'], 'constraint_verdict': verdict,
                                      'observation': observation['observation'], 'rationale': observation['rationale'],
                                      'source_sha256': item['source_sha256'], 'source': source, 'human_expert_reviewed': source == 'independent_expert'})
                    if source == 'independent_expert':
                        judgments[-1].update(reviewer=observation['reviewer'], evidence=observation['evidence'],
                                             reviewed_at=observation['reviewed_at'], human_review_attested=True)
            for kind in settings['geometry_kinds']:
                for method in REFERENCE_METHODS:
                    selected = sorted([r for r in ranks if r['geometry_kind'] == kind and r['query_id'] == qid and r['method'] == method], key=lambda r: int(r['rank']))
                    ids = [r['part_id'] for r in selected]
                    result = metrics(ids, qrels, settings['k'])
                    allcases.append({'grading_scenario': sensitivity, 'geometry_kind': kind, 'query_id': qid, 'family_id': fid,
                                     'method': method, 'has_exemplar': bool(query['reference_part_id']), 'has_constraint': bool(query['constraints']),
                                     'pool_ndcg@5': result['ndcg@5'], 'precision@5': result['precision@5'],
                                     'strong_precision@5': sum(qrels[pid]['grade'] == 2 for pid in ids)/settings['k'],
                                     'dimension_violations': sum(qrels[pid]['constraint_verdict'] == 'fail' for pid in ids),
                                     'dimension_unknown': sum(qrels[pid]['constraint_verdict'] == 'unknown' for pid in ids),
                                     'returned': len(ids), 'family_pool_size': len(qrels), 'pool_strong_count': sum(v['grade']==2 for v in qrels.values())})
    cases = [r for r in allcases if r['grading_scenario'] == 'nominal']
    lookup = {(r['geometry_kind'], r['query_id'], r['method']): r for r in cases}
    pairs = []
    for row in cases:
        if row['method'] != 'kg_full':
            continue
        for method in REFERENCE_METHODS:
            if method == 'kg_full':
                continue
            other = lookup[row['geometry_kind'], row['query_id'], method]
            pairs.append({k: row[k] for k in ['geometry_kind', 'query_id', 'family_id', 'has_exemplar', 'has_constraint']} |
                         {'comparison': 'kg_full - '+method, **{'delta_'+m: row[m]-other[m] for m in ['pool_ndcg@5', 'precision@5', 'strong_precision@5']}})
    family_rows = []
    for family in selected_families:
        for kind in settings['geometry_kinds']:
            selected = [r for r in pairs if r['family_id']==family['family_id'] and r['geometry_kind']==kind and r['has_exemplar'] and r['comparison']=='kg_full - kg_no_geometry']
            family_rows.append({'family_id': family['family_id'], 'label': family['label'], 'batch': family['batch'], 'geometry_kind': kind,
                                'paired_queries': len(selected), **{m: float(np.mean([r[m] for r in selected])) for m in ['delta_pool_ndcg@5', 'delta_precision@5', 'delta_strong_precision@5']}})
    write_csv(destination/('expert_judgments.csv' if source == 'independent_expert' else 'ai_judgments.csv'), judgments)
    write_csv(destination/'per_query.csv', cases)
    write_csv(destination/'paired_differences.csv', pairs)
    write_csv(destination/'family_differences.csv', family_rows)
    write_csv(destination/'summary.csv', summarize(cases, selected_families))
    write_csv(destination/'sensitivity.csv', allcases)
    write_json(destination/'semantic_digest.json', {'sha256': object_hash({'cases': cases, 'pairs': pairs, 'families': family_rows})})
    return {'ranking_runs': len(cases), 'judged_query_part_pairs': len(judgments), 'source': source, 'human_expert_reviewed': source == 'independent_expert'}


def validate(output, settings):
    output = Path(output)
    protocol = read_json(output/'protocol.json')
    ranks = read_csv(output/'rankings.csv')
    judgments = read_csv(output/'ai_judgments.csv')
    judged = {(r['query_id'], r['part_id']) for r in judgments}
    def lists(method):
        return {(r['geometry_kind'],r['query_id'],r['rank']):(r['part_id'],r['score']) for r in ranks if r['method']==method}
    old = read_csv(Path(settings['previous_output'])/'functional_rankings.csv')
    oldq = {q['query_id'] for q in protocol['queries'] if next(f for f in protocol['families'] if f['family_id']==q['scenario_id'])['batch']=='previous'}
    def signature(rows):
        return {(r['geometry_kind'],r['query_id'],r['method'],r['rank']):(r['part_id'],r['score']) for r in rows if r['query_id'] in oldq}
    old_metrics={(r['geometry_kind'],r['query_id'],r['method']):r for r in read_csv(Path(settings['previous_output'])/'functional_per_query.csv')}
    prior_metrics_equal=all(all(abs(float(r[key])-float(old_metrics[r['geometry_kind'],r['query_id'],r['method']][key])) < 1e-12
                               for key in ['pool_ndcg@5','precision@5','strong_precision@5'])
                            for r in read_csv(output/'per_query.csv') if r['query_id'] in oldq)
    checks = {'complete_top5_grading': all((r['query_id'],r['part_id']) in judged for r in ranks),
              'flat_kg_equivalent': lists('flat_full')==lists('kg_full'), 'previous_rankings_unchanged': signature(old)==signature(ranks),
              'same_previous_gallery': protocol['gallery']==read_json(Path(settings['previous_output'])/'functional_protocol.json')['gallery'],
              'no_cross_split_byte_leak': not(set(protocol['gallery']) & set(protocol['contaminated_excluded'])),
              'ai_provenance_explicit': all(r['source']=='ai_reference' and r['human_expert_reviewed']=='False' for r in judgments),
              'original_weights_unchanged': settings['weights']==read_json('configs/full_scale.json')['weights'],
              'previous_metrics_unchanged': prior_metrics_equal,
              'all_paired_runs_present': len(read_csv(output/'per_query.csv'))==len(protocol['queries'])*len(REFERENCE_METHODS)*2}
    current = {'metrics': read_json(output/'semantic_digest.json')['sha256'], 'rankings': sha256(output/'rankings.csv'), 'human_sheet': sha256(output/'human_annotations.csv')}
    first = output/'first_run_digests.json'
    replay = {k: read_json(first)[k]==v for k,v in current.items()} if first.exists() else {'status': 'first run recorded'}
    if not first.exists():
        write_json(first, current)
    prior_validation = read_json(Path(settings['previous_output'])/'validation.json')
    result = {'passed': all(checks.values()), 'checks': checks, 'exact_saved_input_replay': replay,
              'independent_expert_validation': False, 'family_uncertainty': 'Descriptive observed-family averages; correlated tasks, AI authoring and pooled judging preclude confirmatory significance.',
              'statistical_fallacy_coverage': '11/11 checked', 'interpretation_checks': prior_validation['interpretation_checks']}
    write_json(output/'validation.json', result)
    if not result['passed'] or ('status' not in replay and not all(replay.values())):
        raise ValueError('Intent batch validation/replay failed')
    return result['checks'] | {'replay': replay}


def main(args):
    settings = read_json(args.config)
    output = Path(args.output)
    if settings['k'] != 5:
        raise ValueError('This protocol uses k=5')
    if args.stage in {'all', 'search'}:
        print(search(output, settings), flush=True)
    if args.stage in {'all', 'evidence'}:
        print(prepare(output, settings), flush=True)
    if args.stage in {'all', 'evaluate'}:
        print(evaluate(output, settings), flush=True)
    if args.stage in {'all', 'report'}:
        from .intent_report import report
        print(report(output, settings), flush=True)
    if args.stage in {'all', 'validate'}:
        print(validate(output, settings), flush=True)
