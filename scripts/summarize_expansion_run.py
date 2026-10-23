"""Record actual labeling coverage and independently reproduced matrix results."""
from pathlib import Path

from cadkg.io import read_csv, read_json, write_json, sha256


def main():
    root = Path('outputs/intent_batch2_expanded_2x')
    run = Path('outputs/intent_batch2_matrix_reproduction')
    data = read_json('annotations/intent_batch2_expanded_2x/judgments.json')
    original = read_json('annotations/intent_batch2/judgments.json')
    rows = data['judgments']
    pairs = {(r['family_id'],r['part_id']) for r in rows}
    completed = [r for r in rows if r['grade'] is not None]
    originals = {(r['family_id'],r['part_id']):r for r in original['judgments']}
    gallery = set(read_json('outputs/intent_batch2/protocol.json')['gallery'])
    assert len(rows) == len(pairs) == 14850
    assert len({r['part_id'] for r in rows}) == 1350
    assert {r['part_id'] for r in rows} <= gallery
    assert len(completed) == len(originals) == 674
    for row in completed:
        prior = originals[row['family_id'],row['part_id']]
        assert all(row[k] == prior[k] for k in ['grade','observation','rationale','source_sha256'])
    assert all(r['grade'] is None for r in rows if (r['family_id'],r['part_id']) not in originals)
    manifest = read_csv(root/'pool_manifest.csv')
    rendered = 0
    for item in manifest:
        path = root/'evidence'/f"{item['inspection_id']}.json"
        if not path.exists():
            continue
        evidence = read_json(path)
        assert evidence['source_sha256'] == item['source_sha256']
        assert evidence['part_id'] == item['part_id']
        if all(path.with_suffix(suffix).exists() for suffix in ['.png','.npz']):
            rendered += 1
    coverage = read_json(root/'coverage.json')
    coverage.update(actual_records=len(rows),actual_completed_assessments=len(completed),
                    actual_pending_assessments=len(rows)-len(completed),
                    rendered_source_cads=rendered,
                    duplicate_pairs=0,valid_for_expanded_relevance_evaluation=False)
    write_json(root/'coverage.json',coverage)
    cases = read_csv(run/'per_query.csv')
    sensitivity = read_csv(run/'sensitivity.csv')
    assert len(cases) == 784 and len(sensitivity) == 2352
    query_strong = {r['query_id']: int(r['pool_strong_count']) for r in cases}
    top5_invariance = (len(query_strong) == 49 and min(query_strong.values()) >= 5)
    comparisons = {}
    for filename in ['rankings.csv','summary.csv','per_query.csv','sensitivity.csv','semantic_digest.json']:
        comparisons[filename] = sha256(run/filename) == sha256(Path('outputs/intent_batch2')/filename)
    assert all(comparisons.values())
    write_json(run/'expansion_audit.json',dict(
        actual_relevance_label_count=674, expanded_target=14850,
        expanded_evaluation_completed=False,
        fresh_full_gallery_searches=784, grading_sensitivity_cases=2352,
        minimum_strong_candidates_per_query=min(query_strong.values()),
        fixed_top5_metrics_invariant_to_added_labels=top5_invariance,
        invariance_conditions='Original labels and top-five rankings retained; added labels use the same 0/1/2 rubric. Every returned part is already judged and every query IDCG@5 already attains the maximum.',
        original_run_exact_reproduction=comparisons,
        expanded_source='review_queue_pending_visual_inspection',
        invalid_previous_category_proxy_labels_excluded=True))
    summaries = [r for r in read_csv(run/'summary.csv') if r['subset']=='combined' and r['variant']=='all']
    table = '| Representation | Method | Pool nDCG@5 | P@5 | Strong P@5 |\n|---|---|---:|---:|---:|\n'
    for row in summaries:
        table += f"| {row['geometry_kind']} | {row['method']} | {float(row['macro_pool_ndcg@5']):.4f} | {float(row['macro_precision@5']):.4f} | {float(row['macro_strong_precision@5']):.4f} |\n"
    text = f'''# Expanded annotation target and experiment status

The requested 14,850 completed assessments have **not been achieved**.
The corrected cohort contains 1,350 eligible held-out CADs × 11 unchanged tasks.
It has 674 preserved prior AI inspection judgments and 14,176 unreviewed pairs.
Currently {rendered}/1350 source CADs have complete hash-bound renders and meshes.
Generating a CAD view is not a relevance assessment.

## Correction of the previous completion claim

The previous added 6,751 grades were assigned by membership of the catalog
category in a per-task ALIGNMENT map. Their observation strings claimed visible
evidence that was not individually inspected. Those added grades were withdrawn.
An explicitly invalid archive preserves what was previously generated.
The original 674 judgments remain intact. Twenty-eight CADs in the former
675-part cohort were outside the frozen 1,816-part evaluation gallery; the new
cohort excludes those CADs. Category strata influence sampling only.

## Experiments actually completed

A fresh search reproduced 49 queries × 8 methods × 2 representations = 784
ranking runs over the original 1,816-part held-out gallery. All three conditional
grading scenarios were evaluated (2,352 metric cases). The 674 original AI
judgments provide 2,704 query-part labels after applying query variants and
numeric screening; these are not 2,704 independently inspected parts.
Rankings, summaries, per-query metrics, sensitivity metrics, and semantic
digests exactly match the original saved run. Existing validation checks pass.

These experiments use **674 prior AI judgments**, not the pending 14,850-row
cohort. They do not satisfy the requested enlarged-label evaluation.

## Why adding labels cannot change the current top-five scores

Every returned top-five result is already judged. Each of the 49 queries
already has at least five grade-2 candidates after numeric screening
(the minimum is {min(query_strong.values())}). Its ideal DCG@5 therefore already
attains the largest possible value under the 0/1/2 rubric. Keeping the original
labels and rankings fixed, adding grades for other parts cannot change either
the retrieved DCG@5 or its ideal denominator. Precision@5, strong precision@5,
and dimension violation counts are also fixed. Thus the matrix's reported
top-five scores are invariant to this additive annotation expansion.

The expanded dataset would be useful for relevance-coverage analysis and
metrics at deeper retrieval ranks once it is actually labeled. It does not
increase the number of independent task families or independently validate
the existing AI judgments.

Family-balanced results across all query variants:

{table}

Full with descriptors scores 0.7202 compared with 0.6334 without exemplar
geometry; Full with embeddings scores 0.6075. The flat table and graph produce
identical rankings with matched information and scoring. There is no measured
graph-storage advantage. The embedding benefit reverses under alternative
conditional-grade interpretations; see report/sensitivity_summary.json.

These are descriptive pooled results from AI-authored references. Unknown
full-gallery recall, author bias, correlated query variants, and incomplete
relevance coverage limit interpretation. Increasing a queue size does not
reduce those limitations until new evidence-based judgments are completed.

Full figures and family-level results: ../intent_batch2_matrix_reproduction/report/REPORT.md
Coverage: coverage.json
Reproduction audit: ../intent_batch2_matrix_reproduction/expansion_audit.json
'''
    (root/'STATUS.md').write_text(text,encoding='utf-8')
    print(coverage,flush=True)


if __name__ == '__main__':
    main()
