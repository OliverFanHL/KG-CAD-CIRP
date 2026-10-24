# Experimental results

All files describe the study's 264-query primary panel, 1,816-candidate gallery, two geometric representations, and three observation collections per observer.

| File | Contents |
|---|---|
| [summary.csv](summary.csv) | Method/representation means and sample SDs across three collections |
| [per_query.csv](per_query.csv) | 17,424 query × method × representation × collection metric records |
| [family_metrics.csv](family_metrics.csv) | Role-level means for each method, representation, and collection |
| [run_metrics.csv](run_metrics.csv) | Aggregate metric for each complete collection |
| [paired_tests.csv](paired_tests.csv) | Four primary paired contrasts, unadjusted bootstrap intervals, exact p-values, and Holm-adjusted p-values |
| [family_differences.csv](family_differences.csv) | Role-level effects underlying the primary contrasts |
| [component_controls.csv](component_controls.csv) | Fixed-encoder component controls repeated in three execution orders |
| [top5_rankings.json](top5_rankings.json) | Evaluated top-five gallery indices, joined through `data/rankings/cohorts.json` after preparation |
| [validation/](validation/) | Reproduction and publication-integrity checks |

## Aggregation and uncertainty

Means average six size conditions within each exemplar, four exemplars within each role, then eleven roles equally. Sample SD is calculated from the three collection means with denominator two. Lower and upper fields represent uncertainty from unresolved relevance judgments. These bounds coincide for the primary nDCG@5 results.

The primary comparisons use eleven role effects after averaging collections. Exact sign-flip tests enumerate 2,048 patterns, with one Holm correction across four tests. Bootstrap intervals use 20,000 role resamples with seed 20261023 and are unadjusted. Interpret mean differences alongside the corrected p-values and uncertainty intervals; the reported Luna comparisons with the holistic control remain inconclusive after correction.

## Reproduction

Run `python -m scripts.reproduce` to rebuild the results in `results/reproduced/` and verify their hashes. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for setup and the metrics-only workflow.
