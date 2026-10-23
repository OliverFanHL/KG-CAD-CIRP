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
| [figure_annotations.json](figure_annotations.json) | Full-precision numerical annotations underlying the publication figures |
| [validation/](validation/) | Reproduction and publication-integrity checks |

Means weight six size conditions within each exemplar, four exemplars within each role, then eleven roles equally. SD uses the three collection means with denominator two. Lower/upper fields retain uncertainty from unresolved reference judgments. The primary nDCG@5 endpoints are point identified under those bounds.

The primary comparisons use eleven role effects after averaging collections. Exact sign-flip tests enumerate 2,048 patterns; one Holm correction covers four tests. Bootstrap intervals use 20,000 role resamples with seed 20261023 and are unadjusted. Positive mean effects do not establish superiority after the specified correction.

`python -m scripts.reproduce` rebuilds the results and checks exact hashes. Generated files appear in a separate `results/reproduced/` directory. The checked-in numerical tables remain unchanged.
