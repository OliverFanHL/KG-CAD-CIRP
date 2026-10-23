# Direct effectiveness evaluation of the KG-constrained method

## Material Passport

This follow-up responds to the user's explicit request to evaluate the proposed KG-constrained method. The random annotation audit is complete and the audited-v2 labels are frozen. We re-evaluate the existing task/query matrix, with no expansion to new effectiveness task families. The previous systems-only gate limited the preceding scalability study; this authorized follow-up is a descriptive/exploratory effectiveness re-evaluation, not an assertion that global annotation quality or independent human validity has been established.

## Method and hypotheses

The actual Full implementation fuses exemplar geometry, predicted classifier-distribution agreement and admitted class–function priors, followed by verified numeric filtering and stable ranking. Here the constraints are declared largest-axis envelope bounds, not verified interface mating, material/load limits or an independently learned constraint graph.

Evaluate Full on the 1,350 completely judged held-out CADs, separately for:

- All 38 exemplar query variants, retaining the previously declared eleven family-level method/representation tests.
- The **19 exemplar queries with explicit dimensional constraints**: Full versus unconstrained Full; filtered geometry; geometry+classifier with the same filter; intent prior with the same filter; reciprocal rank fusion with the same filter; geometry+intent with the same filter; and Full without exemplar geometry.
- The corresponding nineteen unconstrained exemplar queries to distinguish dimension-screening gains from ranking/fusion gains.
- Text-only cases separately, avoiding the input-availability confound of comparing geometry methods without an exemplar.

Primary functional outcomes: family-balanced graded nDCG@5, strong P@5, P@5, recall@20 and full AP. Report surface-reference fail/unknown counts and returned slots; zero fail does not imply all passes. Constraints also enter the independent qrel construction, so the expected mechanical advantage of screening must be separated from any contribution of KG priors.

The focused constrained family comprises seven contrasts × two representations (14 exact two-sided task-family sign-flip tests), adjusted together with Holm. Preserve all wins/losses, cluster-bootstrap intervals, spring joint-flip sensitivity and leave-one-family-out ranges. These tasks are authored, related and AI-graded; inferential results are exploratory. Keep the original all-exemplar 11-test family separate and identify it explicitly; neither family is a retrospective confirmatory study.

No tuning, extra unvalidated requests, new annotations or new methods. Existing fourteen methods and score weights remain unchanged. Sampled annotation errors and the unsampled population remain limitations. Analyze the 926 added candidates, strict grade-2 relevance and existing robustness outputs. Report whether the method beats strong matched-information controls, rather than presenting its score in isolation.

## Reproducibility

Run `scripts.run_verified_evaluation` with `configs/kg_constrained_audited_v2.json`, repeat into a replay subdirectory, validate source hashes/metrics, and create the focused report with `scripts.report_kg_constrained`. Preserve the v1 studies and their annotation hashes. The statistics follow the paired comparison and multiplicity principles documented in the earlier evaluation plan; all new outcomes remain exploratory.
