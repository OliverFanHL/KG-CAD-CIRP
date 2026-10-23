# Evaluation plan fixed before execution

## Material Passport

- Artifact: verified-evaluation-v1; created 2026-10-20.
- Purpose: evaluate corrected AI visual annotations and isolate contributions of the existing CADKG scoring method.
- Sources: frozen intent-batch2 queries, priors, models, measured constraints; verified annotation release.
- Status: exploratory design fixed before computing new outcomes; historical outcomes are already known. This is not an externally preregistered confirmatory study.
- Authority: user authorized implementation and execution, including additional justified comparisons.

## Questions and evidence gaps

1. Does combining geometry, classifier-reference agreement and the class-function prior improve retrieval with identical candidate and constraint access?
2. Does the function prior add value beyond geometry and classifier-reference agreement?
3. Are results sensitive to representation, conditional judgments, original pool membership, exemplars, family dependence, or synthetic classifier errors?
4. Do existing equality results follow from the implementation, and is screening consistent with separately read source surfaces?

Old results use only 674 pooled judgments. `kg_no_class` removes two terms together. Category-only retrieval lacks the full method's filter. The matched flat control and representation-free ablation have structurally identical formulas. Annotating more candidates cannot change those equalities.

## Fixed inputs and comparisons

Use all 1,350 annotated candidates for every existing task, with all 49 frozen queries and both representations. These are complete closed-gallery qrels. Keep the original 1,816-candidate gallery as a separate continuity evaluation, with coverage and bounds for unjudged returns. Never silently treat unjudged candidates as rejects.

Preserve all eight methods. Add fixed equal-weight controls: intent prior + filter; intent-to-category + filter; classifier-reference + filter; geometry + classifier-reference + filter (no function prior); geometry + intent prior + filter (no classifier-reference term). Add reciprocal rank fusion of available geometry/reference/prior branches with constant 60, equal branch weights and average ranks for exact score ties. Use a uniformly random feasible baseline as an analytic expectation. Geometry-only baselines without an exemplar abstain and are excluded from fair exemplar-only comparisons; text results remain separate.

All methods share candidates, fixed classifier, train-only exemplars, descriptor normalization fit only on train, constraints and unknown policy. Unfiltered methods remain explicit ablations. No additional learned models are trained: no independently annotated development set exists for fair functional training or tuning. Catalog class labels may describe sampling and auxiliary strata; they never set functional grades or serving scores.

## Metrics and statistical analysis

Primary: equal-family mean nDCG@5 among exemplar queries. Secondary: all/text subsets, strong P@5, P@5, nDCG@10/20, recall@5/10/20, truncated AP@20, full AP, result count, and dimension failure/unknown rates. Closed-gallery AP uses all known positive grades; truncated AP divides by min(total relevant,k), with full AP reported separately.

Ten method contrasts (five controls × two representations), plus Full descriptor minus embedding, form one Holm family. Compute two-sided exact sign-flip p-values on the 11 family means, 20,000 seeded family-bootstrap intervals, absolute and relative effects, paired wins/ties/losses and leave-one-family-out ranges. Cluster the two spring families jointly in a sensitivity analysis, retaining the same equal-family statistic. These inference models assume exchangeable task clusters; tasks were authored and share candidates, so results are benchmark-conditional rather than population confirmation. A 0.02 absolute nDCG difference is an explicit descriptive practical threshold, not an established engineering utility boundary.

Report all comparisons, including losses. Do not infer equivalence from a nonsignificant difference. Exact formula/ranking equality is a separate integrity fact.

## Targeted robustness and generalization diagnostics

- Grade-1 sensitivity: map conditional grades to 0 and 2 as alternatives to nominal 1. These are hypothetical interpretations, not annotation confidence intervals.
- Evaluate the 926 newly added candidates separately from the 424 original pooled candidates. Candidate-set changes alter retrieval difficulty and denominators; compare methods within each cohort.
- Report train-exemplar1 versus exemplar2 effects in the eight families with two inspected exemplars. Do not create arbitrary new exemplars or tune their selection.
- Synthetic classifier errors: independently select 10% or 30% of candidate/reference distributions and permute their class coordinates, using five fixed seeds shared across methods and representations. This probes reliance on classifier information, not real out-of-domain generalization.
- Exact-score tie sensitivity: five seeded shared candidate tie orders; no score jitter and no test-label-selected order.
- Near-duplicate diagnostic: union train/test descriptors within 1e-6 Euclidean distance after the fixed train transform. Remove test candidates matching train, collapse gallery groups to one lexicographic representative, and report exclusions/label discordance. This is a narrow numerical descriptor signature, not proof that all near-duplicate designs were found.
- Numeric reference guards 0, 0.2 and 0.5 mm: change only reference uncertainty, never production constraints. Same-kernel surface measurements cannot establish independent metrology or manufacturing suitability.

## Reproducibility and claim limits

Save complete ranked IDs/scores, input/code hashes, frozen configuration, qrel statistics, per-query results, family summaries, multiplicity corrections and a machine-readable integrity report. Compare legacy top5 IDs to the serving implementation; rerun deterministic results and require exact artifact digests. Do not use historical-metric equality as a validator after label correction.

AI-only annotations, authored task families, partial retrieval-based cohort selection and lack of an independent external dataset remain limitations. Graph storage alone, unrestricted language reasoning, certified compatibility and broad industrial generalization are outside the evidence.

Methodological sources: [Smucker et al., paired randomization for IR](https://ciir-publications.cs.umass.edu/getpdf.php?id=744); [NIST, pooled versus sampled judgments](https://www.nist.gov/publications/comparison-pooled-and-sampled-relevance-judgments). These motivate paired comparisons and explicit treatment of pool coverage; the present small-cluster analysis remains exploratory.
