# Frozen query expansion: 110 variants

## Objective and scope

The requested approximately 108-query effectiveness evaluation uses **110 variants,
ten for each of the eleven existing task families**. Equal family coverage is more
useful than an exact total of 108. Preserve the original 49 query objects exactly;
add 61 before inspecting their retrieval outcomes. This is an extension of the
audited-v2 relevance evaluation, separate from the 1,000-request systems workload.

Each family has three text-only queries (one original unconstrained and two new
constrained) and seven exemplar queries. The original nineteen inspected training
exemplars remain fixed. No new task semantics, annotation grades, models, learned
weights, ranking methods or geometry predicates are introduced.

## Selection without test-outcome tuning

Use the frozen class-function mapping only to choose validation-set size
distributions. Categories are not suitability judgments. Text-only additions use
the 25th and 75th percentiles of finite positive largest stored-axis extents in
that validation category. Fill each family's remaining exemplar slots using its
existing inspected training references, cycling references and the prespecified
quantile sequence 0.25, 0.75, 0.10, 0.90, 0.50, 0.05, 0.95. Rotate the starting
reference by family index to avoid always favoring the first reference. Round
bounds to six decimals. Skip exact duplicate structured queries; if necessary
continue through a fixed 0.01-spaced quantile grid. If distinct queries cannot be
constructed, fail rather than inventing new semantic tasks. Save the validation
source IDs, sizes, quantiles, thresholds and input hashes.

All new constraints are hypothetical `aabb_large_mm <= value` screening requests
in mm. Their semantics agree with the existing numeric reference and original
rubric. The unmodified family-level shape grades are combined with a separate
per-query surface-measurement constraint verdict; this reuse does not constitute
new independent expert annotation. Retain unanswerable or low-relevance queries
in the reported results. Never pick queries using test grades, predictions or
ranking performance.

## Fair comparison and inference

Use the same 1,816-part held-out common gallery, the same fully judged 1,350-part
cohort, 14 methods, embedding/descriptor transforms, equal scoring weights,
unknown exclusion policy, numeric-reference guard of 0.2 mm, cutoffs 5/10/20,
twenty thousand family bootstrap draws, seeds, paired comparisons and robustness
settings as audited-v2. No weight or model tuning. Preserve the original 49-query
results exactly. Directly check the new queries against the actual serving
implementation for all eight original methods and both representations.

Report family-balanced effectiveness, constraint compliance, matched-filter
controls, original versus added queries, numeric selectivity, grade sensitivity,
new-only/deduplicated cohorts and the existing robustness conditions. Retain the
original eleven primary comparisons (Holm correction); the focused constrained
analysis remains a separately corrected, exploratory fourteen-comparison family.
Do not claim confirmatory significance from this sequential benchmark expansion.
Variants, annotations and CAD records are not independent task replications.
Keep the two spring families together in the dependence sensitivity analysis.

## Appropriateness, cost and limits

This roughly doubles query coverage at about 2.25 times the original evaluation
cost, without retraining or rendering/annotating additional CADs. Measure actual
wall time and serving-check cost; a deterministic replay doubles evaluation
execution for reproducibility. Stop at this scale: further variants offer less
insight than independently reviewed new task families or industrial queries.
The design is balanced within the authored task space, not a probability sample
of industrial workloads. The two spring roles share a source category. Extent
bounds do not evaluate assembly fit, strength, material or interface constraints.
The prior random annotation audit is retained, including its corrections and
remaining AI-authored-label uncertainty. The earlier audit's systems-only
recommendation is superseded for this explicit user-requested, scoped variant
extension; it is not reinterpreted as independent human validation.

110 queries improve assessment across exemplar and size conditions. They do not
increase the independent task count beyond eleven or prove scalability by
themselves. Candidate-gallery and concurrency scaling remain separate systems
measurements.
