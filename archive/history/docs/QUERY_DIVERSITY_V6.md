# Diverse-reference queries and seven controlled methods, v6

## Material Passport

- User-authorized query expansion and method reduction, 2026-10-21.
- Development inputs: existing train/validation splits, raw source views and the
  frozen v5 operational rubric. Candidate relevance labels remain immutable v5.
- AI source audits and authored queries; no independent human certification.
- Historical v5 outcomes are known. This is a sequential exploratory extension,
  not a newly untouched confirmatory population test.
- Plan written before inspecting any v6 retrieval result; query/exemplar,
  analysis and code hashes will be sealed before execution.

## Question and controlled methods

Assess whether Full helps across previously unused training-reference shapes
and tight, typical, loose and absent size requirements. Do not select methods
or queries to manufacture a favorable result. Keep the strongest existing
same-filter controls even if Full loses. All methods share each query, reference,
candidate gallery, predictions, priors and constraint values.

| Method ID | Display / experimental purpose |
|---|---|
| kg_full | Proposed Full: geometry + exemplar-category agreement + function prior + size filter |
| geometry_class_filter | Full without function prior; single-component ablation |
| geometry_intent_filter | Full without exemplar-category agreement; single-component ablation |
| kg_no_geometry | Full without geometric similarity; single-component ablation |
| kg_no_constraints | Full without size filtering; single-component ablation |
| kg_no_class | Geometry + same filter; geometry-only filtered baseline |
| rrf_filter | Same three signals + same filter with reciprocal-rank fusion; fusion control |

Seven methods replace fourteen. Omit category/reference-only variations and
other unfiltered mixtures from the comparison table. Flat/graph equality remains
an implementation-integrity check, not an eighth performance competitor. It
cannot establish graph-specific retrieval quality. No new model, weight,
function mapping, size predicate, fusion recipe or geometry representation is
introduced. Equal signal weights and the existing RRF constant remain fixed.

## Query construction and source-only reference audit

Preserve the 110 old query objects exactly as a continuity block. Add thirty
queries per family, yielding forty per family and 440 in total:

- Four previously unused training references per family, each crossed with six
  conditions: no constraint and validation 10th/25th/50th/75th/90th percentile
  largest-axis size bounds. This adds 264 exemplar queries.
- Six additional text-only bounds per family add 66 text queries. Use the fixed
  quantile order .05, .10, .35, .50, .65, .90. Skip exact structured duplicates
  against earlier text bounds and continue the fixed 1%-spaced grid if needed.
  Do not count paraphrases with the same structured query as new experiments.

The complete design has 341 exemplar queries, 99 text queries, 278 constrained
exemplar queries and 366 constrained queries overall. The new-reference panel
has 264 queries, including 220 constrained queries. The new-text panel has 66
queries. Query text does not test a new language parser; retrieval consumes the
existing explicit function and constraint fields.

Reference candidates are selected from training only, using mapped source
classes for sampling rather than suitability truth. Before any source audit,
freeze descriptor-diverse farthest-point order with source-byte deduplication,
exclusion of every val/test source hash and exclusion of the old nineteen
references. Select twelve candidates per family: six initial and six reserve.
Farthest-point sampling is descriptor-conditioned, not a probability sample or
an unbiased benchmark for choosing between the two representations. Primary
comparisons are within each representation; do not infer a representation winner
from this sampling design.

Review neutral IDs and source-bound six-view images with categories, intended
family, previous answers, ranks and scores hidden. Apply all eleven frozen roles
and record per-feature evidence. Inspect enlarged images for decisive threads,
keyways, seats, tooth profiles and spring attachments. Fine mesh edges never
prove a helix. Source judgments precede family unblinding and reference selection.
This is separate-context AI review, not human agreement.

For each family, take the first four candidates with visible strong or
conditional task suitability (grade 2 or 1) in the frozen FPS order. Do not rank
by retrieval score or preferentially keep grade 2 outcomes. Preserve conditional
reference grades and report their prevalence and performance separately. Reject
grade 0/null references as unsupported query exemplars. If fewer than four
initial candidates qualify, audit the frozen reserve in order. If that still
fails, preserve the insufficiency and prepare an additional training-only source
pool before any evaluation; do not invent labels or silently duplicate references.
For the depleted support_body and tension_return pools, continue the identical
training normalization and seeded FPS sequence through positions 13–24, preserving
all original 132 candidate positions and writing separate additive manifests.
Freeze these 24 additional sources and their order before additive source audit;
no v6 retrieval scores have been inspected when preparing this addition.

Validation size distributions use finite positive computed extents from mapped
validation classes, excluding exact test-source overlaps. Rehash source files
and save measurements/quantiles/bounds. Never use test relevance, candidate
selectivity, test rank or performance to choose bounds. Retain empty-return and
zero-relevance queries in all denominators. The eleven task scopes and test
labels are unchanged; do not reinterpret new references as new relevance labels.

## Frozen evaluation and primary analysis

Retain all 1,816 eligible test candidates, both geometry backends and the existing
cohort definitions, cutoffs 5/10/20, conditional grading sensitivities, size guards,
classifier corruption fractions and seeds. New references never enter the gallery.
For corruption, preserve the old candidate/reference probability draw universe;
extend perturbations only to new references using an independent fixed stream.
All methods receive the same perturbed inputs for each paired condition.

Primary endpoint: closed-cohort nominal nDCG@5 on the 264 new-reference queries.
Average six conditions within each reference, four references within each family,
then eleven families equally. Test Full against all six retained alternatives,
separately for embedding and descriptor: twelve paired comparisons in one Holm
family. Use the existing exact two-sided family sign-flip implementation, fixed
seed 20261020 and 20,000 family bootstrap replicates. Report effect sizes,
unadjusted exploratory 95% intervals, all p-values, family wins/ties/losses and
leave-one-family-out ranges. Keep compression/tension families together in the
existing joint-flip and cluster-bootstrap sensitivity.

The 110 legacy queries, their 77 exemplar subset, all 341 exemplar queries,
constrained/unconstrained new-reference panels, 99 text queries, per-reference
effects, source-quality strata, candidate cohorts and all robustness conditions
are descriptive secondary analyses. Do not create additional favorable test
families after inspecting results. Do not treat 264 or 440 variants as independent
tasks or claim narrower population uncertainty from their count alone.

Retain the two null relevance labels and candidates. Pass all admissible raw
grades 0..2 through the unchanged conditional and dimension transforms. Use
conservative metric envelopes where needed. Run point tests only if every
targeted per-query nDCG@5 is exactly identified across all seven methods and both
representations; otherwise retain all twelve comparison envelopes and withhold
point significance. Do not select a favorable label completion.

## Validation and claim limits

Reconstruct the new queries from frozen development inputs and independently
validate count, signatures, family/function mapping, reference split/hash/audit
eligibility and validation-only numeric bounds. Verify original 110 query objects
and all retained nominal orders/scores and bounds against v5. Verify common
legacy corruption/tie/guard cases where methods overlap. Execute the revised
matrix twice, compare all ranking arrays and deterministic tables, recheck raw
source/code hashes, and independently validate metric and task/constraint logic.

The new panel probes transfer across additional reference shapes and size
requirements within the authored eleven-family CAD distribution. It does not
test new industrial task families, unseen source domains, natural-language
understanding, material/load/fit compatibility or human annotation truth. Report
failures, zero-relevance cases, conditional dependence and corruption degradation
alongside any positive contribution. Query diversity is evidence to assess, not
permission to claim that Full must win.
