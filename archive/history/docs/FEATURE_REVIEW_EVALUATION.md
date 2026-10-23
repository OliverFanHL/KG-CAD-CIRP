# Source-feature review and fixed-framework rerun, v5

## Scope and provenance

This is an AI engineering-shape audit, not an independent human audit or a
manufacturing qualification. Historical test outcomes are known to the
coordinator. First-pass reviewers receive only a frozen rubric, neutral IDs and
source-bound images, in separate contexts. They cannot see old labels, source
categories, method names, ranks, scores or each other's decisions. All 1,816
eligible test CADs are reviewed against all eleven existing roles. A second
source-view audit emphasizes historical positive/conditional and retrieval
disagreement cases; its selection is frozen before new-label analysis. Its
agreement is AI consistency evidence, never human inter-rater reliability.

The operational rubric is frozen after a 44-CAD validation-only pilot. Exact
STEP hashes shared with test are excluded from that pilot. Rubric refinement
and better views are retrospective benchmark revisions. Their effect cannot be
described as pure annotation-error correction or a new untouched confirmatory
test. No favorable performance outcome is a reason to change a label.

## Evidence and adjudication

All CADs receive six source-bound surface/cutaway views with mesh edges hidden.
Navigation sheets are supplemented by enlarged individual images where needed.
Each positive, conditional or cannot-assess role requires named per-feature
states, view locators and concrete observations. Incompatible roles are
explicitly considered and supported by the observed form; they do not receive
invented feature facts. Full source observations replace legacy generic prose.

The prior release, old decisions and source hashes remain preserved. First-pass
notes are frozen before unblinding old judgments. Disagreements are logged with
old/new grades, evidence, operative boundary, adjudicator and disposition. A
second blind sample is compared only after both judgments are sealed. Source
ambiguities and feature/rationale diagnostics trigger enlarged review, never
automatic promotion. No annotation is chosen using an evaluation score.

Geometric conditional suitability is grade 1. Decisive unreadable evidence is
null/cannot-assess, with a reason. A visibly smooth simplified fastener may be
conditional under the rubric; it is not evidence of actual threading. Fine
faceting or section-edge serrations never prove a helix.

## Locked experiment and interpretation

Retain all 110 queries, eleven families, fourteen methods, embedding and
descriptor representations, candidate eligibility, exemplars, weights,
constraints, random seeds, cutoffs and robustness conditions from v4. Do not
train, tune or add configurations. Candidate count is 1,816, versus the prior
1,350 and original 424; all eligible held-out candidates are included. Cohort
diagnostics retain their earlier definitions and are descriptive.

The proposed method is **kg_full**, reported separately for both existing
representations. `flat_full` is its matched flat implementation control, not an
additional proposed method. `geometry_intent_filter`, `geometry_class_filter`,
`intent_filter` and `rrf_filter` are controls, not names to substitute for the
proposed method if they score higher.

Primary analysis retains equal-family nDCG@5 over the existing 58 exemplar
queries, eleven predeclared Holm comparisons, family-paired exact sign flips,
20,000 bootstrap replicates and spring-dependence sensitivity. Retain the
existing fourteen focused ablation comparisons with their separate Holm family.
Report effects and uncertainty against both component ablations and strongest
matched controls. Report all-query/text, conditional-grade, tie, size-guard,
classifier-corruption, deduplication and candidate-cohort diagnostics. All
families stay in the denominator, including zero-relevance cases.

If null judgments remain, retain those candidates and judgments. Do not silently
score them as 0, remove them or publish a complete-label point estimate as
truth. Publish conservative metric intervals over admissible grades 0..2 after
the unchanged dimension/conditional transforms. Lower/upper annotation
completions are computational envelopes, never new ground truth or new
experimental configurations. Statistical point claims are withheld wherever
label uncertainty prevents a defensible comparison. When no null remains, use
the unchanged complete-qrel evaluator directly.

Freeze final annotations and input/code hashes before scoring. Repeat the run,
compare all ranking arrays and deterministic output hashes, independently
recompute metrics and task/constraint transforms, and rehash raw STEP sources.
Rankings should match v4 exactly for identical cohorts: only judgments/evidence
are revised. Label-induced descriptor-duplicate discordance is reported rather
than silently resolved by label copying.

The report must distinguish: (1) useful shortlisting on this benchmark,
(2) incremental advantage of the full combination over matched alternatives,
(3) graph representation contribution, (4) robustness on the existing tests,
and (5) generalization not established outside these authored tasks and source
distribution. An nDCG value alone proves none of the latter claims. The scale
increase tests more candidates, not more independent task families.
