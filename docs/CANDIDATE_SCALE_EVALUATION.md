# Candidate-only expansion, fixed before new scoring

## Material Passport

- Created 2026-10-21 under the user's instruction to audit annotations and increase candidate coverage without expanding configurations.
- Inputs: audited-v2 annotations, frozen 110-query v3 protocol, frozen classifiers/representations/priors and training exemplars.
- Status: exploratory sequential extension; historical outcomes are known. No external preregistration or independent human validation is claimed.

## Fixed scope and order

1. Mechanically audit every one of the existing 14,850 records, source bytes, evidence bindings, decisions, observation consistency, scope and split. Inspect a new seeded simple random sample of 100 CADs against all eleven rubrics, plus evidence-based flags. Keep the existing 166-CAD audit as historical evidence; do not pool dependent audits into an error estimate.
2. Preserve the existing 1,350 candidates. Add **every remaining 466 member** of the frozen 1,816-part eligible held-out gallery. Do not add training or validation candidates to relevance evaluation. No rank-based candidate selection or category-derived labels.
3. Read neutral source views and record observations before comparing old grades or computing new effectiveness. Existing task definitions and documented criteria clarifications apply. Explicitly list all plausible roles; omitted roles mean visually incompatible under the stated rubric. Use null for inability to assess, never substitute reject. Enlarge ambiguous evidence before finalization. No performance target guides correction.
4. Publish a new version, retaining every parent record, original grade, changed grade, evidence/observation, reason and source hash. Freeze its hash before scoring. Preserve original-v2 results. Run the existing 110 queries, fourteen methods, two representations, cutoffs, equal weights, filters, train exemplars, seed, statistical comparisons and robustness conditions unchanged.
5. Evaluate the original 1,350 cohort with corrected labels to isolate correction effects from gallery changes. Evaluate all 1,816 and the 466 added candidates separately. Keep the existing original-pool/new-only/deduplicated diagnostics as well. Use actual complete qrels for recall and AP; if any unresolved annotations remain, report incompleteness rather than manufacture closed-gallery point estimates.
6. Independently recompute metrics, rehash STEP files, rerun deterministic evaluation, compare arrays and artifact digests, verify fixed query/method/input hashes and serving rankings. No annotation is inserted into the serving graph or classifier. No model or weight is tuned on these labels.

## Interpretation fixed in advance

Primary inference remains the existing eleven family-paired exemplar contrasts, with the same Holm family, exact sign flips and family bootstrap. Preserve the existing focused constrained-exemplar analysis and its fourteen-comparison Holm family. Added-cohort, correction-impact and conditional-grade analyses are descriptive; do not add significance tests or choose subsets after outcomes. Eleven authored families, not 110 variants or 19,976 task-part rows, determine inferential scope. Correlated spring families retain the joint-cluster sensitivity.

The 0.6767 reference is descriptor Full constrained-exemplar nDCG@5, not accuracy. Grade 1, numerical screening uncertainty, exemplar choice, candidate composition, class priors and representation limitations can affect it. A score below one does not diagnose bad annotations. Report losses against same-filter controls, task concentration and deep-ranking quality. Same-model rereview disagreement is not a ground-truth error estimate. More candidates improve coverage within these tasks, not unseen-task or external-domain validity.

The earlier audit's conservative recommendation to limit subsequent claims is retained in history. The present user authorizes exploratory candidate expansion; it does not supply human validation or permission for stronger claims. No new configuration, query, task, model, tuning procedure or engineering suitability claim is introduced.
