# Methodology and evidence boundaries

## Feasibility

TMCAD supplies 9,799 STEP models in ten classes; local inventory matches the [dataset description](https://github.com/Qiang-Zou/BRT/blob/main/DATASET.md). Verified functions, mating interfaces, tolerances, materials, ratings and compatibility are absent. Category folders cannot generate engineering ground truth.

Reused TMCAD exports are a supervised local-surface encoder, 128-dimensional embeddings, ten classifier logits and numeric descriptors. The neighboring JEPA tiny checkpoint uses synthetic shapes and is not a TMCAD JEPA result. Checkpoints/exports retain producer metadata and hashes. Legacy descriptor dimensions lack unit/backend attestation; fresh checked STEP measurements supply physical-unit constraints.

## Knowledge and retrieval

Typed JSON nodes cover Part, Class, Function, Measurement and reviewed Interface. Sources/status distinguish dataset observations, predictions, computed geometry, provisional manufacturer-supported priors and expert-curated knowledge. Dataset labels do not enter serving scores. A predicted category supports a search prior, not a verified instance function.

Function support sums predicted category probabilities times admitted class–function weights. Optional exemplar cosine and reference category-distribution agreement join other available signals at fixed weights. Unavailable signals have no denominator weight. Measured constraints filter candidates. Matched graph/flat information deliberately produces equal rankings.

Default `build` admits explicitly curated rows with reviewer/evidence. Studies opt into `source_supported` priors while retaining provisional status. Human relevance labels never enter graph construction. Reviewed generic priors can create a new development bundle; they do not modify the frozen test graph.

JSON queries and a small alias/constraint parser are supported, not unrestricted engineering language. Thread visibility, hook form and gear subtype are review criteria, not automatic geometry predicates. Spring roles sharing one category prior can remain indistinguishable without informative geometric evidence.

## Supported reasoning

- Preliminary intent discovery with scoped engineering priors.
- Example-shape retrieval using descriptors or existing embeddings.
- Measured largest stored-axis extent screening, without an optimized packing claim.
- Optional bore/shaft clearance intervals when experts supply actual interface roles and drawing-backed tolerance bounds.

The same-kernel surface reference uses a provisional 0.2 mm near-boundary guard, not manufacturing tolerance or independent metrology. Unverified units, failed imports, quarantined bounds and ambiguous interfaces stay unknown. A cylinder is not automatically a bore. Assembly compatibility, strength, fatigue, ratings, thread mating and verified gear meshing require further evidence and are not claimed.

## Evaluation

### Annotation audit and efficiency scope

The latest random audit reviewed a 100-CAD simple random sample across all eleven roles, plus 66 supplemental grade-stratified CADs. It corrected 36 grades in a versioned release. The 2% primary rereview disagreement is not an independent human accuracy estimate; conditional decisions changed more frequently. Existing effectiveness tables retain their annotation-version provenance.

The subsequent serving study is label-independent: 1,000 distinct structured parameter combinations characterize computational coverage, while five actual gallery sizes characterize cost growth. They remain eleven semantic families. Graph size, residency, serialization, candidate scanning and thread batches are reported separately. The measured range ends at 9,798 usable catalog parts and uses an exhaustive ranker. Timing measurements do not extend relevance validity or demonstrate unrestricted language understanding.

### Corrected complete-cohort study

The latest study uses 14,850 AI visual records across all eleven tasks and 1,350 CADs, including 424 original pooled candidates and 926 class-balanced additions. Closed-cohort nDCG, recall and full AP now have complete qrel denominators. Full 1,816-gallery results still require explicit missing-judgment bounds. No candidate receives a functional grade from its catalog class.

The design isolates geometry, exemplar classifier agreement and intent priors with matched filters, and compares fixed reciprocal rank fusion. Inference uses eleven family-level paired differences, with Holm adjustment across the eleven primary nDCG@5 tests; a joint spring-cluster sensitivity addresses one known dependency. The authored families and AI-only judgments prevent independent human or population validation claims. More task–part records do not increase the number of independent task families.

See [the fixed design](VERIFIED_EVALUATION_PLAN.md) and [the complete result report](../outputs/verified_evaluation_v1/REPORT.md). The historical pooled study described below remains preserved and should not be confused with the corrected complete-cohort evaluation.

Auxiliary category/constraint targets use dataset labels only for that expressly auxiliary task. Functional evaluation uses separate AI inspection or genuine human records. Cross-split byte contamination and within-gallery byte duplicates are excluded; near-duplicate designs remain possible. Exemplars come from train, bounds from validation, and paired methods share the test gallery.

The expanded matrix has 11 families, 49 queries, two representations and eight methods. Summaries average within then across families; text-only, exemplar, previous and new cases remain separate. Negative deltas are retained. Related variants are not independent observations.

Functional pools are the union of every system's top5 plus controls. P@5 and strong P@5 describe returned shortlist quality; pooled nDCG uses that finite pool. Exhaustive relevance and full-gallery recall are unknown. Human Cannot assess is null and blocks its family's metrics until resolved. Partial evaluation includes only complete, task-approved families with exclusions reported.

The supplied AI study was graded by its implementation author. It is exploratory, not independent expert truth. Conditional-grade sensitivity is a hypothetical reinterpretation, not a confidence interval. Broader results favor descriptor geometry in aggregate but show lower aggregate embedding performance than cosine removal. Universal improvement and graph-storage superiority are unsupported.

Human reviewers evaluate requests/rubrics and candidates without displayed AI grades, predictions or method ranks. KG authoring and test judging remain separate, ideally with different experts. Reviewer identity is self-attested; qualifications are not authenticated by software. Agreement/adjudication improve consistency without supplying missing engineering specifications.

See [experiment commands](EXPERIMENTS.md) and [the human workflow](ANNOTATION.md).
