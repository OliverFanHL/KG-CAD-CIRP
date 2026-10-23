# Methodology

The task is preliminary functional CAD discovery from an exemplar, an engineering role, and optional upper size constraints. The primary evaluation crosses eleven roles, four training exemplars per role, and six size conditions, against a 1,816-part held-out gallery.

The initial retrieval score averages available geometric similarity G, predicted exemplar-category agreement A, and category-derived function support F. Measured dimensions determine eligibility C. Embeddings and thirteen standardized geometric descriptors provide separate representations. The encoder, category probabilities, normalization, weights, query definitions, and gallery are fixed.

The refinement pool is the union of the top ten eligible candidates from three fixed retrieval branches. A visual observer receives neutral CAD views and four role-specific geometric clauses with component bindings. Yes/no/unknown states map to 1/0/0.5. Their mean gives coverage H. Within the pool, the refined score is `(3 × Full + H) / 4`; the unchanged Full tail is appended. Ties use the fixed gallery index.

The matched holistic observer receives the same CAD views and plain role purpose, using a nine-level J score with the same range and increments as H. Full+J uses the identical combination. Score-only controls retain the same inherited candidate pool. The comparison therefore concerns requirement-guided observation and aggregation, rather than graph storage alone.

AI-reference role judgments and an independent numerical relevance rule define evaluation. Extent failures receive grade zero; missing/quarantined surface evidence or proximity within 0.2 mm of a requested bound caps relevance at grade one. Two unresolved functional grades retain all admissible values. The primary reported nDCG@5 bounds coincide.

Metrics average size conditions within exemplar, exemplars within role, then roles equally. Three response collections quantify acquisition variation; local ranking order repeats are deterministic under the fixed encoder. Exact paired tests use eleven role effects, one four-test Holm family, and separate unadjusted role-bootstrap intervals. See [results](../results/README.md) for statistical interpretation and the [supplement](../publication/supplement/supplement.tex) for complete definitions.

AI-derived reference judgments, authored role scope, descriptor-conditioned exemplar selection, a bounded refinement pool, and the absence of encoder retraining limit generalization. Materials, loading, fit, and tolerances require subsequent engineering assessment.
