# Methodology

The method retrieves CAD candidates for preliminary functional shortlisting from an exemplar, an engineering role, and optional upper size constraints. The primary evaluation combines eleven roles, four training exemplars per role, and six size conditions, giving 264 queries against a held-out gallery of 1,816 parts.

## Initial retrieval

The initial score, Full, averages the available geometric similarity G, predicted exemplar-category agreement A, and category-derived function support F. Measured dimensions determine eligibility C. Embeddings and thirteen standardized geometric descriptors provide separate representations. The encoder, category probabilities, normalization, weights, queries, and gallery are fixed throughout evaluation.

## Visual refinement

The refinement pool is the union of the top ten eligible candidates from three fixed retrieval branches. A visual observer receives neutral CAD views and four role-specific geometric clauses with component bindings. Yes/no/unknown states map to 1/0/0.5. Their mean gives coverage H. Within the pool, the refined score is `(3 × Full + H) / 4`; the unchanged Full tail is appended. Ties use the fixed gallery index.

The matched holistic observer receives the same CAD views and the role's purpose, and assigns a nine-level score J with the same range and increments as H. Full+J uses the same scoring formula. Score-only controls use the same candidate pool. The comparison evaluates requirement-guided observation and aggregation together; graph storage is not isolated as an experimental factor.

## Relevance and metrics

Evaluation combines role–part relevance judgments with an independent numerical relevance rule. Extent failures receive grade zero. Missing or quarantined surface evidence, or proximity within 0.2 mm of a requested bound, caps relevance at grade one. Two unresolved functional grades retain all admissible values; the resulting lower and upper bounds coincide for the primary nDCG@5 results.

Metrics average size conditions within each exemplar, exemplars within each role, then roles equally. Three response collections measure acquisition variability; reranking is deterministic under the fixed encoder. Exact paired tests use eleven role effects and a single Holm correction across four tests. Role-bootstrap intervals are reported separately without adjustment. See the [results guide](../results/README.md) for statistical definitions and the [study specification](../configs/study.json) for evaluation settings.

## Scope and limitations

Reference uncertainty, authored role scope, descriptor-conditioned exemplar selection, a bounded refinement pool, and the absence of encoder retraining limit generalization. Materials, loading, fit, and tolerances require subsequent engineering assessment.
