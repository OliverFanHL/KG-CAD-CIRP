# Supporting data

The study uses TMCAD geometry and category labels, authored engineering roles and requirements, relevance judgments, measured geometry, and saved visual-model observations.

Run `python -m scripts.prepare_data` to verify the dataset packages and unpack the larger inputs into the locations below. Queries, judgments, metadata, annotation templates, and representative CAD models can be browsed directly.

| Location | Contents |
|---|---|
| `queries/queries.json` | The 264 primary queries: eleven roles × four exemplars × six size conditions |
| `annotations/judgments.json` | Supplied release of 19,976 role–part judgments; two unresolved grades remain explicit |
| `annotations/templates/` | Blank relevance annotation templates |
| `study/knowledge_rules.json` | The four visual requirements and bindings for each role |
| `study/prior_relations.csv` | The eleven source-supported category–function relations used by the initial ranker |
| `study/role_only_contract.json` | Matched holistic-observer contract |
| `study/relevance_rubric.md` | Relevance assessment rubric |
| `study/qualitative_diagnostics.json` | Source-linked diagnostics supporting the qualitative examples |
| `catalog/parts.csv` | Part IDs, source category labels, split membership, STEP paths, and source hashes |
| `observations/` (prepared) | Neutral source images, raw model responses, clause evidence, and H/J coverage |
| `geometry/evidence/` (prepared) | Independent surface/dimensional reference evidence |
| `rankings/` (prepared) | Initial ranking arrays for the 264 primary queries and the ordered candidate gallery |
| `encoder/` | Selected checkpoint, cohort, model metadata, and prepared descriptor/embedding exports |
| `cad_examples/` | Ten unmodified STEP examples, one per category |

## Identifiers and manifests

`cad_examples_manifest.json` maps examples to original part IDs, categories, splits, source paths, and hashes. Primary queries join observations and judgments through `query_id`, `scenario_id`/`family_id`, and `part_id`. Ranking indices refer to the ordered gallery in `rankings/cohorts.json`.

`manifest.json` records each input's path, size, SHA-256, and distribution location: directly in the repository or in a dataset package. `prepared_inputs.json` records successful preparation. See the [observation guide](observations/README.md) for response-level join keys.

## Annotation provenance

The supplied judgment release records `source: expert` and `human_expert_reviewed: true`. Its 19,976 grades, including two unresolved values, are the labels used for the reported numerical results. The [v5 rubric](study/relevance_rubric.md) documents the earlier AI shape-review protocol.

## Source geometry and observations

The source corpus contains 9,799 parts across ten categories. TMCAD is described by Zou and Zhu, *Bringing attention to CAD: Boundary representation learning via transformer*, Computer-Aided Design 189 (2025), 103940 ([DOI](https://doi.org/10.1016/j.cad.2025.103940)). Obtain the complete raw STEP corpus through the original source, subject to its dataset terms and attribution requirements. Category labels describe part classes; functional relevance and compatibility require separate assessment.

The saved response collections contain 18,756 observations across the KG-directed Luna, KG-directed Sol, and holistic Luna arms, including neutral fallbacks in their assigned collections. All observations can be inspected and reproduced offline. Large inputs are distributed in numbered [dataset packages](packages/README.md), which data preparation verifies and unpacks into the documented locations.

See the [extension guide](../docs/EXTENDING_EXPERIMENTS.md) to add LLM observers or collect additional CAD relevance assessments.
