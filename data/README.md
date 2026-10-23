# Supporting data

The study uses TMCAD geometry and category labels, authored engineering roles and requirements, relevance judgments, measured geometry, and saved visual-model observations.

Run `python -m scripts.prepare_data` once. This verifies the final study's dataset packages and prepares the larger inputs in the locations below. Queries, judgments, metadata, annotation templates, and representative CAD models are directly browsable without preparation.

| Location | Contents |
|---|---|
| `queries/queries.json` | The 264 primary queries: eleven roles × four exemplars × six size conditions |
| `annotations/judgments.json` | Supplied release of 19,976 role–part judgments; two unresolved grades remain explicit |
| `annotations/templates/` | Blank annotation templates; no fabricated expert responses |
| `study/knowledge_rules.json` | The four visual requirements and bindings for each role |
| `study/role_only_contract.json` | Matched holistic-observer contract |
| `study/relevance_rubric.md` | Relevance assessment rubric |
| `study/qualitative_diagnostics.json` | Source-linked diagnostics supporting the qualitative examples |
| `catalog/parts.csv` | Part IDs, source category labels, split membership, STEP paths, and source hashes |
| `catalog/catalog.json` (prepared) | Representations, classifier probabilities, geometry measurements, and source metadata |
| `knowledge/priors/` | Source-supported engineering category–role priors |
| `knowledge/graph.json`, `knowledge/knowledge.json` (prepared) | Retrieval knowledge graph and supporting records |
| `observations/` (prepared) | Neutral source images, raw model responses, clause evidence, and H/J coverage |
| `geometry/evidence/` (prepared) | Independent surface/dimensional reference evidence |
| `rankings/` (prepared) | Initial ranking arrays for the 264 primary queries and the ordered candidate gallery |
| `encoder/` | Selected checkpoint, cohort, model metadata, and prepared descriptor/embedding exports |
| `cad_examples/` | Ten unmodified STEP examples, one per category |

`cad_examples_manifest.json` joins examples to original part IDs, categories, splits, source paths, and hashes. Primary queries join observations and judgments through `query_id`, `scenario_id`/`family_id`, and `part_id`. Ranking indices refer to the ordered gallery in `rankings/cohorts.json`. `manifest.json` records every input's path, size, SHA-256, and whether it is directly checked out or supplied in a dataset package. `prepared_inputs.json` records successful preparation.

The supplied judgment file declares `source: expert` and `human_expert_reviewed: true`. Its 19,976 grades and unresolved values are unchanged from those underlying the recorded numerical results; review-provenance fields reflect the supplied release.

The 9,799-part source corpus has ten categories. The original TMCAD source is Zou and Zhu, *Bringing attention to CAD: Boundary representation learning via transformer*, Computer-Aided Design 189 (2025), 103940 ([DOI](https://doi.org/10.1016/j.cad.2025.103940)). Obtain the complete raw STEP corpus through the original source. Category labels are not verified functions or compatibility labels. Original dataset terms and attribution continue to apply.

The saved response collections contain 18,756 observations across the KG-directed Luna, KG-directed Sol, and holistic Luna arms. Recorded neutral fallbacks remain in their assigned collections. No network request is needed to inspect or reproduce these data. Large inputs are distributed losslessly in the numbered files under `packages/`; preparation verifies their hashes and unpacks only the final study inputs into their documented locations.

For new LLM observers or additional CAD relevance assessments, see [extension workflows](../docs/EXTENDING_EXPERIMENTS.md). The helpers export separate plans and blank packets without modifying these supplied data.
