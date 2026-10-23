# Research data and provenance

Run `python -m scripts.restore_artifacts` from the repository root. Numbered `research_artifacts.zip.partNNN` files form one lossless ZIP, split below GitHub's per-file limit. The manifest lists every archived and directly checked-out source file with its SHA-256 and original location. Paths under `upstream/jev_tmcad_v0/` refer to the original neighboring encoder project.

| Restored location | Experimental role |
|---|---|
| `annotations/source_supported/` | Sourced engineering category–role associations |
| `annotations/intent_feature_review_v5/` | Immutable 1,816 × 11 AI-reference judgments, including two unresolved grades |
| `annotations/intent_queries440_v6/` | Query definitions and manifest identifying the 264-query primary panel |
| `outputs/intent_batch2_matrix_reproduction/bundle/` | Catalog, labels, splits, representations, probabilities, knowledge graph and integrity manifest |
| `outputs/feature_review_v5/evidence/` | Source-linked dimensional/surface evidence and inspection provenance |
| `outputs/query_diversity_v6/evaluation/` | Frozen initial ranking/score arrays, gallery order and evaluation protocol |
| `outputs/llm_geometry_matched_v1/` | First Luna collection, neutral CAD images, clauses, prompts, responses and pool definitions |
| `outputs/llm_geometry_sol_matched_v1/` | First Sol collection with the matched observation contract |
| `outputs/llm_robustness_v7/` | Repeated KG/holistic collections, raw journals, H/J support, ranking indices, metrics and tests |
| `outputs/robustness*_v7/` | Fixed-encoder local execution repeats and component controls |
| `data/encoder/` | Original epoch-17 checkpoint, exports, descriptors, cohort and selection metadata |

The repeated study uses 14,588 new responses and 4,168 historical observations, giving 18,756 observations. IDs join queries to roles, roles/parts to judgments and observations, and ranking indices to the frozen cohort order. Files named `protocol_freeze.json`, `evaluation_freeze.json` and `consumed_evidence_freeze.json` bind source bytes. Historical absolute paths are provenance, resolved in memory by the portable replay adapter.

`cad_examples/` contains ten original STEP files, one per class. `cad_examples_manifest.json` records their original relative paths, part IDs, labels, splits and source hashes. These are illustrative examples, not a replacement for the full TMCAD corpus. The original source is identified in the manuscript bibliography under `zou2025brt` (BRep-based retrieval/TMCAD). Obtain the corpus from that source and respect its original terms. No additional dataset license is asserted here.

Earlier annotation versions and compact historical reports/tables are retained for traceability. Repeated replay arrays, large regenerable sensitivity arrays, full old result trees and the 5.98 GB raw corpus are excluded. The primary offline replay has its full required saved inputs. The inventory preserves relative source locations; it does not contain credentials.
