# Experiment map

The current manuscript is revision 8. Revision 8 changes presentation, using unchanged revision-7 experimental data.

| Stage | Main implementation | Configuration / data | Status in current manuscript |
|---|---|---|---|
| Source geometry / encoder | `preprocessing/encoder/src/jev_tmcad/geometry_encoder.py`, `study_cli.py`; `cadkg/geometry.py` | `data/encoder/`; `configs/full_scale.json` | Fixed representation and dimensional evidence |
| Shared initial retrieval | `cadkg/retrieval.py`, `graph.py`, `knowledge.py` | Source-supported priors and frozen bundle | G/A/F signals and dimensional screening |
| v5 feature review | `scripts/prepare_feature_review.py`, `validate_feature_review.py` | `annotations/intent_feature_review_v5/` | AI-reference judgments; no human validation |
| v6 query diversity | `scripts/run_query_diversity.py` | `configs/query_diversity_v6.json` | Initial arrays; 264 new-exemplar primary queries selected from 440 |
| Initial Luna/Sol | `scripts/evaluate_llm_geometry_matched.py`, `evaluate_llm_geometry_sol_matched_v1.py` | Matching `outputs/llm_geometry*_v1/` | First collection per KG observer |
| Repeated visual evidence | `scripts/complete_llm_robustness_v7.py`, `evaluate_llm_robustness_v7.py` | `outputs/llm_robustness_v7/` | Main three-collection results and holistic control |
| Local repeats / ablation | `outputs/robustness_local_v7/replay_local.py`, `scripts/repeat_remaining_local_candidates_v7.py` | `outputs/robustness_local_v7/`, `robustness_remaining_local_v7/` | Deterministic fixed-encoder controls |
| Publication | `manuscript/revision8/scripts/` | Packaged CSVs, table sources and bibliography | Current figures and manuscript |

Earlier verified-v1, audited-v2, query-v3, candidate-v4 and feature-v5 reports are historical evidence. Their galleries, labels, query panels and comparison families differ. Do not combine their numerical results. Full old run trees remain in the original workspace; this repository retains versioned code/configuration, annotation releases and compact reports, with the complete saved inputs needed for the current primary replay.

Frozen scripts intentionally retain separate versioned implementations where protocol hashes require them. `scripts/reproduce_release.py` consolidates the supported offline workflow without changing their method or numeric behavior.
