# Experiments and outputs

Run from the repository root; substitute the existing Python path from the README if needed. Source inputs remain at `TMCAD/` and `../jepa_cad/jev_tmcad_v0/outputs/`. Manifests record versions and input/code hashes.

## Current studies

### Direct KG-constrained performance with audited-v2 judgments

Plan: `docs/KG_CONSTRAINED_EVALUATION.md`; config: `configs/kg_constrained_audited_v2.json`; report: `outputs/kg_constrained_audited_v2/REPORT.md`. Run the existing evaluator with `--config configs/kg_constrained_audited_v2.json`, repeat using its `--output outputs/kg_constrained_audited_v2/replay`, run the validator with the same config, and generate `scripts.report_kg_constrained`.

The focused analysis separates nineteen constrained queries, nineteen unconstrained exemplar queries and eleven text cases. It tests Full against seven controls per representation with identical candidate access, saving family differences, Holm-adjusted tests, strict-grade sensitivity, robustness and source-linked top5 evidence. This re-evaluates existing queries with newer audited labels; it does not expand the effectiveness task set. The previous scalability-only restriction applies to that study, not to this newly authorized descriptive re-evaluation. Remaining AI-only and sampled-review limitations are preserved.

### Annotation-gated systems evaluation

The [random annotation audit](../outputs/annotation_quality_audit_v2/REPORT.md) precedes this workload. It publishes a separate audited annotation version; no relevance score is used by the systems benchmark.

```powershell
python -m scripts.benchmark_audited_scalability
python -m scripts.report_audited_scalability
python -m pytest -q tests/test_scalability_workload.py --basetemp outputs/test-temp-scalability-audit
```

Design: `docs/SCALABILITY_AUDIT_PLAN.md`; configuration: `configs/scalability_audited_v2.json`; report: `outputs/scalability_audited_v2/REPORT.md`. The benchmark uses the actual serving ranker plus explanations and JSON serialization, not the vectorized effectiveness runner. It measures 1,000 structured signatures, two randomized passes, candidate scans at 250/1000/1816/4000/9798 real parts, matched graph/flat/filter controls, k=5/20/100, bounded thread batches, graph construction and separate allocation tracing. It has a 30-minute total timeout. Catalogs mixing splits are systems-only; do not compute test quality from them.

Outputs preserve queries, nested candidate lists, per-call wall/CPU/serialization times, response sizes, memory/hardware metadata, hashes and exact result comparisons. The study measures up to the available ~10k catalog on one Windows machine. It does not establish distributed capacity, an industrial SLA, or graph-specific speed superiority.

### Corrected annotations: comprehensive paired evaluation

```powershell
python -m scripts.run_verified_evaluation
python -m scripts.run_verified_evaluation --output outputs/verified_evaluation_v1/replay
python -m scripts.validate_verified_evaluation
python -m scripts.report_verified_evaluation
python -m pytest -q tests/test_verified_evaluation.py --basetemp outputs/test-temp-verified-evaluation
```

Configuration: `configs/verified_evaluation.json`. Design: `docs/VERIFIED_EVALUATION_PLAN.md`. Output: `outputs/verified_evaluation_v1/REPORT.md`. This evaluates 1,350 CADs with complete AI functional grades for all eleven existing tasks. It adds same-filter isolated scoring controls, fixed reciprocal rank fusion, separate added-candidate and numerical-signature cohorts, conditional-grade sensitivity, exemplar checks, classifier-coordinate corruption, exact-tie checks, and reference-guard sensitivity. Train models, exemplars, task scopes and score weights stay frozen.

Primary comparisons are exemplar-only equal-family nDCG@5, with eleven exact family sign-flip tests, Holm adjustment and exploratory bootstrap intervals. Text-only and deeper retrieval metrics remain separate. Full 1,816-gallery results report missing-judgment bounds. Complete closed-gallery recall/AP are valid only for the annotated cohort. AI authorship and small authored-task diversity remain explicit limitations.

New outputs include `primary_comparisons.csv`, `family_metrics.csv`, `complete_rankings.npz`, `cohorts.json`, `robustness_summary.csv`, `exemplar_sensitivity.csv`, `full_gallery_bounds.csv`, `run_manifest.json`, `integrity.json`, and `reproducibility.json`. The validator rehashes all 1,350 actual STEP files, compares independent metric calculations, and requires exact table/ranking replay.

### Frozen historical studies

The following commands intentionally retain the original historical labels and output formats.

```powershell
python -m cadkg full-scale --stage all
python -m cadkg intent-batch --stage all
```

Full-scale stages: `prepare`, `measure`, `build`, `class`, `functional-search`, `pool-evidence`, `functional-evaluate`, `report`, `validate`. Config: `configs/full_scale.json`; output: `outputs/full_scale`. Geometry caches are resumable/source-checked; native import workers have timeouts. The auxiliary category/constraint matrix has 319,248 paired runs. Category targets are not engineering relevance truth.

Expanded stages: `search`, `evidence`, `evaluate`, `report`, `validate`. Config: `configs/intent_batch2.json`; output: `outputs/intent_batch2`. Reuses the full-scale bundle: 11 families, 49 queries, two representations, eight methods and the same 1,816 held-out candidates. Train exemplars and validation bounds precede test ranking. Old primary cases remain exactly comparable.

| Method | Contribution |
|---|---|
| Intent → category | Function mapping followed by category probability |
| Exemplar cosine | Representation similarity |
| Cosine + classifier | Similarity and classifier/reference distribution |
| Matched flat table | Same information and score as the graph |
| Remove exemplar cosine | Remove direct cosine and its weight; retain classifier/reference distributions, function priors and filter |
| Remove classifier/function branch | Exemplar cosine and dimension filter |
| Remove dimension filter | Full scoring without the hard filter |
| Full | All available signals and the filter |

Embeddings are supervised local-surface encoder exports, not newly trained TMCAD JEPA. Descriptors are train-normalized. The classifier stays fixed across representations. Text-only cases lack exemplar cosine and are separate. Flat/full equality is expected; graph storage alone is not a performance contribution.

## Outputs

| File/directory | Meaning |
|---|---|
| `bundle/` | Integrity-checked catalog, knowledge, graph and manifest |
| `protocol.json`, `run_manifest.json` | Frozen inputs/settings and provenance |
| `rankings.csv`, `explanations.json` | Full-gallery top5 and component evidence |
| `evidence/`, `pool_manifest.csv`, `family_pools.json` | Source surfaces/views, neutral IDs and judged pools |
| `per_query.csv`, `summary.csv` | Previous/new/combined and exemplar/text metrics |
| `paired_differences.csv`, `family_differences.csv` | Signed differences, including losses |
| `ai_judgments.csv`, `sensitivity.csv` | Explicit AI labels and alternative conditional grading |
| `equality_audit.csv` | Identical rankings versus equal grade sequences/metrics |
| `report/REPORT.md`, figures | Labeled results, CAD examples and schematic |
| `validation.json` | Integrity and exact replay |

The original full-scale study prefixes functional outputs with `functional_`. Existing human CSV work is never overwritten. The app stores human work separately in `outputs/expert_review/`.

Every returned candidate is judged in the union of all systems' top5 plus controls. Family-balanced P@5 counts grades 1/2; strong P@5 counts grade 2; pooled nDCG uses the finite judged pool's ideal ordering. This does not establish full-gallery recall. Correlated variants, same-author AI grading and pool selection limit inference; no confirmatory population significance is claimed.

## Human evaluation and validation

```powershell
python -m cadkg annotate --open
python -m cadkg annotate --action export --reviewer expert_a
python -m cadkg expert-evaluate --review outputs/expert_review/exports/expert_a/review.json
python -m pytest -q --basetemp outputs/test-temp-local
python -m cadkg full-scale --stage validate
python -m cadkg intent-batch --stage evaluate
python -m cadkg intent-batch --stage validate
```

See [ANNOTATION.md](ANNOTATION.md). Expert outputs add `coverage.csv/json`, `expert_judgments.csv`, `REPORT.md` and `review_provenance.json` to the shared metric files. Incomplete reviews produce coverage, not fabricated scores. `--allow-partial` selects only complete families.

Tests check geometry, provenance, ranking, unknown handling, source changes, knowledge/evaluation separation, persistence and metric coverage with synthetic fixtures. Full-scale validation checks 480 serving comparisons and exact replay. Expanded validation checks nine invariants and exact metrics/rankings/human worksheet replay. Pytest's explicitly named workspace temp directory is disposable.

## Historical commands retained

| Command | Purpose |
|---|---|
| `demo --output outputs/demo` | Fictional plumbing fixture |
| `pilot --output outputs/pilot` | Bounded real import and measurement |
| `technical --output outputs/technical` | 60-part cohort, 288 technical runs and blank pack |
| `technical-replay --run outputs/technical` | Replay without measuring again |
| `reference-prepare` | Original bounded inspection evidence |
| `reference-run` | Original AI three-family study |
| `reference-report` | Original plots and examples |

These studies live under `cadkg/experiments/`; they are not duplicate annotation apps. All existing CLI arguments remain supported. Core utilities `inventory`, `import-legacy`, `measure`, `build`, `query`, `train-probe`, `evaluate`, `reason`, `evaluate-reasoning`, `annotate-init` remain available; use their `--help`. Generic `evaluate` still requires independent closed-gallery labels; `expert-evaluate` is the new pooled-family path.

Cleanup preserves datasets, model exports, frozen inputs and result directories. Historical guides were consolidated into three guides. Originals are recoverable from `outputs/maintenance/before_review_cleanup.zip`; original hashes and the mechanical maintenance scripts are alongside it. Those scripts are not required to reproduce experiments.
