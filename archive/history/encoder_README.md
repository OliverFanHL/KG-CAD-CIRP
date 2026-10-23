# TMCAD: local geometry and auxiliary semantic decisions

The [CLM benchmark extension](research_v2/CLM_BENCHMARK.md) adds runnable
official CLM inference and matched classification/retrieval outputs on the
frozen full cohort. Its separate configuration is `configs/study_clm.yaml`.
The [Kaggle CLM package](kaggle_clm/README.md) runs these experiments without
installing vLLM or the Qwen3-8B model on Windows.

This project compares global CAD descriptors, a learned local surface representation, JEV, Laya, and their combinations for mechanical-part classification and expert-judged retrieval. `run.py` is the single entry point. The [executable experiment matrix](research_v2/EXPERIMENT_MATRIX.md) maps every method to one ID, command, feature recipe, implementation path, and output. The [artifact audit](research_v2/RESULTS_AUDIT.md) records what has actually completed and its interpretation limits. A draft [LaTeX manuscript](research_v2/manuscript/main.tex) and [framework schematic](research_v2/figures/framework_gpt_image_2.png) are in `research_v2/`. Existing configs, source data, and results remain in place.

## Structure and configurations

| Location | Responsibility |
| --- | --- |
| `run.py`, `src/jev_tmcad/study_cli.py` | Main CLI and orchestration |
| `src/jev_tmcad/study_data.py`, `study_audit.py`, `provenance.py` | Cohort, duplicate audit, original paths/labels |
| `cad_features.py`, `semantic_state.py`, `mesh_data.py` | Descriptors, compact semantic state, STEP surface neighborhoods |
| `geometry_encoder.py` | Supervised local geometry training and embeddings |
| `jev_client.py`, `laya_client.py` | External JEV requests and local Laya inference |
| `study_laya_finetune.py`, `study_ground_truth.py` | Typed Choice fine-tuning and blinded retrieval ground truth |
| `study_experiments.py` | Authoritative experiment IDs and feature recipes |
| `study_evaluation.py`, `study_retrieval.py`, `efficiency.py` | Classification, judged retrieval, and observed timing |
| `configs/study.yaml` | Historical 200-part full-state JEV pilot |
| `configs/study_compact.yaml` | Historical 200-part matched compact JEV/Laya pilot |
| `configs/study_full.yaml` | Full 9,798-part compact protocol and all full-study paths |
| `configs/study_full_ft.yaml` | Same frozen cohort and base caches; separate fine-tuned Laya, expert judgments, and evaluation outputs |
| `configs/study_full_ft_kaggle.yaml`, `kaggle_laya/` | Kaggle notebook, verified small upload ZIP, and separate local import/evaluation paths |
| `configs/questions_compact.yaml` | Shared Choice/Noul question wording for full JEV/Laya comparison |

The original `scan`, `extract`, `inspect-state`, `jev`, `evaluate`, and `qualitative` commands and their output files remain available for reproducing the legacy pilot. They are separate from the `study-*` matrix.

## Installation

Use a Python environment with CadQuery/OpenCASCADE, then install the project:

```powershell
conda create -n laya-tmcad -c conda-forge python=3.11 cadquery
conda activate laya-tmcad
pip install -e ".[geometry,laya]"
```

New JEV calls require `TYPESAFE_API_KEY` in the environment or local `.env`. Laya requires its model checkpoint. The JEV SDK performs bounded retries and records API latency; Laya uses a local checkpoint and records warm inference separately. The implementation validates typed responses, questions, and complete per-sample coverage before evaluating. When `jev.model: jev-latest` is used, the actual returned model revision is retained in the response record; for a publication rerun, pin the model/checkpoint versions.

## Full-study execution

The full cohort, descriptor CSV, mesh cache, geometry checkpoint/embeddings, JEV JSONL, and Laya JSONL already exist. The commands below describe the dependency order for a fresh run. **Do not rerun `study-plan --force` against these artifacts.** `study-plan` normally refuses to overwrite a frozen cohort; `study-pool` refuses to overwrite an annotation file. All `study-*` commands read `configs/study_full.yaml` when passed as shown.

```powershell
python run.py --config configs/study_full.yaml study-plan
python run.py --config configs/study_full.yaml study-audit
python run.py --config configs/study_full.yaml study-mesh
python run.py --config configs/study_full.yaml study-train-geometry
python run.py --config configs/study_full.yaml study-embed
python run.py --config configs/study_full.yaml study-jev --dry-run
python run.py --config configs/study_full.yaml study-jev
python run.py --config configs/study_full.yaml study-laya
python run.py --config configs/study_full.yaml study-experiments
```

`study-jev` **acquires semantic answers**: it builds a compact state from 14 selected deterministic descriptor fields rounded to three decimals, asks the fixed Choice and Noul questions, caches JEV answers by payload hash, and saves per-sample records. The quantile schema is passed through this stage but is used by the separate full-state protocol, not by `cad.state_variant: compact`. `study-jev` trains no model. `--dry-run` only writes request previews and estimates cache/API work; it produces no result records. `study-laya` supplies the same state and question protocol to a local typed-decisions model; it also trains no model. `study-train-geometry` is the supervised training stage: 512 sampled surface points with normals and 16 nearest-neighbor indices enter two local EdgeBlocks; a class head is optimized with cross-entropy on train labels. The best checkpoint is chosen by validation macro F1, with early stopping. `geometry.epochs` is a maximum.

Run a **single classification matrix row** with its exact ID, for example:

```powershell
python run.py --config configs/study_full.yaml study-evaluate --experiment classification/descriptor_only
python run.py --config configs/study_full.yaml study-evaluate --experiment classification/geometry_plus_jev
```

Each single-row run writes under `outputs/study/full/evaluation/experiments/classification/<ID>/`. Its `semantic_response_alignment.csv` records the original sample/path/label alongside each backend's request hash, response sample ID, reuse origin, and cache source. The historical aggregate command remains:

```powershell
python run.py --config configs/study_full.yaml study-evaluate --classification-only
```

The previous `jev request hash belongs to another sample` error was caused by indexing results by hash alone. Compact descriptors can produce identical request payloads for distinct samples; 9,798 JEV rows contain 9,182 unique hashes. The revised evaluator recomputes and validates every sample's payload hash, indexes JEV by `(sample_id, request_hash)`, and checks each record's saved split and label. The original Laya run saved only one row per unique payload, leaving 616 sample IDs without rows. For those IDs only, evaluation reuses a response after independently verifying the identical payload; it records the origin sample ID and excludes this reuse from inference latency. The original result JSONLs are unmodified. Future Laya runs save per-sample records for shared payloads.

## Retrieval and evaluation

The full judged retrieval study is pending. `outputs/study/full/retrieval_judgments.csv` has already been generated as an **ungraded template**: it contains 130,412 pairs over 2,940 validation/test queries, and every `grade` cell is blank. Do not rerun `study-pool` against this file. For a fresh, separately configured run, the pool command is:

```powershell
python run.py --config configs/study_full.yaml study-pool
```

For each validation/test query, the pool combines top neighbors from available modalities with random candidates. Expert annotators should view anonymized geometry and assign grade 0 (unrelated), 1 (broad form/role overlap), 2 (similar interfaces, local form, or manufacturing role), or 3 (close functional and geometric substitute). Source paths are deliberately preserved for traceability but reveal class folders; use an anonymized presentation during blinded annotation. A query must have at least one graded positive and negative. Only judged within-split pairs are evaluated; unjudged pairs are not assumed irrelevant.

Fill the existing `grade` column with expert-assigned integers 0–3, preserving every query/candidate ID and split. All 130,412 rows must be graded under the current full-query protocol, and each query needs at least one grade 0–1 and one grade 2–3. Do not copy folder labels or similarity scores into the grade column: that would make the retrieval metric circular or merely class-based. Once the annotation is complete, run one exact retrieval row, for example:

```powershell
python run.py --config configs/study_full.yaml study-evaluate --experiment retrieval/geometry_plus_jev
```

The unqualified `study-evaluate` command runs the complete classification and judged-retrieval matrix and requires the judgment file. Use `--classification-only` when retrieval annotations are unavailable.

Classification reports accuracy, macro F1, balanced accuracy, negative log likelihood, Brier score, per-sample predictions, and confusion matrices with original source-label mappings. The linear probe C is selected on validation macro F1, then fitted on train plus validation before test. Retrieval reports nDCG, precision, recall, MAP, and MRR on the common judged pool; fusion weight is selected on validation nDCG and applied once to test. The evaluator retains sample ID, original relative/absolute path, source label, canonical label, and split in manifests and prediction files. Efficiency output separates legacy descriptor timing when available, STEP preprocessing/cache lookup, geometry training and batched inference, JEV API latency, and Laya warm inference. Compare those timings only with the same hardware, software, batch size, cache policy, and concurrency; external round trips and local model time have different scopes.

## Research status and limits

The full classification summary is available at `outputs/study/full/evaluation/classification/summary.json`; the [audit](research_v2/RESULTS_AUDIT.md) gives selected metrics. Descriptor plus geometry is strongest among the current evaluated methods (test macro F1 0.915). Adding JEV or Laya semantic signatures to geometry does not exceed that combined baseline in the existing run. This result concerns semantic reasoning over the project's **compressed descriptor state**, not native B-Rep understanding by the semantic models. No full expert-judged retrieval result exists yet.

The duplicate audit found 183 byte-identical STEP groups spanning splits and 25 groups with conflicting labels. The existing full split and its already inspected test results are descriptive. A confirmatory paper claim needs a new duplicate/family-aware split, label reconciliation, locked train/validation choices, and a fresh test evaluation. The draft manuscript states this limitation explicitly and contains only Introduction, Related Work, and Methodology.

To compile the manuscript on a machine with a LaTeX distribution, run `latexmk -pdf main.tex` from `research_v2/manuscript/`; the adjacent `references.bib` supplies its verified paper and software citations. A TeX compiler is not installed in this workspace, so no PDF is included.

## Expert retrieval labels and fine-tuned Laya extension

Use `configs/study_full_ft.yaml` for the new extension. It reads the **same** 9,798-part cohort, descriptor cache, geometry features, JEV records, base Laya records, and 10-class Choice/Noul wording as `study_full.yaml`. It writes new artifacts under `outputs/study/full_ft/` and expert annotations under `outputs/study/full/annotations/`. The existing results remain intact. The experiment registry printed by `study-experiments` is authoritative; the extension rows are listed in [the matrix](research_v2/EXPERIMENT_MATRIX.md).

### What the semantic systems receive

For each sample, `semantic_state.py` converts 14 numerical CAD descriptor fields into a compact JSON object: `object: unlabeled mechanical CAD part` plus rounded extent, fill, count, surface, symmetry, periodicity, and axis facts. It contains **no** sample ID, file path, class folder, or ground-truth label. This compact protocol ignores the quantile schema and the `include_numeric_context` and `include_profile_arrays` flags. `configs/questions_compact.yaml` supplies one `part_class` typed Choice with the exact instruction “Choose the mechanical part class supported by the geometry facts.” and ten named class criteria; it also supplies 22 fixed typed Noul questions about form, function, and manufacturing plausibility. JEV receives `state` and SDK `Choice`/`Noul` objects through `TypeSafeClient.system_one`. Base and fine-tuned Laya receive the same `state` and equivalent typed question dictionaries through local `agent.predict`.

Each backend returns a Choice label and ten probabilities, plus one `[0,1]` score per Noul question. The Choice label is compared with the preserved folder-derived canonical class on the same train/validation/test split. Noul scores have **no independent binary ground truth** in this dataset: they are used as features for validation-selected probes and as retrieval vectors, not scored as if the class label answered each question. Geometry embeddings come from local sampled STEP surfaces and are separate from the descriptor-only semantic state. Every semantic result is validated against its sample ID, request hash, model identity, exact state, and exact questions before evaluation.

### Expert relevance protocol

The package has already been prepared with 50 validation and 100 test queries, five and ten per class respectively, and 6,645 within-split pairs. All methods rank the **same** fixed candidates. Class labels and model scores do not define relevance. Two independent mechanical-CAD experts should open the opaque assets named in `rater_a.csv` and `rater_b.csv`, with no access to the private manifest or each other's ratings. `study-grade-assets` copies these files into `blinded_cad/<sample_id>.stp`; this optional step needs about 1.46 GB of storage. The private selected-pair and asset manifests contain original paths and labels for audit and must not be distributed to raters.

Grade each pair by **engineering substitutability and local geometric/interface similarity**: 0 unrelated; 1 broad form or role overlap; 2 similar local geometry, interfaces, or manufacturing role; 3 close functional and geometric substitute. Use 2–3 as relevant for precision, recall, MAP, and MRR; nDCG retains all four grades. After both raters finish, an adjudicator fills `adjudication.csv` **for every disagreement**, using the same IDs. The finalization command rejects missing grades, changed pair IDs, and queries lacking both a relevant and a nonrelevant candidate. It produces an immutable `judged_pairs.csv`, per-pair provenance, and agreement statistics. If coverage fails, expand the pool and obtain new expert ratings before evaluating; do not infer missing grades from folders or scores.

```powershell
# Already completed once for this workspace; do not repeat unless creating a new protocol.
python run.py --config configs/study_full_ft.yaml study-grade-prepare

# Optional: materialize 2,643 opaque STEP files (~1.46 GB) for the raters.
python run.py --config configs/study_full_ft.yaml study-grade-assets

# Run only after experts have filled rater_a.csv, rater_b.csv, and disagreements in adjudication.csv.
python run.py --config configs/study_full_ft.yaml study-grade-finalize
```

`study_full.yaml` retains the historical all-query pool protocol. The extension explicitly sets `retrieval.query_scope: preselected`, and its evaluator checks the finalized file hash and exact query IDs against the annotation protocol. Retrieval cannot run until experts provide grades. The current sampled pool was frozen before fine-tuned Laya inference; it is common to every method, although it was originally assembled using the existing representations and may favor their retrieved neighbors. The duplicate audit and previously inspected test split also limit confirmatory claims; treat this extension as a matched exploratory comparison until a new duplicate-aware split is locked.

### Laya fine-tuning and execution

The local preparation records are already available: 6,181 training, 677 train-only calibration, and 980 validation examples. Identical compact states remain in the same train/calibration partition. Where distinct training parts share an identical compact state but disagree on class, the **train-only empirical class distribution** is the soft target. Validation keeps original hard labels; test labels are never written into fine-tuning data. Training adapts the official typed-decision checkpoint using soft cross-entropy plus Laya's group-relative proper-scoring RLCD objective for **Choice only**. The best epoch is selected by validation macro F1. Temperature is fitted on the separate train-only calibration partition. Unlabeled Noul questions are not pseudo-labeled; the fine-tuned model answers them only at inference. See the [official Laya repository](https://github.com/NandhaKishorM/laya) and its [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) for the source method.

```powershell
# Already prepared in this workspace; rerun only with a new output directory.
python run.py --config configs/study_full_ft.yaml study-laya-ft-prepare

# These commands have NOT been run here. They download the pinned base checkpoint if absent.
python run.py --config configs/study_full_ft.yaml study-laya-ft-train
python run.py --config configs/study_full_ft.yaml study-laya-ft-infer
python run.py --config configs/study_full_ft.yaml study-evaluate --classification-only

# After study-grade-finalize, run one retrieval row or all rows:
python run.py --config configs/study_full_ft.yaml study-evaluate --experiment retrieval/geometry_plus_laya_ft
python run.py --config configs/study_full_ft.yaml study-evaluate
```

The checkpoint directory contains Laya-compatible model weights, encoder, tokenizer, config, and training provenance. `study-laya-ft-infer` writes one record per sample; a repeated identical state can reuse a validated payload result while retaining its own sample ID and original metadata. The extension adds `laya_ft_direct_choice`, `laya_ft_choice_probabilities`, `geometry_plus_laya_ft_choice_probabilities`, `laya_ft_all_questions`, and `geometry_plus_laya_ft` classification rows, plus `laya_ft` and `geometry_plus_laya_ft` retrieval rows. Classification uses the original 6,858/980/1,960 train/validation/test cohort, metrics, validation selection, and source-label confusion matrix. Retrieval uses the common expert-judged pair set and validation-selected fusion weight. `efficiency.json` separately reports base/fine-tuned warm local inference, model load, preflight, and the existing geometry/JEV timings. Use the same hardware, cache state, and batch settings to compare latency.

### Monitoring Laya fine-tuning

The trainer now shows an inner batch progress bar for each epoch and writes `outputs/study/full_ft/laya_checkpoint.progress.json` every 25 batches. The file reports the stage (base-checkpoint loading, tokenizing, training, validating, calibrating, saving, or complete), process ID, elapsed time, epoch, batches completed, samples seen, and recent loss. In another PowerShell window, inspect it with:

```powershell
Get-Content outputs/study/full_ft/laya_checkpoint.progress.json | ConvertFrom-Json | Format-List
Get-Process python | Select-Object Id,CPU,StartTime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

If the fine-tuning process was started **before this progress change**, it keeps its already loaded code and cannot emit the new progress file. In that run, increasing Python CPU time and sustained GPU utilization indicate active computation. The original outer progress bar advances only after all approximately 3,091 mini-batches of an epoch, so it may remain at 0% for a long time. Do not start a second training job while the first is active.

### Kaggle execution

For a free-GPU run, use the ready-to-upload [Kaggle package](kaggle_laya/README.md). It contains a self-contained notebook and a verified 1.9 MB upload ZIP with the prepared training partitions and all 9,798 compact states. Kaggle saves selected weights, epoch snapshots, logs, direct Choice reports, and all-sample predictions to one downloadable results ZIP. The local import helper checks those artifacts and places them under `outputs/study/full_ft_kaggle/`; `configs/study_full_ft_kaggle.yaml` evaluates them without overwriting the existing local training run.

The **v2 Kaggle bundle** corrects a first-epoch validation failure: tokenized Choice targets are integer option IDs, so validation F1 now compares them with integer predicted IDs. A metric check runs before training, and an epoch weight snapshot is saved before validation. Older uploaded bundles must be replaced with v2.
