# Reproduction guide

## Primary publication results

Use the root README installation and restore commands first. Python must run from the repository root. The replay adapter resolves original Windows paths against this checkout in memory and leaves archived files and frozen script hashes unchanged.

```text
python -m scripts.reproduce_release --metrics-only --output outputs/check_metrics
python -m scripts.reproduce_release --output outputs/check_full
```

Choose a new output directory for each execution. The metrics path recomputes five aggregate CSVs from the 17,424 archived query/method/collection metric rows. The full path reconstructs clause observations and holistic support from saved responses, repeats reranking from v6 complete arrays, applies independent numerical relevance rules and recomputes all outputs. SHA comparisons identify exact matches; zero new API calls are made. Original replay scripts refuse existing sealed outputs and use historical path assumptions, so use this adapter for a relocated checkout.

The manuscript's four primary exact sign-flip tests use eleven role effects after collection averaging, 2,048 sign patterns and one Holm family. Bootstrap intervals use 20,000 resamples with seed 20261023. Collection SDs use denominator two. AI labels and the two unknown judgments remain unchanged. No significance claim is added by this release.

## Geometry, initial retrieval and retraining

`cadkg` provides `full-scale`, `intent-batch` and review commands (`python -m cadkg --help`). Frozen study configs are under `configs/`. Upstream geometry/encoder implementation and its original configuration/README are under `preprocessing/encoder/`; its optional environment is separate from core replay. The included checkpoint is the original validation-selected epoch 17, with seed 42 and exported embeddings. Retraining is not part of exact numerical replay, and no training-seed variation is claimed.

The legacy `full-scale --stage prepare` import expects a legacy root containing `jev_tmcad_v0/outputs/manifest.csv`, `descriptors/tmcad_descriptors.csv` and `study/full/geometry_embeddings.csv`; `--legacy-root` selects that root. The supplied exports live under `data/encoder/` and can be placed into that original layout for a raw-data rebuild. Original TMCAD files must occupy `TMCAD/<category>/<file>.stp` and match source hashes. Other preprocessing stages and encoder training use the upstream configuration and optional CAD/PyTorch dependencies. These expensive raw-data/training workflows were not rerun during consolidation; fixed outputs support the verified primary replay.

## Artwork and Word documents

Install `manuscript/revision8/requirements.txt`, change to that directory and run `python scripts/generate_figures.py`, then optionally `python scripts/build_word.py`. Scripts use packaged numerical CSVs and do not collect data. The original private visual-QA validator depends on omitted Word-rendering intermediates; it is retained as history and is not the public numerical validation command.

## Fresh model acquisition

Do not run API executors to reproduce saved results. Original acquisition scripts require an explicit execution flag and process-local credential. Prompts, schemas, limits, job pools and accounting are archived in each study directory. Fresh responses may differ, and a new acquisition must have its own protocol and output directory. No credentials or paid calls are needed by the supported release workflow.
