# Reproduce the study

Run commands from the repository root in an activated Python 3.12 environment.

```text
python -m pip install -e ".[technical,dev]"
python -m scripts.prepare_data
python -m scripts.reproduce --metrics-only
python -m scripts.reproduce
python -m pytest -q
```

Preparation verifies the distributed artifact parts and all restored file hashes, then exposes the study inputs under `data/`. Primary queries and final metric tables are already checked in. Full offline reproduction requires neither the raw STEP corpus, encoder retraining, model credentials, nor additional API calls.

The metrics-only workflow regenerates role means, collection means, summary statistics, role effects, and paired tests from saved per-query metrics. The full workflow reconstructs observations from saved responses, reranks complete fixed source arrays, applies numerical relevance rules, and recalculates metrics and statistics. Nine output artifacts are checked against their recorded hashes. Saved files retain their numerical values and encoding conventions.

Outputs are written to `results/reproduced_metrics/` and `results/reproduced/`. The program refuses an existing destination; use, for example, `--output results/reproduced_check` for another execution. Each destination contains `release_replay_audit.json`, and a full run records execution details in `replay.log`.

## Figures and documents

```text
python -m pip install -r publication/requirements.txt
python -m scripts.generate_figures
```

This regenerates Figures 2–3 from `results/summary.csv` and `results/family_metrics.csv`. It makes no model calls and does not rerun statistical tests. `python -m scripts.build_documents` rebuilds editable Word documents from the final LaTeX sources and tables; Word must update its fields and pagination. The supplied reviewed documents and artwork are available directly in [publication](../publication/README.md).

## Geometry and encoder

`data/encoder/` includes the selected epoch-17 checkpoint, selection metadata, cohort, and descriptor/embedding exports. Encoder and geometry preprocessing source/configurations are under `preprocessing/encoder/`. Raw-data rebuilding requires the original TMCAD corpus, its source hashes and splits, and the upstream optional CAD/PyTorch environment. It is distinct from reproducing the primary saved-input analysis. Neither retraining nor fresh model collection was performed for this repository organization.

The study specification is in `configs/study.json`; `requirements-validated.txt` records the tested environment, while `pyproject.toml` declares supported core dependencies. The internal sealed execution workspace is maintained separately for byte-level provenance; readers need only the public commands above.
