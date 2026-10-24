# Reproduce the study

Run commands from the repository root in an activated Python 3.12 environment.

```text
python -m pip install -e ".[technical,dev]"
python -m scripts.prepare_data
python -m scripts.reproduce --metrics-only
python -m scripts.reproduce
python -m pytest -q
```

Data preparation verifies the dataset packages and input hashes, then unpacks the study inputs under `data/`. Primary queries and final metric tables are included directly in the repository. Both reproduction workflows run offline from saved inputs.

## Reproduction workflows

| Command | Workflow |
|---|---|
| `python -m scripts.reproduce --metrics-only` | Recalculate role means, collection means, summary statistics, role effects, and paired tests from saved per-query metrics |
| `python -m scripts.reproduce` | Reconstruct observations from saved responses, rerank the complete source arrays, apply numerical relevance rules, and recalculate metrics and statistics |

The full workflow checks nine output files against their recorded hashes, including numerical values and encoding conventions.

The default output directories are `results/reproduced_metrics/` and `results/reproduced/`, respectively. Each run requires a fresh destination; use `--output results/reproduced_check` for another run. Each destination contains `release_replay_audit.json`, and full runs also write execution details to `replay.log`.

## Geometry and encoder

`data/encoder/` includes the selected epoch-17 checkpoint, selection metadata, cohort, and descriptor and embedding exports. The [preprocessing guide](../preprocessing/encoder/README.md) describes the encoder implementation and dependencies. Rebuilding from raw geometry requires the original TMCAD corpus, source hashes, split assignments, and optional CAD/PyTorch dependencies. The primary reproduction uses the supplied exports and saved responses.

## Configuration and input records

| Location | Purpose |
|---|---|
| `configs/study.json` | Study specification |
| `requirements-validated.txt` | Tested dependency versions |
| `pyproject.toml` | Supported core dependencies |
| `cadkg/study/` | Evidence aggregation, deterministic reranking, partial-label metrics, and role-level statistics |
| `data/manifest.json` | Path, size, and SHA-256 for every distributed input |

## New experiments

See the [extension guide](EXTENDING_EXPERIMENTS.md) to evaluate new observers on the fixed panel or prepare new annotation and gallery releases.
