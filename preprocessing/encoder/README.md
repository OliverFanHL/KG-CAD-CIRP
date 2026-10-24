# Geometry preprocessing and encoder

This directory contains the geometry preprocessing and supervised local-surface encoder used to produce the study representations. The study uses one validation-selected checkpoint (epoch 17, seed 42). Its weights, selection metadata, cohort, and exported representations are in `data/encoder/` at the repository root.

## Rebuild representations

Paths in this section are relative to this directory. Training and extraction are implemented in `src/jev_tmcad/geometry_encoder.py`, with workflow commands in `src/jev_tmcad/study_cli.py`. Dependencies are declared in `pyproject.toml` and `environment.yml`.

Rebuilding from raw STEP files requires the original dataset, source hashes, split assignments, and optional CAD/PyTorch packages. The reported study uses the selected checkpoint throughout; variability across encoder training runs was not measured.

## Reproduce retrieval results

Use the repository-level `scripts.prepare_data` and `scripts.reproduce` commands with the supplied representations, as described in the [reproduction guide](../../docs/REPRODUCIBILITY.md). Upstream license terms are in [LICENSE](LICENSE).
