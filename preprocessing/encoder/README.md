# Geometry preprocessing and encoder

This directory contains the geometry preprocessing and supervised local-surface encoder used to produce the study representations. The active study uses one validation-selected checkpoint (epoch 17, seed 42); its weights, selection metadata, cohort, and exported representations are in `../../data/encoder/`.

The training and extraction implementation is in `src/jev_tmcad/geometry_encoder.py`, with orchestration in `src/jev_tmcad/study_cli.py`. Dependencies are declared in `pyproject.toml` and `environment.yml`; optional CAD/PyTorch packages are needed for rebuilding from raw STEP files. The original dataset, source hashes, and split membership are required. Training variation was not measured in the reported study.

To reproduce the reported retrieval results, use the repository-level `scripts.prepare_data` and `scripts.reproduce` commands; no retraining is required. Upstream license terms are preserved in `LICENSE`.
