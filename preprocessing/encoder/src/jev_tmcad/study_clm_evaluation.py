"""Assemble CLM-only matched features for remote evaluation.

Uses the same frozen cohort, descriptor/geometry columns, semantic validator,
and registered evaluators as the local full-study matrix. No baseline inference
files or Windows-specific checkpoint paths are required on Kaggle.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cad_features import numeric_feature_columns
from .clm_client import file_sha256
from .provenance import with_source_labels
from .semantic_state import QuantileSchema
from .study_data import check_cohort_hash, valid_descriptors
from .study_evaluation import _backend_frame


def prepare_clm_features(cohort: pd.DataFrame, descriptor_path: Path,
                         geometry_path: Path, geometry_checkpoint: Path,
                         clm_results: Path, clm_checkpoint: Path,
                         schema_path: Path, questions: dict, *, model: str = "clm-latest",
                         state_variant: str = "compact", include_numeric: bool = True,
                         include_profiles: bool = False) -> pd.DataFrame:
    cohort = with_source_labels(cohort)
    desc = valid_descriptors(descriptor_path)
    state_rows = cohort[["sample_id"]].merge(desc, on="sample_id", validate="one_to_one")
    if len(state_rows) != len(cohort):
        raise ValueError("Descriptor coverage differs from frozen cohort")
    for key in ("split", "label", "source_label", "relative_path", "absolute_path"):
        actual = state_rows[key].astype(str).to_numpy()
        expected = cohort[key].astype(str).to_numpy()
        if not (actual == expected).all():
            raise ValueError(f"Descriptor {key} differs from frozen cohort")
    numeric = [c for c in numeric_feature_columns(list(desc))
               if pd.api.types.is_numeric_dtype(desc[c])]
    frame = cohort.merge(desc[["sample_id", *numeric]], on="sample_id", validate="one_to_one")
    frame = frame.rename(columns={c: f"desc__{c}" for c in numeric})
    check_cohort_hash(cohort, geometry_path.with_suffix(".meta.json"))
    geometry_meta = json.loads(geometry_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if file_sha256(geometry_checkpoint) != geometry_meta["checkpoint_sha256"]:
        raise ValueError("Geometry embeddings do not match the bundled checkpoint")
    geom = pd.read_csv(geometry_path, dtype={"sample_id": str})
    if geom["sample_id"].duplicated().any():
        raise ValueError("Duplicate geometry embedding sample IDs")
    matched = cohort[["sample_id", "split", "label"]].merge(
        geom[["sample_id", "split", "label"]], on="sample_id",
        suffixes=("_cohort", "_geometry"), validate="one_to_one")
    if len(matched) != len(cohort) or any(
        matched[f"{key}_cohort"].astype(str).ne(matched[f"{key}_geometry"].astype(str)).any()
        for key in ("split", "label")
    ):
        raise ValueError("Geometry embedding split or label differs from frozen cohort")
    for key in ("source_label", "relative_path", "absolute_path"):
        if key in geom:
            values = cohort[["sample_id", key]].merge(geom[["sample_id", key]],
                                                        on="sample_id", suffixes=("_cohort", "_geometry"),
                                                        validate="one_to_one")
            if values[f"{key}_cohort"].astype(str).ne(values[f"{key}_geometry"].astype(str)).any():
                raise ValueError(f"Geometry embedding {key} differs from frozen cohort")
    geom_cols = [c for c in geom if c.startswith("geom__") or c.startswith("geom_logit__")]
    frame = frame.merge(geom[["sample_id", *geom_cols]], on="sample_id", validate="one_to_one")
    schema = QuantileSchema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
    semantic = _backend_frame(clm_results, state_rows, questions, schema, backend="clm",
                              model=model, required=True, include_numeric=include_numeric,
                              include_profiles=include_profiles, state_variant=state_variant,
                              checkpoint_sha256=file_sha256(clm_checkpoint))
    frame = frame.merge(semantic.drop(columns=["split", "label"]), on="sample_id", validate="one_to_one")
    if len(frame) != len(cohort):
        raise ValueError("CLM feature frame does not cover frozen cohort")
    return frame
