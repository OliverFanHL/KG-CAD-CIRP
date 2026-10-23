"""Frozen, split-aware cohorts and provenance for matched experiments."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .provenance import with_source_labels
from .utils import read_jsonl, sha256_json, write_json


def valid_descriptors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"sample_id": str})
    valid = df[df["extract_error"].fillna("").astype(str).str.strip().eq("")].copy()
    return with_source_labels(valid)


def cohort_hash(df: pd.DataFrame) -> str:
    fields = ["sample_id", "split", "label", "relative_path"]
    return sha256_json(df[fields].sort_values("sample_id").to_dict("records"))


def load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "split", "label", "absolute_path", "relative_path"}
    if not required.issubset(df):
        raise ValueError(f"Cohort is missing columns: {sorted(required - set(df))}")
    if df["sample_id"].duplicated().any() or not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Cohort needs unique sample IDs and train/val/test splits")
    return with_source_labels(df)


def create_cohort(descriptor_path: Path, jev_path: Path, output_path: Path, *, source: str,
                  per_class: dict[str, int], seed: int) -> pd.DataFrame:
    desc = valid_descriptors(descriptor_path)
    if source == "existing_jev":
        rows = read_jsonl(jev_path)
        if not rows:
            raise ValueError("No cached JEV results for existing_jev cohort")
        latest = rows[-1].get("experiment_hash")
        cached = [r for r in rows if r.get("experiment_hash") == latest]
        ids = {str(r["sample_id"]) for r in cached}
        cohort = desc[desc["sample_id"].astype(str).isin(ids)].copy()
        if len(cohort) != len(ids):
            raise ValueError("Cached JEV IDs do not all have valid descriptor rows")
        metadata = {str(r["sample_id"]): (str(r.get("split")), str(r.get("true_label"))) for r in cached}
        for _, row in cohort.iterrows():
            if metadata[str(row["sample_id"])] != (str(row["split"]), str(row["label"])):
                raise ValueError(f"Cached JEV split/label differs for {row['sample_id']}")
    elif source == "all":
        # The descriptor table already carries the frozen scan splits. Keep
        # every successfully extracted sample without resampling any class.
        cohort = desc.copy()
    elif source == "stratified":
        parts = []
        for (split, label), group in desc.groupby(["split", "label"], sort=True):
            count = int(per_class.get(split, 0))
            if count and len(group) < count:
                raise ValueError(f"Insufficient {split}/{label} examples: {len(group)} < {count}")
            if count:
                parts.append(group.sample(n=count, random_state=seed))
        if not parts:
            raise ValueError("No samples selected")
        cohort = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(f"Unknown cohort source: {source}")
    cohort = with_source_labels(cohort)
    cohort = cohort[["sample_id", "label", "source_label", "split", "relative_path", "absolute_path"]]
    cohort = cohort.sort_values(["split", "label", "sample_id"]).reset_index(drop=True)
    if cohort.empty or cohort["sample_id"].duplicated().any():
        raise ValueError("Cohort must contain at least one row and unique sample IDs")
    if set(cohort["split"]) != {"train", "val", "test"}:
        raise ValueError("Cohort must contain train, val, and test samples only")
    if source in {"stratified", "all"} and (cohort.groupby("split")["label"].nunique() < desc["label"].nunique()).any():
        raise ValueError("Every selected split must contain every class")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)
    write_json(output_path.with_suffix(".meta.json"), {
        "source": source, "seed": seed, "per_class": per_class,
        "cohort_hash": cohort_hash(cohort),
        "counts": {f"{s}/{label}": int(n) for (s, label), n in cohort.groupby(["split", "label"]).size().items()},
    })
    return cohort


def check_cohort_hash(cohort: pd.DataFrame, metadata_path: Path) -> None:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta["cohort_hash"] != cohort_hash(cohort):
        raise ValueError(f"Stale artifact: cohort hash differs in {metadata_path}")
