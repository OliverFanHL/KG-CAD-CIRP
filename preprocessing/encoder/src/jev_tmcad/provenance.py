"""Keep source CAD identity and class names alongside normalized study labels."""
from __future__ import annotations

import pandas as pd

from .dataset import canonical_label


def with_source_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover the original class-folder name without changing existing labels."""
    required = {"sample_id", "label", "relative_path"}
    if not required.issubset(frame):
        raise ValueError(f"Source metadata missing columns: {sorted(required - set(frame))}")
    result = frame.copy()
    folders = result["relative_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    if folders.eq("").any() or folders.eq("nan").any():
        raise ValueError("Cannot recover original labels from empty relative paths")
    if "source_label" in result:
        saved = result["source_label"].fillna("").astype(str)
        if saved.ne("").any() and saved[saved.ne("")].ne(folders[saved.ne("")]).any():
            raise ValueError("Stored source_label disagrees with the original class folder")
    result["source_label"] = folders
    derived = result["source_label"].map(canonical_label)
    if derived.isna().any() or derived.ne(result["label"].astype(str)).any():
        raise ValueError("Original class folder and normalized label disagree")
    return result


def class_label_mapping(frame: pd.DataFrame, classes: list[str]) -> dict[str, list[str]]:
    source = with_source_labels(frame)
    mapping = {label: sorted(source.loc[source["label"] == label, "source_label"].unique().tolist())
               for label in classes}
    if any(not names for names in mapping.values()):
        raise ValueError("At least one configured class lacks source samples")
    return mapping


def display_label(label: str, mapping: dict[str, list[str]]) -> str:
    if label not in mapping:
        raise ValueError(f"Unknown normalized class label: {label}")
    return " / ".join(mapping[label])


def source_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    source = with_source_labels(frame)
    columns = ["sample_id", "split", "label", "source_label", "relative_path", "absolute_path"]
    if not set(columns).issubset(source):
        raise ValueError("Source manifest requires original CAD paths and split metadata")
    result = source[columns].copy()
    if result["sample_id"].duplicated().any():
        raise ValueError("Source manifest requires unique sample IDs")
    return result.sort_values("sample_id").reset_index(drop=True)
