from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ensure_parent, sha1_text

LOGGER = logging.getLogger(__name__)

TMCAD_CLASSES = [
    "bearing",
    "bolt-screw",
    "bracket",
    "coupling",
    "flange",
    "gear",
    "nut",
    "pulley",
    "shaft",
    "spring",
]

_FOLDER_ALIASES = {
    "bearing": "bearing",
    "bearings": "bearing",
    "bolt-screw": "bolt-screw",
    "bolt_screw": "bolt-screw",
    "bolt screw": "bolt-screw",
    "boltscrew": "bolt-screw",
    "bolt": "bolt-screw",
    "bolts": "bolt-screw",
    "screw": "bolt-screw",
    "screws": "bolt-screw",
    "bracket": "bracket",
    "brackets": "bracket",
    "coupling": "coupling",
    "couplings": "coupling",
    "flange": "flange",
    "flanges": "flange",
    "gear": "gear",
    "gears": "gear",
    "nut": "nut",
    "nuts": "nut",
    "pulley": "pulley",
    "pulleys": "pulley",
    "shaft": "shaft",
    "shafts": "shaft",
    "spring": "spring",
    "springs": "spring",
}


def canonical_label(folder_name: str) -> str | None:
    return _FOLDER_ALIASES.get(folder_name.strip().lower())


def find_dataset_root(root: Path) -> Path:
    """Accept either TMCAD root or a parent containing TMCAD_v2."""
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    direct = {canonical_label(p.name) for p in root.iterdir() if p.is_dir()}
    if len({x for x in direct if x}) >= 5:
        return root

    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in {"tmcad_v2", "tmcad-v2", "tmcad"}]
    for cand in candidates:
        labels = {canonical_label(p.name) for p in cand.iterdir() if p.is_dir()}
        if len({x for x in labels if x}) >= 5:
            return cand.resolve()

    raise RuntimeError(
        f"Could not identify a TMCAD class-folder structure under {root}. "
        "Expected folders such as bearing, bolt-screw, bracket, ..., spring."
    )


def scan_tmcad(root: str | Path) -> pd.DataFrame:
    dataset_root = find_dataset_root(Path(root))
    rows: list[dict[str, str]] = []
    for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        label = canonical_label(class_dir.name)
        if label is None:
            continue
        for step_path in sorted(class_dir.rglob("*")):
            if not step_path.is_file() or step_path.suffix.lower() not in {".stp", ".step"}:
                continue
            rel = step_path.relative_to(dataset_root).as_posix()
            rows.append(
                {
                    "sample_id": sha1_text(rel, 20),
                    "label": label,
                    "source_label": class_dir.name,
                    "relative_path": rel,
                    "absolute_path": str(step_path.resolve()),
                }
            )
    if not rows:
        raise RuntimeError(f"No .stp/.step files found under {dataset_root}")

    df = pd.DataFrame(rows).sort_values(["label", "relative_path"]).reset_index(drop=True)
    duplicates = df["sample_id"].duplicated().sum()
    if duplicates:
        raise RuntimeError(f"Unexpected sample-id collisions: {duplicates}")
    LOGGER.info("Scanned %d STEP models from %s", len(df), dataset_root)
    return df


def add_stratified_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("train + val + test fractions must sum to 1")

    work = df.copy()
    idx_train, idx_temp = train_test_split(
        work.index,
        test_size=(1.0 - train_fraction),
        random_state=seed,
        stratify=work["label"],
    )
    temp = work.loc[idx_temp]
    val_share_of_temp = val_fraction / (val_fraction + test_fraction)
    idx_val, idx_test = train_test_split(
        temp.index,
        train_size=val_share_of_temp,
        random_state=seed + 1,
        stratify=temp["label"],
    )
    work["split"] = ""
    work.loc[idx_train, "split"] = "train"
    work.loc[idx_val, "split"] = "val"
    work.loc[idx_test, "split"] = "test"
    return work


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    p = ensure_parent(path)
    df.to_csv(p, index=False)


def select_subset(
    df: pd.DataFrame,
    splits: Iterable[str] | None = None,
    max_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    if splits:
        wanted = set(splits)
        work = work[work["split"].isin(wanted)]
    if max_per_class is not None:
        parts = []
        for (split, label), grp in work.groupby(["split", "label"], sort=True):
            n = min(max_per_class, len(grp))
            parts.append(grp.sample(n=n, random_state=seed))
        work = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    return work.sort_values(["split", "label", "sample_id"]).reset_index(drop=True)
