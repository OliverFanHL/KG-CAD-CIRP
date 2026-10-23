"""Exact-file leakage audit; near-duplicate family review remains a manual step."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .utils import write_json


def audit_exact_duplicates(cohort: pd.DataFrame, output_path: Path) -> dict:
    rows = []
    for item in cohort.to_dict("records"):
        digest = hashlib.sha256()
        with Path(item["absolute_path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"sample_id": item["sample_id"], "split": item["split"],
                     "label": item["label"], "sha256": digest.hexdigest()})
    table = pd.DataFrame(rows)
    repeats = []
    for digest, group in table.groupby("sha256"):
        if len(group) > 1:
            repeats.append({"sha256": digest, "samples": group.drop(columns="sha256").to_dict("records"),
                            "cross_split": group["split"].nunique() > 1,
                            "conflicting_labels": group["label"].nunique() > 1})
    result = {"n_samples": len(table), "n_duplicate_groups": len(repeats),
              "n_cross_split_groups": sum(x["cross_split"] for x in repeats),
              "n_conflicting_label_groups": sum(x["conflicting_labels"] for x in repeats),
              "groups": repeats}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result
