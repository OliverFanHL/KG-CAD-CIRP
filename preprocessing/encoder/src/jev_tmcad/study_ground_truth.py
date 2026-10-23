"""Blinded, reproducible expert grading of CAD retrieval pairs.

Folder labels and model similarities never become relevance ground truth.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .provenance import with_source_labels
from .utils import sha256_json, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(split: str, query_id: str, candidate_id: str) -> str:
    return sha256_json([str(split), str(query_id), str(candidate_id)])[:24]


def _grade_column(frame: pd.DataFrame, path: Path) -> pd.Series:
    if "grade" not in frame:
        raise ValueError(f"Missing grade column: {path}")
    raw = frame["grade"].astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = raw.isna() | raw.eq("") | ~numeric.isin([0, 1, 2, 3])
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} missing or invalid expert grades; "
                         "only integer values 0, 1, 2, 3 are accepted")
    return numeric.astype("int8")


def prepare_annotation_files(cohort: pd.DataFrame, full_pool: Path, directory: Path, *,
                             per_class: dict[str, int], seed: int,
                             copy_assets: bool = False) -> dict:
    """Freeze query selection before rating; all methods share these pairs."""
    cohort = with_source_labels(cohort)
    targets = [directory / name for name in ("selected_pairs.csv", "rater_a.csv",
                                                 "rater_b.csv", "adjudication.csv", "protocol.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Annotation package already exists in {directory}; preserve ratings")
    pool = pd.read_csv(full_pool, dtype={"split": str, "query_id": str,
                                        "candidate_id": str, "grade": "string"})
    required = {"split", "query_id", "candidate_id", "grade"}
    if not required.issubset(pool) or pool.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Source pair pool has missing columns or duplicate pairs")
    if pool["grade"].notna().any():
        raise ValueError("Source pool already has grades; preserve it and use a fresh ungraded pool")
    rng = np.random.default_rng(seed)
    selected_ids: dict[str, list[str]] = {}
    cohort_index = cohort.set_index("sample_id")
    for split in ("val", "test"):
        n_per_class = int(per_class[split])
        if n_per_class < 1:
            raise ValueError("At least one query per class and split is required")
        ids = []
        for label in sorted(cohort["label"].unique()):
            choices = sorted(cohort.loc[(cohort["split"] == split) &
                                        (cohort["label"] == label), "sample_id"].astype(str))
            if len(choices) < n_per_class:
                raise ValueError(f"{split}/{label} has only {len(choices)} queries")
            ids.extend(str(value) for value in rng.choice(choices, n_per_class, replace=False))
        selected_ids[split] = sorted(ids)
    selected = pool[((pool["split"] == "val") & pool["query_id"].isin(selected_ids["val"])) |
                    ((pool["split"] == "test") & pool["query_id"].isin(selected_ids["test"]))].copy()
    for split, ids in selected_ids.items():
        if set(selected.loc[selected["split"] == split, "query_id"].astype(str)) != set(ids):
            raise ValueError(f"Pair pool does not cover every preselected {split} query")
    if selected.empty:
        raise ValueError("No annotation pairs selected")
    for column in ("query_id", "candidate_id"):
        info = cohort_index.loc[selected[column].astype(str)]
        if (info["split"].to_numpy() != selected["split"].to_numpy()).any():
            raise ValueError("Source pair pool contains a cross-split candidate")
    selected.insert(0, "pair_id", [pair_key(s, q, c) for s, q, c in
                                   zip(selected["split"], selected["query_id"], selected["candidate_id"])])
    if selected["pair_id"].duplicated().any():
        raise ValueError("Pair IDs are not unique")
    directory.mkdir(parents=True, exist_ok=True)
    selected.to_csv(targets[0], index=False)
    # Annotators receive opaque CAD filenames and pair IDs. Original paths and
    # class labels stay in the private selected_pairs/source manifests.
    public = pd.DataFrame({
        "pair_id": selected["pair_id"],
        "query_asset": selected["query_id"].astype(str) + ".stp",
        "candidate_asset": selected["candidate_id"].astype(str) + ".stp",
        "grade": "", "evidence": "",
    })
    for target in targets[1:4]:
        public.to_csv(target, index=False)
    asset_ids = sorted(set(selected["query_id"].astype(str)) |
                       set(selected["candidate_id"].astype(str)))
    asset_manifest = cohort_index.loc[asset_ids, ["split", "label", "source_label",
                                                 "relative_path", "absolute_path"]].reset_index()
    asset_manifest.to_csv(directory / "private_asset_manifest.csv", index=False)
    protocol = {
        "rubric": {"0": "Unrelated form and mechanical role",
                   "1": "Broad form or role overlap only",
                   "2": "Similar local geometry, interfaces, or manufacturing role",
                   "3": "Close functional and geometric substitute"},
        "selection": "fixed-seed, class-stratified query sample; common candidate pool for every method",
        "seed": seed, "per_class": per_class, "selected_query_ids": selected_ids,
        "n_pairs": len(selected), "n_assets": len(asset_ids),
        "source_pool": str(full_pool), "source_pool_sha256": file_sha256(full_pool),
        "selected_pairs_sha256": file_sha256(targets[0]),
        "annotation": "two independent blinded raters; adjudicate every disagreement",
    }
    write_json(targets[4], protocol)
    if copy_assets:
        materialize_blinded_assets(directory)
    return protocol


def materialize_blinded_assets(directory: Path) -> dict:
    """Copy opaque CAD files from the frozen private mapping without exposing labels."""
    import json
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if file_sha256(directory / "selected_pairs.csv") != protocol["selected_pairs_sha256"]:
        raise ValueError("Selected pair list changed after annotation preparation")
    manifest = pd.read_csv(directory / "private_asset_manifest.csv", dtype={"sample_id": str})
    if manifest["sample_id"].duplicated().any() or len(manifest) != protocol["n_assets"]:
        raise ValueError("Private asset mapping differs from the frozen protocol")
    target_dir = directory / "blinded_cad"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in manifest.iterrows():
        source = Path(row["absolute_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = target_dir / f"{row['sample_id']}.stp"
        if not target.exists():
            shutil.copy2(source, target)
            copied += 1
    return {"assets": len(manifest), "copied": copied, "directory": str(target_dir)}


def finalize_annotation_files(directory: Path, output_csv: Path) -> dict:
    """Create the immutable evaluation truth only after independent ratings."""
    selected_path = directory / "selected_pairs.csv"
    protocol_path = directory / "protocol.json"
    import json
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if file_sha256(selected_path) != protocol["selected_pairs_sha256"]:
        raise ValueError("Selected pair identities changed after annotation preparation")
    if file_sha256(Path(protocol["source_pool"])) != protocol["source_pool_sha256"]:
        raise ValueError("Source candidate pool changed after query selection")
    selected = pd.read_csv(selected_path, dtype={"pair_id": str, "split": str,
                                                 "query_id": str, "candidate_id": str})
    ratings = {}
    for name in ("rater_a", "rater_b"):
        path = directory / f"{name}.csv"
        frame = pd.read_csv(path, dtype={"pair_id": str, "grade": "string"})
        if frame["pair_id"].duplicated().any() or set(frame["pair_id"]) != set(selected["pair_id"]):
            raise ValueError(f"{name} pair IDs differ from the frozen pool")
        frame = frame.set_index("pair_id").loc[selected["pair_id"]]
        ratings[name] = _grade_column(frame, path).to_numpy(dtype=int)
    a, b = ratings["rater_a"], ratings["rater_b"]
    disagreement = a != b
    path = directory / "adjudication.csv"
    adjudication = pd.read_csv(path, dtype={"pair_id": str, "grade": "string"})
    if adjudication["pair_id"].duplicated().any() or set(adjudication["pair_id"]) != set(selected["pair_id"]):
        raise ValueError("Adjudication pair IDs differ from the frozen pool")
    adjudication = adjudication.set_index("pair_id").loc[selected["pair_id"]]
    final = a.copy()
    if disagreement.any():
        final[disagreement] = _grade_column(adjudication.iloc[np.flatnonzero(disagreement)], path).to_numpy(dtype=int)
    if output_csv.exists():
        raise FileExistsError(f"Finalized judgments already exist: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = selected.drop(columns=["grade"]).copy()
    result["grade"] = final
    invalid = [str(qid) for _, part in result.groupby("split") for qid, group in
               part.groupby("query_id") if len(group) < 2 or
               not group["grade"].ge(2).any() or not group["grade"].lt(2).any()]
    if invalid:
        raise ValueError(f"{len(invalid)} queries lack both relevant (2-3) and nonrelevant (0-1) "
                         f"candidates, including {invalid[:5]}; expand the frozen pool and obtain expert ratings")
    result.to_csv(output_csv, index=False)
    provenance = selected[["pair_id", "split", "query_id", "candidate_id"]].copy()
    provenance["rater_a_grade"] = a
    provenance["rater_b_grade"] = b
    provenance["adjudicated"] = disagreement
    provenance["final_grade"] = final
    provenance.to_csv(output_csv.with_suffix(".provenance.csv"), index=False)
    kappa = float(cohen_kappa_score(a, b, weights="quadratic"))
    summary = {"label_source": "two independent expert ratings with adjudication of disagreements",
               "rubric": protocol["rubric"], "selection_protocol": str(protocol_path),
               "selected_pairs_sha256": protocol["selected_pairs_sha256"],
               "judgments_sha256": file_sha256(output_csv),
               "selected_query_ids": protocol["selected_query_ids"],
               "n_pairs": len(result), "n_disagreements": int(disagreement.sum()),
               "exact_agreement": float((a == b).mean()),
               "quadratic_weighted_kappa": kappa if np.isfinite(kappa) else None}
    write_json(output_csv.with_suffix(".meta.json"), summary)
    return summary
