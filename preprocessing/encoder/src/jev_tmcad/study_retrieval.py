"""Judged-pair retrieval with validation-selected fusion and no class-label relevance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import TMCAD_CLASSES
from .provenance import class_label_mapping, source_manifest, with_source_labels
from .study_experiments import RETRIEVAL_METHODS, feature_blocks
from .utils import write_json


def _similarities(frame: pd.DataFrame, split: str) -> tuple[list[str], dict[str, np.ndarray]]:
    test = frame[frame["split"] == split].sort_values("sample_id")
    train = frame[frame["split"] == "train"]
    ids = test["sample_id"].astype(str).tolist()
    blocks = feature_blocks(frame)
    groups = {name: blocks[recipe[0]] for name, recipe in RETRIEVAL_METHODS.items()
              if len(recipe) == 1}
    matrices = {}
    for name, cols in groups.items():
        if not cols:
            continue
        prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
        prep.fit(train[cols])
        matrices[name] = cosine_similarity(prep.transform(test[cols]))
    return ids, matrices


def make_pair_pool(frame: pd.DataFrame, output_csv: Path, *, top_k: int, random_k: int,
                   seed: int) -> pd.DataFrame:
    frame = with_source_labels(frame)
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("val", "test"):
        ids, matrices = _similarities(frame, split)
        paths = frame[frame["split"] == split].set_index("sample_id")["relative_path"].to_dict()
        for i, qid in enumerate(ids):
            candidates = set()
            for scores in matrices.values():
                order = np.argsort(-scores[i])
                selected = [int(j) for j in order if j != i][:top_k]
                candidates.update(selected)
            remainder = [j for j in range(len(ids)) if j != i and j not in candidates]
            if remainder:
                candidates.update(int(j) for j in rng.choice(remainder, size=min(random_k, len(remainder)), replace=False))
            for j in sorted(candidates):
                rows.append({"split": split, "query_id": qid, "candidate_id": ids[j],
                             "query_relative_path": paths[qid],
                             "candidate_relative_path": paths[ids[j]],
                             "grade": ""})
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    # Keep the full source mapping separate from pair grades. The relative
    # paths above still reveal class folders; use an anonymized viewer for
    # genuinely blinded expert annotation.
    source_manifest(frame).to_csv(output_csv.with_suffix(".sources.csv"), index=False)
    return result


def _metrics(grades: np.ndarray, ranking: np.ndarray, ks: list[int]) -> dict[str, float]:
    ranked = grades[ranking]
    positive = ranked >= 2
    total_positive = int((grades >= 2).sum())
    out = {}
    for k in ks:
        top = ranked[:k]
        gains = 2.0 ** top - 1.0
        discounts = np.log2(np.arange(2, len(top) + 2))
        ideal = np.sort(grades)[::-1][:k]
        ideal_gain = (2.0 ** ideal - 1.0) / np.log2(np.arange(2, len(ideal) + 2))
        denom = float(ideal_gain.sum())
        out[f"ndcg@{k}"] = float((gains / discounts).sum() / denom) if denom else 0.0
        out[f"precision@{k}"] = float((top >= 2).sum() / min(k, len(grades)))
        out[f"recall@{k}"] = float((top >= 2).sum() / total_positive) if total_positive else 0.0
    positions = np.flatnonzero(positive)
    out["mrr"] = float(1.0 / (positions[0] + 1)) if len(positions) else 0.0
    if total_positive:
        out["map"] = float(np.mean([positive[:p + 1].mean() for p in positions]))
    else:
        out["map"] = 0.0
    return out


def _evaluate(split: str, pairs: pd.DataFrame, ids: list[str], matrix: np.ndarray,
              ks: list[int], *, query_scope: str = "all") -> tuple[dict, pd.DataFrame]:
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    if query_scope == "all" and set(pairs.loc[pairs["split"] == split, "query_id"].astype(str)) != set(ids):
        raise ValueError(f"Judgment file does not cover every {split} query")
    rows = []
    invalid_queries = []
    for qid, group in pairs[pairs["split"] == split].groupby("query_id"):
        if qid not in index:
            raise ValueError(f"Unknown retrieval query {qid}")
        candidates = group["candidate_id"].astype(str).tolist()
        if any(c not in index or c == qid for c in candidates):
            raise ValueError(f"Invalid candidate in query {qid}")
        grades = group["grade"].to_numpy(dtype=int)
        if len(grades) < 2 or not (grades >= 2).any() or not (grades < 2).any():
            invalid_queries.append(qid)
            continue
        scores = np.array([matrix[index[qid], index[c]] for c in candidates])
        rank = np.argsort(-scores, kind="stable")
        rows.append({"query_id": qid, "judged_candidates": len(grades), **_metrics(grades, rank, ks)})
    if invalid_queries:
        raise ValueError(f"{split} has {len(invalid_queries)} queries without both positive and negative judged candidates; "
                         "expand and annotate the pool before evaluation")
    if not rows:
        raise ValueError(f"No evaluable {split} queries with judged positive and negative candidates")
    table = pd.DataFrame(rows)
    metrics = {k: float(table[k].mean()) for k in table if k != "query_id"}
    metrics["n_queries"] = len(table)
    return metrics, table


def load_judgments(path: Path) -> pd.DataFrame:
    """Load expert grades without silently interpreting an empty pool as truth."""
    pairs = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str,
                                     "split": str, "grade": "string"})
    required = {"split", "query_id", "candidate_id", "grade"}
    if not required.issubset(pairs):
        raise ValueError(f"Judgments missing columns: {required - set(pairs)} in {path}")
    raw_grade = pairs["grade"].str.strip()
    missing = raw_grade.isna() | raw_grade.eq("")
    numeric = pd.to_numeric(raw_grade, errors="coerce")
    invalid = ~missing & ~numeric.isin([0, 1, 2, 3])
    if missing.any() or invalid.any():
        examples = (pairs.index[missing | invalid][:5] + 2).tolist()  # CSV header is line 1.
        raise ValueError(
            f"Retrieval judgments are incomplete in {path}: {int(missing.sum()):,} blank "
            f"and {int(invalid.sum()):,} invalid grades among {len(pairs):,} pairs "
            f"(example CSV lines: {examples}). The study-pool command creates an "
            "ungraded annotation template. Expert-assign grade 0, 1, 2, or 3 to "
            "every selected pair before retrieval evaluation; do not infer "
            "relevance from class folders or model scores."
        )
    pairs["grade"] = numeric.astype("int8")
    return pairs


def evaluate_retrieval(frame: pd.DataFrame, judgments_csv: Path, output_dir: Path,
                       *, ks: list[int], weights: list[float], only: str | None = None,
                       query_scope: str = "all") -> dict:
    if only is not None and only not in RETRIEVAL_METHODS:
        raise ValueError(f"Unknown retrieval experiment: {only}")
    frame = with_source_labels(frame)
    pairs = load_judgments(judgments_csv)
    if query_scope not in {"all", "preselected"}:
        raise ValueError(f"Unknown retrieval query scope: {query_scope}")
    if query_scope == "preselected":
        import json
        from .study_ground_truth import file_sha256
        meta_path = judgments_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["judgments_sha256"] != file_sha256(judgments_csv):
            raise ValueError("Finalized expert judgments changed after adjudication")
        for split in ("val", "test"):
            if set(pairs.loc[pairs["split"] == split, "query_id"].astype(str)) != set(meta["selected_query_ids"][split]):
                raise ValueError(f"Finalized {split} query IDs differ from annotation protocol")
    if pairs.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate judged query/candidate pair")
    if not pairs["split"].isin(["val", "test"]).all():
        raise ValueError("Retrieval judgments must use validation or test split")
    for split in ("val", "test"):
        ids = set(frame.loc[frame["split"] == split, "sample_id"].astype(str))
        selected = pairs[pairs["split"] == split]
        if not set(selected["query_id"]).issubset(ids) or not set(selected["candidate_id"]).issubset(ids):
            raise ValueError(f"Judgments for {split} contain IDs from another split")
    val_ids, val = _similarities(frame, "val")
    test_ids, test = _similarities(frame, "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest(frame).to_csv(output_dir / "source_manifest.csv", index=False)
    write_json(output_dir / "label_mapping.json", {
        "canonical_to_source": class_label_mapping(frame, TMCAD_CLASSES),
        "class_order": TMCAD_CLASSES,
    })
    identity = frame.set_index("sample_id")[["label", "source_label", "relative_path", "absolute_path"]]

    def save_query_table(table: pd.DataFrame, name: str) -> None:
        enriched = table.join(identity, on="query_id", validate="many_to_one")
        enriched = enriched.rename(columns={"label": "query_label",
                                            "source_label": "query_source_label",
                                            "relative_path": "query_relative_path",
                                            "absolute_path": "query_absolute_path"})
        enriched.to_csv(output_dir / f"{name}_per_query.csv", index=False)

    results = {}
    for name in val:
        if only is not None and name != only:
            continue
        metric, table = _evaluate("test", pairs, test_ids, test[name], ks, query_scope=query_scope)
        results[name] = metric
        save_query_table(table, name)
    for semantic in ("jev", "laya", "laya_ft", "clm"):
        if semantic not in val:
            continue
        if only is not None and only != f"geometry_plus_{semantic}":
            continue
        selected = None
        choices = []
        for weight in weights:
            matrix = (1 - weight) * val["geometry"] + weight * val[semantic]
            metric, _ = _evaluate("val", pairs, val_ids, matrix, ks, query_scope=query_scope)
            score = metric[f"ndcg@{max(ks)}"]
            choices.append({"weight": weight, "validation_ndcg": score})
            if selected is None or score > selected[0]:
                selected = (score, weight)
        assert selected is not None
        name = f"geometry_plus_{semantic}"
        matrix = (1 - selected[1]) * test["geometry"] + selected[1] * test[semantic]
        metric, table = _evaluate("test", pairs, test_ids, matrix, ks, query_scope=query_scope)
        metric["selected_semantic_weight"] = selected[1]
        metric["validation_selection"] = choices
        results[name] = metric
        save_query_table(table, name)
    if only is not None and only not in results:
        raise ValueError(f"Required features are unavailable for retrieval/{only}")
    write_json(output_dir / "summary.json", {"relevance": "judged engineering grades; 2-3 relevant",
                                              "query_scope": query_scope,
                                              "candidate_scope": "judged within-split pairs only",
                                              "experiment": only,
                                              "methods": results})
    return results
