from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cad_features import numeric_feature_columns
from .dataset import TMCAD_CLASSES
from .provenance import class_label_mapping, source_manifest, with_source_labels
from .utils import ensure_dir, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


def _signature_df(jev_path: str | Path) -> pd.DataFrame:
    rows = read_jsonl(jev_path)
    if not rows:
        return pd.DataFrame()
    latest = pd.DataFrame(rows)
    if "experiment_hash" in latest.columns and latest["experiment_hash"].notna().any():
        latest_hash = str(latest.loc[latest["experiment_hash"].notna(), "experiment_hash"].iloc[-1])
        latest = latest[latest["experiment_hash"].astype(str) == latest_hash].copy()
        models = sorted(latest.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if len(models) > 1:
            LOGGER.warning("Latest experiment contains multiple resolved Jev model versions: %s", models)
    latest = latest.drop_duplicates("sample_id", keep="last")
    out = []
    for _, r in latest.iterrows():
        sig = r.get("semantic_signature", {})
        if isinstance(sig, str):
            sig = json.loads(sig)
        row = {"sample_id": str(r["sample_id"]), "split": r.get("split"), "label": r.get("true_label")}
        row.update({f"sig__{k}": float(v) for k, v in sig.items()})
        out.append(row)
    return pd.DataFrame(out)


def _fit_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Xtr = pipe.fit_transform(train[cols])
    Xte = pipe.transform(test[cols])
    return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)


def _metrics_for_ranking(labels: np.ndarray, q_idx: int, ranking: np.ndarray, ks: list[int]) -> dict[str, float]:
    q_label = labels[q_idx]
    relevant = (labels == q_label)
    relevant[q_idx] = False
    total_rel = int(relevant.sum())
    ranked_rel = relevant[ranking].astype(int)

    out: dict[str, float] = {}
    for k in ks:
        top = ranked_rel[:k]
        out[f"precision@{k}"] = float(top.mean()) if len(top) else 0.0
        out[f"recall@{k}"] = float(top.sum() / total_rel) if total_rel > 0 else 0.0
        dcg = float(np.sum(top / np.log2(np.arange(2, len(top) + 2)))) if len(top) else 0.0
        ideal_n = min(k, total_rel)
        idcg = float(np.sum(np.ones(ideal_n) / np.log2(np.arange(2, ideal_n + 2)))) if ideal_n > 0 else 0.0
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    hit_positions = np.flatnonzero(ranked_rel)
    out["mrr"] = float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0
    if total_rel > 0:
        precisions = []
        hits = 0
        for rank, rel in enumerate(ranked_rel, start=1):
            if rel:
                hits += 1
                precisions.append(hits / rank)
        out["ap"] = float(sum(precisions) / total_rel)
    else:
        out["ap"] = 0.0
    return out


def _evaluate_similarity(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    sim: np.ndarray,
    method: str,
    ks: list[int],
    top_k_save: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows = []
    top_rows = []
    n = len(sample_ids)
    for i in range(n):
        scores = sim[i].copy()
        scores[i] = -np.inf
        ranking = np.argsort(-scores)
        ranking = ranking[ranking != i]
        metrics = _metrics_for_ranking(labels, i, ranking, ks)
        query_rows.append({"method": method, "query_id": sample_ids[i], "query_label": labels[i], **metrics})
        for rank, j in enumerate(ranking[:top_k_save], start=1):
            top_rows.append(
                {
                    "method": method,
                    "query_id": sample_ids[i],
                    "query_label": labels[i],
                    "rank": rank,
                    "candidate_id": sample_ids[j],
                    "candidate_label": labels[j],
                    "score": float(scores[j]),
                    "relevant": int(labels[j] == labels[i]),
                }
            )
    return pd.DataFrame(query_rows), pd.DataFrame(top_rows)


def evaluate_retrieval(
    descriptor_csv: str | Path,
    jev_jsonl: str | Path,
    output_dir: str | Path,
    ks: list[int] | None = None,
    semantic_weight: float = 0.5,
    eval_split: str = "test",
) -> dict[str, Any]:
    ks = ks or [1, 5, 10]
    out = ensure_dir(output_dir)
    desc = pd.read_csv(descriptor_csv, low_memory=False)
    desc = desc[desc["extract_error"].fillna("").astype(str).str.len() == 0].copy()
    desc = with_source_labels(desc)
    source_manifest(desc).to_csv(out / "source_manifest.csv", index=False)
    write_json(out / "label_mapping.json", {"canonical_to_source": class_label_mapping(desc, TMCAD_CLASSES),
                                            "class_order": TMCAD_CLASSES})
    sig = _signature_df(jev_jsonl)
    if sig.empty:
        raise RuntimeError("No Jev results found; retrieval evaluation needs semantic signatures.")
    label_check = sig[["sample_id", "split", "label"]].merge(
        desc[["sample_id", "split", "label"]], on=["sample_id", "split"],
        suffixes=("_jev", "_source"), validate="one_to_one")
    if len(label_check) != len(sig) or label_check["label_jev"].astype(str).ne(label_check["label_source"].astype(str)).any():
        raise ValueError("JEV retrieval labels differ from original descriptor labels")

    # Restrict retrieval to the same API-evaluated examples for fair method comparisons.
    merged = desc.merge(sig.drop(columns=["label"], errors="ignore"), on=["sample_id", "split"], how="inner")
    descriptor_train = desc[desc["split"] == "train"].copy()
    signature_train = sig[sig["split"] == "train"].copy()
    test = merged[merged["split"] == eval_split].copy().sort_values("sample_id").reset_index(drop=True)
    if len(test) < 2:
        raise RuntimeError("Need at least two evaluated test samples for retrieval.")

    sample_ids = test["sample_id"].astype(str).to_numpy()
    labels = test["label"].astype(str).to_numpy()
    all_query_results = []
    all_top = []
    similarity_matrices: dict[str, np.ndarray] = {}

    numeric_cols = numeric_feature_columns(list(desc.columns))
    numeric_cols = [c for c in numeric_cols if c in merged and pd.api.types.is_numeric_dtype(merged[c])]
    if numeric_cols:
        fit_ref = descriptor_train if not descriptor_train.empty else test
        prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        prep.fit(fit_ref[numeric_cols])
        X = prep.transform(test[numeric_cols])
        sim_desc = cosine_similarity(X)
        similarity_matrices["deterministic_descriptors"] = sim_desc

    sig_cols = [c for c in merged.columns if c.startswith("sig__")]
    geom_sig_cols = [c for c in sig_cols if c.startswith("sig__geom__")]
    for method_name, cols in {
        "jev_geometry_signature": geom_sig_cols,
        "jev_full_signature": sig_cols,
    }.items():
        if not cols:
            continue
        fit_ref = signature_train if (not signature_train.empty and all(c in signature_train.columns for c in cols)) else test
        if fit_ref is test:
            LOGGER.warning(
                "No Jev training signatures are available; semantic-signature scaling is being fit on the evaluation subset. "
                "For a final benchmark, run Jev on the train split as well."
            )
        prep_sig = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        prep_sig.fit(fit_ref[cols])
        S = prep_sig.transform(test[cols])
        similarity_matrices[method_name] = cosine_similarity(S)

    if "deterministic_descriptors" in similarity_matrices and "jev_full_signature" in similarity_matrices:
        w = float(np.clip(semantic_weight, 0.0, 1.0))
        similarity_matrices["hybrid_score_fusion"] = (
            (1.0 - w) * similarity_matrices["deterministic_descriptors"]
            + w * similarity_matrices["jev_full_signature"]
        )

    for method, sim in similarity_matrices.items():
        query_df, top_df = _evaluate_similarity(sample_ids, labels, sim, method, ks, top_k_save=max(max(ks), 10))
        all_query_results.append(query_df)
        all_top.append(top_df)

    query_results = pd.concat(all_query_results, ignore_index=True)
    top_results = pd.concat(all_top, ignore_index=True)
    lookup = desc.set_index("sample_id")[["source_label", "relative_path", "absolute_path"]]
    query_lookup = lookup.rename(columns={c: f"query_{c}" for c in lookup})
    candidate_lookup = lookup.rename(columns={c: f"candidate_{c}" for c in lookup})
    query_results.join(query_lookup, on="query_id", validate="many_to_one").to_csv(
        out / "retrieval_per_query.csv", index=False)
    top_results.join(query_lookup, on="query_id", validate="many_to_one").join(
        candidate_lookup, on="candidate_id", validate="many_to_one").to_csv(
        out / "retrieval_topk.csv", index=False)

    metric_cols = [c for c in query_results.columns if c not in {"method", "query_id", "query_label"}]
    summary_df = query_results.groupby("method", as_index=False)[metric_cols].mean()
    summary_df.insert(1, "n_queries", query_results.groupby("method").size().values)
    summary_df.to_csv(out / "retrieval_summary.csv", index=False)

    per_class = query_results.groupby(["method", "query_label"], as_index=False)[metric_cols].mean()
    per_class.to_csv(out / "retrieval_per_class.csv", index=False)

    summary = {row["method"]: {k: float(row[k]) if k != "n_queries" else int(row[k]) for k in row.index if k != "method"} for _, row in summary_df.iterrows()}
    write_json(out / "retrieval_summary.json", summary)
    LOGGER.info("Retrieval evaluation written to %s", out)
    return summary
