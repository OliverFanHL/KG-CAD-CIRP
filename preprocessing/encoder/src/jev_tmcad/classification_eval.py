from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cad_features import numeric_feature_columns
from .dataset import TMCAD_CLASSES
from .provenance import class_label_mapping, source_manifest, with_source_labels
from .reporting import prediction_frame, save_confusion
from .utils import ensure_dir, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


def _latest_jev_df(path: str | Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "experiment_hash" in df.columns and df["experiment_hash"].notna().any():
        latest_hash = str(df.loc[df["experiment_hash"].notna(), "experiment_hash"].iloc[-1])
        df = df[df["experiment_hash"].astype(str) == latest_hash].copy()
        models = sorted(df.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if len(models) > 1:
            LOGGER.warning("Latest experiment contains multiple resolved Jev model versions: %s", models)
    return df.drop_duplicates("sample_id", keep="last").reset_index(drop=True)


def _signature_frame(jev: pd.DataFrame) -> pd.DataFrame:
    if jev.empty:
        return pd.DataFrame()
    sig_rows = []
    for _, r in jev.iterrows():
        sig = r.get("semantic_signature", {})
        if isinstance(sig, str):
            sig = json.loads(sig)
        base = {"sample_id": str(r["sample_id"]), "split": r.get("split"), "true_label": r.get("true_label")}
        base.update({f"sig__{k}": float(v) for k, v in sig.items()})
        sig_rows.append(base)
    return pd.DataFrame(sig_rows)


def _multiclass_probability_matrix(jev: pd.DataFrame, classes: list[str]) -> np.ndarray:
    rows = []
    for _, r in jev.iterrows():
        probs = r.get("class_probabilities", {})
        if isinstance(probs, str):
            probs = json.loads(probs)
        arr = np.array([float(probs.get(c, 0.0)) for c in classes], dtype=float)
        total = arr.sum()
        arr = arr / total if total > 0 else np.full(len(classes), 1.0 / len(classes))
        rows.append(arr)
    return np.vstack(rows) if rows else np.empty((0, len(classes)))


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == y_pred[mask])
        conf = np.mean(confidence[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _fit_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    numeric_cols = list(X_train.columns)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=4000, solver="lbfgs"),
            ),
        ]
    )
    pipe.fit(X_train[numeric_cols], y_train)
    return pipe


def evaluate_classification(
    descriptor_csv: str | Path,
    jev_jsonl: str | Path,
    output_dir: str | Path,
    test_split: str = "test",
    ece_bins: int = 10,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    figures = ensure_dir(out / "figures")
    predictions_dir = ensure_dir(out / "predictions")

    desc = pd.read_csv(descriptor_csv, low_memory=False)
    desc = desc[desc["extract_error"].fillna("").astype(str).str.len() == 0].copy()
    desc = with_source_labels(desc)
    source_manifest(desc).to_csv(out / "source_manifest.csv", index=False)
    mapping = class_label_mapping(desc, TMCAD_CLASSES)
    write_json(out / "label_mapping.json", {"canonical_to_source": mapping,
                                            "class_order": TMCAD_CLASSES,
                                            "source": "original class-folder component of relative_path"})
    jev = _latest_jev_df(jev_jsonl)
    summary: dict[str, Any] = {}

    # 1) Jev zero-shot Choice evaluation on whichever test examples were actually queried.
    if not jev.empty:
        jt = jev[jev["split"] == test_split].copy()
        if not jt.empty:
            jt = jt.merge(desc[["sample_id", "label", "source_label", "relative_path", "absolute_path"]],
                          on="sample_id", validate="one_to_one")
            if jt["true_label"].astype(str).ne(jt["label"].astype(str)).any():
                raise ValueError("JEV true labels differ from original descriptor labels")
            y_true = jt["true_label"].astype(str).to_numpy()
            y_pred = jt["predicted_class"].astype(str).to_numpy()
            probs = _multiclass_probability_matrix(jt, TMCAD_CLASSES)
            conf = probs.max(axis=1)
            class_to_idx = {c: i for i, c in enumerate(TMCAD_CLASSES)}
            y_onehot = np.zeros_like(probs)
            for i, y in enumerate(y_true):
                if y in class_to_idx:
                    y_onehot[i, class_to_idx[y]] = 1.0
            m = _metrics(y_true, y_pred)
            m.update(
                {
                    "n": int(len(jt)),
                    "negative_log_likelihood": float(log_loss(y_true, probs, labels=TMCAD_CLASSES)),
                    "multiclass_brier": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
                    "ece": expected_calibration_error(y_true, y_pred, conf, bins=ece_bins),
                }
            )
            summary["jev_zero_shot"] = m
            pred_out = prediction_frame(jt, y_pred, probs, TMCAD_CLASSES, mapping,
                                        true_column="true_label", prediction_column="predicted_class")
            pred_out["class_confidence"] = jt["class_confidence"].to_numpy()
            pred_out["max_probability"] = conf
            pred_out.to_csv(predictions_dir / "jev_zero_shot.csv", index=False)
            write_json(out / "jev_classification_report.json", classification_report(y_true, y_pred, output_dict=True, zero_division=0))
            save_confusion(y_true, y_pred, TMCAD_CLASSES, mapping,
                           "Jev zero-shot classification", figures / "confusion_jev.png")

    # 2) Deterministic-descriptor logistic regression baseline. Train on full train split,
    #    but compare on the exact Jev test subset when available so the comparison is fair.
    feature_cols = numeric_feature_columns(list(desc.columns))
    # Keep only actual numeric columns after CSV loading.
    feature_cols = [c for c in feature_cols if c in desc and pd.api.types.is_numeric_dtype(desc[c])]
    train = desc[desc["split"] == "train"].copy()
    test = desc[desc["split"] == test_split].copy()
    if not jev.empty:
        jev_test_ids = set(jev.loc[jev["split"] == test_split, "sample_id"].astype(str))
        if jev_test_ids:
            test = test[test["sample_id"].astype(str).isin(jev_test_ids)]
    if not train.empty and not test.empty and feature_cols:
        pipe = _fit_logreg(train[feature_cols], train["label"])
        pred = pipe.predict(test[feature_cols])
        summary["descriptor_logreg"] = {"n": int(len(test)), **_metrics(test["label"].to_numpy(), pred)}
        prediction_frame(test, pred, None, TMCAD_CLASSES, mapping,
                         true_column="true_label", prediction_column="predicted_class").to_csv(
            predictions_dir / "descriptor_logreg.csv", index=False)
        save_confusion(
            test["label"].to_numpy(), pred, TMCAD_CLASSES, mapping,
            "Deterministic descriptor + logistic regression", figures / "confusion_descriptor_logreg.png"
        )

    # 3) Logistic regression on Jev semantic signatures (requires train + test API outputs).
    #    We report a geometry-only signature separately from the full signature so
    #    taxonomy-aligned functional questions do not get hidden inside one score.
    sig = _signature_frame(jev)
    if not sig.empty:
        sig = sig.merge(desc[["sample_id", "split", "label", "source_label", "relative_path", "absolute_path"]],
                        on=["sample_id", "split"], validate="one_to_one")
        if sig["true_label"].astype(str).ne(sig["label"].astype(str)).any():
            raise ValueError("JEV signature labels differ from original descriptor labels")
    sig_cols = [c for c in sig.columns if c.startswith("sig__")]
    geom_sig_cols = [c for c in sig_cols if c.startswith("sig__geom__")]
    signature_sets = {
        "geometry_signature_logreg": geom_sig_cols,
        "full_signature_logreg": sig_cols,
    }
    for method_name, cols in signature_sets.items():
        if not cols:
            continue
        sig_train = sig[sig["split"] == "train"].copy()
        sig_test = sig[sig["split"] == test_split].copy()
        if sig_train.empty or sig_test.empty:
            continue
        pipe = _fit_logreg(sig_train[cols], sig_train["true_label"])
        pred = pipe.predict(sig_test[cols])
        summary[method_name] = {"n": int(len(sig_test)), **_metrics(sig_test["true_label"].to_numpy(), pred)}
        prediction_frame(sig_test, pred, None, TMCAD_CLASSES, mapping,
                         true_column="true_label", prediction_column="predicted_class").to_csv(
            predictions_dir / f"{method_name}.csv", index=False)
        save_confusion(
            sig_test["true_label"].to_numpy(),
            pred,
            TMCAD_CLASSES,
            mapping,
            method_name.replace("_", " "),
            figures / f"confusion_{method_name}.png",
        )

    # 4) Hybrid deterministic descriptor + full semantic signature classifier.
    if sig_cols:
        merged = desc.merge(sig[["sample_id", "split"] + sig_cols], on=["sample_id", "split"], how="inner")
        h_train = merged[merged["split"] == "train"].copy()
        h_test = merged[merged["split"] == test_split].copy()
        hybrid_cols = feature_cols + sig_cols
        if not h_train.empty and not h_test.empty and hybrid_cols:
            pipe = _fit_logreg(h_train[hybrid_cols], h_train["label"])
            pred = pipe.predict(h_test[hybrid_cols])
            summary["hybrid_logreg"] = {"n": int(len(h_test)), **_metrics(h_test["label"].to_numpy(), pred)}
            prediction_frame(h_test, pred, None, TMCAD_CLASSES, mapping,
                             true_column="true_label", prediction_column="predicted_class").to_csv(
                predictions_dir / "hybrid_logreg.csv", index=False)
            save_confusion(
                h_test["label"].to_numpy(), pred, TMCAD_CLASSES, mapping,
                "Descriptor + full Jev signature logistic regression", figures / "confusion_hybrid_logreg.png"
            )

    write_json(out / "classification_summary.json", summary)
    if summary:
        pd.DataFrame(summary).T.reset_index(names="method").to_csv(out / "classification_summary.csv", index=False)
    LOGGER.info("Classification evaluation written to %s", out)
    return summary
