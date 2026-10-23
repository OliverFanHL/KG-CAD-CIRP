"""Traceable classification tables and confusion matrices."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .provenance import display_label, with_source_labels
from .utils import write_json


def prediction_frame(samples: pd.DataFrame, predictions, probabilities: np.ndarray | None,
                     classes: list[str], mapping: dict[str, list[str]], *,
                     true_column: str = "label", prediction_column: str = "prediction") -> pd.DataFrame:
    source = with_source_labels(samples)
    required = {"sample_id", "split", "absolute_path", "relative_path"}
    if not required.issubset(source):
        raise ValueError(f"Prediction provenance missing: {sorted(required - set(source))}")
    predicted = np.asarray(predictions, dtype=str)
    if len(predicted) != len(source) or not set(predicted).issubset(classes):
        raise ValueError("Predictions do not align with source samples or configured classes")
    result = source[["sample_id", "split", "relative_path", "absolute_path", "source_label", "label"]].copy()
    result = result.rename(columns={"source_label": "true_source_label", "label": true_column})
    result[prediction_column] = predicted
    result["predicted_source_label"] = [display_label(label, mapping) for label in predicted]
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(source), len(classes)) or not np.isfinite(p).all():
            raise ValueError("Probability matrix does not align with samples and classes")
        for index, label in enumerate(classes):
            result[f"prob__{label}"] = p[:, index]
    return result


def save_confusion(y_true, y_pred, classes: list[str], mapping: dict[str, list[str]],
                   title: str, path: Path) -> None:
    """Save counts, source-name mapping, and a plot from the same ordered labels."""
    true = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    if len(true) != len(predicted) or not set(true).issubset(classes) or not set(predicted).issubset(classes):
        raise ValueError("Confusion matrix labels are misaligned or unknown")
    counts = confusion_matrix(true, predicted, labels=classes)
    source_names = [display_label(label, mapping) for label in classes]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(counts, columns=[f"predicted:{label}" for label in classes])
    table.insert(0, "true_source_label", source_names)
    table.insert(0, "true_canonical_label", classes)
    table.to_csv(path.with_suffix(".csv"), index=False)
    write_json(path.with_suffix(".labels.json"), {
        "axis_order_canonical": classes,
        "axis_order_source": source_names,
        "canonical_to_source": mapping,
        "rows": "true class", "columns": "predicted class",
    })
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)), labels=source_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), labels=source_names)
    ax.set_xlabel("Predicted original class folder")
    ax.set_ylabel("True original class folder")
    ax.set_title(title)
    threshold = counts.max() / 2 if counts.size else 0
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(counts[row, col]), ha="center", va="center",
                    color="white" if counts[row, col] > threshold else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
