"""Optional small train-only ridge probe, with a validation-selected regularizer."""
from __future__ import annotations

import numpy as np

from .catalog import CLASSES, FEATURE_KEYS
from .io import write_json


def fit_probe(catalog, output, alphas=(0.1, 1.0, 10.0)):
    train = [p for p in catalog["parts"] if p["split"] == "train"]
    val = [p for p in catalog["parts"] if p["split"] == "val"]
    if not train or not val or {p["label"] for p in train} != set(CLASSES):
        raise ValueError("Probe needs train+val data and all ten classes in train")
    keys = [k for k in FEATURE_KEYS if any(p["features"].get(k) is not None for p in train)]
    if not keys:
        raise ValueError("Missing descriptors")
    def raw(parts):
        return np.asarray([[np.nan if p["features"].get(k) is None else p["features"][k] for k in keys] for p in parts])
    tx = raw(train)
    median = np.nanmedian(tx, axis=0)
    tx = np.where(np.isfinite(tx), tx, median)
    mean, scale = tx.mean(0), tx.std(0)
    scale[scale < 1e-8] = 1
    def transform(parts):
        x = raw(parts)
        x = (np.where(np.isfinite(x), x, median)-mean)/scale
        return np.column_stack([x, np.ones(len(x))])
    x, vx = transform(train), transform(val)
    y = np.eye(len(CLASSES))[[CLASSES.index(p["label"]) for p in train]]
    vy = np.array([CLASSES.index(p["label"]) for p in val])
    trials = []
    for alpha in alphas:
        regularizer = np.eye(x.shape[1])*alpha
        regularizer[-1, -1] = 0
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        pred = (vx @ weights).argmax(1)
        macro = np.mean([np.mean(pred[vy == c] == c) for c in np.unique(vy)])
        trials.append((float(macro), float(alpha), weights))
    score, alpha, weights = max(trials, key=lambda t: (t[0], -t[1]))
    state = {"kind": "ridge_multiclass_probe", "classes": CLASSES, "keys": keys, "median": median.tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(), "alpha": alpha,
             "selection": "val balanced accuracy", "val_balanced_accuracy": score,
             "calibration": "softmax of ridge scores; uncalibrated", "train_ids": [p["part_id"] for p in train],
             "val_ids": [p["part_id"] for p in val]}
    write_json(output, state)
    return state


def predict_probe(catalog, state, source):
    keys = state["keys"]
    for part in catalog["parts"]:
        if not any(part["features"].get(k) is not None for k in keys):
            continue
        raw = np.array([np.nan if part["features"].get(k) is None else part["features"][k] for k in keys])
        x = (np.where(np.isfinite(raw), raw, state["median"])-state["mean"])/state["scale"]
        logits = np.append(x, 1) @ np.asarray(state["weights"])
        probs = np.exp(logits-logits.max())
        probs /= probs.sum()
        part["prediction"] = {"probabilities": dict(zip(state["classes"], probs.tolist())), "status": "predicted",
                              "source": str(source), "model": state["kind"], "calibration": state["calibration"]}
