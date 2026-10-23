"""Observed timings, separated by work stage and cache status."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import write_json


def _summary(values) -> dict:
    x = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean_seconds": float(x.mean()),
            "median_seconds": float(np.median(x)), "p95_seconds": float(np.quantile(x, .95)),
            "total_seconds": float(x.sum())}


def summarize_efficiency(descriptor_path: Path, mesh_manifest_path: Path, embeddings_path: Path,
                         geometry_checkpoint: Path, semantic_frame: pd.DataFrame,
                         output_path: Path, cohort_ids: set[str]) -> dict:
    out = {"conditions": {"python": platform.python_version(), "platform": platform.platform(),
                          "unit": "wall-clock seconds", "note": "Warm/cached calls are excluded from cold stage summaries; compare hardware and batch sizes before attributing speed differences."}}
    desc = pd.read_csv(descriptor_path, low_memory=False)
    desc = desc[desc["sample_id"].astype(str).isin(cohort_ids)]
    out["descriptor_extraction"] = _summary(desc["descriptor_seconds"]) if "descriptor_seconds" in desc else {"n": 0, "note": "Legacy descriptor cache has no per-sample timing"}
    mesh = pd.read_csv(mesh_manifest_path)
    out["mesh_preprocessing_uncached"] = _summary(mesh.loc[mesh["status"] == "computed", "elapsed_seconds"])
    out["mesh_cache_lookup"] = _summary(mesh.loc[mesh["status"] == "cached", "elapsed_seconds"])
    mesh_execution = mesh_manifest_path.parent / "execution.json"
    if mesh_execution.exists():
        out["mesh_execution"] = json.loads(mesh_execution.read_text(encoding="utf-8"))
    geom = json.loads(embeddings_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    timings = geom["timings"]
    out["geometry_feature_extraction"] = {
        "device": geom["device"], "configured_batch_size": geom["batch_size"],
        "batch_load": _summary(t["load_seconds"] for t in timings),
        "batch_transfer": _summary(t["transfer_seconds"] for t in timings),
        "batch_inference": _summary(t["inference_seconds"] for t in timings),
        "batch_end_to_end": _summary(t["end_to_end_seconds"] for t in timings),
        "per_sample_inference": _summary(t["per_sample_inference_seconds"] for t in timings),
        "weighted_mean_per_sample_inference_seconds": (
            sum(t["inference_seconds"] for t in timings) / sum(t["batch_size"] for t in timings)
        ),
    }
    train = json.loads(geometry_checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    out["geometry_training"] = {"total_seconds": train["training_total_seconds"],
                                "epochs": len(train["history"]), "device": train["device"],
                                "cache_in_memory": train.get("cache_in_memory", False),
                                "mesh_cache_bytes": train.get("mesh_cache_bytes", 0),
                                "mesh_preload_seconds": train.get("mesh_preload_seconds", 0)}
    for name, source in (("jev", "api"), ("laya", "local_inference"),
                         ("laya_ft", "local_inference"), ("clm", "local_inference")):
        if f"{name}__latency_seconds" in semantic_frame:
            selected = semantic_frame[semantic_frame[f"{name}__cache_source"] == source]
            out[name] = _summary(selected[f"{name}__latency_seconds"])
            out[name]["timing_scope"] = ("API round trip including retries" if name == "jev" else
                                          "local CLM server round trip, including embedding and head inference; excludes server startup" if name == "clm" else
                                          "warm local predict call; excludes model load")
            if name.startswith("laya"):
                out[f"{name}_preflight"] = _summary(selected[f"{name}__preflight_seconds"])
                out[f"{name}_model_load"] = _summary(selected[f"{name}__model_load_seconds"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, out)
    return out
