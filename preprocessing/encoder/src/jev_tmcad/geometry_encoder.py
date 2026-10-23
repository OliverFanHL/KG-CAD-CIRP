"""Local-neighborhood surface encoder and split-safe training/inference."""
from __future__ import annotations

import json
import hashlib
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import TMCAD_CLASSES
from .study_data import cohort_hash
from .utils import sha256_json, write_json


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Geometry experiments require `pip install -e .[geometry]`.") from exc
    return torch


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        _torch().cuda.synchronize()


def mesh_cache_hash(manifest: pd.DataFrame) -> str:
    return sha256_json(manifest[["sample_id", "cache_key"]].sort_values("sample_id").to_dict("records"))


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(embedding_dim: int):
    torch = _torch()
    nn = torch.nn

    class EdgeBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv2d(channels_in * 2, channels_out, 1), nn.BatchNorm2d(channels_out), nn.ReLU(),
                nn.Conv2d(channels_out, channels_out, 1), nn.BatchNorm2d(channels_out), nn.ReLU(),
            )

        def forward(self, x, neighbors):
            batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
            near = x[batch, neighbors]
            center = x[:, :, None, :].expand_as(near)
            edge = torch.cat([center, near - center], dim=-1).permute(0, 3, 1, 2)
            return self.mlp(edge).amax(dim=-1).transpose(1, 2)

    class LocalSurfaceEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge1 = EdgeBlock(6, 64)
            self.edge2 = EdgeBlock(64, 128)
            self.embedding = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2),
                                           nn.Linear(256, embedding_dim))
            self.classifier = nn.Linear(embedding_dim, len(TMCAD_CLASSES))

        def forward(self, points, neighbors):
            x = self.edge1(points, neighbors)
            x = self.edge2(x, neighbors)
            pooled = torch.cat([x.amax(dim=1), x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=1)
            z = self.embedding(pooled)
            return self.classifier(z), z

    return LocalSurfaceEncoder()


def _batch_arrays(rows: list[dict], cache: dict | None = None):
    features, neighbors = [], []
    for row in rows:
        if cache is None:
            with np.load(row["mesh_path"]) as data:
                features.append(data["features"])
                neighbors.append(data["knn"])
        else:
            point_array, neighbor_array = cache[str(row["sample_id"])]
            features.append(point_array)
            neighbors.append(neighbor_array)
    return np.stack(features), np.stack(neighbors)


def _preload_meshes(rows: list[dict]) -> tuple[dict, int]:
    cache = {}
    total_bytes = 0
    for row in tqdm(rows, desc="Loading geometry training meshes"):
        with np.load(row["mesh_path"]) as data:
            points = data["features"]
            neighbors = data["knn"]
        cache[str(row["sample_id"])] = (points, neighbors)
        total_bytes += points.nbytes + neighbors.nbytes
    return cache, total_bytes


def _iter_batches(rows: list[dict], batch_size: int, shuffle: bool, seed: int):
    order = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start:start + batch_size]]


def _predict(model, rows: list[dict], batch_size: int, device: str,
             cache: dict | None = None):
    torch = _torch()
    model.eval()
    outputs, embeddings, timings = [], [], []
    with torch.inference_mode():
        for batch in _iter_batches(rows, batch_size, False, 0):
            start = time.perf_counter()
            points, neighbors = _batch_arrays(batch, cache)
            load_seconds = time.perf_counter() - start
            transfer_start = time.perf_counter()
            x = torch.from_numpy(points).to(device)
            k = torch.from_numpy(neighbors).to(device)
            _synchronize(device)
            transfer_seconds = time.perf_counter() - transfer_start
            infer_start = time.perf_counter()
            logits, z = model(x, k)
            _synchronize(device)
            infer_seconds = time.perf_counter() - infer_start
            outputs.extend(logits.cpu().numpy())
            embeddings.extend(z.cpu().numpy())
            timings.append({"batch_size": len(batch), "load_seconds": load_seconds,
                            "transfer_seconds": transfer_seconds,
                            "inference_seconds": infer_seconds,
                            "end_to_end_seconds": load_seconds + transfer_seconds + infer_seconds,
                            "per_sample_inference_seconds": infer_seconds / len(batch)})
    return np.asarray(outputs), np.asarray(embeddings), timings


def train_geometry(cohort: pd.DataFrame, mesh_manifest: pd.DataFrame, checkpoint: Path, *,
                   seed: int, device: str, epochs: int, batch_size: int, learning_rate: float,
                   weight_decay: float, embedding_dim: int, patience: int,
                   cache_in_memory: bool = False) -> dict:
    torch = _torch()
    from sklearn.metrics import f1_score

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rows = cohort.merge(mesh_manifest[["sample_id", "mesh_path"]], on="sample_id", validate="one_to_one")
    train = rows[rows["split"] == "train"].to_dict("records")
    val = rows[rows["split"] == "val"].to_dict("records")
    if not train or not val:
        raise ValueError("Geometry training requires train and validation rows")
    preload_start = time.perf_counter()
    mesh_cache, cache_bytes = _preload_meshes(train + val) if cache_in_memory else (None, 0)
    preload_seconds = time.perf_counter() - preload_start if cache_in_memory else 0.0
    model = _model(embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best = -1.0
    wait = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()
    stop_reason = "max_epochs"
    with tqdm(total=epochs, desc=f"Geometry epochs ({device})") as epoch_progress:
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            model.train()
            losses = []
            for batch in _iter_batches(train, batch_size, True, seed + epoch):
                points, neighbors = _batch_arrays(batch, mesh_cache)
                x = torch.from_numpy(points).to(device)
                k = torch.from_numpy(neighbors).to(device)
                y = torch.tensor([TMCAD_CLASSES.index(r["label"]) for r in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x, k)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_logits, _, _ = _predict(model, val, batch_size, device, mesh_cache)
            val_true = [r["label"] for r in val]
            val_pred = [TMCAD_CLASSES[i] for i in val_logits.argmax(axis=1)]
            score = float(f1_score(val_true, val_pred, labels=TMCAD_CLASSES, average="macro", zero_division=0))
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                            "val_macro_f1": score, "epoch_seconds": time.perf_counter() - epoch_started})
            if score > best:
                best, wait = score, 0
                torch.save({"state_dict": model.state_dict(), "embedding_dim": embedding_dim,
                            "cohort_hash": cohort_hash(cohort), "mesh_hash": mesh_cache_hash(mesh_manifest),
                            "epoch": epoch + 1,
                            "classes": TMCAD_CLASSES}, checkpoint)
            else:
                wait += 1
            epoch_progress.update(1)
            epoch_progress.set_postfix(val_macro_f1=f"{score:.3f}", best=f"{best:.3f}",
                                       patience=f"{wait}/{patience}")
            if wait >= patience:
                stop_reason = "early_stopping"
                break
    details = {"best_val_macro_f1": best, "best_epoch": history[int(np.argmax([h["val_macro_f1"] for h in history]))]["epoch"],
               "history": history, "device": device, "torch": torch.__version__,
               "configured_epochs": epochs, "completed_epochs": len(history),
               "stop_reason": stop_reason, "patience": patience,
               "epochs_without_improvement": wait,
               "python": platform.python_version(), "seed": seed,
               "cache_in_memory": cache_in_memory, "mesh_cache_bytes": cache_bytes,
               "mesh_preload_seconds": preload_seconds,
               "cohort_hash": cohort_hash(cohort),
               "training_total_seconds": time.perf_counter() - training_started}
    write_json(checkpoint.with_suffix(".json"), details)
    return details


def extract_embeddings(cohort: pd.DataFrame, mesh_manifest: pd.DataFrame, checkpoint: Path,
                       output_csv: Path, *, device: str, batch_size: int) -> pd.DataFrame:
    torch = _torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["cohort_hash"] != cohort_hash(cohort):
        raise ValueError("Geometry checkpoint was trained on a different cohort")
    if saved["mesh_hash"] != mesh_cache_hash(mesh_manifest):
        raise ValueError("Geometry checkpoint was trained on a different mesh cache")
    model = _model(saved["embedding_dim"])
    model.load_state_dict(saved["state_dict"])
    model = model.to(device)
    rows = cohort.merge(mesh_manifest[["sample_id", "mesh_path"]], on="sample_id", validate="one_to_one")
    ordered = rows.sort_values("sample_id").to_dict("records")
    logits, embeddings, timings = _predict(model, ordered, batch_size, device)
    records = []
    for row, logit, vector in zip(ordered, logits, embeddings):
        rec = {"sample_id": row["sample_id"], "split": row["split"], "label": row["label"],
               "source_label": row.get("source_label", ""),
               "relative_path": row.get("relative_path", ""),
               "absolute_path": row.get("absolute_path", "")}
        rec.update({f"geom__{j:03d}": float(v) for j, v in enumerate(vector)})
        rec.update({f"geom_logit__{name}": float(v) for name, v in zip(TMCAD_CLASSES, logit)})
        records.append(rec)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output_csv, index=False)
    write_json(output_csv.with_suffix(".meta.json"), {
        "cohort_hash": cohort_hash(cohort), "checkpoint": str(checkpoint),
        "mesh_hash": mesh_cache_hash(mesh_manifest),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_epoch": saved["epoch"], "device": device, "batch_size": batch_size,
        "timings": timings,
    })
    return result
