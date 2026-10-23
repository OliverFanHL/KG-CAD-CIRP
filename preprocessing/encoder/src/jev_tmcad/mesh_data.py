"""Deterministic local surface samples from STEP B-Reps.

The representation retains point neighborhoods and surface normals; labels and
paths are never part of the tensors passed to the geometry encoder.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cad_features import _load_shape
from .utils import sha256_json, write_json


MESH_VERSION = "surface-knn-v1"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_surface(path: Path, *, points: int, neighbors: int, tolerance: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    if points < 32 or not 1 <= neighbors < points:
        raise ValueError("Require points >= 32 and 1 <= neighbors < points")
    shape = _load_shape(path)
    bb = shape.BoundingBox()
    extent = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6)
    vertices, triangles = shape.tessellate(max(extent * tolerance, 1e-5))
    xyz = np.asarray([v.toTuple() for v in vertices], dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[0] < 4 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("STEP tessellation contains no usable triangular surface")
    faces = xyz[tri]
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 1e-12
    faces, cross, area2 = faces[keep], cross[keep], area2[keep]
    if not len(faces):
        raise ValueError("STEP tessellation has only degenerate triangles")
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), size=points, p=area2 / area2.sum())
    chosen = faces[face_ids]
    u = np.sqrt(rng.random(points))
    v = rng.random(points)
    sampled = ((1 - u)[:, None] * chosen[:, 0]
               + (u * (1 - v))[:, None] * chosen[:, 1]
               + (u * v)[:, None] * chosen[:, 2])
    normals = cross[face_ids] / area2[face_ids, None]
    sampled -= sampled.mean(axis=0)
    radius = max(float(np.linalg.norm(sampled, axis=1).max()), 1e-8)
    sampled /= radius
    features = np.concatenate([sampled, normals], axis=1).astype(np.float32)
    dist2 = ((sampled[:, None, :] - sampled[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(dist2, np.inf)
    knn = np.argpartition(dist2, kth=neighbors - 1, axis=1)[:, :neighbors]
    knn = np.take_along_axis(knn, np.argsort(np.take_along_axis(dist2, knn, axis=1)), axis=1)
    return features, knn.astype(np.int64)


def _mesh_one(item: dict, output_dir: Path, cfg: dict) -> dict:
    """One independent OpenCASCADE job; safe to dispatch to a process."""
    started = time.perf_counter()
    path = Path(item["absolute_path"])
    sample_id = str(item["sample_id"])
    target = output_dir / f"{sample_id}.npz"
    cache_key = ""
    status = "cached"
    error = ""
    try:
        cache_key = sha256_json({"source_sha256": _file_digest(path), "config": cfg})
        if target.exists():
            with np.load(target) as old:
                reusable = str(old["cache_key"]) == cache_key
        else:
            reusable = False
        if not reusable:
            features, knn = sample_surface(
                path, points=cfg["points"], neighbors=cfg["neighbors"],
                tolerance=cfg["tolerance"],
                seed=(cfg["seed"] + int(sample_id[:8], 16)) % (2**32),
            )
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, features=features, knn=knn, cache_key=cache_key)
            tmp.replace(target)
            status = "computed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return {"sample_id": sample_id, "split": item["split"], "label": item["label"],
            "source_label": item.get("source_label", path.parent.name),
            "relative_path": item.get("relative_path", ""), "absolute_path": str(path),
            "cache_key": cache_key, "mesh_path": str(target), "status": status,
            "error": error, "elapsed_seconds": time.perf_counter() - started}


def _save_manifest(rows: list[dict], path: Path) -> None:
    # Only the parent process writes the manifest. A restart can still reuse
    # finished .npz files even when it was interrupted before this checkpoint.
    tmp = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def build_mesh_cache(cohort: pd.DataFrame, output_dir: Path, *, points: int,
                     neighbors: int, tolerance: float, seed: int,
                     workers: int = 1, checkpoint_every: int = 100) -> pd.DataFrame:
    if workers < 1 or checkpoint_every < 1:
        raise ValueError("workers and checkpoint_every must be positive")
    run_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"version": MESH_VERSION, "points": points, "neighbors": neighbors,
           "tolerance": tolerance, "seed": seed}
    items = cohort.to_dict("records")
    rows = []
    manifest_path = output_dir / "manifest.csv"

    def collect(row: dict, progress) -> None:
        rows.append(row)
        progress.update(1)
        if len(rows) % checkpoint_every == 0:
            _save_manifest(rows, manifest_path)

    with tqdm(total=len(items), desc=f"STEP surface samples ({workers} processes)") as progress:
        if workers == 1:
            for item in items:
                collect(_mesh_one(item, output_dir, cfg), progress)
        else:
            # Bound submitted jobs to limit memory use on Windows. Processes
            # isolate OpenCASCADE state; threads would share it.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                source = iter(items)
                pending = {}

                def submit_next() -> bool:
                    try:
                        item = next(source)
                    except StopIteration:
                        return False
                    pending[pool.submit(_mesh_one, item, output_dir, cfg)] = item
                    return True

                for _ in range(min(len(items), workers * 2)):
                    submit_next()
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        item = pending.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = {"sample_id": str(item["sample_id"]), "split": item["split"],
                                   "label": item["label"],
                                   "source_label": item.get("source_label", Path(item["absolute_path"]).parent.name),
                                   "relative_path": item.get("relative_path", ""),
                                   "absolute_path": str(item["absolute_path"]), "cache_key": "",
                                   "mesh_path": str(output_dir / f"{item['sample_id']}.npz"),
                                   "status": "failed", "error": f"WorkerError: {type(exc).__name__}: {exc}",
                                   "elapsed_seconds": float("nan")}
                        collect(row, progress)
                        submit_next()
    # Stable manifest order makes hashes and timing reports easy to compare.
    order = {str(item["sample_id"]): index for index, item in enumerate(items)}
    rows.sort(key=lambda row: order[row["sample_id"]])
    _save_manifest(rows, manifest_path)
    write_json(output_dir / "config.json", cfg)
    write_json(output_dir / "execution.json", {"workers": workers,
                                                "checkpoint_every": checkpoint_every,
                                                "wall_seconds": time.perf_counter() - run_started,
                                                "computed": sum(r["status"] == "computed" for r in rows),
                                                "cached": sum(r["status"] == "cached" for r in rows),
                                                "failed": sum(r["status"] == "failed" for r in rows)})
    return pd.DataFrame(rows)


def load_mesh_manifest(path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"sample_id": str}).fillna("")
    ids = set(cohort["sample_id"].astype(str))
    if manifest["sample_id"].duplicated().any() or set(manifest["sample_id"].astype(str)) != ids:
        raise ValueError("Mesh cache does not cover the frozen cohort")
    source = cohort.set_index("sample_id")
    indexed = manifest.set_index("sample_id")
    for column in ("split", "label", "source_label", "relative_path", "absolute_path"):
        if column in indexed and column in source and indexed[column].astype(str).ne(
                source.loc[indexed.index, column].astype(str)).any():
            raise ValueError(f"Mesh manifest {column} differs from the frozen cohort")
    failures = manifest[manifest["status"] == "failed"]
    if not failures.empty:
        raise ValueError(f"Mesh preprocessing failed for {len(failures)} samples; inspect {path}")
    return manifest
