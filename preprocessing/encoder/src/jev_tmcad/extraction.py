from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .cad_features import extract_descriptor
from .utils import ensure_parent

LOGGER = logging.getLogger(__name__)


def _extract_one_worker(
    item: dict[str, Any],
    relative_tolerance: float,
    angular_bins: int,
    axial_bins: int,
) -> dict[str, Any]:
    base = dict(item)
    started = time.perf_counter()
    try:
        desc = extract_descriptor(
            item["absolute_path"],
            relative_tolerance=relative_tolerance,
            angular_bins=angular_bins,
            axial_bins=axial_bins,
        )
        for key, value in list(desc.items()):
            if isinstance(value, list):
                desc[key] = json.dumps(value)
        base.update(desc)
        base["extract_error"] = ""
    except Exception as exc:
        base["extract_error"] = f"{type(exc).__name__}: {exc}"
    base["descriptor_seconds"] = time.perf_counter() - started
    return base


def extract_dataset_descriptors(
    manifest: pd.DataFrame,
    output_csv: str | Path,
    relative_tolerance: float = 0.005,
    angular_bins: int = 72,
    axial_bins: int = 12,
    resume: bool = True,
    workers: int = 1,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        if "sample_id" in existing:
            done = set(existing["sample_id"].astype(str))
            LOGGER.info("Resuming descriptor extraction with %d cached samples", len(done))

    todo = [r for r in manifest.to_dict(orient="records") if str(r["sample_id"]) not in done]
    if not todo:
        return existing if existing is not None else manifest.iloc[0:0].copy()

    new_rows: list[dict[str, Any]] = []

    def checkpoint(force: bool = False) -> None:
        nonlocal existing, new_rows, done
        if not new_rows:
            return
        if not force and len(new_rows) < checkpoint_every:
            return
        _merge_and_save(existing, new_rows, output_csv)
        existing = pd.read_csv(output_csv)
        done = set(existing["sample_id"].astype(str))
        new_rows = []

    workers = max(1, int(workers))
    if workers == 1:
        iterator = tqdm(todo, total=len(todo), desc="Extracting STEP descriptors")
        for item in iterator:
            row = _extract_one_worker(item, relative_tolerance, angular_bins, axial_bins)
            if row.get("extract_error"):
                LOGGER.warning("Descriptor extraction failed for %s: %s", item["absolute_path"], row["extract_error"])
            new_rows.append(row)
            checkpoint()
    else:
        # Process-based parallelism avoids sharing OpenCASCADE state between threads.
        # On Windows this is safe when invoked through `python -m jev_tmcad ...` because
        # the CLI entry point is protected by the normal module main guard.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _extract_one_worker,
                    item,
                    relative_tolerance,
                    angular_bins,
                    axial_bins,
                ): item
                for item in todo
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting STEP descriptors ({workers} processes)"):
                item = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = dict(item)
                    row["extract_error"] = f"WorkerError: {type(exc).__name__}: {exc}"
                if row.get("extract_error"):
                    LOGGER.warning("Descriptor extraction failed for %s: %s", item["absolute_path"], row["extract_error"])
                new_rows.append(row)
                checkpoint()

    checkpoint(force=True)
    return pd.read_csv(output_csv)


def _merge_and_save(existing: pd.DataFrame | None, new_rows: list[dict], output_csv: Path) -> None:
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("sample_id", keep="last").sort_values(["label", "sample_id"])
    ensure_parent(output_csv)
    merged.to_csv(output_csv, index=False)
