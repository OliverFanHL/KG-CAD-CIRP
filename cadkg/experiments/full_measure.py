"""Resumable full-corpus STEP measurement with isolated, time-limited workers."""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from collections import Counter
from pathlib import Path

from ..geometry import extract_step
from ..io import read_json, sha256, write_json


def _worker(jobs, results):
    while True:
        item = jobs.get()
        if item is None:
            return
        pid, path, expected = item
        started = time.monotonic()
        try:
            if sha256(path) != expected:
                raise ValueError("Source bytes changed since import")
            data = extract_step(path)
            results.put((pid, {"part_id": pid, "file_sha256": expected, "status": "ok",
                               "data": data, "seconds": time.monotonic()-started}))
        except Exception as exc:
            results.put((pid, {"part_id": pid, "file_sha256": expected, "status": "error",
                               "error": f"{type(exc).__name__}: {exc}"}))


def measure_full(catalog, output, workers=4, timeout_seconds=90):
    """Cache each source-bound result. A timed-out native kernel cannot block the run."""
    if workers < 1 or timeout_seconds < 1:
        raise ValueError("Positive worker count and timeout required")
    output = Path(output)
    cache = output / "measurement_cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = []
    for part in catalog["parts"]:
        target = cache / f"{part['part_id']}.json"
        if target.exists():
            value = read_json(target)
            if value["file_sha256"] != part["file_sha256"] or sha256(part['absolute_path']) != part['file_sha256']:
                raise ValueError("Measurement cache belongs to different source bytes")
        else:
            todo.append(part)
    # Heavy files start early, so the end of the run is not dominated by one file.
    todo.sort(key=lambda p: Path(p["absolute_path"]).stat().st_size, reverse=True)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    slots = []
    def spawn():
        jobs = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(jobs, results), daemon=True)
        proc.start()
        return {"process": proc, "jobs": jobs, "part": None, "started": None}
    slots = [spawn() for _ in range(min(workers, max(1, len(todo))))]
    pending, completed = iter(todo), len(catalog["parts"])-len(todo)
    exhausted = False
    last_progress = 0
    try:
        while not exhausted or any(s["part"] for s in slots):
            for slot in slots:
                if slot["part"] is None and not exhausted:
                    part = next(pending, None)
                    if part is None:
                        exhausted = True
                        break
                    slot["part"], slot["started"] = part, time.monotonic()
                    slot["jobs"].put((part["part_id"], part["absolute_path"], part["file_sha256"]))
            try:
                pid, value = results.get(timeout=.5)
                slot = next((s for s in slots if s["part"] and s["part"]["part_id"] == pid), None)
                if slot is not None:
                    write_json(cache / f"{pid}.json", value)
                    slot["part"] = None
                    completed += 1
            except queue.Empty:
                pass
            for i, slot in enumerate(slots):
                part = slot["part"]
                if part is not None and (not slot["process"].is_alive() or time.monotonic()-slot["started"] > timeout_seconds):
                    status = "timeout" if slot["process"].is_alive() else "worker_crash"
                    slot["process"].terminate()
                    slot["process"].join(5)
                    write_json(cache / f"{part['part_id']}.json", {"part_id": part["part_id"],
                               "file_sha256": part["file_sha256"], "status": status,
                               "error": f"isolated worker {status}; limit={timeout_seconds}s"})
                    slots[i] = spawn()
                    completed += 1
            if time.monotonic()-last_progress > 30:
                print(f"STEP measurement: {completed}/{len(catalog['parts'])} cached; active={sum(s['part'] is not None for s in slots)}", flush=True)
                last_progress = time.monotonic()
    finally:
        for slot in slots:
            slot["process"].terminate()
            slot["process"].join(5)
    counts, errors = Counter(), []
    # Preserve previously found analytical/tessellation disagreements. Never restore
    # quarantined dimensions merely because the same analytical extractor was rerun.
    previous = output / "previous_quarantine.json"
    quarantined = read_json(previous) if previous.exists() else {}
    for part in catalog["parts"]:
        value = read_json(cache / f"{part['part_id']}.json")
        counts[value["status"]] += 1
        if value["status"] == "ok":
            part.update(value["data"])
            if part["part_id"] in quarantined:
                for attribute, measurement in part["measurements"].items():
                    if attribute.startswith("aabb_"):
                        measurement["status"] = "computed_unverified"
                        measurement["quarantine"] = quarantined[part["part_id"]]
        else:
            errors.append(value)
    catalog["measurement_errors"] = errors
    audit = {"requested": len(catalog["parts"]), "outcomes": dict(counts), "workers": workers,
             "timeout_seconds": timeout_seconds, "cached_results_are_source_hash_bound": True,
             "quarantined_parts": list(quarantined),
             "independence": "OCCT analytical measurements, not independent engineering metrology"}
    catalog["audit"]["full_measurement"] = audit
    write_json(output / "measured_catalog.json", catalog)
    write_json(output / "measurement_audit.json", audit)
    return audit


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    print(json.dumps(measure_full(read_json(args.catalog), args.output, args.workers, args.timeout)))


if __name__ == "__main__":
    main()
