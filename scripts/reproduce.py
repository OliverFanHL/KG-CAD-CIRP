"""Replay immutable publication results offline from any checkout location.

The archived evaluators and journals remain byte-for-byte unchanged. This
adapter only resolves historical workspace paths and selects a fresh output.
No collection runner is imported or called.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
REPO = PUBLIC_ROOT / 'archive/workspace'
sys.path.insert(0, str(REPO))
HISTORICAL_ROOT = "D:/Postdoc/research/cad_kg/"
ARTIFACTS = (
    "graph_support.json", "graph_evidence.jsonl", "per_query.csv",
    "family_metrics.csv", "run_metrics.csv", "summary.csv", "paired_tests.csv",
    "family_differences.csv", "top5_rankings.json",
)


def local_path(value):
    normalized = str(value).replace("\\", "/")
    public_root = PUBLIC_ROOT.as_posix().rstrip("/") + "/"
    if normalized.lower().startswith(public_root.lower()):
        return Path(normalized)
    current_root = REPO.as_posix().rstrip("/") + "/"
    if normalized.lower().startswith(current_root.lower()):
        return Path(normalized)
    if normalized.lower().startswith(HISTORICAL_ROOT.lower()):
        return REPO / normalized[len(HISTORICAL_ROOT):]
    path = Path(normalized)
    if not path.is_absolute() and not (len(normalized) > 1 and normalized[1] == ":"):
        return REPO / path
    return path


def read(value):
    path = local_path(value)
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    # Frozen consumed-evidence keys use Windows separators. Normalize only the
    # in-memory lookup keys; the archived JSON bytes and digest remain intact.
    if path.name == "consumed_evidence_freeze.json":
        obj["input_sha256"] = {
            str(local_path(key)): digest for key, digest in obj["input_sha256"].items()
        }
    return obj


@lru_cache(maxsize=None)
def _sha_snapshot(path, size, modified_ns):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha(value):
    path = local_path(value).resolve()
    stat = path.stat()
    # Sealed protocol bytes are checked thousands of times by the original
    # evidence builder. Cache only while file size and modification time agree.
    return _sha_snapshot(path, stat.st_size, stat.st_mtime_ns)


def preserve_archived_newlines(source, target, names):
    """Restore the archived text encoding convention on non-Windows hosts."""
    for name in names:
        original = (source / name).read_bytes()
        generated = (target / name).read_bytes()
        use_crlf = b"\r\n" in original
        normalized = generated.replace(b"\r\n", b"\n")
        if use_crlf:
            normalized = normalized.replace(b"\n", b"\r\n")
        if normalized != generated:
            (target / name).write_bytes(normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Fresh destination (defaults to results/reproduced or results/reproduced_metrics).")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute family/run means, SDs and paired tests from frozen per-query metrics.")
    args = parser.parse_args()
    os.chdir(REPO)
    source = REPO / "outputs/llm_robustness_v7"
    destination = args.output or Path("results/reproduced_metrics" if args.metrics_only
                                      else "results/reproduced")
    target = (PUBLIC_ROOT / destination).resolve()
    if target.exists():
        parser.error("Output already exists; choose a fresh --output directory.")
    from scripts import complete_llm_robustness_v7 as evidence
    from scripts import evaluate_llm_robustness_v7 as evaluation
    evidence.read = evaluation.read = evaluation.old.read = read
    evidence.sha = evaluation.sha = evaluation.old.sha = sha
    target.mkdir(parents=True)
    if args.metrics_only:
        with (source / "per_query.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["replicate"] = int(row["replicate"])
            for metric in evaluation.METRICS:
                for endpoint in ("lower", "upper"):
                    key = metric + "_" + endpoint
                    row[key] = float(row[key])
        family, run, summary, tests, differences, identified = evaluation.analyze(rows)
        if not identified:
            raise ValueError("Primary metrics unexpectedly not point identified")
        names = ("family_metrics.csv", "run_metrics.csv", "summary.csv",
                 "paired_tests.csv", "family_differences.csv")
        for name, records in zip(names, (family, run, summary, tests, differences)):
            evaluation.old.write_csv(target / name, records)
    else:
        for name in ("study.json", "protocol_freeze.json", "evaluation_freeze.json",
                     "pool_manifest.json", "api_attempts.jsonl", "api_responses.jsonl"):
            shutil.copyfile(source / name, target / name)
        evidence.ROOT = target
        with (target / "replay.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                evidence.main()
                evaluation.evaluate(target)
        names = ARTIFACTS
    preserve_archived_newlines(source, target, names)
    if not args.metrics_only:
        # These are fresh receipts with fresh timestamps, never archive inputs.
        receipt = read(target / "evidence_completion.json")
        for name in ("graph_support.json", "graph_evidence.jsonl"):
            receipt[name.replace(".jsonl", "").replace(".json", "") + "_sha256"] = sha(target / name)
        evidence.write(target / "evidence_completion.json", receipt)
        result = read(target / "results.json")
        result["evidence_completion_sha256"] = sha(target / "evidence_completion.json")
        result["artifact_sha256"] = {name: sha(target / name) for name in result["artifact_sha256"]}
        evidence.write(target / "results.json", result)
    comparisons = {
        name: {"expected_sha256": sha(source / name), "actual_sha256": sha(target / name),
               "exact": sha(source / name) == sha(target / name)}
        for name in names
    }
    report = {"status": "PASS" if all(c["exact"] for c in comparisons.values()) else "FAIL",
              "mode": "metrics-only" if args.metrics_only else "full-offline-replay",
              "api_calls": 0, "comparisons": comparisons,
              "path_adapter": "Historical paths resolved in memory; frozen bytes preserved."}
    (target / "release_replay_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(target),
                      "artifacts_compared": len(comparisons), "api_calls": 0}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
