from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def receipt(command, inputs, settings=None):
    files = {str(Path(p).resolve()): sha256(p) for p in inputs if p and Path(p).is_file()}
    versions = {}
    for name in ["numpy", "cadquery-ocp", "pytest", "matplotlib", "pillow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = {str(p): sha256(p) for p in sorted((p for p in Path(__file__).parent.rglob("*") if p.suffix in {".py", ".html", ".js", ".css"} and "__pycache__" not in p.parts))}
    return {"command": command, "input_sha256": files, "settings": settings or {},
            "python": sys.version, "platform": platform.platform(), "versions": versions,
            "code_sha256": source}
