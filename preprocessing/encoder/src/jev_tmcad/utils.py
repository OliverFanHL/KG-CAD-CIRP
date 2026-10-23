from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def sha1_text(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def write_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        value = float(x)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) < 1e-12 else a / b


def normalized_entropy(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[arr > 0]
    if arr.size <= 1:
        return 0.0
    p = arr / arr.sum()
    h = -(p * np.log(p + 1e-12)).sum()
    return float(h / np.log(len(p)))


def resolve_from(base_dir: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()
