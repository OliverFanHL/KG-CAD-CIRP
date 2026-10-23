"""Only explicitly reviewed rows enter engineering knowledge; evaluation judgments never do."""
from __future__ import annotations

import math
from pathlib import Path

from .catalog import canonical_class, finite
from .io import object_hash, read_csv


def reviewed(row):
    return row.get("status") == "curated" and bool(row.get("reviewer", "").strip()) and bool(row.get("evidence", "").strip())


def load_knowledge(directory, catalog, policy="curated_only"):
    if policy not in {"curated_only", "include_source_supported"}:
        raise ValueError("Unknown knowledge admission policy")
    directory = Path(directory)
    functions, relations, interfaces, excluded = [], [], [], []
    for filename, destination in [("functions.csv", functions), ("class_functions.csv", relations), ("interfaces.csv", interfaces)]:
        path = directory / filename
        if not path.exists():
            continue
        for index, row in enumerate(read_csv(path), 2):
            if not any(v.strip() for v in row.values()):
                continue
            sourced = (policy == "include_source_supported" and filename != "interfaces.csv"
                       and row.get("status") == "source_supported" and row.get("evidence", "").startswith("https://")
                       and row.get("source_id", "").strip())
            if not reviewed(row) and not sourced:
                excluded.append({"file": filename, "line": index, "reason": "not explicitly curated with reviewer/evidence"})
                continue
            destination.append({**row, "source": f"{path.resolve()}:{index}"})
    by_function = {}
    aliases = {}
    for row in functions:
        fid = row["function_id"].strip()
        if not fid or fid in by_function or not row["label"].strip():
            raise ValueError("Empty/duplicate function ID or label")
        by_function[fid] = row
        for alias in [row["label"], *row.get("aliases", "").split("|")]:
            alias = alias.strip().lower()
            if alias and alias in aliases and aliases[alias] != fid:
                raise ValueError(f"Ambiguous function alias: {alias}")
            if alias:
                aliases[alias] = fid
    seen = set()
    for row in relations:
        row["class_id"] = canonical_class(row["class_id"])
        if row["function_id"] not in by_function:
            raise ValueError("Class-function row refers to unknown/unreviewed function")
        key = row["class_id"], row["function_id"]
        if key in seen:
            raise ValueError(f"Duplicate class-function row {key}")
        seen.add(key)
        for field in ["relevance", "confidence"]:
            val = finite(row[field])
            if val is None or not 0 <= val <= 1:
                raise ValueError(f"{field} must be finite in [0,1]")
            row[field] = val
        if not row.get("scope", "").strip() or not row.get("caveat", "").strip():
            raise ValueError("Class-function priors require scope and caveat")
    parts = {p["part_id"]: p for p in catalog["parts"]}
    seen = set()
    for row in interfaces:
        key = row["part_id"], row["interface_id"]
        if row["part_id"] not in parts or not row["interface_id"].strip() or key in seen:
            raise ValueError(f"Unknown part or empty/duplicate interface {key}")
        seen.add(key)
        if row["role"] not in {"bore", "shaft"} or row.get("unit") != "mm":
            raise ValueError("Interface role must be bore/shaft, with explicitly verified mm")
        value = finite(row["diameter_mm"])
        lower, upper = finite(row.get("min_mm")), finite(row.get("max_mm"))
        if value is None or value <= 0:
            raise ValueError("Interface diameter must be positive")
        if bool(row.get("min_mm")) != bool(row.get("max_mm")):
            raise ValueError("Provide both tolerance bounds or neither")
        if row.get("min_mm") and (lower is None or upper is None or not 0 < lower <= value <= upper):
            raise ValueError("Invalid interface tolerance interval")
        row.update(diameter_mm=value, min_mm=lower, max_mm=upper)
        parts[row["part_id"]]["interfaces"].append(row)
    result = {"functions": list(by_function.values()), "class_functions": relations, "admission_policy": policy,
              "evidence_note": "source_supported rows are documented generic priors, not expert-reviewed part facts",
              "interfaces": interfaces, "excluded_rows": excluded}
    result["sha256"] = object_hash(result)
    return result
