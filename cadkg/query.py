"""Deterministic vocabulary/constraint parsing. JSON is the canonical interface."""
from __future__ import annotations

import math
import re

from .catalog import CLASSES, canonical_class

ATTRIBUTES = {"aabb_x_mm": "mm", "aabb_y_mm": "mm", "aabb_z_mm": "mm", "aabb_small_mm": "mm",
              "aabb_mid_mm": "mm", "aabb_large_mm": "mm", "area_mm2": "mm2", "volume_mm3": "mm3",
              "bore_diameter_mm": "mm", "shaft_diameter_mm": "mm"}


def validate_query(query, knowledge):
    allowed = {"query_id", "scenario_id", "text", "function_ids", "category_ids", "reference_part_id", "constraints", "split", "parsed_from_text"}
    extra = set(query) - allowed
    if extra:
        raise ValueError(f"Unknown query fields: {sorted(extra)}")
    query = {"query_id": "interactive", "text": "", "function_ids": [], "category_ids": [],
             "reference_part_id": None, "constraints": [], "split": "test", **query}
    if query["split"] not in {"dev", "test", "demo"}:
        raise ValueError("Query split must be dev/test/demo")
    if not isinstance(query["query_id"], str) or not query["query_id"].strip():
        raise ValueError("Query ID must be a nonempty string")
    query.setdefault("scenario_id", query["query_id"])
    if not isinstance(query["scenario_id"], str) or not query["scenario_id"].strip():
        raise ValueError("Scenario ID must be a nonempty string")
    for field in ["function_ids", "category_ids", "constraints"]:
        if not isinstance(query[field], list):
            raise ValueError(f"{field} must be a list")
    ids = {f["function_id"] for f in knowledge["functions"]}
    if any(fid not in ids for fid in query["function_ids"]):
        raise ValueError("Unknown or unreviewed function ID")
    query["category_ids"] = [canonical_class(c) for c in query["category_ids"]]
    for constraint in query["constraints"]:
        if set(constraint) - {"attribute", "op", "value", "unit", "interface_id"}:
            raise ValueError("Unknown constraint fields")
        attr = constraint["attribute"]
        if attr not in ATTRIBUTES or constraint.get("unit") != ATTRIBUTES[attr]:
            raise ValueError(f"Unsupported attribute or wrong/missing unit: {attr}")
        op = constraint["op"]
        if op not in {"<=", ">=", "between"}:
            raise ValueError("Use <=, >=, or between")
        values = constraint["value"] if op == "between" else [constraint["value"]]
        if not isinstance(values, list) or len(values) != (2 if op == "between" else 1):
            raise ValueError("Invalid constraint bound count")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Constraint bounds must be finite nonnegative numbers")
        if op == "between" and values[0] > values[1]:
            raise ValueError("Reversed interval")
    return query


def parse_text(text, knowledge):
    lowered = text.lower()
    if re.search(r"\b(no|not|without|except|exclude|avoid)\b", lowered):
        raise ValueError("Negated intent is unsupported by this small parser; specify positive intent and explicit constraints")
    functions = []
    function_spans = []
    for function in knowledge["functions"]:
        aliases = [function["label"], *function.get("aliases", "").split("|")]
        matches = [match for alias in aliases if alias.strip() for match in
                   re.finditer(r"(?<!\w)" + re.escape(alias.strip().lower()) + r"(?!\w)", lowered)]
        if matches:
            functions.append(function["function_id"])
            function_spans.extend(match.span() for match in matches)
    category_text = list(lowered)
    for start, end in function_spans:
        category_text[start:end] = " "*(end-start)
    category_text = "".join(category_text)
    categories = [c for c in CLASSES if re.search(r"(?<!\w)" + "[-_ ]".join(re.escape(p) for p in c.split("-")) + r"(?!\w)", category_text)]
    constraints = []
    pattern = r"([a-z_]+mm[23]?)\s*(<=|>=)\s*(\d+(?:\.\d+)?)\s*(mm[23]?)(?!\w)"
    for match in re.finditer(pattern, lowered):
        attr, op, value, unit = match.groups()
        constraints.append({"attribute": attr, "op": op, "value": float(value), "unit": unit})
    remainder = re.sub(pattern, "", lowered)
    if re.search(r"\d", remainder):
        raise ValueError("Unparsed numeric requirement; use explicit attribute <= value unit syntax or query JSON")
    if not functions and not categories and not constraints:
        raise ValueError("No supported intent/category/constraint found; add reviewed aliases or use query JSON")
    return validate_query({"text": text, "function_ids": functions, "category_ids": categories,
                           "constraints": constraints, "parsed_from_text": True}, knowledge)
