"""Join local STEP files and legacy exports by relative-path SHA1, never filename alone."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_csv, sha256

CLASSES = ["bearing", "bolt-screw", "bracket", "coupling", "flange", "gear", "nut", "pulley", "shaft", "spring"]
FEATURE_KEYS = ["dim_small_large_ratio", "dim_mid_large_ratio", "bbox_fill_ratio",
                "surface_to_bbox_surface_ratio", "face_count", "edge_count", "solid_count",
                "face_count_frac_plane", "face_count_frac_cylinder", "face_count_frac_cone",
                "face_count_frac_torus", "face_count_frac_bspline", "edge_count_frac_circle"]


def canonical_class(label):
    label = label.strip().lower().replace("_", "-")
    if label not in CLASSES:
        raise ValueError(f"Unsupported class {label!r}")
    return label


def sample_id(relative):
    return hashlib.sha1(relative.replace("\\", "/").encode()).hexdigest()[:20]


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def inventory(root, legacy_manifest=None, per_class_split=None, seed=17):
    root = Path(root).resolve()
    previous = {}
    if legacy_manifest:
        for row in read_csv(legacy_manifest):
            sid = row["sample_id"]
            if sid in previous:
                raise ValueError(f"Duplicate legacy manifest ID {sid}")
            previous[sid] = row
    parts, counts = [], Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".stp", ".step"}:
            continue
        relative = path.relative_to(root).as_posix()
        label = canonical_class(relative.split("/")[0])
        sid = sample_id(relative)
        old = previous.get(sid)
        if old:
            if old["relative_path"].replace("\\", "/") != relative or canonical_class(old["label"]) != label:
                raise ValueError(f"Legacy manifest mismatch for {relative}")
            split = old["split"]
        else:
            bucket = int(hashlib.sha256(f"{seed}:{relative}".encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 70 else "val" if bucket < 85 else "test"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {sid}: {split}")
        counts[label] += 1
        parts.append({"part_id": sid, "relative_path": relative, "absolute_path": str(path),
                      "label": label, "split": split, "file_sha256": None,
                      "measurements": {}, "features": {}, "interfaces": [],
                      "prediction": None, "embedding": None})
    if not parts:
        raise ValueError(f"No STEP files in {root}")
    if per_class_split:
        seen = Counter()
        selected = []
        # Seeded hash ordering prevents always choosing the first indexed part.
        for part in sorted(parts, key=lambda p: hashlib.sha256(f"{seed}:{p['part_id']}".encode()).hexdigest()):
            key = part["label"], part["split"]
            if seen[key] < per_class_split:
                selected.append(part)
                seen[key] += 1
        parts = sorted(selected, key=lambda p: p["part_id"])
    return {"schema_version": 1, "dataset_root": str(root), "classes": CLASSES,
            "parts": parts, "audit": {"available_counts": dict(counts), "available_total": sum(counts.values()),
                                      "selected_total": len(parts), "split_source": str(legacy_manifest) if legacy_manifest else "path-hash; no family guarantee",
                                      "seed": seed, "per_class_split": per_class_split}, "artifacts": {}}


def verify_legacy_part(part, row):
    relative = row.get("relative_path", "").replace("\\", "/")
    if not relative or relative != part["relative_path"]:
        raise ValueError(f"Legacy relative path mismatch for {part['part_id']}")
    if row.get("label") and canonical_class(row["label"]) != part["label"]:
        raise ValueError("Legacy label mismatch")
    if row.get("split") and row["split"] != part["split"]:
        raise ValueError("Legacy split mismatch")
    old_path = Path(row["absolute_path"])
    if not old_path.is_file():
        raise ValueError(f"Cannot verify legacy file bytes: {old_path}")
    current = sha256(part["absolute_path"])
    if current != sha256(old_path):
        raise ValueError(f"Legacy STEP bytes differ: {part['relative_path']}")
    part["file_sha256"] = current


def import_descriptors(catalog, path):
    by_id = {p["part_id"]: p for p in catalog["parts"]}
    seen = set()
    for row in read_csv(path):
        sid = row["sample_id"]
        if sid not in by_id:
            continue
        if sid in seen:
            raise ValueError(f"Duplicate descriptor ID {sid}")
        seen.add(sid)
        part = by_id[sid]
        verify_legacy_part(part, row)
        if row.get("extract_error", "").strip():
            part["legacy_extract_error"] = row["extract_error"]
            continue
        part["features"] = {k: finite(row.get(k)) for k in FEATURE_KEYS}
        if len(seen) % 1000 == 0:
            print(f"Descriptor source verification: {len(seen)} parts", flush=True)
        # Units/backend not attested by this CSV. Preserve values without enabling mm checks.
        for key, field in [("aabb_x_mm", "bbox_x"), ("aabb_y_mm", "bbox_y"), ("aabb_z_mm", "bbox_z"),
                           ("volume_mm3", "volume"), ("area_mm2", "surface_area")]:
            value = finite(row.get(field))
            if value is not None:
                part["measurements"][key] = {"value": value, "unit": "unknown", "status": "imported_unverified",
                                             "source": str(Path(path).resolve()), "field": field}
    catalog["artifacts"]["descriptors"] = {"path": str(Path(path).resolve()), "sha256": sha256(path), "matched": len(seen)}


def import_embeddings(catalog, path):
    import numpy as np
    by_id = {p["part_id"]: p for p in catalog["parts"]}
    matched = set()
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"].astype(str).tolist()
            vectors = data["embeddings"]
            if len(ids) != len(vectors) or len(ids) != len(set(ids)):
                raise ValueError("Invalid or duplicate embedding IDs")
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError("Invalid embeddings")
            for sid, vector in zip(ids, vectors):
                if sid in by_id:
                    by_id[sid]["embedding"] = vector.tolist()
                    matched.add(sid)
    else:
        for row in read_csv(path):
            sid = row["sample_id"]
            if sid not in by_id:
                continue
            if sid in matched:
                raise ValueError(f"Duplicate embedding ID {sid}")
            part = by_id[sid]
            verify_legacy_part(part, row)
            keys = sorted(k for k in row if k.startswith("geom__"))
            vector = [finite(row[k]) for k in keys]
            if not vector or any(v is None for v in vector):
                raise ValueError(f"Invalid embedding {sid}")
            part["embedding"] = vector
            logit_keys = [f"geom_logit__{c}" for c in CLASSES]
            if all(k in row for k in logit_keys):
                logits = np.array([float(row[k]) for k in logit_keys])
                if not np.isfinite(logits).all():
                    raise ValueError(f"Invalid classifier logits {sid}")
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                part["prediction"] = {"probabilities": dict(zip(CLASSES, probs.tolist())), "status": "predicted",
                                      "source": str(Path(path).resolve()), "model": "legacy_supervised_local_surface_head",
                                      "calibration": "not_recalibrated"}
            matched.add(sid)
            if len(matched) % 1000 == 0:
                print(f"Embedding source verification: {len(matched)} parts", flush=True)
    if not matched:
        raise ValueError("No embedding IDs match local CAD; synthetic CAD-JEPA embeddings are not TMCAD")
    dims = {len(p["embedding"]) for p in by_id.values() if p["embedding"] is not None}
    if len(dims) != 1:
        raise ValueError("Inconsistent embedding dimensions")
    catalog["artifacts"]["embeddings"] = {"path": str(Path(path).resolve()), "sha256": sha256(path),
                                          "matched": len(matched), "dimension": dims.pop(),
                                          "npz_provenance": "ID_join_only; producer attestation required" if Path(path).suffix == ".npz" else "source STEP bytes verified"}


def import_predictions(catalog, path):
    by_id = {p["part_id"]: p for p in catalog["parts"]}
    seen = set()
    for row in read_csv(path):
        sid = row["sample_id"]
        if sid not in by_id:
            continue
        if sid in seen:
            raise ValueError(f"Duplicate prediction ID {sid}")
        seen.add(sid)
        part = by_id[sid]
        if row.get("relative_path") and row.get("absolute_path"):
            verify_legacy_part(part, row)
        elif row.get("split") and row["split"] != part["split"]:
            raise ValueError("Prediction split mismatch")
        keys = [f"prob__{c}" for c in CLASSES]
        if all(k in row for k in keys):
            probs = {c: float(row[k]) for c, k in zip(CLASSES, keys)}
            if any(not math.isfinite(v) or v < 0 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-4:
                raise ValueError("Invalid probabilities")
            uncertainty = "not_recalibrated"
        else:
            label = canonical_class(row["predicted_class"])
            probs = {c: float(c == label) for c in CLASSES}
            uncertainty = "hard_label_only; one-hot is encoding, not calibrated confidence"
        by_id[sid]["prediction"] = {"probabilities": probs, "status": "predicted", "source": str(Path(path).resolve()),
                                    "model": "imported_classifier", "calibration": uncertainty}
    if not seen:
        raise ValueError("No matching prediction IDs")
    catalog["artifacts"]["predictions"] = {"path": str(Path(path).resolve()), "sha256": sha256(path), "matched": len(seen),
                                            "provenance": "source STEP bytes checked when paths supplied; otherwise ID join only"}


def leakage_audit(catalog):
    groups = defaultdict(list)
    for part in catalog["parts"]:
        if part["file_sha256"]:
            groups[part["file_sha256"]].append(part)
    return {"exact_byte_cross_split_duplicates": [[p["part_id"] for p in group] for group in groups.values()
                                                if len({p["split"] for p in group}) > 1],
            "near_duplicate_family_audit": "not_available; human family IDs recommended",
            "hashed_parts": sum(len(g) for g in groups.values())}
