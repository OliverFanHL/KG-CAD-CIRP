from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .utils import normalized_entropy, safe_div

LOGGER = logging.getLogger(__name__)

FACE_TYPES = ["PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS", "BSPLINE", "BEZIER", "REVOLUTION", "EXTRUSION", "OFFSET"]
EDGE_TYPES = ["LINE", "CIRCLE", "ELLIPSE", "BSPLINE", "BEZIER", "HYPERBOLA", "PARABOLA", "OFFSET"]


def _load_shape(path: str | Path):
    try:
        import cadquery as cq
    except ImportError as exc:
        raise ImportError(
            "CadQuery is required for STEP parsing. Install with conda/mamba (recommended) "
            "or `pip install cadquery`."
        ) from exc

    wp = cq.importers.importStep(str(path))
    vals = wp.vals()
    if not vals:
        raise RuntimeError(f"CadQuery imported no shapes from {path}")
    if len(vals) == 1:
        return vals[0]
    try:
        return cq.Compound.makeCompound(vals)
    except Exception:
        # importStep commonly returns a compound as the first object already.
        return wp.val()


def _round_list(values: np.ndarray | list[float], ndigits: int = 5) -> list[float]:
    return [round(float(x), ndigits) for x in values]


def _fraction_map(values: dict[str, float], denom: float, names: list[str], prefix: str) -> dict[str, float]:
    out = {}
    used = 0.0
    for name in names:
        val = float(values.get(name, 0.0))
        used += val
        out[f"{prefix}_{name.lower()}"] = safe_div(val, denom)
    out[f"{prefix}_other"] = max(0.0, safe_div(max(0.0, denom - used), denom))
    return out


def _geom_type_name(obj: Any) -> str:
    try:
        value = obj.geomType()
        return str(value).upper()
    except Exception:
        return "OTHER"


def _tessellate(shape: Any, relative_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    bb = shape.BoundingBox()
    max_extent = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6)
    tol = max(max_extent * float(relative_tolerance), 1e-5)
    vertices, triangles = shape.tessellate(tol)
    xyz = np.asarray([v.toTuple() for v in vertices], dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[0] < 4:
        raise RuntimeError("Too few tessellated vertices for shape-profile analysis")
    return xyz, tri


def _surface_sample_points(xyz: np.ndarray, tri: np.ndarray, n_samples: int = 8192) -> np.ndarray:
    """Deterministically sample approximately uniform points over the triangulated surface.

    Using only tessellation vertices badly overestimates rotational symmetry for flat
    low-poly shapes (for example, a box has only corner vertices). Area-weighted
    barycentric sampling provides a much denser and more faithful angular/axial profile
    while remaining label-free and deterministic.
    """
    if tri.ndim != 2 or tri.shape[0] == 0:
        return xyz
    polys = xyz[tri]
    cross = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return xyz
    probs = areas / total
    rng = np.random.default_rng(0)
    n = int(max(2048, min(n_samples, max(2048, tri.shape[0] * 16))))
    chosen = rng.choice(tri.shape[0], size=n, replace=True, p=probs)
    t = polys[chosen]
    u = rng.random(n)
    v = rng.random(n)
    su = np.sqrt(u)
    a = 1.0 - su
    b = su * (1.0 - v)
    c = su * v
    samples = a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2]
    # Include original mesh vertices so sharp extrema are preserved.
    if xyz.shape[0] <= 20000:
        samples = np.vstack([samples, xyz])
    return samples


def _profile_for_axis(
    centered: np.ndarray,
    basis: np.ndarray,
    axis_idx: int,
    angular_bins: int,
    axial_bins: int,
) -> dict[str, Any]:
    axis = basis[:, axis_idx]
    others = [i for i in range(3) if i != axis_idx]
    u_axis = basis[:, others[0]]
    v_axis = basis[:, others[1]]

    z = centered @ axis
    u = centered @ u_axis
    v = centered @ v_axis
    r = np.sqrt(u * u + v * v)
    theta = np.mod(np.arctan2(v, u), 2 * np.pi)

    z_extent = max(float(np.ptp(z)), 1e-12)
    u_extent = max(float(np.ptp(u)), 1e-12)
    v_extent = max(float(np.ptp(v)), 1e-12)
    radial_extent = 0.5 * (u_extent + v_extent)

    angular_edges = np.linspace(0.0, 2 * np.pi, angular_bins + 1)
    angular_profile = np.zeros(angular_bins, dtype=float)
    global_r = float(np.percentile(r, 95)) if r.size else 1.0
    for i in range(angular_bins):
        mask = (theta >= angular_edges[i]) & (theta < angular_edges[i + 1])
        angular_profile[i] = float(np.percentile(r[mask], 95)) if np.any(mask) else global_r
    angular_profile /= max(float(np.mean(angular_profile)), 1e-12)
    angular_cv = float(np.std(angular_profile))
    rotational_symmetry = float(np.clip(1.0 - angular_cv, 0.0, 1.0))

    centered_profile = angular_profile - float(np.mean(angular_profile))
    spectrum = np.abs(np.fft.rfft(centered_profile))
    max_order = min(32, max(2, spectrum.size - 1))
    if max_order >= 2 and spectrum[1 : max_order + 1].sum() > 1e-12:
        segment = spectrum[2 : max_order + 1]
        if segment.size:
            rel_idx = int(np.argmax(segment))
            periodicity_order = rel_idx + 2
            periodicity_strength = float(segment[rel_idx] / (spectrum[1 : max_order + 1].sum() + 1e-12))
        else:
            periodicity_order = 0
            periodicity_strength = 0.0
    else:
        periodicity_order = 0
        periodicity_strength = 0.0

    z_edges = np.linspace(float(np.min(z)), float(np.max(z)), axial_bins + 1)
    axial_profile = np.zeros(axial_bins, dtype=float)
    for i in range(axial_bins):
        if i == axial_bins - 1:
            mask = (z >= z_edges[i]) & (z <= z_edges[i + 1])
        else:
            mask = (z >= z_edges[i]) & (z < z_edges[i + 1])
        axial_profile[i] = float(np.percentile(r[mask], 95)) if np.any(mask) else 0.0
    max_axial = max(float(np.max(axial_profile)), 1e-12)
    axial_norm = axial_profile / max_axial
    axial_variation = float(np.std(axial_norm))
    end_n = max(1, axial_bins // 6)
    end_asymmetry = float(abs(np.mean(axial_norm[:end_n]) - np.mean(axial_norm[-end_n:])))

    return {
        "axis_idx": int(axis_idx),
        "axis_extent": z_extent,
        "transverse_extent_mean": radial_extent,
        "elongation": safe_div(z_extent, radial_extent),
        "discness": safe_div(radial_extent, z_extent),
        "rotational_symmetry": rotational_symmetry,
        "angular_cv": angular_cv,
        "periodicity_order": int(periodicity_order),
        "periodicity_strength": periodicity_strength,
        "axial_variation": axial_variation,
        "end_asymmetry": end_asymmetry,
        "angular_profile": _round_list(angular_profile),
        "axial_profile": _round_list(axial_norm),
    }


def _pca_profiles(
    xyz: np.ndarray,
    angular_bins: int,
    axial_bins: int,
) -> dict[str, Any]:
    center = np.mean(xyz, axis=0)
    centered = xyz - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    basis = eigvecs[:, order]
    eig_norm = eigvals / max(float(eigvals[0]), 1e-12)

    profiles = [
        _profile_for_axis(centered, basis, axis_idx=i, angular_bins=angular_bins, axial_bins=axial_bins)
        for i in range(3)
    ]
    # Prefer high rotational symmetry; use a small tie-breaker favoring an axis
    # whose two transverse extents are more alike.
    scores = []
    for p in profiles:
        i = p["axis_idx"]
        other = [j for j in range(3) if j != i]
        a = math.sqrt(max(float(eigvals[other[0]]), 1e-12))
        b = math.sqrt(max(float(eigvals[other[1]]), 1e-12))
        transverse_similarity = min(a, b) / max(a, b)
        scores.append(p["rotational_symmetry"] + 0.15 * transverse_similarity)
    best = profiles[int(np.argmax(scores))]

    return {
        "pca_eig_ratio_1": float(eig_norm[1]) if eig_norm.size > 1 else 0.0,
        "pca_eig_ratio_2": float(eig_norm[2]) if eig_norm.size > 2 else 0.0,
        "best_rot_axis": int(best["axis_idx"]),
        "best_rotational_symmetry": float(best["rotational_symmetry"]),
        "best_axis_elongation": float(best["elongation"]),
        "best_axis_discness": float(best["discness"]),
        "best_axis_periodicity_order": int(best["periodicity_order"]),
        "best_axis_periodicity_strength": float(best["periodicity_strength"]),
        "best_axis_axial_variation": float(best["axial_variation"]),
        "best_axis_end_asymmetry": float(best["end_asymmetry"]),
        "best_axis_angular_profile": best["angular_profile"],
        "best_axis_axial_profile": best["axial_profile"],
    }


def extract_descriptor(
    step_path: str | Path,
    relative_tolerance: float = 0.005,
    angular_bins: int = 72,
    axial_bins: int = 12,
) -> dict[str, Any]:
    """Extract label-free deterministic B-rep/mesh descriptors from one STEP file.

    The output intentionally does not include the filename, directory, or class label.
    Those fields are kept in the manifest only and never sent to Jev.
    """
    shape = _load_shape(step_path)
    bb = shape.BoundingBox()
    xlen, ylen, zlen = float(bb.xlen), float(bb.ylen), float(bb.zlen)
    dims = np.sort(np.asarray([xlen, ylen, zlen], dtype=float))
    d0, d1, d2 = [max(float(x), 1e-12) for x in dims]

    solids = list(shape.Solids())
    shells = list(shape.Shells())
    faces = list(shape.Faces())
    edges = list(shape.Edges())
    vertices = list(shape.Vertices())

    volumes = []
    for solid in solids:
        try:
            volumes.append(max(float(solid.Volume()), 0.0))
        except Exception:
            pass
    volume = float(sum(volumes))

    face_area_by_type: dict[str, float] = {}
    face_count_by_type: dict[str, float] = {}
    face_areas: list[float] = []
    for face in faces:
        t = _geom_type_name(face)
        try:
            area = max(float(face.Area()), 0.0)
        except Exception:
            area = 0.0
        face_areas.append(area)
        face_area_by_type[t] = face_area_by_type.get(t, 0.0) + area
        face_count_by_type[t] = face_count_by_type.get(t, 0.0) + 1.0
    surface_area = float(sum(face_areas))

    edge_length_by_type: dict[str, float] = {}
    edge_count_by_type: dict[str, float] = {}
    edge_lengths: list[float] = []
    for edge in edges:
        t = _geom_type_name(edge)
        try:
            length = max(float(edge.Length()), 0.0)
        except Exception:
            length = 0.0
        edge_lengths.append(length)
        edge_length_by_type[t] = edge_length_by_type.get(t, 0.0) + length
        edge_count_by_type[t] = edge_count_by_type.get(t, 0.0) + 1.0
    total_edge_length = float(sum(edge_lengths))

    xyz, triangles = _tessellate(shape, relative_tolerance)
    profile_points = _surface_sample_points(xyz, triangles)
    pca = _pca_profiles(profile_points, angular_bins=angular_bins, axial_bins=axial_bins)

    bbox_volume = xlen * ylen * zlen
    bbox_surface = 2.0 * (xlen * ylen + xlen * zlen + ylen * zlen)

    out: dict[str, Any] = {
        "bbox_x": xlen,
        "bbox_y": ylen,
        "bbox_z": zlen,
        "dim_small": d0,
        "dim_mid": d1,
        "dim_large": d2,
        "dim_small_large_ratio": safe_div(d0, d2),
        "dim_mid_large_ratio": safe_div(d1, d2),
        "dim_small_mid_ratio": safe_div(d0, d1),
        "volume": volume,
        "surface_area": surface_area,
        "bbox_fill_ratio": float(np.clip(safe_div(volume, bbox_volume), 0.0, 1.0)) if bbox_volume > 0 else 0.0,
        "surface_to_bbox_surface_ratio": safe_div(surface_area, bbox_surface),
        "solid_count": len(solids),
        "shell_count": len(shells),
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "mesh_vertex_count": int(xyz.shape[0]),
        "mesh_triangle_count": int(len(triangles)),
        "face_area_entropy": normalized_entropy(face_areas),
        "edge_length_entropy": normalized_entropy(edge_lengths),
        "largest_face_area_fraction": safe_div(max(face_areas) if face_areas else 0.0, surface_area),
        "largest_edge_length_fraction": safe_div(max(edge_lengths) if edge_lengths else 0.0, total_edge_length),
    }
    out.update(_fraction_map(face_count_by_type, max(float(len(faces)), 1.0), FACE_TYPES, "face_count_frac"))
    out.update(_fraction_map(face_area_by_type, max(surface_area, 1e-12), FACE_TYPES, "face_area_frac"))
    out.update(_fraction_map(edge_count_by_type, max(float(len(edges)), 1.0), EDGE_TYPES, "edge_count_frac"))
    out.update(_fraction_map(edge_length_by_type, max(total_edge_length, 1e-12), EDGE_TYPES, "edge_length_frac"))
    out.update(pca)

    # Compact summaries are useful for JSON state construction and debugging.
    out["surface_type_counts_json"] = json.dumps({k: int(v) for k, v in sorted(face_count_by_type.items())})
    out["edge_type_counts_json"] = json.dumps({k: int(v) for k, v in sorted(edge_count_by_type.items())})
    return out


def numeric_feature_columns(df_columns: list[str]) -> list[str]:
    excluded = {
        "sample_id",
        "label",
        "split",
        "relative_path",
        "absolute_path",
        "extract_error",
        "descriptor_seconds",
        "surface_type_counts_json",
        "edge_type_counts_json",
        "best_axis_angular_profile",
        "best_axis_axial_profile",
    }
    return [c for c in df_columns if c not in excluded]
