from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


COUNT_FEATURES = ["face_count", "edge_count", "vertex_count", "solid_count", "shell_count"]
DISTRIBUTIONAL_FEATURES = [
    "best_rotational_symmetry",
    "best_axis_periodicity_strength",
    "best_axis_axial_variation",
    "best_axis_end_asymmetry",
    "bbox_fill_ratio",
    "surface_to_bbox_surface_ratio",
]


@dataclass
class QuantileSchema:
    cuts: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {"cuts": self.cuts}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "QuantileSchema":
        return cls(cuts={k: [float(x) for x in v] for k, v in obj["cuts"].items()})


def fit_quantile_schema(train_df: pd.DataFrame) -> QuantileSchema:
    cuts: dict[str, list[float]] = {}
    for feature in COUNT_FEATURES + DISTRIBUTIONAL_FEATURES:
        if feature not in train_df:
            continue
        values = pd.to_numeric(train_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        q = np.quantile(values, [0.10, 0.30, 0.70, 0.90]).astype(float)
        # Preserve non-decreasing cuts even for discrete/degenerate features.
        cuts[feature] = [float(x) for x in q]
    return QuantileSchema(cuts=cuts)


def _qcat(value: float, feature: str, schema: QuantileSchema) -> str:
    cuts = schema.cuts.get(feature)
    if not cuts:
        return "unknown"
    q10, q30, q70, q90 = cuts
    if abs(q90 - q10) < 1e-12:
        return "typical"
    if value < q10:
        return "very_low"
    if value < q30:
        return "low"
    if value <= q70:
        return "typical"
    if value <= q90:
        return "high"
    return "very_high"


def _fraction_cat(x: float) -> str:
    if x < 0.02:
        return "absent_or_negligible"
    if x < 0.15:
        return "minor"
    if x < 0.40:
        return "substantial"
    if x < 0.65:
        return "major"
    return "dominant"


def _ratio_cat(x: float) -> str:
    if x < 0.15:
        return "very_small"
    if x < 0.35:
        return "small"
    if x < 0.65:
        return "moderate"
    if x < 0.85:
        return "large"
    return "very_large"




def _axis_ratio_cat(x: float) -> str:
    if x < 0.5:
        return "much_smaller"
    if x < 0.8:
        return "smaller"
    if x < 1.25:
        return "comparable"
    if x < 2.5:
        return "larger"
    return "much_larger"

def _profile_cat(x: float) -> str:
    if x < 0.20:
        return "very_narrow"
    if x < 0.45:
        return "narrow"
    if x < 0.70:
        return "medium"
    if x < 0.90:
        return "wide"
    return "very_wide"


def _global_form(row: pd.Series) -> str:
    small_large = float(row.get("dim_small_large_ratio", 0.0))
    mid_large = float(row.get("dim_mid_large_ratio", 0.0))
    if mid_large < 0.38:
        return "strongly_elongated"
    if mid_large < 0.58:
        return "moderately_elongated"
    if small_large < 0.18 and mid_large > 0.60:
        return "thin_disk_or_plate_like"
    if small_large > 0.68 and mid_large > 0.78:
        return "compact_three_dimensional"
    return "intermediate_proportions"


def _load_profile(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        try:
            return [float(x) for x in json.loads(value)]
        except Exception:
            return []
    return []


def build_semantic_state(
    row: pd.Series,
    schema: QuantileSchema,
    include_numeric_context: bool = True,
    include_profile_arrays: bool = True,
) -> dict[str, Any]:
    """Build a label-free Jev state from deterministic CAD descriptors.

    IMPORTANT: This function deliberately excludes sample_id, path, filename, folder,
    class label, and split. Ground-truth metadata must never appear in the model state.
    """
    plane = float(row.get("face_area_frac_plane", 0.0))
    cylinder = float(row.get("face_area_frac_cylinder", 0.0))
    cone = float(row.get("face_area_frac_cone", 0.0))
    torus = float(row.get("face_area_frac_torus", 0.0))
    bspline = float(row.get("face_area_frac_bspline", 0.0))
    circle_edge = float(row.get("edge_length_frac_circle", 0.0))
    line_edge = float(row.get("edge_length_frac_line", 0.0))

    angular = _load_profile(row.get("best_axis_angular_profile", []))
    axial = _load_profile(row.get("best_axis_axial_profile", []))

    state: dict[str, Any] = {
        "object": "one unlabeled mechanical CAD part represented by deterministic geometry descriptors",
        "global_geometry": {
            "principal_extent_form": _global_form(row),
            "small_to_large_extent": _ratio_cat(float(row.get("dim_small_large_ratio", 0.0))),
            "middle_to_large_extent": _ratio_cat(float(row.get("dim_mid_large_ratio", 0.0))),
            "pca_secondary_extent": _ratio_cat(float(np.sqrt(max(float(row.get("pca_eig_ratio_1", 0.0)), 0.0)))),
            "pca_tertiary_extent": _ratio_cat(float(np.sqrt(max(float(row.get("pca_eig_ratio_2", 0.0)), 0.0)))),
            "bounding_volume_fill": _qcat(float(row.get("bbox_fill_ratio", 0.0)), "bbox_fill_ratio", schema),
        },
        "topology": {
            "face_count_level": _qcat(float(row.get("face_count", 0.0)), "face_count", schema),
            "edge_count_level": _qcat(float(row.get("edge_count", 0.0)), "edge_count", schema),
            "vertex_count_level": _qcat(float(row.get("vertex_count", 0.0)), "vertex_count", schema),
            "solid_count_level": _qcat(float(row.get("solid_count", 0.0)), "solid_count", schema),
            "shell_count_level": _qcat(float(row.get("shell_count", 0.0)), "shell_count", schema),
            "face_area_distribution": (
                "concentrated" if float(row.get("largest_face_area_fraction", 0.0)) > 0.45
                else "distributed"
            ),
        },
        "surface_character": {
            "planar_area": _fraction_cat(plane),
            "cylindrical_area": _fraction_cat(cylinder),
            "conical_area": _fraction_cat(cone),
            "toroidal_area": _fraction_cat(torus),
            "freeform_bspline_area": _fraction_cat(bspline),
            "straight_edge_length": _fraction_cat(line_edge),
            "circular_edge_length": _fraction_cat(circle_edge),
        },
        "principal_axis_profile": {
            "rotational_symmetry_level": _qcat(
                float(row.get("best_rotational_symmetry", 0.0)), "best_rotational_symmetry", schema
            ),
            "circumferential_periodicity_level": _qcat(
                float(row.get("best_axis_periodicity_strength", 0.0)), "best_axis_periodicity_strength", schema
            ),
            "dominant_periodicity_order": int(round(float(row.get("best_axis_periodicity_order", 0.0)))),
            "axial_radius_variation": _qcat(
                float(row.get("best_axis_axial_variation", 0.0)), "best_axis_axial_variation", schema
            ),
            "end_to_end_asymmetry": _qcat(
                float(row.get("best_axis_end_asymmetry", 0.0)), "best_axis_end_asymmetry", schema
            ),
            "axis_extent_relative_to_transverse_extent": _axis_ratio_cat(float(row.get("best_axis_elongation", 0.0))),
            "transverse_extent_relative_to_axis_extent": _axis_ratio_cat(float(row.get("best_axis_discness", 0.0))),
        },
    }

    if include_profile_arrays:
        state["principal_axis_profile"]["normalized_angular_radius_pattern"] = [_profile_cat(x) for x in angular]
        state["principal_axis_profile"]["normalized_axial_radius_pattern"] = [_profile_cat(x) for x in axial]

    if include_numeric_context:
        # These are scale-invariant or count facts only. They are supporting evidence;
        # semantic categories above remain the primary representation for Jev.
        state["selected_numeric_facts"] = {
            "solid_count": int(round(float(row.get("solid_count", 0)))),
            "face_count": int(round(float(row.get("face_count", 0)))),
            "edge_count": int(round(float(row.get("edge_count", 0)))),
            "small_to_large_extent_ratio": round(float(row.get("dim_small_large_ratio", 0.0)), 4),
            "middle_to_large_extent_ratio": round(float(row.get("dim_mid_large_ratio", 0.0)), 4),
            "bounding_volume_fill_ratio": round(float(row.get("bbox_fill_ratio", 0.0)), 4),
            "planar_area_fraction": round(plane, 4),
            "cylindrical_area_fraction": round(cylinder, 4),
            "circular_edge_length_fraction": round(circle_edge, 4),
            "rotational_symmetry_score": round(float(row.get("best_rotational_symmetry", 0.0)), 4),
            "periodicity_strength": round(float(row.get("best_axis_periodicity_strength", 0.0)), 4),
            "periodicity_order": int(round(float(row.get("best_axis_periodicity_order", 0.0)))),
            "axial_profile_variation": round(float(row.get("best_axis_axial_variation", 0.0)), 4),
            "end_asymmetry": round(float(row.get("best_axis_end_asymmetry", 0.0)), 4),
        }

    return state


def build_compact_semantic_state(row: pd.Series, schema: QuantileSchema, **_: Any) -> dict[str, Any]:
    """Short, label-free evidence for bounded-context typed-decision models."""
    fields = {
        "extent_small_large": "dim_small_large_ratio",
        "extent_mid_large": "dim_mid_large_ratio",
        "fill": "bbox_fill_ratio",
        "face_count": "face_count",
        "solid_count": "solid_count",
        "plane_area": "face_area_frac_plane",
        "cylinder_area": "face_area_frac_cylinder",
        "circle_edge": "edge_length_frac_circle",
        "rotation_symmetry": "best_rotational_symmetry",
        "periodicity_strength": "best_axis_periodicity_strength",
        "periodicity_order": "best_axis_periodicity_order",
        "axial_variation": "best_axis_axial_variation",
        "end_asymmetry": "best_axis_end_asymmetry",
        "axis_elongation": "best_axis_elongation",
    }
    facts = {name: round(float(row.get(column, 0.0)), 3) for name, column in fields.items()}
    return {"object": "unlabeled mechanical CAD part", "geometry": facts}
