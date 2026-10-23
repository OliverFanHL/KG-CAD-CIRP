"""Authoritative IDs and feature recipes for the matched study matrix."""
from __future__ import annotations

from .dataset import TMCAD_CLASSES


# Each tuple names a feature block, not a separately fitted model. All linear
# probes use the same train/validation/test cohort and C grid.
CLASSIFICATION_PROBES = {
    "descriptor_only": ("descriptor",),
    "geometry_embedding": ("geometry",),
    "descriptor_plus_geometry": ("descriptor", "geometry"),
    "jev_choice_probabilities": ("jev_choice",),
    "geometry_plus_jev_choice_probabilities": ("geometry", "jev_choice"),
    "jev_geometry_questions": ("jev_geometry",),
    "jev_all_questions": ("jev_signature",),
    "geometry_plus_jev": ("geometry", "jev_signature"),
    "descriptor_plus_jev": ("descriptor", "jev_signature"),
    "geometry_plus_jev_geometry_questions": ("geometry", "jev_geometry"),
    "laya_choice_probabilities": ("laya_choice",),
    "geometry_plus_laya_choice_probabilities": ("geometry", "laya_choice"),
    "laya_geometry_questions": ("laya_geometry",),
    "laya_all_questions": ("laya_signature",),
    "geometry_plus_laya": ("geometry", "laya_signature"),
    "descriptor_plus_laya": ("descriptor", "laya_signature"),
    "geometry_plus_laya_geometry_questions": ("geometry", "laya_geometry"),
    "geometry_plus_jev_plus_laya": ("geometry", "jev_signature", "laya_signature"),
    "clm_choice_probabilities": ("clm_choice",),
    "geometry_plus_clm_choice_probabilities": ("geometry", "clm_choice"),
    "clm_geometry_questions": ("clm_geometry",),
    "clm_all_questions": ("clm_signature",),
    "geometry_plus_clm": ("geometry", "clm_signature"),
    "descriptor_plus_clm": ("descriptor", "clm_signature"),
    "geometry_plus_clm_geometry_questions": ("geometry", "clm_geometry"),
    "laya_ft_choice_probabilities": ("laya_ft_choice",),
    "geometry_plus_laya_ft_choice_probabilities": ("geometry", "laya_ft_choice"),
    "laya_ft_all_questions": ("laya_ft_signature",),
    "geometry_plus_laya_ft": ("geometry", "laya_ft_signature"),
}

CLASSIFICATION_DIRECT = {
    "geometry_encoder_head": "validation-selected geometry checkpoint logits",
    "jev_direct_choice": "typed JEV Choice answer",
    "laya_direct_choice": "typed Laya Choice answer",
    "laya_ft_direct_choice": "validation-selected fine-tuned Laya Choice answer",
    "clm_direct_choice": "official CLM typed Choice answer",
}

RETRIEVAL_METHODS = {
    "descriptor": ("descriptor",),
    "geometry": ("geometry",),
    "jev": ("jev_signature",),
    "laya": ("laya_signature",),
    "geometry_plus_jev": ("geometry", "jev_signature"),
    "geometry_plus_laya": ("geometry", "laya_signature"),
    "laya_ft": ("laya_ft_signature",),
    "geometry_plus_laya_ft": ("geometry", "laya_ft_signature"),
    "clm": ("clm_signature",),
    "geometry_plus_clm": ("geometry", "clm_signature"),
}


def feature_blocks(frame):
    """Resolve one explicit recipe against columns in the matched feature frame."""
    columns = list(frame)
    blocks = {
        "descriptor": [c for c in columns if c.startswith("desc__")],
        "geometry": [c for c in columns if c.startswith("geom__")],
    }
    for backend in ("jev", "laya", "laya_ft", "clm"):
        signature = [c for c in columns if c.startswith(f"{backend}__sig__")]
        choice = [f"{backend}__prob__{c}" for c in TMCAD_CLASSES]
        blocks[f"{backend}_signature"] = signature
        blocks[f"{backend}_geometry"] = [c for c in signature if "__geom__" in c]
        blocks[f"{backend}_choice"] = choice if all(c in columns for c in choice) else []
    return blocks


def classification_ids():
    return tuple(CLASSIFICATION_DIRECT) + tuple(CLASSIFICATION_PROBES)


def experiment_catalog():
    return {
        "classification": {
            **{name: {"estimator": "direct", "input": description}
               for name, description in CLASSIFICATION_DIRECT.items()},
            **{name: {"estimator": "validation-selected logistic regression",
                      "feature_blocks": list(recipe)}
               for name, recipe in CLASSIFICATION_PROBES.items()},
        },
        "retrieval": {
            name: {"input": list(recipe),
                   "ranking": "train-standardized cosine similarity" if len(recipe) == 1
                   else "validation-selected weighted geometry/semantic cosine similarity"}
            for name, recipe in RETRIEVAL_METHODS.items()
        },
    }
