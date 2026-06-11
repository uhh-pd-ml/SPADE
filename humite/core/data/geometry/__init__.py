from __future__ import annotations

from typing import Callable, Dict, Tuple

import awkward as ak

from .ecal_cartesian import standardize_ecal_cartesian

GeometryStandardizer = Callable[[ak.Array], ak.Array]
GeometrySpatialFields = Tuple[str, str, str]

GEOMETRY_STANDARDIZERS: Dict[str, GeometryStandardizer] = {
    "ECAL": standardize_ecal_cartesian,
}

GEOMETRY_SPATIAL_FIELDS: Dict[str, GeometrySpatialFields] = {
    "ECAL": ("x", "y", "z"),
}

# Per-axis vocabulary sizes are taken from the dataset config (nbins_*) for ECAL,
# so no static geometry codebook is needed here.
GEOMETRY_SPATIAL_CODEBOOK_SIZES: Dict[str, Dict[str, int]] = {}


def get_spatial_field_names(dataset_type: str) -> GeometrySpatialFields:
    try:
        return GEOMETRY_SPATIAL_FIELDS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_type: {dataset_type}") from exc


def get_spatial_codebook_sizes(dataset_type: str) -> Dict[str, int]:
    return GEOMETRY_SPATIAL_CODEBOOK_SIZES.get(dataset_type, {})


__all__ = [
    "GEOMETRY_STANDARDIZERS",
    "GEOMETRY_SPATIAL_FIELDS",
    "GeometryStandardizer",
    "GeometrySpatialFields",
    "standardize_ecal_cartesian",
    "get_spatial_field_names",
    "get_spatial_codebook_sizes",
]
