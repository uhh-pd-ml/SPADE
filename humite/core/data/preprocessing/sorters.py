from typing import Sequence

import awkward as ak
import numpy as np


def _select_longitudinal_field(spatial_fields: Sequence[str]) -> str:
    if "layer" in spatial_fields:
        return "layer"
    if len(spatial_fields) >= 3:
        return spatial_fields[2]
    return spatial_fields[-1]


def sort_showers(
    data_showers,
    mode: str = "energy",
    logger=None,
    spatial_fields: Sequence[str] = ("x", "y", "z"),
    interaction_point: Sequence[float] = (14.5, 14.5, 0.0),
):
    longitudinal_field = _select_longitudinal_field(spatial_fields)
    reference_field = spatial_fields[0]
    if mode == "energy":
        if logger:
            logger.info("Sorting by energy")
        return data_showers[ak.argsort(data_showers.energy, axis=1, ascending=False)]

    elif mode == "layer":
        return data_showers[ak.argsort(data_showers[longitudinal_field], axis=1, ascending=True)]

    elif mode == "layer_energy":
        idx_energy = ak.argsort(data_showers.energy, axis=1, ascending=False, stable=True)
        data_showers = data_showers[idx_energy]
        idx_z = ak.argsort(data_showers[longitudinal_field], axis=1, ascending=True, stable=True)
        return data_showers[idx_z]

    elif mode == "layer_desc_energy":
        idx_energy = ak.argsort(data_showers.energy, axis=1, ascending=False, stable=True)
        data_showers = data_showers[idx_energy]
        idx_z = ak.argsort(data_showers[longitudinal_field], axis=1, ascending=False, stable=True)
        return data_showers[idx_z]

    elif mode == "layer_random":
        lengths = ak.num(data_showers[reference_field], axis=1)
        rand = ak.Array([np.random.uniform(size=int(n)) for n in lengths])
        idx_rand = ak.argsort(rand, axis=1, stable=True)
        data_showers = data_showers[idx_rand]
        idx_z = ak.argsort(data_showers[longitudinal_field], axis=1, ascending=True, stable=True)
        return data_showers[idx_z]

    elif mode == "random":
        lengths = ak.num(data_showers[reference_field], axis=1)
        rand = ak.Array([np.random.uniform(size=int(n)) for n in lengths])
        return data_showers[ak.argsort(rand, axis=1)]

    elif mode == "layer_x":
        idx_x = ak.argsort(data_showers[spatial_fields[0]], axis=1, ascending=True, stable=True)
        data_showers = data_showers[idx_x]
        idx_z = ak.argsort(data_showers[spatial_fields[2]], axis=1, ascending=True, stable=True)
        return data_showers[idx_z]

    elif mode == "radius_from_ip":
        ip_x, ip_y, ip_z = interaction_point
        dx = data_showers[spatial_fields[0]] - ip_x
        dy = data_showers[spatial_fields[1]] - ip_y
        dz = data_showers[spatial_fields[2]] - ip_z
        radial_distance_sq = dx * dx + dy * dy + dz * dz
        idx_radius = ak.argsort(radial_distance_sq, axis=1, ascending=True, stable=True)
        return data_showers[idx_radius]

    # First sort by layer, then by distance from IP
    elif mode == "layer_radius_from_ip":
        ip_x, ip_y, ip_z = interaction_point
        dx = data_showers[spatial_fields[0]] - ip_x
        dy = data_showers[spatial_fields[1]] - ip_y
        dz = data_showers[spatial_fields[2]] - ip_z
        radial_distance_sq = dx * dx + dy * dy + dz * dz
        idx_radial = ak.argsort(radial_distance_sq, axis=1, ascending=True, stable=True)
        data_showers = data_showers[idx_radial]
        idx_layer = ak.argsort(
            data_showers[spatial_fields[2]], axis=1, ascending=True, stable=True
        )
        return data_showers[idx_layer]

    elif mode == "none":
        return data_showers

    else:
        raise ValueError(f"Unknown sequence sorting: {mode}")
