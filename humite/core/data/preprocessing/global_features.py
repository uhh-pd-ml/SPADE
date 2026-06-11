"""Single source of truth for global shower observables from point clouds."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import awkward as ak
import numpy as np


def _weighted_std(values: ak.Array, weights: ak.Array, weighted_mean: ak.Array) -> ak.Array:
    denom = ak.sum(weights, axis=-1)
    safe = ak.where(denom > 0, denom, 1.0)
    second_moment = ak.sum(weights * values * values, axis=-1) / safe
    variance = second_moment - weighted_mean * weighted_mean
    variance = ak.values_astype(variance, np.float64)
    variance = ak.where(variance > 0, variance, 0.0)
    return np.sqrt(variance)


def _layer_index(z: ak.Array, n_layers: int, z_max: Optional[float]) -> ak.Array:
    z_max_val = float(z_max if z_max is not None else n_layers)
    scaled = ak.values_astype(z, np.float64) * n_layers / z_max_val
    idx = ak.values_astype(np.floor(scaled), np.int64)
    idx = ak.where(idx >= 0, idx, 0)
    return ak.where(idx <= n_layers - 1, idx, n_layers - 1)


def _layer_observables(
    x: ak.Array, y: ak.Array, energy: ak.Array, layer_idx: ak.Array, n_layers: int
) -> tuple[np.ndarray, np.ndarray]:
    n_evt = len(x)
    e_layer = np.zeros((n_evt, n_layers), dtype=np.float64)
    sigma_r_layer = np.zeros((n_evt, n_layers), dtype=np.float64)
    for lid in range(n_layers):
        mask = layer_idx == lid
        e = ak.where(mask, energy, 0.0)
        e_sum = ak.sum(e, axis=-1)
        safe = ak.where(e_sum > 0, e_sum, 1.0)
        mx = ak.sum(e * x, axis=-1) / safe
        my = ak.sum(e * y, axis=-1) / safe
        vx = ak.sum(e * x * x, axis=-1) / safe - mx * mx
        vy = ak.sum(e * y * y, axis=-1) / safe - my * my
        vr = ak.values_astype(vx + vy, np.float64)
        vr = ak.where(vr > 0, vr, 0.0)
        e_layer[:, lid] = np.asarray(ak.fill_none(e_sum, 0.0))
        sigma_r_layer[:, lid] = np.asarray(np.sqrt(vr))
    return e_layer, sigma_r_layer


def _radial_containment(
    x: ak.Array, y: ak.Array, energy: ak.Array, cog_x: ak.Array, cog_y: ak.Array, e_total: ak.Array
) -> tuple[ak.Array, ak.Array]:
    cog_x_col = ak.Array(np.asarray(ak.fill_none(cog_x, 0.0), dtype=np.float64)[:, None])
    cog_y_col = ak.Array(np.asarray(ak.fill_none(cog_y, 0.0), dtype=np.float64)[:, None])
    r = np.sqrt((x - cog_x_col) ** 2 + (y - cog_y_col) ** 2)
    order = ak.argsort(r, axis=-1)
    r_sorted = r[order]
    e_sorted = energy[order]
    max_hits = int(ak.max(ak.num(e_sorted, axis=-1), axis=0, initial=0))
    r_pad = ak.fill_none(ak.pad_none(r_sorted, max_hits, axis=-1, clip=True), 0.0)
    e_pad = ak.fill_none(ak.pad_none(e_sorted, max_hits, axis=-1, clip=True), 0.0)
    r_np = np.asarray(ak.to_numpy(r_pad), dtype=np.float64)
    e_np = np.asarray(ak.to_numpy(e_pad), dtype=np.float64)
    e_total_np = np.asarray(ak.fill_none(e_total, 1.0), dtype=np.float64)
    cum_frac = np.cumsum(e_np, axis=-1) / np.maximum(e_total_np[:, None], 1e-12)
    n_evt = r_np.shape[0]

    if r_np.shape[1] == 0:
        return ak.Array(np.zeros(n_evt, dtype=np.float64)), ak.Array(
            np.zeros(n_evt, dtype=np.float64)
        )

    idx = np.arange(n_evt, dtype=np.int64)
    r50_idx = np.argmax(cum_frac >= 0.5, axis=-1)
    r90_idx = np.argmax(cum_frac >= 0.9, axis=-1)
    r50 = np.where(np.any(cum_frac >= 0.5, axis=-1), r_np[idx, r50_idx], 0.0)
    r90 = np.where(np.any(cum_frac >= 0.9, axis=-1), r_np[idx, r90_idx], 0.0)
    return ak.Array(np.ascontiguousarray(r50)), ak.Array(np.ascontiguousarray(r90))


def compute_global_features(
    ak_points: ak.Array,
    incident_energy: Optional[Union[np.ndarray, ak.Array]] = None,
    energy_threshold: float = 0.0,
    n_layers: int = 48,
    z_max: Optional[float] = None,
    hit_thresholds: Optional[Sequence[float]] = None,
    include_layer_profile: bool = True,
) -> ak.Array:
    """
    Compute global shower observables from point cloud.

    Args:
        ak_points: Awkward array with fields x, y, z, energy (per-hit).
        incident_energy: Incident particle energy [GeV]. If None, incident_energy and
            energy_sum_over_incident_energy are omitted from the output.
        energy_threshold: Minimum hit energy [MeV] for feature computation.
        n_layers: Number of longitudinal layers for E_layer, sigma_r_layer.
        z_max: Max z for layer mapping. Default: n_layers.
        hit_thresholds: Extra n_hits_E>X counts, e.g. [0.1] for n_hits_01.
        include_layer_profile: If True, compute E_layer and sigma_r_layer.

    Returns:
        ak.Array with scalar and optional per-layer global features.
    """
    if len(ak_points) == 0:
        return ak.Array([])

    x = ak.values_astype(ak_points.x, np.float64)
    y = ak.values_astype(ak_points.y, np.float64)
    z = ak.values_astype(ak_points.z, np.float64)
    raw_e = ak_points.energy
    e = ak.where(raw_e > energy_threshold, ak.values_astype(raw_e, np.float64), 0.0)

    e_total = ak.sum(e, axis=-1)
    safe_e = ak.where(e_total > 0, e_total, 1.0)
    n_hits = ak.sum(e > 0.0, axis=-1)

    cog_x = ak.sum(e * x, axis=-1) / safe_e
    cog_y = ak.sum(e * y, axis=-1) / safe_e
    cog_z = ak.sum(e * z, axis=-1) / safe_e

    sigma_x = _weighted_std(x, e, cog_x)
    sigma_y = _weighted_std(y, e, cog_y)
    sigma_z = _weighted_std(z, e, cog_z)

    cog_x_bc = ak.broadcast_arrays(x, cog_x)[1]
    cog_y_bc = ak.broadcast_arrays(y, cog_y)[1]
    radial_ak = np.sqrt((x - cog_x_bc) ** 2 + (y - cog_y_bc) ** 2)
    radial_mean = ak.sum(e * radial_ak, axis=-1) / safe_e
    sigma_r = _weighted_std(radial_ak, e, radial_mean)

    e_per_hit = safe_e / ak.where(n_hits > 0, n_hits, 1.0)
    e_max = ak.max(e, axis=-1)
    e_max_frac = e_max / safe_e

    depth_max = ak.values_astype(
        ak.fill_none(ak.firsts(z[ak.argmax(raw_e, axis=-1, keepdims=True)]), 0.0), np.float64
    )
    z_span = ak.max(z, axis=-1) - ak.min(z, axis=-1)
    z_span = ak.where(ak.num(z) > 0, z_span, 0.0)

    out: dict[str, ak.Array] = {
        "n_hits": ak.values_astype(ak.fill_none(n_hits, 0.0), np.float64),
        "energy_sum": ak.values_astype(ak.fill_none(e_total, 0.0), np.float64),
        "E_total": ak.values_astype(ak.fill_none(e_total, 0.0), np.float64),
        "cog_x": ak.values_astype(ak.fill_none(cog_x, 0.0), np.float64),
        "cog_y": ak.values_astype(ak.fill_none(cog_y, 0.0), np.float64),
        "cog_z": ak.values_astype(ak.fill_none(cog_z, 0.0), np.float64),
        "sigma_x": ak.values_astype(ak.fill_none(sigma_x, 0.0), np.float64),
        "sigma_y": ak.values_astype(ak.fill_none(sigma_y, 0.0), np.float64),
        "sigma_z": ak.values_astype(ak.fill_none(sigma_z, 0.0), np.float64),
        "sigma_r": ak.values_astype(ak.fill_none(sigma_r, 0.0), np.float64),
        "E_total_per_hit": ak.values_astype(ak.fill_none(e_per_hit, 0.0), np.float64),
        "E_max_over_E_total": ak.values_astype(ak.fill_none(e_max_frac, 0.0), np.float64),
        "depth_of_shower_max": ak.values_astype(ak.fill_none(depth_max, 0.0), np.float64),
        "z_span": ak.values_astype(ak.fill_none(z_span, 0.0), np.float64),
    }
    if incident_energy is not None:
        inc = incident_energy
        if isinstance(inc, ak.Array):
            inc = np.asarray(ak.fill_none(inc, 0.0))
        inc = np.asarray(inc, dtype=np.float64).reshape(-1)
        if inc.shape[0] < len(ak_points):
            inc = np.pad(inc, (0, len(ak_points) - inc.shape[0]), constant_values=0.0)
        inc_safe = np.where(inc > 0, inc, 1.0)
        ratio = np.asarray(ak.fill_none(e_total, 0.0)) / (inc_safe * 1e3)
        out["incident_energy"] = ak.Array(inc[: len(ak_points)])
        out["energy_sum_over_incident_energy"] = ak.Array(ratio[: len(ak_points)])

    for thresh in hit_thresholds or [0.1]:
        key = "n_hits_01" if thresh == 0.1 else f"n_hits_E{thresh}"
        out[key] = ak.values_astype(ak.num(e[e > thresh]), np.float64)

    r50, r90 = _radial_containment(x, y, e, cog_x, cog_y, safe_e)
    out["R50"] = ak.values_astype(ak.fill_none(r50, 0.0), np.float64)
    out["R90"] = ak.values_astype(ak.fill_none(r90, 0.0), np.float64)

    if include_layer_profile:
        layer_idx = _layer_index(z, n_layers, z_max)
        e_layer, sigma_r_layer = _layer_observables(x, y, e, layer_idx, n_layers)
        out["E_layer"] = ak.Array(np.ascontiguousarray(e_layer))
        out["sigma_r_layer"] = ak.Array(np.ascontiguousarray(sigma_r_layer))

    return ak.Array(out)
