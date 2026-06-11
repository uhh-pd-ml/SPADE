from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

from ..data.preprocessing.global_features import compute_global_features
from ..plotting import utils as plot_utils
from ..plotting.plotting import plot_features
from ..utils.pylogger import get_pylogger

_logger = get_pylogger(__name__)

# --- Constants ---

TOKEN_KEYS = ("token_ids", "interleaved_token", "spatial_token")
MASK_KEYS = ("hit_mask", "attention_mask")
ENERGY_KEYS = ("raw_energy_physical", "energy", "energy_values", "raw_energy")
ENERGY_TOKEN_KEYS = ("energy_token", "energy_ids")
INCIDENT_ENERGY_KEYS = ("incident_energy", "shower_energy")

DATASET_NAME_GEN = "OmniJet-alpha v2"
DATASET_NAME_REF = "Geant4"


@dataclass
class PlotInputs:
    """Prepared data for plotting."""

    features_gen: ak.Array
    features_ref: Optional[ak.Array]
    phys_gen: ak.Array
    phys_ref: Optional[ak.Array]
    glob_gen: ak.Array
    glob_ref: Optional[ak.Array]


def is_main_process(trainer: Any) -> bool:
    try:
        return bool(getattr(trainer, "is_global_zero", False))
    except Exception:
        return True


def get_first_tensor(data: Dict[str, Any], keys: Sequence[str]) -> Optional[torch.Tensor]:
    """Helper to retrieve the first matching key from a dictionary."""
    for k in keys:
        if k in data and torch.is_tensor(data[k]):
            return data[k]
    return None


def physical_to_ak(
    hit_features: torch.Tensor,
    hit_mask: torch.Tensor,
    incident_energy: torch.Tensor,
    tokens: Optional[torch.Tensor] = None,
) -> Tuple[ak.Array, ak.Array, ak.Array]:
    """
    Converts physical hit features to Awkward Arrays for plotting.

    Args:
        hit_features: [B, N, 4] Tensor (x, y, z, E)
        hit_mask: [B, N] Boolean Tensor
        incident_energy: [B] Tensor
        tokens: [B, N] Tensor (optional, for feature histograms)

    Returns:
        (ak_phys, ak_global, ak_features)
    """
    # Defensive checks
    if hit_features.ndim != 3 or hit_features.shape[-1] != 4:
        # Return empty arrays structure if input is invalid
        empty = ak.Array([])
        return empty, empty, empty

    B = hit_features.shape[0]

    # Ensure consistency
    if hit_mask.shape != hit_features.shape[:2]:
        # Simple slicing/padding to match features
        T_feat = hit_features.shape[1]
        T_mask = hit_mask.shape[1]
        if T_mask > T_feat:
            hit_mask = hit_mask[:, :T_feat]
        elif T_mask < T_feat:
            padding = torch.zeros(
                (B, T_feat - T_mask), dtype=hit_mask.dtype, device=hit_mask.device
            )
            hit_mask = torch.cat([hit_mask, padding], dim=1)

    # Prepare counts
    counts = hit_mask.long().sum(dim=1).cpu().numpy()
    mask_flat = hit_mask.reshape(-1).bool().cpu().numpy()

    # --- 1. Physics Features (Point Cloud) ---
    # Extract flattened data using mask
    feat_flat = hit_features.reshape(-1, 4).detach().cpu().numpy()
    # Apply mask only after moving to numpy to ensure sync
    feat_masked = feat_flat[mask_flat]

    # Defensive: generation can produce non-finite hit features under
    # ill-trained models (e.g. epoch 0). Replace inf/nan with 0 so plotting
    # math (np.log, ak multiply/divide, np.histogram) does not blow up.
    if feat_masked.size > 0 and not np.all(np.isfinite(feat_masked)):
        n_bad = int(np.sum(~np.isfinite(feat_masked)))
        _logger.warning(
            "physical_to_ak: replaced %d non-finite values in hit_features "
            "(shape=%s); model output likely degenerate at this step.",
            n_bad,
            tuple(feat_masked.shape),
        )
        feat_masked = np.nan_to_num(feat_masked, nan=0.0, posinf=0.0, neginf=0.0)

    if feat_masked.size > 0:
        x, y, z, e = feat_masked[:, 0], feat_masked[:, 1], feat_masked[:, 2], feat_masked[:, 3]
    else:
        x, y, z, e = np.array([]), np.array([]), np.array([]), np.array([])

    ak_x = ak.unflatten(x, counts)
    ak_y = ak.unflatten(y, counts)
    ak_z = ak.unflatten(z, counts)
    ak_e = ak.unflatten(e, counts)

    ak_phys = ak.Array({"x": ak_x, "y": ak_y, "z": ak_z, "energy": ak_e})

    # Derived physics fields
    if len(ak_phys) > 0:
        e_pos = ak.where(ak_phys["energy"] > 0, ak_phys["energy"], 0.01)
        ak_phys = ak.with_field(ak_phys, np.log(e_pos), "log_energy")
        ak_phys = ak.with_field(ak_phys, e_pos, "energy_logscale")
    else:
        # Handle empty case safely?
        pass

    inc = incident_energy.detach().cpu().numpy().reshape(-1)
    if inc.shape[0] > B:
        inc = inc[:B]

    if len(ak_phys) > 0:
        ak_global = compute_global_features(
            ak_phys,
            incident_energy=inc,
            include_layer_profile=False,
        )
    else:
        ak_global = ak.Array([])

    # --- 3. Token/Raw Features (Histograms) ---
    # For histograms, we plot the 'token_id' and 'energy' distribution

    # Token IDs
    if tokens is not None:
        # Align shape
        if tokens.shape != hit_mask.shape:
            # Try to match length
            min_t = min(tokens.shape[1], hit_mask.shape[1])
            tokens = tokens[:, :min_t]
            temp_mask_flat = hit_mask[:, :min_t].reshape(-1).bool().cpu().numpy()
        else:
            temp_mask_flat = mask_flat

        tok_flat = tokens.reshape(-1).detach().cpu().numpy()[temp_mask_flat]
    else:
        tok_flat = np.zeros_like(x, dtype=np.int32)

    # We need to unflatten tokens with counts that match the slice used
    # If we sliced tokens, counts might differ. For simplicity assume tokens aligned or we rely on 'counts' from hit_mask
    # If tokens were None, we use 'x' length which matches 'counts'

    if len(tok_flat) == sum(counts):
        ak_tok = ak.unflatten(tok_flat, counts)
    else:
        # Fallback if shapes wildly mismatched (should happen rarely with new logic)
        ak_tok = ak.unflatten(np.zeros(sum(counts), dtype=np.int32), counts)

    ak_features = ak.Array({"token_id": ak_tok, "energy": ak_e})

    return ak_phys, ak_global, ak_features


def _resolve_bins(
    key: str, default_bins: Union[np.ndarray, Sequence], overrides: Optional[Dict[str, Any]]
) -> Union[np.ndarray, Sequence]:
    if overrides is None or key not in overrides:
        return default_bins

    cfg = overrides[key]
    if isinstance(cfg, (list, tuple)):
        # Format: [min, max, count]
        if len(cfg) == 3:
            return np.linspace(cfg[0], cfg[1], int(cfg[2]))
        # Alternatively, if it's a list of edges
        return np.array(cfg)

    if isinstance(cfg, dict):
        if "edges" in cfg:
            return np.array(cfg["edges"])
        if "min" in cfg and "max" in cfg:
            count = int(cfg.get("count", 50))
            return np.linspace(cfg["min"], cfg["max"], count)

    return default_bins


def _extract_flat_z_energy(
    physical_features: Optional[ak.Array],
) -> Tuple[np.ndarray, np.ndarray]:
    if physical_features is None or len(physical_features) == 0:
        empty_values = np.array([], dtype=np.float64)
        return empty_values, empty_values

    z_values = ak.to_numpy(ak.flatten(physical_features.z, axis=None))
    energy_values = ak.to_numpy(ak.flatten(physical_features.energy, axis=None))
    finite_mask = np.isfinite(z_values) & np.isfinite(energy_values)
    if not np.any(finite_mask):
        empty_values = np.array([], dtype=np.float64)
        return empty_values, empty_values
    return z_values[finite_mask], energy_values[finite_mask]


def _resolve_z_bin_edges(
    z_gen_values: np.ndarray,
    z_ref_values: np.ndarray,
    spatial_nbins: Optional[Tuple[int, int, int]],
    bins_overrides: Optional[Dict[str, Any]],
    override_key: str,
) -> np.ndarray:
    if spatial_nbins is not None:
        n_z_bins = int(spatial_nbins[2])
        z_bin_edges_default = np.linspace(-0.5, n_z_bins - 0.5, n_z_bins + 1)
    else:
        concatenated_z_values = (
            np.concatenate([arr for arr in [z_gen_values, z_ref_values] if arr.size > 0])
            if (z_gen_values.size > 0 or z_ref_values.size > 0)
            else np.array([0.0, 1.0])
        )
        z_min = float(np.floor(np.min(concatenated_z_values))) - 0.5
        z_max = float(np.ceil(np.max(concatenated_z_values))) + 0.5
        if z_max <= z_min:
            z_max = z_min + 1.0
        z_bin_edges_default = np.linspace(z_min, z_max, int(max(3, round(z_max - z_min) + 2)))

    z_bin_edges = np.asarray(
        _resolve_bins(override_key, z_bin_edges_default, bins_overrides),
        dtype=np.float64,
    )
    z_bin_edges = np.unique(z_bin_edges)
    if z_bin_edges.size < 2:
        z_bin_edges = np.array([0.0, 1.0], dtype=np.float64)
    return z_bin_edges


def _plot_total_energy_per_layer_impl(
    inputs: PlotInputs,
    out_dir: str,
    epoch: int,
    spatial_nbins: Optional[Tuple[int, int, int]],
    should_save: bool,
    bins_overrides: Optional[Dict[str, Any]] = None,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
) -> List[Tuple[object, str]]:
    z_gen_values, e_gen_values = _extract_flat_z_energy(inputs.phys_gen)
    z_ref_values, e_ref_values = _extract_flat_z_energy(inputs.phys_ref)
    z_bin_edges = _resolve_z_bin_edges(
        z_gen_values,
        z_ref_values,
        spatial_nbins,
        bins_overrides,
        "total_energy_per_z",
    )
    number_of_bins = int(z_bin_edges.size - 1)
    z_bin_centers = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])

    def total_energy_per_bin(z_values: np.ndarray, energy_values: np.ndarray) -> np.ndarray:
        positive_energy_mask = np.isfinite(energy_values) & (energy_values > 0.0)
        valid_mask = np.isfinite(z_values) & positive_energy_mask
        if not np.any(valid_mask):
            return np.zeros(number_of_bins, dtype=np.float64)
        selected_z = z_values[valid_mask]
        selected_energy = energy_values[valid_mask]
        bin_indices = np.digitize(selected_z, z_bin_edges) - 1
        in_range_mask = (bin_indices >= 0) & (bin_indices < number_of_bins)
        if not np.any(in_range_mask):
            return np.zeros(number_of_bins, dtype=np.float64)
        return np.bincount(
            bin_indices[in_range_mask],
            weights=selected_energy[in_range_mask],
            minlength=number_of_bins,
        )

    total_energy_gen = total_energy_per_bin(z_gen_values, e_gen_values)
    total_energy_ref = total_energy_per_bin(z_ref_values, e_ref_values)

    fig, (axis_main, axis_ratio) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.35], "hspace": 0.05},
    )

    axis_main.step(
        z_bin_centers,
        total_energy_gen,
        where="mid",
        linewidth=2.0,
        color="C0",
        label=dataset_name_gen,
    )
    axis_main.step(
        z_bin_centers,
        total_energy_ref,
        where="mid",
        linewidth=2.0,
        color="C1",
        label=dataset_name_ref,
    )

    ratio_values = np.divide(
        total_energy_gen,
        total_energy_ref,
        out=np.full_like(total_energy_gen, np.nan, dtype=np.float64),
        where=total_energy_ref > 0.0,
    )
    valid_ratio_mask = np.isfinite(ratio_values)
    if np.any(valid_ratio_mask):
        axis_ratio.step(
            z_bin_centers[valid_ratio_mask],
            ratio_values[valid_ratio_mask],
            where="mid",
            linewidth=1.8,
            color="C1",
        )

    axis_main.set_ylabel("Total energy [MeV]")
    axis_main.set_yscale("log")
    axis_main.legend(frameon=False)
    axis_main.set_xlim(float(z_bin_edges[0]), float(z_bin_edges[-1]))
    plot_utils.decorate_ax(axis_main, yscale=1.6)

    axis_ratio.axhline(1.0, color="C0", linewidth=1.8)
    axis_ratio.set_ylabel("Ratio")
    axis_ratio.set_xlabel("z [layer]")
    axis_ratio.set_ylim(0.5, 1.8)
    axis_ratio.set_xlim(float(z_bin_edges[0]), float(z_bin_edges[-1]))
    plot_utils.decorate_ax(axis_ratio, yscale=1.2)

    filename = f"val_gen_total_energy_per_layer_epoch{epoch}.png"
    if should_save:
        fig.savefig(os.path.join(out_dir, filename), dpi=160, bbox_inches="tight")
    return [(fig, filename)]


def _plot_last_layers_energy_spectrum_impl(
    inputs: PlotInputs,
    out_dir: str,
    epoch: int,
    should_save: bool,
    bins_overrides: Optional[Dict[str, Any]] = None,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
) -> List[Tuple[object, str]]:
    min_last_layer_z = 27.0
    if isinstance(bins_overrides, dict) and "last_layer_min_z" in bins_overrides:
        min_last_layer_z = float(bins_overrides["last_layer_min_z"])

    z_gen_values, e_gen_values = _extract_flat_z_energy(inputs.phys_gen)
    z_ref_values, e_ref_values = _extract_flat_z_energy(inputs.phys_ref)

    e_gen_last = e_gen_values[(z_gen_values >= min_last_layer_z) & (e_gen_values > 0.0)]
    e_ref_last = e_ref_values[(z_ref_values >= min_last_layer_z) & (e_ref_values > 0.0)]
    combined = np.concatenate([arr for arr in [e_gen_last, e_ref_last] if arr.size > 0])

    if combined.size == 0:
        return []

    positive_combined = combined[combined > 0.0]
    if positive_combined.size == 0:
        return []

    energy_min = float(np.quantile(positive_combined, 0.001))
    energy_max = float(np.quantile(positive_combined, 0.999))
    energy_min = max(energy_min, 1e-4)
    if energy_max <= energy_min:
        energy_max = energy_min * 10.0

    default_energy_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), 70)
    energy_bins = np.asarray(
        _resolve_bins("last_layers_energy_spectrum", default_energy_bins, bins_overrides),
        dtype=np.float64,
    )
    energy_bins = np.unique(energy_bins)
    if energy_bins.size < 2:
        energy_bins = default_energy_bins

    hist_gen, _ = np.histogram(e_gen_last, bins=energy_bins, density=True)
    hist_ref, _ = np.histogram(e_ref_last, bins=energy_bins, density=True)
    energy_bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    fig, (axis_main, axis_ratio) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.35], "hspace": 0.05},
    )

    axis_main.step(
        energy_bin_centers,
        hist_gen,
        where="mid",
        linewidth=2.0,
        color="C0",
        label=dataset_name_gen,
    )
    axis_main.step(
        energy_bin_centers,
        hist_ref,
        where="mid",
        linewidth=2.0,
        color="C1",
        label=dataset_name_ref,
    )

    ratio_values = np.divide(
        hist_gen,
        hist_ref,
        out=np.full_like(hist_gen, np.nan, dtype=np.float64),
        where=hist_ref > 0.0,
    )
    valid_ratio_mask = np.isfinite(ratio_values)
    if np.any(valid_ratio_mask):
        axis_ratio.step(
            energy_bin_centers[valid_ratio_mask],
            ratio_values[valid_ratio_mask],
            where="mid",
            linewidth=1.8,
            color="C1",
        )

    axis_main.set_xscale("log")
    axis_main.set_yscale("log")
    axis_main.set_ylabel("Density")
    axis_main.legend(frameon=False)
    plot_utils.decorate_ax(axis_main, yscale=1.6)

    axis_ratio.axhline(1.0, color="C0", linewidth=1.8)
    axis_ratio.set_ylabel("Ratio")
    axis_ratio.set_xlabel(f"Hit energy [MeV] (z ≥ {min_last_layer_z:.0f})")
    axis_ratio.set_ylim(0.5, 1.8)
    axis_ratio.set_xscale("log")
    plot_utils.decorate_ax(axis_ratio, yscale=1.2)

    filename = f"val_gen_last_layers_energy_spectrum_epoch{epoch}.png"
    if should_save:
        fig.savefig(os.path.join(out_dir, filename), dpi=160, bbox_inches="tight")
    return [(fig, filename)]


def _plot_mean_energy_impl(
    inputs: PlotInputs,
    out_dir: str,
    epoch: int,
    spatial_nbins: Optional[Tuple[int, int, int]],
    should_save: bool,
    bins_overrides: Optional[Dict[str, Any]] = None,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
) -> List[Tuple[object, str]]:
    energy_threshold = 0.1

    def extract_flat_xyz_energy(
        physical_features: Optional[ak.Array],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if physical_features is None or len(physical_features) == 0:
            empty_values = np.array([], dtype=np.float64)
            return empty_values, empty_values, empty_values, empty_values

        x_values = ak.to_numpy(ak.flatten(physical_features.x, axis=None))
        y_values = ak.to_numpy(ak.flatten(physical_features.y, axis=None))
        z_values = ak.to_numpy(ak.flatten(physical_features.z, axis=None))
        energy_values = ak.to_numpy(ak.flatten(physical_features.energy, axis=None))

        finite_mask = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & np.isfinite(z_values)
            & np.isfinite(energy_values)
        )
        if not np.any(finite_mask):
            empty_values = np.array([], dtype=np.float64)
            return empty_values, empty_values, empty_values, empty_values

        return (
            x_values[finite_mask],
            y_values[finite_mask],
            z_values[finite_mask],
            energy_values[finite_mask],
        )

    def calculate_mean_energy_per_bin(
        coordinate_values: np.ndarray,
        energy_values: np.ndarray,
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        selected_mask = np.isfinite(coordinate_values) & np.isfinite(energy_values)
        selected_mask = selected_mask & (energy_values > energy_threshold)
        if not np.any(selected_mask):
            return np.full(len(bin_edges) - 1, np.nan, dtype=np.float64)

        selected_coordinate_values = coordinate_values[selected_mask]
        selected_energy_values = energy_values[selected_mask]

        bin_indices = np.digitize(selected_coordinate_values, bin_edges) - 1
        number_of_bins = len(bin_edges) - 1
        in_range_mask = (bin_indices >= 0) & (bin_indices < number_of_bins)
        if not np.any(in_range_mask):
            return np.full(number_of_bins, np.nan, dtype=np.float64)

        valid_bin_indices = bin_indices[in_range_mask]
        valid_energy_values = selected_energy_values[in_range_mask]

        summed_energy_per_bin = np.bincount(
            valid_bin_indices,
            weights=valid_energy_values,
            minlength=number_of_bins,
        )
        number_of_hits_per_bin = np.bincount(valid_bin_indices, minlength=number_of_bins)

        mean_energy_per_bin = np.divide(
            summed_energy_per_bin,
            number_of_hits_per_bin,
            out=np.full(number_of_bins, np.nan, dtype=np.float64),
            where=number_of_hits_per_bin > 0,
        )
        return mean_energy_per_bin

    x_gen_values, y_gen_values, z_gen_values, e_gen_values = extract_flat_xyz_energy(
        inputs.phys_gen
    )
    x_ref_values, y_ref_values, z_ref_values, e_ref_values = extract_flat_xyz_energy(
        inputs.phys_ref
    )

    if spatial_nbins is not None:
        n_x_bins, n_y_bins, n_z_bins = spatial_nbins
        z_bin_edges_default = np.linspace(-0.5, n_z_bins - 0.5, n_z_bins + 1)
        center_x = (float(n_x_bins) - 1.0) / 2.0
        center_y = (float(n_y_bins) - 1.0) / 2.0
        max_delta_x = max(abs(0.0 - center_x), abs((float(n_x_bins) - 1.0) - center_x))
        max_delta_y = max(abs(0.0 - center_y), abs((float(n_y_bins) - 1.0) - center_y))
        max_radius = float(np.sqrt(max_delta_x**2 + max_delta_y**2))
        radius_bin_edges_default = np.linspace(0.0, max(max_radius, 1.0), 100)
    else:
        concatenated_z_values = (
            np.concatenate([arr for arr in [z_gen_values, z_ref_values] if arr.size > 0])
            if (z_gen_values.size > 0 or z_ref_values.size > 0)
            else np.array([0.0, 1.0])
        )
        z_min = float(np.floor(np.min(concatenated_z_values))) - 0.5
        z_max = float(np.ceil(np.max(concatenated_z_values))) + 0.5
        if z_max <= z_min:
            z_max = z_min + 1.0
        z_bin_edges_default = np.linspace(z_min, z_max, int(max(3, round(z_max - z_min) + 2)))
        center_x = 0.0
        center_y = 0.0
        max_radius = 1.0
        radius_bin_edges_default = np.linspace(0.0, max_radius, 100)

    radius_gen_values = np.sqrt((x_gen_values - center_x) ** 2 + (y_gen_values - center_y) ** 2)
    radius_ref_values = np.sqrt((x_ref_values - center_x) ** 2 + (y_ref_values - center_y) ** 2)

    if spatial_nbins is None:
        concatenated_radius_values = (
            np.concatenate([arr for arr in [radius_gen_values, radius_ref_values] if arr.size > 0])
            if (radius_gen_values.size > 0 or radius_ref_values.size > 0)
            else np.array([0.0, 1.0])
        )
        radius_max = float(np.max(concatenated_radius_values))
        radius_bin_edges_default = np.linspace(0.0, max(radius_max, 1.0), 100)

    z_bin_edges = np.asarray(
        _resolve_bins("mean_energy_per_z", z_bin_edges_default, bins_overrides),
        dtype=np.float64,
    )
    radius_bin_edges = np.asarray(
        _resolve_bins("mean_energy_per_radius", radius_bin_edges_default, bins_overrides),
        dtype=np.float64,
    )
    z_bin_edges = np.unique(z_bin_edges)
    radius_bin_edges = np.unique(radius_bin_edges)
    if z_bin_edges.size < 2:
        z_bin_edges = np.array([0.0, 1.0], dtype=np.float64)
    if radius_bin_edges.size < 2:
        radius_bin_edges = np.array([0.0, 1.0], dtype=np.float64)

    mean_energy_per_z_gen = calculate_mean_energy_per_bin(z_gen_values, e_gen_values, z_bin_edges)
    mean_energy_per_z_ref = calculate_mean_energy_per_bin(z_ref_values, e_ref_values, z_bin_edges)
    mean_energy_per_radius_gen = calculate_mean_energy_per_bin(
        radius_gen_values,
        e_gen_values,
        radius_bin_edges,
    )
    mean_energy_per_radius_ref = calculate_mean_energy_per_bin(
        radius_ref_values,
        e_ref_values,
        radius_bin_edges,
    )

    z_bin_centers = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])
    radius_bin_centers = 0.5 * (radius_bin_edges[:-1] + radius_bin_edges[1:])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 0.35], "hspace": 0.05, "wspace": 0.25},
    )

    axis_z_main = axes[0, 0]
    axis_r_main = axes[0, 1]
    axis_z_ratio = axes[1, 0]
    axis_r_ratio = axes[1, 1]

    def plot_profile(
        axis_main: Any,
        axis_ratio: Any,
        coordinate_centers: np.ndarray,
        coordinate_edges: np.ndarray,
        mean_energy_gen: np.ndarray,
        mean_energy_ref: np.ndarray,
        x_label: str,
    ) -> None:
        valid_gen_mask = np.isfinite(mean_energy_gen)
        valid_ref_mask = np.isfinite(mean_energy_ref)

        if np.any(valid_gen_mask):
            axis_main.step(
                coordinate_centers,
                mean_energy_gen,
                where="mid",
                linewidth=2.0,
                label=dataset_name_gen,
                color="C0",
            )
        if np.any(valid_ref_mask):
            axis_main.step(
                coordinate_centers,
                mean_energy_ref,
                where="mid",
                linewidth=2.0,
                label=dataset_name_ref,
                color="C1",
            )

        ratio_values = np.divide(
            mean_energy_gen,
            mean_energy_ref,
            out=np.full_like(mean_energy_gen, np.nan, dtype=np.float64),
            where=np.isfinite(mean_energy_ref) & (mean_energy_ref > 0.0),
        )
        valid_ratio_mask = np.isfinite(ratio_values)
        ratio_min, ratio_max = 0.5, 1.8
        if np.any(valid_ratio_mask):
            axis_ratio.step(
                coordinate_centers[valid_ratio_mask],
                ratio_values[valid_ratio_mask],
                where="mid",
                linewidth=1.8,
                color="C1",
            )
            ratio_overflow_mask = valid_ratio_mask & (ratio_values > ratio_max)
            ratio_underflow_mask = valid_ratio_mask & (ratio_values < ratio_min)
            if np.any(ratio_overflow_mask):
                axis_ratio.plot(
                    coordinate_centers[ratio_overflow_mask],
                    np.full(np.sum(ratio_overflow_mask), ratio_max),
                    "^",
                    color="C1",
                    markersize=5,
                    clip_on=False,
                )
            if np.any(ratio_underflow_mask):
                axis_ratio.plot(
                    coordinate_centers[ratio_underflow_mask],
                    np.full(np.sum(ratio_underflow_mask), ratio_min),
                    "v",
                    color="C1",
                    markersize=5,
                    clip_on=False,
                )

        axis_main.set_xlim(float(coordinate_edges[0]), float(coordinate_edges[-1]))
        axis_ratio.set_xlim(float(coordinate_edges[0]), float(coordinate_edges[-1]))
        axis_main.set_ylabel("Mean energy [MeV]")
        axis_main.set_yscale("log")
        axis_ratio.set_ylabel("Ratio")
        axis_ratio.set_xlabel(x_label)
        axis_ratio.axhline(1.0, color="C0", linewidth=1.8)
        axis_ratio.set_ylim(ratio_min, ratio_max)
        plot_utils.decorate_ax(axis_main, yscale=1.6)
        plot_utils.decorate_ax(axis_ratio, yscale=1.2)

    plot_profile(
        axis_z_main,
        axis_z_ratio,
        z_bin_centers,
        z_bin_edges,
        mean_energy_per_z_gen,
        mean_energy_per_z_ref,
        "z [layer]",
    )
    plot_profile(
        axis_r_main,
        axis_r_ratio,
        radius_bin_centers,
        radius_bin_edges,
        mean_energy_per_radius_gen,
        mean_energy_per_radius_ref,
        "r [cells]",
    )

    axis_z_main.legend(frameon=False)
    axis_r_main.legend(frameon=False)

    filename = f"val_gen_features_epoch{epoch}.png"
    if should_save:
        fig.savefig(os.path.join(out_dir, filename), dpi=160, bbox_inches="tight")

    return [(fig, filename)]


def _plot_physics_impl(
    inputs: PlotInputs,
    spatial_nbins: Tuple[int, int, int],
    out_dir: str,
    epoch: int,
    should_save: bool,
    bins_overrides: Optional[Dict[str, Any]] = None,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
) -> List[Tuple[object, str]]:
    if inputs.phys_gen is None:
        return []
    ak_phys = {}
    ak_glob = {}
    if inputs.phys_ref is not None:
        ak_phys[dataset_name_ref] = inputs.phys_ref
    ak_phys[dataset_name_gen] = inputs.phys_gen
    if inputs.glob_ref is not None:
        ak_glob[dataset_name_ref] = inputs.glob_ref
    ak_glob[dataset_name_gen] = inputs.glob_gen

    figures = []

    # Point Cloud Plots
    all_energies = []
    for arr in ak_phys.values():
        if "energy" in arr.fields:
            flat = ak.to_numpy(ak.flatten(arr.energy, axis=None))
            if flat.size > 0:
                all_energies.append(flat)

    if not all_energies:
        all_energies.append(np.array([1e-4, 1.0]))

    concat_energy = np.concatenate(all_energies)
    e_lo = float(np.quantile(concat_energy, 0.001))
    e_hi = float(np.quantile(concat_energy, 0.999))
    e_lo = max(e_lo, 1e-4)
    if e_hi <= e_lo:
        e_hi = e_lo + 1.0

    energy_bins = np.linspace(e_lo, e_hi, 60)
    energy_bins_log = np.logspace(np.log10(e_lo), np.log10(e_hi), 60)
    log_energy_bins = np.linspace(np.log(e_lo), np.log(e_hi), 60)

    bins_points = {
        "x": _resolve_bins(
            "x", np.linspace(-0.5, spatial_nbins[0] - 0.5, spatial_nbins[0] + 1), bins_overrides
        ),
        "y": _resolve_bins(
            "y", np.linspace(-0.5, spatial_nbins[1] - 0.5, spatial_nbins[1] + 1), bins_overrides
        ),
        "z": _resolve_bins(
            "z", np.linspace(-0.5, spatial_nbins[2] - 0.5, spatial_nbins[2] + 1), bins_overrides
        ),
        "energy": _resolve_bins("energy", energy_bins, bins_overrides),
        "energy_logscale": _resolve_bins("energy_logscale", energy_bins_log, bins_overrides),
        "log_energy": _resolve_bins("log_energy", log_energy_bins, bins_overrides),
    }

    fig_p, _ = plot_features(
        ak_array_dict=ak_phys,
        names={
            "x": "Hit x",
            "y": "Hit y",
            "z": "Hit z",
            "energy": "Hit energy [MeV]",
            "energy_logscale": "Hit energy [MeV]",
            "log_energy": "Logarithm of hit energy",
        },
        flatten=True,
        legend_kwargs={"loc": "upper right"},
        decorate_ax_kwargs=dict(yscale=1.6),
        ax_rows=2,
        logscale_features=["energy", "energy_logscale"],
        logscale_xaxis_features=["energy_logscale"],
        ratio=True,
        ylabel="Normalized",
        bins_dict=bins_points,  # type: ignore
    )
    fname_p = f"val_gen_points_epoch{epoch}.png"
    if should_save:
        fig_p.savefig(os.path.join(out_dir, fname_p), dpi=160, bbox_inches="tight")
    figures.append((fig_p, fname_p))

    # Global Observables Plots
    def _wrap_singletons(arr):
        return ak.Array({k: ak.singletons(arr[k]) for k in arr.fields})

    ak_glob_plot = {k: _wrap_singletons(v) for k, v in ak_glob.items()}

    # Determine incident energy range using GEN incident energy as reference
    inc_arr = inputs.glob_gen.incident_energy
    inc_np = ak.to_numpy(ak.flatten(inc_arr, axis=None))
    inc_min = float(np.min(inc_np)) if inc_np.size > 0 else 0.0
    inc_max = float(np.max(inc_np)) if inc_np.size > 0 else 100.0
    if inc_min == inc_max:
        inc_min -= 1.0
        inc_max += 1.0

    bins_global = {
        "cog_x": _resolve_bins("cog_x", np.linspace(14.0, 15.0, 35), bins_overrides),
        "cog_y": _resolve_bins("cog_y", np.linspace(14.0, 15.0, 35), bins_overrides),
        "cog_z": _resolve_bins("cog_z", np.linspace(8.0, 22.0, 35), bins_overrides),
        "n_hits": _resolve_bins("n_hits", np.linspace(0, max(10.0, 1700.0), 35), bins_overrides),
        "energy_sum": _resolve_bins("energy_sum", np.linspace(100.0, 2000.0, 35), bins_overrides),
        "energy_sum_over_incident_energy": _resolve_bins(
            "energy_sum_over_incident_energy", np.linspace(0.0, 0.05, 35), bins_overrides
        ),
        "incident_energy": _resolve_bins(
            "incident_energy", np.linspace(inc_min, inc_max, 35), bins_overrides
        ),
        "n_hits_01": _resolve_bins(
            "n_hits_01", np.linspace(0, max(10.0, 1500.0), 35), bins_overrides
        ),
    }

    fig_g, _ = plot_features(
        ak_array_dict=ak_glob_plot,
        names={
            "cog_x": "Center of gravity x",
            "cog_y": "Center of gravity y",
            "cog_z": "Center of gravity z",
            "n_hits": "Number of hits",
            "energy_sum": "Energy sum [MeV]",
            "energy_sum_over_incident_energy": "Energy sum / incident energy",
            "incident_energy": "Incident energy [GeV]",
            "n_hits_01": "Hits with E>0.1 MeV",
        },
        flatten=True,
        decorate_ax_kwargs=dict(yscale=1.6),
        ax_rows=2,
        ratio=True,
        ylabel="Normalized",
        logscale_features=["energy_sum_over_incident_energy"],
        bins_dict=bins_global,  # type: ignore
    )
    fname_g = f"val_gen_globals_epoch{epoch}.png"
    if should_save:
        fig_g.savefig(os.path.join(out_dir, fname_g), dpi=160, bbox_inches="tight")
    figures.append((fig_g, fname_g))

    return figures


def _plot_heatmaps_impl(
    inputs: PlotInputs,
    spatial_nbins: Tuple[int, int, int],
    out_dir: str,
    epoch: int,
    should_save: bool,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
) -> List[Tuple[object, str]]:
    ak_phys = {dataset_name_gen: inputs.phys_gen}
    if inputs.phys_ref is not None:
        ak_phys[dataset_name_ref] = inputs.phys_ref

    if not ak_phys:
        return []

    figures = []
    data_list = []
    labels = []

    # Order: Real, Gen
    if dataset_name_ref in ak_phys:
        data_list.append(ak_phys[dataset_name_ref])
        labels.append(dataset_name_ref)

    if dataset_name_gen in ak_phys:
        data_list.append(ak_phys[dataset_name_gen])
        labels.append(dataset_name_gen)

    fontsize = 20
    nx, ny, nz = spatial_nbins
    bins_x = np.linspace(-0.5, nx - 0.5, nx + 1)
    bins_y = np.linspace(-0.5, ny - 0.5, ny + 1)
    bins_z = np.linspace(-0.5, nz - 0.5, nz + 1)

    # 1. Heatmaps (XY, XZ, YZ)
    fig, axes = plt.subplots(
        len(data_list),
        3,
        figsize=(15, 3 * len(data_list)),
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.3},
    )
    if len(data_list) == 1:
        axes = np.expand_dims(axes, axis=0)

    # Pre-compute statistics to get global max for consistent colorbar
    all_stats = []
    for d in data_list:
        x = ak.to_numpy(ak.flatten(d.x))
        y = ak.to_numpy(ak.flatten(d.y))
        z = ak.to_numpy(ak.flatten(d.z))
        e = ak.to_numpy(ak.flatten(d.energy))

        stat_xy, _, _, _ = binned_statistic_2d(x, y, e, statistic="mean", bins=[bins_x, bins_y])  # type: ignore
        stat_xz, _, _, _ = binned_statistic_2d(x, z, e, statistic="mean", bins=[bins_x, bins_z])  # type: ignore
        stat_yz, _, _, _ = binned_statistic_2d(y, z, e, statistic="mean", bins=[bins_y, bins_z])  # type: ignore
        all_stats.extend([stat_xy, stat_xz, stat_yz])

    valid = []
    for s in all_stats:
        v = s[np.isfinite(s)]
        valid.append(v)

    concat = np.concatenate(valid) if valid else np.array([])
    pos = concat[concat > 0]

    if pos.size == 0:
        # fallback: linear norm if everything is empty/zero
        norm = None
    else:
        vmin = np.percentile(pos, 1)  # or 0.5
        vmax = np.percentile(pos, 99)  # or 99.5
        norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for i, data in enumerate(data_list):
        x = ak.to_numpy(ak.flatten(data.x))
        y = ak.to_numpy(ak.flatten(data.y))
        z = ak.to_numpy(ak.flatten(data.z))
        e = ak.to_numpy(ak.flatten(data.energy))

        # XY
        stat, x_edge, y_edge, _ = binned_statistic_2d(
            x,
            y,
            e,
            statistic="mean",
            bins=[bins_x, bins_y],  # type: ignore
        )
        im = axes[i, 0].imshow(
            stat.T,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            aspect="auto",
            norm=norm,
        )
        axes[i, 0].set_ylabel(labels[i], fontsize=fontsize)
        axes[i, 0].set_title("Mean Energy (x-y)", fontsize=fontsize - 2)

        # XZ
        stat, x_edge, z_edge, _ = binned_statistic_2d(
            x,
            z,
            e,
            statistic="mean",
            bins=[bins_x, bins_z],  # type: ignore
        )
        im = axes[i, 1].imshow(
            stat.T,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], z_edge[0], z_edge[-1]],
            aspect="auto",
            norm=norm,
        )
        axes[i, 1].set_title("Mean Energy (x-z)", fontsize=fontsize - 2)

        # YZ
        stat, y_edge, z_edge, _ = binned_statistic_2d(
            y,
            z,
            e,
            statistic="mean",
            bins=[bins_y, bins_z],  # type: ignore
        )
        im = axes[i, 2].imshow(
            stat.T,
            origin="lower",
            extent=[y_edge[0], y_edge[-1], z_edge[0], z_edge[-1]],
            aspect="auto",
            norm=norm,
        )
        axes[i, 2].set_title("Mean Energy (y-z)", fontsize=fontsize - 2)

    axes[-1, 0].set_xlabel("x [cells]", fontsize=fontsize)
    axes[-1, 1].set_xlabel("x [cells]", fontsize=fontsize)
    axes[-1, 2].set_xlabel("y [cells]", fontsize=fontsize)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right")
    cbar.set_label("Mean Energy [MeV]", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    fname_heat = f"val_gen_mean_energy_heatmap_epoch{epoch}.png"
    if should_save:
        fig.savefig(os.path.join(out_dir, fname_heat), bbox_inches="tight")
    figures.append((fig, fname_heat))

    return figures


def _plot_individual_showers_impl(
    inputs: PlotInputs,
    out_dir: str,
    epoch: int,
    should_save: bool,
    dataset_name_gen: str = DATASET_NAME_GEN,
) -> List[Tuple[object, str]]:
    if inputs.phys_gen is None:
        return []

    ak_gen = inputs.phys_gen
    if len(ak_gen) == 0:
        return []

    ak_ref = inputs.phys_ref
    inc_en_gen = (
        inputs.glob_gen.incident_energy if hasattr(inputs.glob_gen, "incident_energy") else None
    )
    inc_en_ref = (
        inputs.glob_ref.incident_energy
        if inputs.glob_ref is not None and hasattr(inputs.glob_ref, "incident_energy")
        else None
    )

    if inc_en_gen is None:
        return []

    energy_cut = 0.1

    def apply_energy_cut_to_array(ak_array):
        mask = ak_array.energy > energy_cut
        return ak.Array(
            {
                "x": ak_array.x[mask],
                "y": ak_array.y[mask],
                "z": ak_array.z[mask],
                "energy": ak_array.energy[mask],
            }
        )

    ak_gen_cut = apply_energy_cut_to_array(ak_gen)
    ak_ref_cut = (
        apply_energy_cut_to_array(ak_ref) if ak_ref is not None and len(ak_ref) > 0 else None
    )

    figures = []

    def plot_2d_showers_projection(ak_array, inc_en, coord_field, label, size_n=5):
        n1 = int(ak.num(ak_array.x, axis=0))
        if n1 == 0:
            return []

        if size_n > n1:
            size_n = n1

        idx = np.arange(size_n)
        sorted_inc_en = ak.argsort(inc_en[idx])

        coord_vals = getattr(ak_array, coord_field)[idx][sorted_inc_en]
        z_vals = ak_array.z[idx][sorted_inc_en]
        e_vals = ak_array.energy[idx][sorted_inc_en]
        inc_en_sorted = inc_en[idx][sorted_inc_en]

        return [
            {
                "coord": ak.to_numpy(coord_vals[i]),
                "z": ak.to_numpy(z_vals[i]),
                "energy": ak.to_numpy(e_vals[i]),
                "inc_en": float(inc_en_sorted[i]),
            }
            for i in range(size_n)
        ]

    projections = [("x", "x"), ("y", "y")]

    for coord_name, coord_field in projections:
        fig = plt.figure(figsize=(15, 6), facecolor="white")
        fig.suptitle(f"{coord_name} vs z")

        all_showers = []
        all_labels = []

        if ak_ref_cut is not None and inc_en_ref is not None:
            ref_showers = plot_2d_showers_projection(
                ak_ref_cut, inc_en_ref, coord_field, "Geant4", size_n=5
            )
            for i, shower in enumerate(ref_showers):
                all_showers.append(shower)
                all_labels.append(f"Geant4 {i}")

        gen_showers = plot_2d_showers_projection(
            ak_gen_cut, inc_en_gen, coord_field, "Gen", size_n=5
        )
        for i, shower in enumerate(gen_showers):
            all_showers.append(shower)
            all_labels.append(f"Gen {i}")

        n_total = len(all_showers)
        n_cols = 5
        n_rows = 2

        for i in range(min(n_total, n_rows * n_cols)):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            shower = all_showers[i]

            if len(shower["energy"]) == 0:
                ax.axis("off")
                continue

            ax.scatter(
                shower["coord"],
                shower["z"],
                s=shower["energy"],
                label=str(np.round(shower["inc_en"], 1)) + " [GeV]",
            )
            ax.legend()
            ax.set_ylim(0, 48)
            ax.set_xlabel(f"{coord_name} [mm]")
            ax.set_ylabel("z [layer]")
            ax.set_title(all_labels[i])

        for i in range(n_total, n_rows * n_cols):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            ax.axis("off")

        plt.tight_layout()

        fname = f"val_gen_individual_showers_{coord_name}vsz_epoch{epoch}.png"
        if should_save:
            fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")

        figures.append((fig, fname))

    return figures


def plot_all(
    inputs: PlotInputs,
    out_dir: str,
    epoch: int,
    spatial_nbins: Optional[Tuple[int, int, int]],
    should_save: bool,
    bins_overrides: Optional[Dict[str, Any]] = None,
    dataset_name_gen: str = DATASET_NAME_GEN,
    dataset_name_ref: str = DATASET_NAME_REF,
    plot_individual_showers: bool = False,
    dark_mode: bool = False,
) -> List[Tuple[object, str]]:
    plot_utils.set_mpl_style(darkmode=dark_mode)
    figures = []

    # Feature Plots
    figures.extend(
        _plot_mean_energy_impl(
            inputs,
            out_dir,
            epoch,
            spatial_nbins,
            should_save,
            bins_overrides,
            dataset_name_gen=dataset_name_gen,
            dataset_name_ref=dataset_name_ref,
        )
    )
    figures.extend(
        _plot_total_energy_per_layer_impl(
            inputs,
            out_dir,
            epoch,
            spatial_nbins,
            should_save,
            bins_overrides,
            dataset_name_gen=dataset_name_gen,
            dataset_name_ref=dataset_name_ref,
        )
    )
    figures.extend(
        _plot_last_layers_energy_spectrum_impl(
            inputs,
            out_dir,
            epoch,
            should_save,
            bins_overrides,
            dataset_name_gen=dataset_name_gen,
            dataset_name_ref=dataset_name_ref,
        )
    )

    if spatial_nbins is None:
        return figures

    # Physics Point Cloud Plots
    figures.extend(
        _plot_physics_impl(
            inputs,
            spatial_nbins,
            out_dir,
            epoch,
            should_save,
            bins_overrides,
            dataset_name_gen=dataset_name_gen,
            dataset_name_ref=dataset_name_ref,
        )
    )
    # Heatmaps
    figures.extend(
        _plot_heatmaps_impl(
            inputs,
            spatial_nbins,
            out_dir,
            epoch,
            should_save,
            dataset_name_gen=dataset_name_gen,
            dataset_name_ref=dataset_name_ref,
        )
    )

    if plot_individual_showers:
        figures.extend(
            _plot_individual_showers_impl(
                inputs,
                out_dir,
                epoch,
                should_save,
                dataset_name_gen=dataset_name_gen,
            )
        )

    return figures
