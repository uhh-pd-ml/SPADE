from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch import nn

from ..data.preprocessing.utils import (
    bin_centers_from_edges,
    build_hcal_resolution_bin_edges,
)
from .base_backbone import ARCombinedAxis, ARSplitAxis
from .gpt_decoder import TransformerConfig


def _resolve_energy_binning(
    kwargs: Dict[str, Any], fallback_vocab: Optional[int]
) -> Tuple[str, Optional[int], Optional[Any], Optional[Any]]:
    """Resolve (binning_mode, effective_vocab, bin_centers, bin_edges) from kwargs.

    When mode == "hcal_resolution" and tokenization is enabled, we build the
    edges here so adapter and model agree on num_bins / vocab / centers.
    """
    use_tokenization = bool(kwargs.get("use_energy_tokenization", False))
    mode = str(kwargs.get("energy_binning_mode", "uniform"))
    if not (use_tokenization and mode == "hcal_resolution"):
        return mode, fallback_vocab, None, None

    edges = build_hcal_resolution_bin_edges(
        e_min=float(kwargs.get("energy_hcal_e_min", 0.001)),
        e_max=float(kwargs.get("energy_hcal_e_max", 150.0)),
        A=float(kwargs.get("energy_hcal_A", 0.50)),
        B=float(kwargs.get("energy_hcal_B", 0.0)),
        C=float(kwargs.get("energy_hcal_C", 0.02)),
        step_sigma=float(kwargs.get("energy_hcal_step_sigma", 0.5)),
    )
    centers = bin_centers_from_edges(edges)
    num_bins = int(centers.shape[0])
    # vocab must match adapter: +2 for PAD=0 and MISSING=vocab-1.
    return mode, num_bins + 2, centers, edges


def _optional_int(value: Any) -> Optional[int]:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def build_model_core(*, transformer: TransformerConfig, kwargs: Dict[str, Any]) -> nn.Module:
    """
    Build and return a model subclass instance from a normalized config.

    Supported ``axis_mode`` values:
    - "split":    factorized per-axis (x, y, z) embeddings and heads (SPADE).
    - "combined": a single combined spatial vocabulary and head (Combined baseline).
    """
    mode = kwargs.get("axis_mode", None)

    if mode == "split":
        return _build_split_axis(transformer, kwargs)

    if mode == "combined":
        return _build_combined(transformer, kwargs)

    if mode is None:
        raise ValueError("Model config requires 'axis_mode' in kwargs")

    raise ValueError(f"Unknown model mode '{mode}'")


def _build_split_axis(transformer: TransformerConfig, kwargs: Dict[str, Any]) -> ARSplitAxis:
    pax = kwargs.get("per_axis_vocab_sizes")
    if not isinstance(pax, dict):
        raise ValueError("split requires per_axis_vocab_sizes")
    binning_mode, resolved_vocab, bin_centers, bin_edges = _resolve_energy_binning(
        kwargs, _optional_int(kwargs.get("energy_vocab_size"))
    )
    return ARSplitAxis(
        transformer=transformer,
        per_axis_vocab_sizes={str(k): int(v) for k, v in pax.items()},
        per_axis_embedding_dims=kwargs.get("per_axis_embedding_dims"),
        per_axis_embedding_dropout=float(kwargs.get("per_axis_embedding_dropout", 0.0)),
        hierarchical_conditioning=bool(kwargs.get("hierarchical_conditioning", False)),
        hierarchical_axis_order=kwargs.get("hierarchical_axis_order"),
        use_hit_energy_feature=bool(kwargs.get("use_hit_energy_feature", True)),
        hit_energy_feature_dim=int(kwargs.get("hit_energy_feature_dim", 16)),
        use_incident_energy_feature=bool(kwargs.get("use_incident_energy_feature", True)),
        incident_energy_feature_dim=int(kwargs.get("incident_energy_feature_dim", 16)),
        use_n_hits_feature=bool(kwargs.get("use_n_hits_feature", False)),
        n_hits_feature_dim=int(kwargs.get("n_hits_feature_dim", 32)),
        n_hits_dropout_p=float(kwargs.get("n_hits_dropout_p", 0.0)),
        use_mog_energy=bool(kwargs.get("use_mog_energy", True)),
        mog_components=int(kwargs.get("mog_components", 5)),
        n_registers=int(kwargs.get("n_registers", 0)),
        use_n_hits_register=bool(kwargs.get("use_n_hits_register", False)),
        register_condition_incident_energy=bool(
            kwargs.get("register_condition_incident_energy", True)
        ),
        zero_init_register_incident=bool(kwargs.get("zero_init_register_incident", True)),
        hit_energy_dropout_p=float(kwargs.get("hit_energy_dropout_p", 0.0)),
        incident_energy_dropout_p=float(kwargs.get("incident_energy_dropout_p", 0.0)),
        use_energy_tokenization=bool(kwargs.get("use_energy_tokenization", False)),
        energy_vocab_size=resolved_vocab,
        energy_binning_mode=binning_mode,
        energy_bin_centers=bin_centers,
        energy_bin_edges=bin_edges,
        energy_sample_min=_optional_float(kwargs.get("energy_sample_min", -5.0)),
        energy_sample_max=_optional_float(kwargs.get("energy_sample_max", 5.0)),
        energy_nan_replacement=kwargs.get("energy_nan_replacement"),
        energy_noise_rel_std=float(kwargs.get("energy_noise_rel_std", 0.0)),
        use_spatial_axis_sinusoid=bool(kwargs.get("use_spatial_axis_sinusoid", False)),
        spatial_axis_sinusoid_scale=float(kwargs.get("spatial_axis_sinusoid_scale", 1.0)),
        spatial_axis_sinusoid_axes=kwargs.get("spatial_axis_sinusoid_axes"),
    )


def _build_combined(transformer: TransformerConfig, kwargs: Dict[str, Any]) -> ARCombinedAxis:
    token_vocab_size = kwargs.get("vocab_size_total", None)
    if token_vocab_size is None:
        raise ValueError("combined requires vocab_size_total")
    fallback_vocab = (
        int(kwargs["energy_vocab_size"]) if kwargs.get("energy_vocab_size") is not None else None
    )
    binning_mode, resolved_vocab, bin_centers, bin_edges = _resolve_energy_binning(
        kwargs, fallback_vocab
    )
    return ARCombinedAxis(
        transformer=transformer,
        token_vocab_size=token_vocab_size,
        token_pad_id=int(kwargs.get("token_pad_id", 0)),
        energy_vocab_size=resolved_vocab,
        energy_binning_mode=binning_mode,
        energy_bin_centers=bin_centers,
        energy_bin_edges=bin_edges,
        use_energy_tokenization=bool(kwargs.get("use_energy_tokenization", False)),
        use_hit_energy_feature=bool(kwargs.get("use_hit_energy_feature", True)),
        hit_energy_feature_dim=int(kwargs.get("hit_energy_feature_dim", 16)),
        use_incident_energy_feature=bool(kwargs.get("use_incident_energy_feature", True)),
        incident_energy_feature_dim=int(kwargs.get("incident_energy_feature_dim", 16)),
        use_n_hits_feature=bool(kwargs.get("use_n_hits_feature", False)),
        n_hits_feature_dim=int(kwargs.get("n_hits_feature_dim", 32)),
        n_hits_dropout_p=float(kwargs.get("n_hits_dropout_p", 0.0)),
        use_mog_energy=bool(kwargs.get("use_mog_energy", True)),
        mog_components=int(kwargs.get("mog_components", 5)),
        n_registers=int(kwargs.get("n_registers", 0)),
        hierarchical_conditioning=bool(kwargs.get("hierarchical_conditioning", False)),
        energy_sample_min=_optional_float(kwargs.get("energy_sample_min", -5.0)),
        energy_sample_max=_optional_float(kwargs.get("energy_sample_max", 5.0)),
        use_n_hits_register=bool(kwargs.get("use_n_hits_register", False)),
        register_condition_incident_energy=bool(
            kwargs.get("register_condition_incident_energy", True)
        ),
        zero_init_register_incident=bool(kwargs.get("zero_init_register_incident", True)),
        incident_energy_dropout_p=float(kwargs.get("incident_energy_dropout_p", 0.0)),
        energy_noise_rel_std=float(kwargs.get("energy_noise_rel_std", 0.0)),
    )
