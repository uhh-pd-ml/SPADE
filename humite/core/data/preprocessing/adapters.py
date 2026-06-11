from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import awkward as ak
import torch

from ...config.schemas import DatasetConfig
from ..geometry import get_spatial_codebook_sizes, get_spatial_field_names
from .utils import (
    apply_feature_transform,
    bin_centers_from_edges,
    build_hcal_resolution_bin_edges,
    decompose_combined_spatial_tokens,
    get_valid_energy_mask,
)


class BaseAdapter:
    def __init__(self, dataset: DatasetConfig) -> None:
        self.dataset = dataset
        self.pad_length = dataset.pad_length
        self.raw_spatial_fields = get_spatial_field_names(dataset.dataset_type)
        self.spatial_axis_names = tuple(self.raw_spatial_fields)
        self.axis_to_index = {axis: idx for idx, axis in enumerate(self.spatial_axis_names)}
        feats = self._as_dict(getattr(dataset, "features", {}))
        self.feature_dict = feats
        e_cfg = self._as_dict(feats.get("energy", {}))
        self.energy_threshold = float(getattr(dataset, "energy_threshold", 0.0))
        geometry_nbins = get_spatial_codebook_sizes(dataset.dataset_type)
        self.axis_nbins: Dict[str, int] = {}
        self.axis_vocab_sizes: Dict[str, int] = {}
        self.axis_missing_ids: Dict[str, int] = {}

        self.spatial_vocab: int = 1
        for axis in self.spatial_axis_names:
            axis_cfg = self._as_dict(feats.get(axis, {}))
            nbins_attr = getattr(dataset, f"nbins_{axis}", None)
            if nbins_attr is None:
                nbins_attr = axis_cfg.get(f"nbins_{axis}")
            if nbins_attr is None:
                nbins_attr = axis_cfg.get("nbins")
            if nbins_attr is None:
                nbins_attr = geometry_nbins.get(axis)
            self.axis_nbins[axis] = int(nbins_attr) if nbins_attr is not None else 30

            vocab_attr = getattr(dataset, f"vocab_size_{axis}", None)
            if vocab_attr is None:
                vocab_attr = axis_cfg.get(f"vocab_size_{axis}")
            if vocab_attr is None:
                vocab_attr = axis_cfg.get("vocab_size")
            if vocab_attr is None:
                vocab_attr = self.axis_nbins[axis] + 2
            self.axis_vocab_sizes[axis] = int(vocab_attr)

            # --- Validation & Fix for Missing ID collision ---
            if self.axis_vocab_sizes[axis] < self.axis_nbins[axis] + 2:
                raise ValueError(
                    f"Invalid vocab_size for axis '{axis}': vocab_size={self.axis_vocab_sizes[axis]} "
                    f"but nbins={self.axis_nbins[axis]}. "
                    f"Need vocab_size >= nbins + 2 to reserve PAD=0 and MISSING=nbins+1."
                )

            self.axis_missing_ids[axis] = self.axis_nbins[axis] + 1

            setattr(self, f"nbins_{axis}", self.axis_nbins[axis])
            setattr(self, f"vocab_{axis}", self.axis_vocab_sizes[axis])
            setattr(self, f"missing_id_{axis}", self.axis_missing_ids[axis])

            self.spatial_vocab *= self.axis_nbins[axis]

        self.energy_vocab: int = int(
            getattr(
                dataset,
                "energy_vocab_size",
                e_cfg.get("energy_vocab_size", feats.get("energy_vocab_size", 1000)),
            )
        )
        self.energy_min: float = float(
            getattr(dataset, "energy_min", e_cfg.get("energy_min", 0.01))
        )
        self.energy_max: float = float(
            getattr(dataset, "energy_max", e_cfg.get("energy_max", 100.0))
        )
        self.energy_log: bool = bool(
            getattr(dataset, "energy_log_scale", e_cfg.get("energy_log_scale", True))
        )
        self.use_energy_tokenization: bool = bool(dataset.use_energy_tokenization)

        # HCal resolution-binning (active only when use_energy_tokenization=True).
        # Builds raw-GeV edges once and derives num_bins; energy_vocab is overridden
        # to num_bins + 2 to reserve PAD=0 and MISSING=vocab-1.
        self.energy_binning_mode: str = str(getattr(dataset, "energy_binning_mode", "uniform"))
        self.hcal_bin_edges: Optional[torch.Tensor] = None
        self.hcal_bin_centers: Optional[torch.Tensor] = None
        if self.use_energy_tokenization and self.energy_binning_mode == "hcal_resolution":
            edges_np = build_hcal_resolution_bin_edges(
                e_min=float(getattr(dataset, "energy_hcal_e_min", 0.001)),
                e_max=float(getattr(dataset, "energy_hcal_e_max", 150.0)),
                A=float(getattr(dataset, "energy_hcal_A", 0.50)),
                B=float(getattr(dataset, "energy_hcal_B", 0.0)),
                C=float(getattr(dataset, "energy_hcal_C", 0.02)),
                step_sigma=float(getattr(dataset, "energy_hcal_step_sigma", 0.5)),
            )
            centers_np = bin_centers_from_edges(edges_np)
            num_bins = int(centers_np.shape[0])
            self.hcal_bin_edges = torch.as_tensor(edges_np, dtype=torch.float32)
            self.hcal_bin_centers = torch.as_tensor(centers_np, dtype=torch.float32)
            # Override vocab to match generated bins. Data tokens live in [1, num_bins].
            self.energy_vocab = num_bins + 2

        self.energy_label_smoothing: str = str(getattr(dataset, "energy_label_smoothing", "none"))
        self.hierarchical: bool = bool(dataset.hierarchical)
        self.axis_mode: str = str(dataset.axis_mode)
        self.combine_xyz_tokens: bool = self.axis_mode == "combined"

        self.spatial_vocab += 2  # [PAD, ..., MISSING] (NO STOP)
        self.pad_id: int = 0
        self.missing_id_spatial: int = self.spatial_vocab - 1
        self.missing_id_energy: int = self.energy_vocab - 1

        self._pos_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        # Set up continuous energy bounds (used by decoding uniform bins if not operating on raw_gev)
        # But for uniform tokenization on raw GeV, we should use energy_hcal_e_min / max.
        self.token_e_min = float(getattr(dataset, "energy_hcal_e_min", 0.01))
        self.token_e_max = float(getattr(dataset, "energy_hcal_e_max", 150.0))

        if self.energy_log:
            self._e_min_const: float = float(math.log(self.token_e_min))
            self._e_max_const: float = float(math.log(self.token_e_max))
        else:
            self._e_min_const = float(self.token_e_min)
            self._e_max_const = float(self.token_e_max)
        self._e_denom_const: float = float(max(self._e_max_const - self._e_min_const, 1e-8))

    def preprocess(self, data_showers: ak.Array, incident_energy: Any) -> Dict[str, Any]:
        """Preprocesses raw shower data using feature transforms and energy-threshold hit_mask."""
        mask = get_valid_energy_mask(data_showers, self.energy_threshold)

        valid_hit_counts_per_shower = ak.sum(mask, axis=1)

        processed: Dict[str, Any] = {
            "incident_energy": incident_energy,
            "spatial_axis_names": tuple(self.raw_spatial_fields),
        }

        # Spatial features with transform
        for field in self.raw_spatial_fields:
            axis_cfg = self._as_dict(self.feature_dict.get(field, {}))
            selected_values_flat = ak.flatten(data_showers[field][mask], axis=None)
            masked_field = ak.unflatten(selected_values_flat, valid_hit_counts_per_shower)
            processed[field] = apply_feature_transform(
                masked_field,
                axis_cfg,
                invert=False,
            )

        # Energy features with transform
        energy_cfg = self._as_dict(self.feature_dict.get("energy", {}))
        selected_energy_flat = ak.flatten(data_showers["energy"][mask], axis=None)
        masked_energy = ak.unflatten(selected_energy_flat, valid_hit_counts_per_shower)
        processed["energy"] = apply_feature_transform(
            masked_energy,
            energy_cfg,
            invert=False,
        )

        packed_hit_mask = ak.unflatten(
            ak.ones_like(selected_energy_flat, dtype="bool"), valid_hit_counts_per_shower
        )
        processed["hit_mask"] = packed_hit_mask
        processed["n_hits"] = valid_hit_counts_per_shower

        return processed

    def inverse_transform_energy(self, energy: Any) -> Any:
        """Applies inverse transform to energy data to get physical values (GeV)."""
        energy_cfg = self._as_dict(self.feature_dict.get("energy", {}))
        return apply_feature_transform(energy, energy_cfg, invert=True)

    def _as_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj

        # Supporting OmegaConf DictConfig explicitly or duck-typing
        if hasattr(obj, "items") and callable(obj.items):
            return dict(obj.items())  # type: ignore[arg-type]

        if hasattr(obj, "__dict__"):
            try:
                return {k: getattr(obj, k) for k in vars(obj)}
            except Exception:
                return {}
        return {}

    def _axes_tokens_from_feats(
        self, feats: torch.Tensor, axis_names: Tuple[str, ...]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        axis_tokens: Dict[str, torch.Tensor] = {}
        for idx, axis in enumerate(axis_names):
            nbins = self.axis_nbins[axis]
            axis_tokens[axis] = self._clamp_and_shift_axis(feats[..., idx], nbins)
        energy_index = len(axis_names)
        energy = feats[..., energy_index]
        return axis_tokens, energy

    def _get_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        feats = batch.get("hit_features")
        if feats is None:
            feats = batch["part_features"]
            batch["hit_features"] = feats
        return feats

    def _get_active_axis_names(self, batch: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
        if batch is None:
            return self.spatial_axis_names
        axis_names = batch.get("spatial_axis_names") if isinstance(batch, dict) else None
        if axis_names is None:
            return self.spatial_axis_names
        axis_tuple = tuple(axis_names)
        if len(axis_tuple) != len(self.spatial_axis_names):
            raise ValueError("Inconsistent spatial_axis_names length in batch")
        return axis_tuple

    def _clamp_and_shift_axis(self, t: torch.Tensor, nbins: int) -> torch.Tensor:
        t_int = t.floor().to(torch.long)
        return t_int.clamp(0, nbins - 1) + 1

    def _tokenize_energy(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.energy_binning_mode == "hcal_resolution":
            return self._tokenize_energy_hcal(e, mask)
        return self._tokenize_energy_uniform(e, mask)

    def _tokenize_energy_uniform(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # e is currently transformed; invert it to raw_gev to decouple tokenization from continuous preprocessing
        raw_gev = self.inverse_transform_energy(e)

        # Handle NaNs by replacing them with the minimum energy value
        raw_gev = torch.nan_to_num(
            raw_gev, nan=self.token_e_min, posinf=self.token_e_max, neginf=self.token_e_min
        )
        e_clamped = torch.clamp(raw_gev, min=self.token_e_min, max=self.token_e_max)
        if self.energy_log:
            e_clamped = torch.log(e_clamped)
        norm = (e_clamped - self._e_min_const) / self._e_denom_const
        # Maps [0, 1] -> [0, V-3] (since V-1=Missing, 0=Pad)
        idx = torch.floor(norm * (self.energy_vocab - 2)).to(torch.long)
        idx = idx.clamp(0, self.energy_vocab - 3) + 1
        idx = torch.where(mask, idx, torch.zeros_like(idx))
        return idx

    def _tokenize_energy_hcal(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input `e` is in the transformed continuous space (post apply_feature_transform).
        # Invert to raw GeV, digitize against physical sigma_E-based edges.
        if self.hcal_bin_edges is None or self.hcal_bin_centers is None:
            raise RuntimeError("hcal bin tensors missing; check adapter init")
        raw_gev = self.inverse_transform_energy(e)
        raw_gev = torch.nan_to_num(raw_gev, nan=0.0, posinf=0.0, neginf=0.0)
        edges = self.hcal_bin_edges.to(device=raw_gev.device, dtype=raw_gev.dtype)
        # Use interior edges so bucketize returns indices in [0, num_bins-1]
        # for inputs in [edges[0], edges[-1]]; out-of-range are clamped.
        interior = edges[1:-1]
        idx = torch.bucketize(raw_gev, interior, right=False).to(torch.long)
        num_bins = self.hcal_bin_centers.numel()
        idx = idx.clamp(0, num_bins - 1) + 1  # reserve 0=PAD
        idx = torch.where(mask, idx, torch.zeros_like(idx))
        return idx

    def _compute_schedule_masks(
        self,
        lengths: torch.Tensor,
        seq_len: int,
        schedule: Dict[str, int],  # axis -> k_start (k_stop is implicit/ignored for validity)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = lengths.shape[0]
        device = lengths.device

        pos = self._cached_pos(seq_len, device).expand(bsz, -1)

        max_start = 0
        if schedule:
            max_start = max(schedule.values())

        total_valid_lengths = lengths + max_start
        attention_mask = pos < total_valid_lengths.unsqueeze(1)

        axis_valid_masks = {}
        for axis, k_start in schedule.items():
            start = k_start
            end = k_start + lengths.unsqueeze(1)
            is_valid = (pos >= start) & (pos < end)

            is_valid = is_valid & attention_mask

            axis_valid_masks[axis] = is_valid

        return attention_mask, axis_valid_masks

    def _apply_hierarchical_schedule(
        self,
        tokens: torch.Tensor,
        axis_valid_mask: torch.Tensor,
        k_start: int,
        missing_id: int,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """
        Shifts tokens by k_start and fills invalid regions with missing_id vs pad_id.
        tokens: [B, L_in] (unshifted data)
        axis_valid_mask: [B, L_out] (where tokens should be placed)
        """
        bsz, L_out = axis_valid_mask.shape
        L_in = tokens.shape[1]
        device = tokens.device
        pos = self._cached_pos(L_out, device).expand(bsz, -1)

        # Calculate source index: src = pos - k_start
        src_idx = pos - k_start
        # Clamp for gather safety
        src_idx_clamped = src_idx.clamp(min=0, max=L_in - 1)

        # Gather
        gathered = tokens.gather(1, src_idx_clamped)

        out = torch.where(axis_valid_mask, gathered, torch.full_like(gathered, missing_id))
        return out

    def build_ar_aligned_z_tokens(
        self,
        showers: ak.Array,
        indices: Any,
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds an AR-aligned ground-truth z token tensor for the selected showers.

        Reuses preprocess() so the energy threshold and feature transform match what
        the model was trained on. Assumes a hierarchical-split schedule with
        z_token at AR shift 1.

        Returns:
            z_tokens: [N, max_len] long, with position 0 = MISSING_z, positions
                      [1, 1 + n_hits_i) = physical z tokens for shower i, and the
                      rest = MISSING_z.
            lengths:  [N] long, post-threshold hit count per shower (suitable for
                      passing as `force_n_hits`).
        """
        if "z" not in self.spatial_axis_names:
            raise NotImplementedError(
                "build_ar_aligned_z_tokens requires a 'z' axis in the adapter"
            )

        selected = showers[indices]
        n_sel = int(len(selected))
        if n_sel == 0:
            empty = torch.empty((0, max_len), dtype=torch.long)
            return empty, torch.empty((0,), dtype=torch.long)

        processed = self.preprocess(selected, incident_energy=None)
        z_ak = processed["z"]
        lengths_np = ak.to_numpy(processed["n_hits"]).astype("int64")
        L_pad = int(max(int(lengths_np.max()) if lengths_np.size > 0 else 0, 1))

        z_padded = ak.fill_none(ak.pad_none(z_ak, L_pad, axis=1, clip=True), 0.0)
        z_tensor = torch.from_numpy(ak.to_numpy(z_padded).astype("float32"))

        z_phys_tokens = self._clamp_and_shift_axis(z_tensor, self.axis_nbins["z"])

        lengths = torch.from_numpy(lengths_np).to(dtype=torch.long)
        schedule = {"z": 1, "x": 2, "y": 3, "energy": 4}
        _, axis_valid_masks = self._compute_schedule_masks(lengths, max_len, schedule)
        z_valid_mask = axis_valid_masks["z"]

        z_missing = int(self.axis_missing_ids["z"])
        out = self._apply_hierarchical_schedule(
            z_phys_tokens,
            z_valid_mask,
            k_start=1,
            missing_id=z_missing,
            pad_id=self.pad_id,
        )
        out[:, 0] = z_missing
        return out.to(dtype=torch.long).cpu(), lengths.cpu()

    def _combine_axes(
        self, axis_tokens: Dict[str, torch.Tensor], axis_names: Tuple[str, ...]
    ) -> torch.Tensor:
        result = torch.zeros_like(next(iter(axis_tokens.values())))
        multiplier = torch.ones_like(result)
        for axis in axis_names:
            axis_value = axis_tokens[axis] - 1
            result = result + axis_value * multiplier
            multiplier = multiplier * self.axis_nbins[axis]
        return result + 1

    def _cached_pos(self, seqlen: int, device: torch.device) -> torch.Tensor:
        key = (str(device), seqlen)
        pos = self._pos_cache.get(key)
        if pos is None:
            pos = torch.arange(seqlen, device=device).unsqueeze(0)
            self._pos_cache[key] = pos
        return pos

    def prune_hit_inputs(self, batch: Dict[str, Any]) -> None:
        batch.pop("hit_features", None)
        batch.pop("part_features", None)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_spatial_metadata(batch)
        return batch

    def _ensure_spatial_metadata(self, batch: Dict[str, Any]) -> None:
        batch.setdefault("spatial_axis_names", self.spatial_axis_names)
        batch.setdefault(
            "axis_nbins",
            {axis: int(self.axis_nbins[axis]) for axis in self.spatial_axis_names},
        )

    def decode(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decodes a batch (training/val batches OR model generation output) into physical space.

        Returns:
            hit_features: [B, L, 4]  (x,y,z,energy) in physical bin integers for xyz and physical energy values
            hit_mask:     [B, L]     boolean mask of valid *physical hits*

        Key behavior:
        - Always removes PAD/MISSING via `missing_mask`.
        - If this looks like *generation output* (presence of "stop_step"), it additionally intersects with the
            stop-derived mask, shifted into physical time for hierarchical adapters.
        """
        with torch.no_grad():
            # 1) Realign tensors for hierarchical adapters (undo shifts on token streams / energy streams)
            aligned = self._realign_tensors(batch)

            # 2) Extract structure (spatial indices are 1-based, energy is decoded to physical values)
            spatial_indices, energy_val, missing_mask = self._extract_structure(aligned)

            # 3) Truncate to common length
            L = min(int(spatial_indices.shape[1]), int(energy_val.shape[1]))
            spatial_indices = spatial_indices[:, :L]
            energy_val = energy_val[:, :L]
            missing_mask = missing_mask[:, :L].to(torch.bool)

            # 4) Base mask source:
            #    - training/val: batch["hit_mask"] is physical and should be respected
            #    - generation: batch["hit_mask"] is AR-time stop mask (handled below)
            hit_mask = None
            if "hit_mask" in batch:
                hit_mask = batch["hit_mask"][:, :L].to(torch.bool)

            if bool(getattr(self, "hierarchical", False)) and "attention_mask" in batch:
                am = batch["attention_mask"]

                # Determine max_shift used during training
                max_shift = 0
                if "hsplit_shifts" in batch:
                    # HSplitAdapter stores shifts in batch
                    shifts = batch["hsplit_shifts"]
                    if isinstance(shifts, dict):
                        # Use max shift of spatial+energy axes
                        keys_to_check = list(getattr(self, "spatial_axis_names", [])) + ["energy"]
                        valid_shifts = [int(shifts.get(k, 0)) for k in keys_to_check]
                        if valid_shifts:
                            max_shift = max(valid_shifts)

                if max_shift == 0:
                    # Fallback defaults if shifts not in batch
                    if bool(getattr(self, "combine_xyz_tokens", False)):
                        max_shift = 2  # HCombined (Token=1, Energy=2)
                    else:
                        max_shift = 4  # HSplit (Z=1, X=2, Y=3, Energy=4)

                if am.dim() >= 2:
                    # attention_mask is autoregressive length.
                    # Physical valid region = unshift(AR_mask, max_shift).
                    am_phys = self._unshift_to_physical(am, max_shift)[:, :L].to(torch.bool)

                    if hit_mask is None:
                        hit_mask = am_phys
                    else:
                        hit_mask = hit_mask & am_phys

            if hit_mask is None:
                hit_mask = ~missing_mask

            # Always enforce missing removal (PAD/MISSING/out-of-range)
            hit_mask = hit_mask & (~missing_mask)

            # 4b) If this is generation output, integrate stop-derived mask correctly.
            #     We detect generation by presence of "stop_step".
            if "stop_step" in batch and "hit_mask" in batch:
                stop_mask = batch["hit_mask"][:, :L].to(torch.bool)

                # In hierarchical mode, the generator's stop_mask lives on the AR timeline,
                # while `aligned`/`spatial_indices` are physical. Shift stop_mask to physical.
                if bool(getattr(self, "hierarchical", False)):
                    # Prefer reading shifts if present (HSplitAdapter stores them as 'hsplit_shifts')
                    k_shift = 0
                    if not bool(getattr(self, "combine_xyz_tokens", False)):
                        shifts = batch.get("hsplit_shifts")
                        if isinstance(shifts, dict):
                            spatial_keys = ["z", "x", "y"]
                            spatial_shifts = [int(shifts.get(k, 1)) for k in spatial_keys]
                            k_shift = min(spatial_shifts) if spatial_shifts else 0
                        else:
                            k_shift = 1
                    else:
                        k_shift = 1

                    stop_mask = self._unshift_to_physical(stop_mask, k_shift)[:, :L]

                # Intersect: never overwrite missing handling
                hit_mask = hit_mask & stop_mask

            # 5) Denormalize spatial indices -> physical bin integers 0..nbins-1
            spatial_val = (spatial_indices.to(torch.float32) - 1.0).clamp(min=0.0)

            # 6) Assemble hit_features
            hit_features = torch.cat([spatial_val, energy_val[:, :L].unsqueeze(-1)], dim=-1)

            # 7) Zero out invalid rows
            hit_features[~hit_mask] = 0.0

            out: Dict[str, Any] = {
                "hit_features": hit_features,
                "hit_mask": hit_mask,
            }

            # Pass through conditioning if present
            for k in ("incident_energy", "shower_energy", "cond_info"):
                if k in batch:
                    out[k] = batch[k]
                    break

            return out

    def _unshift_to_physical(self, t: torch.Tensor, k: int) -> torch.Tensor:
        # t [B, L] or [B, L, ...]
        # Robustly handle shapes with dim >= 2 (e.g. [B, L, 1])
        if t.dim() < 2:
            return t

        B, L = t.shape[0], t.shape[1]

        if k >= L:
            return torch.zeros_like(t)
        if k <= 0:
            return t

        valid_part = t[:, k:]

        pad_shape = list(t.shape)
        pad_shape[1] = k
        pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)

        return torch.cat([valid_part, pad], dim=1)

    def _realign_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def _extract_structure(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement _extract_structure")

    def _decode_energy_from_batch(self, batch: Dict[str, Any], length: int) -> torch.Tensor:
        # HCal resolution mode: the continuous `energy` stream is in the
        # transformed space during training, but during generation it may hold
        # a raw-GeV bin center from the classification head. Tokens → bin
        # centers is the authoritative mapping in both cases.
        if (
            self.energy_binning_mode in ("hcal_resolution", "classification")
            and "energy_token" in batch
        ):
            return self._decode_energy_tokens_val(batch["energy_token"])

        if "energy" in batch:
            e = batch["energy"]
            if e.dim() == 3 and e.shape[2] == 1:
                e = e.squeeze(2)
            return self.inverse_transform_energy(e)

        if "energy_token" in batch:
            return self._decode_energy_tokens_val(batch["energy_token"])

        device = next(iter(batch.values())).device if batch else torch.device("cpu")
        bsz = next(iter(batch.values())).shape[0] if batch else 1
        return torch.zeros((bsz, length), device=device)

    def _decode_energy_tokens_val(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.energy_binning_mode == "hcal_resolution":
            if self.hcal_bin_centers is None:
                raise RuntimeError("hcal_bin_centers missing; check adapter init")
            centers = self.hcal_bin_centers.to(device=tokens.device, dtype=torch.float32)
            # Map token id -> bin index (safe clamp for PAD/MISSING/out-of-range).
            safe = (tokens.to(torch.long) - 1).clamp(0, centers.numel() - 1)
            return centers[safe]

        # Inverse of _tokenize_energy_uniform
        vocab_eff = max(self.energy_vocab - 2, 1)
        norm = (tokens.float() - 1.0) / float(vocab_eff)
        norm = norm.clamp(0.0, 1.0)

        if self.energy_log:
            # log_val = norm * (max_const - min_const) + min_const
            log_e = norm * self._e_denom_const + self._e_min_const
            val = torch.exp(log_e)
        else:
            val = norm * self._e_denom_const + self._e_min_const

        return val


class CombinedAdapter(BaseAdapter):
    def _realign_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def _extract_structure(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = batch.get("token_ids")
        if tokens is None:
            raise ValueError("CombinedAdapter.decode: 'token_ids' missing in batch")

        axis_tokens, missing_mask = decompose_combined_spatial_tokens(
            combined_tokens=tokens,
            axis_nbins=self.axis_nbins,
            axis_names=self.spatial_axis_names,
            missing_id_spatial=self.missing_id_spatial,
            spatial_vocab=self.spatial_vocab,
        )
        spatial_indices = torch.stack(
            [axis_tokens[axis] for axis in self.spatial_axis_names], dim=-1
        )

        # 3. Energy
        energy_val = self._decode_energy_from_batch(batch, tokens.shape[1])

        return spatial_indices, energy_val, missing_mask

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feats = self._get_features(batch)
        mask = batch["hit_mask"]
        lengths = batch["n_hits"]
        axis_names = self._get_active_axis_names(batch)
        axis_tokens, energy = self._axes_tokens_from_feats(feats, axis_names)
        spatial_token = self._combine_axes(axis_tokens, axis_names)

        # Apply mask
        spatial_token = torch.where(mask, spatial_token, torch.zeros_like(spatial_token))
        batch["token_ids"] = spatial_token

        batch["energy"] = torch.where(mask, energy.to(dtype=feats.dtype), torch.zeros_like(energy))

        energy_repr = "energy"
        if self.use_energy_tokenization:
            e_tok = self._tokenize_energy(energy, mask)
            # Mask
            e_tok = torch.where(mask, e_tok, torch.zeros_like(e_tok))
            batch["energy_token"] = e_tok
            energy_repr = "energy_token"
        batch["representation"] = "combined"
        batch["energy_representation"] = energy_repr
        batch["attention_mask"] = batch["token_ids"].ne(0)
        self._ensure_spatial_metadata(batch)
        return batch


class SplitAdapter(BaseAdapter):
    def _realign_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def _extract_structure(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stack x, y, z
        tokens_list = []
        missing_mask_list = []

        # Use first axis to determine length/device
        L = 0
        device = torch.device("cpu")

        for axis in self.spatial_axis_names:
            key = f"{axis}_token"
            if key not in batch:
                raise ValueError(f"SplitAdapter.decode: '{key}' missing")
            t = batch[key]

            # Axis-specific missing ID
            missing_id = getattr(self, f"missing_id_{axis}")
            # Missing Mask: PAD (0) or MISSING_ID
            is_miss = (t == 0) | (t == missing_id)

            # Sanitize token for concatenation (replace invalid with 1)
            safe_t = torch.where(is_miss, torch.ones_like(t), t)
            tokens_list.append(safe_t)
            missing_mask_list.append(is_miss)

            L = t.shape[1]
            device = t.device

        spatial_indices = torch.stack(tokens_list, dim=-1)  # [B, L, 3]

        # Combine missing masks (OR)
        if missing_mask_list:
            missing_mask = missing_mask_list[0]
            for m in missing_mask_list[1:]:
                missing_mask = missing_mask | m
        else:
            missing_mask = torch.zeros((1, L), dtype=torch.bool, device=device)

        energy_val = self._decode_energy_from_batch(batch, L)

        return spatial_indices, energy_val, missing_mask

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feats = self._get_features(batch)
        mask = batch["hit_mask"]
        lengths = batch["n_hits"]
        axis_names = self._get_active_axis_names(batch)
        axis_tokens, energy = self._axes_tokens_from_feats(feats, axis_names)
        for axis in axis_names:
            tok = axis_tokens[axis]
            batch[f"{axis}_token"] = torch.where(mask, tok, torch.zeros_like(tok))

        e_val = energy.to(dtype=feats.dtype)
        batch["energy"] = torch.where(mask, e_val, torch.zeros_like(e_val))

        energy_repr = "energy"
        if self.use_energy_tokenization:
            e_tok = self._tokenize_energy(energy, mask)
            e_tok = torch.where(mask, e_tok, torch.zeros_like(e_tok))
            batch["energy_token"] = e_tok
            energy_repr = "energy_token"
        batch["representation"] = "split"
        batch["energy_representation"] = energy_repr
        batch_mask = mask.bool()
        batch["attention_mask"] = batch_mask
        self._ensure_spatial_metadata(batch)
        return batch


class HCombinedAdapter(CombinedAdapter):
    def _realign_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # HCombined Logic:
        # Token Shift: 1
        # Energy Shift: 2
        # Energy Token Shift: 2 (if present)
        out = dict(batch)

        if "token_ids" in batch:
            out["token_ids"] = self._unshift_to_physical(batch["token_ids"], 1)

        if "energy" in batch:
            out["energy"] = self._unshift_to_physical(batch["energy"], 2)

        if "energy_token" in batch:
            out["energy_token"] = self._unshift_to_physical(batch["energy_token"], 2)

        return out

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feats = self._get_features(batch)
        mask = batch["hit_mask"]
        lengths = batch["n_hits"]
        axis_names = self._get_active_axis_names(batch)
        axis_tokens, energy = self._axes_tokens_from_feats(feats, axis_names)

        for axis in axis_names:
            batch[f"physical_{axis}_token"] = axis_tokens[axis].clone()
        batch["physical_energy"] = energy.clone()
        batch["physical_energy_gev"] = self.inverse_transform_energy(energy.clone())

        # 1. Schedule
        schedule = {"token_ids": 1, "energy": 2}

        # 2. Masks
        bsz, L = energy.shape
        attention_mask, axis_valid_masks = self._compute_schedule_masks(lengths, L, schedule)

        batch["hit_mask"] = mask  # Physical
        batch["attention_mask"] = attention_mask  # Hierarchical AR Mask
        batch["axis_valid_mask"] = axis_valid_masks

        # Check for length overflow as requested
        max_shift = 2
        # (Optional assertion logic)

        # 3. Spatial Token
        spatial_token = self._combine_axes(axis_tokens, axis_names)
        k_start = 1
        valid_mask = axis_valid_masks["token_ids"]
        out_ids = self._apply_hierarchical_schedule(
            spatial_token, valid_mask, k_start, self.missing_id_spatial, self.pad_id
        )
        out_ids = torch.where(attention_mask, out_ids, torch.zeros_like(out_ids))
        batch["token_ids"] = out_ids

        # 4. Energy
        k_start_e = 2
        e_valid_mask = axis_valid_masks["energy"]

        # Sanitize energy (No NaNs contract)
        energy_san = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

        pos = self._cached_pos(L, energy.device).expand(bsz, -1)
        src_idx = (pos - k_start_e).clamp(min=0, max=L - 1)
        e_shifted = energy_san.gather(1, src_idx)
        batch["energy"] = torch.where(e_valid_mask, e_shifted, torch.zeros_like(e_shifted))

        energy_repr = "energy"
        if self.use_energy_tokenization:
            e_tok = self._tokenize_energy(energy, mask)
            # Use missing_id_energy
            miss_id_e = self.missing_id_energy
            e_tok_out = self._apply_hierarchical_schedule(
                e_tok, e_valid_mask, k_start_e, miss_id_e, self.pad_id
            )
            e_tok_out = torch.where(attention_mask, e_tok_out, torch.zeros_like(e_tok_out))
            batch["energy_token"] = e_tok_out
            energy_repr = "energy_token"

        batch["representation"] = "hierarchical"
        batch["energy_representation"] = energy_repr
        self.prune_hit_inputs(batch)
        self._ensure_spatial_metadata(batch)
        return batch


class HSplitAdapter(SplitAdapter):
    def _realign_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # HSplit Logic:
        # Z: start 1
        # X: start 2
        # Y: start 3
        # Energy: start 4
        # Energy Token: start 4 (if present)

        shifts = {"z_token": 1, "x_token": 2, "y_token": 3, "energy": 4, "energy_token": 4}

        out = dict(batch)

        for k, shift in shifts.items():
            if k in batch:
                out[k] = self._unshift_to_physical(batch[k], shift)

        return out

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feats = self._get_features(batch)
        mask = batch["hit_mask"]
        lengths = batch["n_hits"]
        axis_names = self._get_active_axis_names(batch)
        axis_tokens, energy = self._axes_tokens_from_feats(feats, axis_names)

        for axis in axis_names:
            batch[f"physical_{axis}_token"] = axis_tokens[axis].clone()
        batch["physical_energy"] = energy.clone()
        batch["physical_energy_gev"] = self.inverse_transform_energy(energy.clone())

        # 1. Define Schedule (Hierarchical Split)
        # z=1, x=2, y=3, energy=4
        schedule = {"z": 1, "x": 2, "y": 3, "energy": 4}
        batch["hsplit_shifts"] = schedule

        # 2. Masks
        bsz, L = energy.shape
        attention_mask, axis_valid_masks = self._compute_schedule_masks(lengths, L, schedule)
        batch["hit_mask"] = mask  # Physical
        batch["attention_mask"] = attention_mask  # Hierarchical AR Mask
        batch["axis_valid_mask"] = axis_valid_masks

        # Verify Lengths
        max_shift = 4
        # (Optional assertion logic)

        # 3. Apply Shifts to Axes (with MISSING_ID)
        for axis in axis_names:
            k_start = schedule.get(axis, 0)
            valid_mask = axis_valid_masks[axis]

            # Use axis-specific missing ID
            missing_id = getattr(self, f"missing_id_{axis}")

            out_tok = self._apply_hierarchical_schedule(
                axis_tokens[axis], valid_mask, k_start, missing_id, self.pad_id
            )
            # Ensure PAD outside attention_mask
            out_tok = torch.where(attention_mask, out_tok, torch.zeros_like(out_tok))
            batch[f"{axis}_token"] = out_tok

        # 4. Apply Shifts to Energy
        k_start_e = schedule["energy"]
        e_valid_mask = axis_valid_masks["energy"]

        # Sanitize energy (No NaNs contract)
        energy_san = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

        # Energy Values (Continuous) - Zero out invalid
        # Manually shift for continuous
        pos = self._cached_pos(L, energy.device).expand(bsz, -1)
        src_idx = (pos - k_start_e).clamp(min=0, max=L - 1)
        e_shifted = energy_san.gather(1, src_idx)
        batch["energy"] = torch.where(e_valid_mask, e_shifted, torch.zeros_like(e_shifted))

        # Energy Tokens (Optional)
        energy_repr = "energy"
        if self.use_energy_tokenization:
            e_tok = self._tokenize_energy(energy, mask)
            # Use missing_id_energy
            miss_id_e = self.missing_id_energy
            e_tok_out = self._apply_hierarchical_schedule(
                e_tok, e_valid_mask, k_start_e, miss_id_e, self.pad_id
            )
            e_tok_out = torch.where(attention_mask, e_tok_out, torch.zeros_like(e_tok_out))
            batch["energy_token"] = e_tok_out
            energy_repr = "energy_token"

        batch["representation"] = "hierarchical"
        batch["energy_representation"] = energy_repr
        self.prune_hit_inputs(batch)
        self._ensure_spatial_metadata(batch)
        return batch


def build_adapter(dataset: DatasetConfig) -> BaseAdapter:
    # 1. Hierarchical (AR) Mode takes precedence
    mode = dataset.axis_mode
    if dataset.hierarchical:
        if mode == "split":
            return HSplitAdapter(dataset)
        elif mode == "combined":
            return HCombinedAdapter(dataset)

    # 2. Standard Modes
    if mode == "split":
        return SplitAdapter(dataset)
    elif mode == "combined":
        return CombinedAdapter(dataset)

    # 3. Default to Split
    return SplitAdapter(dataset)
