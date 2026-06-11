"""Batch collation utilities for turning variable-length Awkward events into padded tensors.

This collator expects per-event dicts containing geometry-native spatial axes,
energy, and incident_energy. It pads to cfg.data.pad_length, regularizes lists,
stacks into a dense NumPy array, and returns torch tensors suitable for the model
input pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import torch

from ...config.schemas import DatasetConfig
from ..geometry import get_spatial_field_names
from .utils import ak_pad, ak_to_np_stack, apply_feature_transform


class BatchCollator:
    def __init__(self, dataset: DatasetConfig) -> None:
        self.pad_length = dataset.pad_length
        self.raw_spatial_fields = get_spatial_field_names(dataset.dataset_type)
        self.feature_config = self._as_dict(getattr(dataset, "features", {}))

    def _as_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "items") and callable(obj.items):
            return dict(obj.items())
        if hasattr(obj, "__dict__"):
            try:
                return {k: getattr(obj, k) for k in vars(obj)}
            except Exception:
                return {}
        return {}

    def __call__(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        spatial_axis_names: Optional[Tuple[str, str, str]] = None
        axis_lists: Dict[str, List[Any]] = {}
        e_list: List[Any] = []
        energy_list: List[float] = []
        hit_mask_list: List[Any] = []
        n_hits_list: List[Any] = []

        for i, event in enumerate(batch_list):
            missing_energy = [k for k in ("energy", "incident_energy") if k not in event]
            if missing_energy:
                available = sorted(event.keys())
                raise ValueError(
                    f"BatchCollator: event {i} is missing required fields {missing_energy}. Available fields: {available}"
                )

            event_axis_names = tuple(event.get("spatial_axis_names", self.raw_spatial_fields))
            if len(event_axis_names) != len(self.raw_spatial_fields):
                raise ValueError(
                    "BatchCollator: spatial_axis_names must contain exactly three entries. "
                    f"Got {event_axis_names}."
                )
            if spatial_axis_names is None:
                spatial_axis_names = event_axis_names
                axis_lists = {axis: [] for axis in spatial_axis_names}
            elif spatial_axis_names != event_axis_names:
                raise ValueError(
                    "BatchCollator: mixed spatial_axis_names within a batch are not supported."
                )

            for axis_name in event_axis_names:
                if axis_name not in event:
                    available = sorted(event.keys())
                    raise ValueError(
                        f"BatchCollator: event {i} missing spatial field '{axis_name}'. Available fields: {available}"
                    )
                axis_lists[axis_name].append(event[axis_name])

            e_list.append(event["energy"])
            energy_list.append(event["incident_energy"])
            hit_mask_list.append(event["hit_mask"])
            n_hits_list.append(event["n_hits"])

        if spatial_axis_names is None:
            spatial_axis_names = tuple(self.raw_spatial_fields)
            axis_lists = {axis: [] for axis in spatial_axis_names}

        showers_dict = {
            axis_name: ak.Array(axis_lists[axis_name]) for axis_name in spatial_axis_names
        }
        showers_dict["energy"] = ak.Array(e_list)
        showers = ak.zip(showers_dict)

        padded_fields = {}
        for field_name in list(spatial_axis_names) + ["energy"]:
            padded_fields[field_name] = ak_pad(
                ak.Array(showers[field_name]), self.pad_length, return_mask=False
            )

        padded = ak.zip(padded_fields)
        mask = ak_pad(ak.Array(hit_mask_list), self.pad_length, fill_value=False)

        padded = ak.to_regular(padded, axis=1)
        mask = ak.to_regular(mask, axis=1)

        np_feats = ak_to_np_stack(
            padded,
            names=list(spatial_axis_names) + ["energy"],
            axis=-1,
        )
        np_feats = np.asarray(np_feats, dtype=np.float32)

        feats_t = torch.from_numpy(np_feats)
        np_mask = ak.to_numpy(mask)
        mask_t = torch.from_numpy(np.asarray(np_mask, dtype=np.bool_))
        # hit_mask is physical hits; attention_mask is derived later in adapters.
        energies_t = torch.from_numpy(np.asarray(energy_list, dtype=np.float32))
        n_hits = torch.from_numpy(np.asarray(n_hits_list, dtype=np.int32)).clamp(
            max=self.pad_length
        )
        incident_cfg = self._as_dict(self.feature_config.get("incident_energy", {}))
        n_hits_cfg = self._as_dict(self.feature_config.get("n_hits", {}))
        incident_energy_cond = apply_feature_transform(energies_t, incident_cfg)
        n_hits_cond = apply_feature_transform(n_hits.to(dtype=torch.float32), n_hits_cfg)

        batch: Dict[str, Any] = {
            "hit_features": feats_t,
            "hit_mask": mask_t,
            "incident_energy": energies_t,
            "incident_energy_cond": incident_energy_cond,
            "spatial_axis_names": spatial_axis_names,
            "n_hits": n_hits,
            "n_hits_cond": n_hits_cond,
        }
        return batch
