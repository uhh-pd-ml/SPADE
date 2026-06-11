from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import time
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from humite.core.utils.pylogger import get_pylogger

from ..config.schemas import DatasetConfig
from ..data.preprocessing.adapters import build_adapter
from .generative_eval_helpers import (
    PlotInputs,
    get_first_tensor,
    is_main_process,
    physical_to_ak,
    plot_all,
)


class GenerativeEvalCallback(L.Callback):
    def __init__(
        self,
        output_dir: Optional[str] = None,
        n_val_gen_showers: int = 8,
        n_val_reference_showers: Optional[int] = None,
        starting_at_epoch: int = 0,
        every_n_epochs: int = 1,
        run_test_at_epoch_zero: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        spatial_nbins: Optional[Tuple[int, int, int]] = None,
        show_inline: bool = False,
        energy_transform_params: Optional[Any] = None,
        artifacts: Optional[Sequence[str]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        bins: Optional[Dict[str, Any]] = None,
        dist_gather: bool = True,
        incident_energies: Optional[Sequence[float]] = None,
        split_mode: str = "stride",  # "stride" or "chunk"
        SanityCheckCallback: bool = False,
        generate_target_inc_energies: bool = False,
        plot_individual_showers: bool = False,
    ) -> None:
        super().__init__()
        self.logger = get_pylogger(__name__)
        self.output_dir = output_dir
        self.n_val_gen_showers = int(n_val_gen_showers)
        default_ref = max(256, self.n_val_gen_showers)
        self.n_val_reference_showers = (
            int(n_val_reference_showers) if n_val_reference_showers is not None else default_ref
        )
        self.starting_at_epoch = int(starting_at_epoch)
        self.every_n_epochs = int(every_n_epochs)
        self.run_test_at_epoch_zero = bool(run_test_at_epoch_zero)

        # Validation Cache (Stored as Physical Tensors on CPU)
        self._cached_ref_features: List[torch.Tensor] = []
        self._cached_ref_mask: List[torch.Tensor] = []
        self._cached_ref_incident: List[torch.Tensor] = []
        self._cached_ref_incident_cond: List[torch.Tensor] = []
        self._cached_ref_n_hits_cond: List[torch.Tensor] = []
        self._current_ref_count: int = 0
        self._ref_cache_full: bool = False
        self._did_prefill_ref_cache: bool = False

        # Configuration
        self.plot_individual_showers = bool(plot_individual_showers)
        self.sampling: Dict[str, Any] = dict(sampling or {})

        self.spatial_nbins = spatial_nbins

        # Artifacts
        self.artifacts = (
            set(str(a) for a in artifacts)
            if artifacts is not None
            else {"figures", "images", "bundle"}
        )
        self.should_save_figures = "figures" in self.artifacts
        self.should_save_images = "images" in self.artifacts
        self.should_save_bundle = "bundle" in self.artifacts
        self.should_save_plots = self.should_save_images or self.should_save_figures

        self.show_inline = bool(show_inline)
        self.energy_transform_params = energy_transform_params
        energy_params = self._resolve_config_dict(self.energy_transform_params)
        self._ref_energy_params = energy_params if energy_params else None
        self.dataset_config = dict(dataset_config) if isinstance(dataset_config, dict) else None

        self.adapter = None
        if self.dataset_config:
            try:
                clean_config = {k: v for k, v in self.dataset_config.items() if k != "_target_"}
                self.adapter = build_adapter(DatasetConfig(**clean_config))
            except Exception as e:
                self.logger.warning(f"GenerativeEvalCallback: Failed to build adapter: {e}")

        self.bins_overrides = self._resolve_config_dict(bins)
        self.dist_gather = bool(dist_gather)
        self.generate_target_inc_energies = bool(generate_target_inc_energies)
        self.incident_energies_cfg = (
            list(incident_energies) if incident_energies is not None else None
        )
        self.split_mode = str(split_mode)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pass

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._did_prefill_ref_cache and not (dist.is_available() and dist.is_initialized()):
            self._prefill_reference_cache(trainer)
        if self._current_ref_count >= self.n_val_reference_showers:
            self._ref_cache_full = True
            return
        self._ref_cache_full = False

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # We process every batch to build up the reference cache
        if batch_idx < 0:
            return  # Sanity check skips

        self._cache_reference_from_batch(batch)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._should_run_epoch(trainer):
            return

        # Ensure Model has correct Energy Transforms
        if self.dataset_config:
            feats = self.dataset_config.get("features", {})
            if (
                hasattr(pl_module, "set_incident_energy_transform_params")
                and "incident_energy" in feats
            ):
                pl_module.set_incident_energy_transform_params(feats["incident_energy"])

        epoch = int(trainer.current_epoch)

        # Consolidate Reference Cache
        ref_feats = (
            torch.cat(self._cached_ref_features, dim=0) if self._cached_ref_features else None
        )
        ref_mask = torch.cat(self._cached_ref_mask, dim=0) if self._cached_ref_mask else None
        ref_inc = (
            torch.cat(self._cached_ref_incident, dim=0) if self._cached_ref_incident else None
        )
        ref_inc_cond = (
            torch.cat(self._cached_ref_incident_cond, dim=0)
            if self._cached_ref_incident_cond
            else None
        )
        ref_n_hits_cond = (
            torch.cat(self._cached_ref_n_hits_cond, dim=0)
            if self._cached_ref_n_hits_cond
            else None
        )
        ref_len = ref_mask.sum(dim=1).to(torch.long) if ref_mask is not None else None
        ref_total = int(ref_feats.shape[0]) if ref_feats is not None else 0

        if self.dist_gather and dist.is_available() and dist.is_initialized():
            payload = {
                "feats": ref_feats.detach().cpu() if ref_feats is not None else None,
                "mask": ref_mask.detach().cpu() if ref_mask is not None else None,
                "inc": ref_inc.detach().cpu() if ref_inc is not None else None,
                "inc_cond": ref_inc_cond.detach().cpu() if ref_inc_cond is not None else None,
                "n_hits_cond": ref_n_hits_cond.detach().cpu()
                if ref_n_hits_cond is not None
                else None,
            }
            gather_list: list[Any] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_list, payload)
            feats_list: list[torch.Tensor] = []
            mask_list: list[torch.Tensor] = []
            inc_list: list[torch.Tensor] = []
            inc_cond_list: list[torch.Tensor] = []
            n_hits_cond_list: list[torch.Tensor] = []
            for item in gather_list:
                if not isinstance(item, dict):
                    continue
                feats_item = item.get("feats")
                mask_item = item.get("mask")
                inc_item = item.get("inc")
                inc_cond_item = item.get("inc_cond")
                n_hits_cond_item = item.get("n_hits_cond")
                if feats_item is not None:
                    feats_list.append(feats_item)
                if mask_item is not None:
                    mask_list.append(mask_item)
                if inc_item is not None:
                    inc_list.append(inc_item)
                if inc_cond_item is not None:
                    inc_cond_list.append(inc_cond_item)
                if n_hits_cond_item is not None:
                    n_hits_cond_list.append(n_hits_cond_item)
            ref_feats = torch.cat(feats_list, dim=0) if feats_list else None
            ref_mask = torch.cat(mask_list, dim=0) if mask_list else None
            ref_inc = torch.cat(inc_list, dim=0) if inc_list else None
            ref_inc_cond = torch.cat(inc_cond_list, dim=0) if inc_cond_list else None
            ref_n_hits_cond = torch.cat(n_hits_cond_list, dim=0) if n_hits_cond_list else None
            ref_len = ref_mask.sum(dim=1).to(torch.long) if ref_mask is not None else None
            ref_total = int(ref_feats.shape[0]) if ref_feats is not None else 0
        else:
            if not is_main_process(trainer):
                return

        if is_main_process(trainer) and ref_total < self.n_val_reference_showers:
            self.logger.info(
                f"[GenerativeEval] Reference cache {ref_total}/{self.n_val_reference_showers} (accumulating)"
            )

        modes_to_run = []

        # Helper to slice reference data
        cache_full = ref_total >= self.n_val_reference_showers

        def get_ref_slice(mask=None, limit=None):
            if ref_feats is None:
                return None, None, None, None, (None, None, None)

            s_feats, s_mask, s_inc, s_len = ref_feats, ref_mask, ref_inc, ref_len
            s_inc_cond, s_n_hits_cond = ref_inc_cond, ref_n_hits_cond
            if mask is not None:
                s_feats, s_mask = s_feats[mask], s_mask[mask]  # type: ignore
                s_inc = s_inc[mask] if s_inc is not None else None  # type: ignore
                s_len = s_len[mask] if s_len is not None else None  # type: ignore
                s_inc_cond = s_inc_cond[mask] if s_inc_cond is not None else None  # type: ignore
                s_n_hits_cond = (
                    s_n_hits_cond[mask] if s_n_hits_cond is not None else None  # type: ignore
                )

            # Limit amount
            if limit is not None and s_feats.shape[0] > limit:
                if cache_full:
                    gen = torch.Generator()
                    gen.manual_seed(int(epoch))
                    idx = torch.randperm(s_feats.shape[0], generator=gen)[:limit]
                    s_feats, s_mask = s_feats[idx], s_mask[idx]  # type: ignore
                    if s_inc is not None:
                        s_inc = s_inc[idx]
                    if s_len is not None:
                        s_len = s_len[idx]
                    if s_inc_cond is not None:
                        s_inc_cond = s_inc_cond[idx]
                    if s_n_hits_cond is not None:
                        s_n_hits_cond = s_n_hits_cond[idx]
                else:
                    s_feats, s_mask = s_feats[:limit], s_mask[:limit]  # type: ignore
                    if s_inc is not None:
                        s_inc = s_inc[:limit]
                    if s_len is not None:
                        s_len = s_len[:limit]
                    if s_inc_cond is not None:
                        s_inc_cond = s_inc_cond[:limit]
                    if s_n_hits_cond is not None:
                        s_n_hits_cond = s_n_hits_cond[:limit]

            return (
                s_inc,
                s_len,
                s_inc_cond,
                s_n_hits_cond,
                physical_to_ak(s_feats, s_mask, s_inc),
            )  # type: ignore

        # 1. Always Run Full Spectrum (Matched)
        (
            fs_custom_inc,
            fs_custom_len,
            fs_custom_inc_cond,
            fs_custom_n_hits_cond,
            fs_ref_ak,
        ) = get_ref_slice(limit=self.n_val_reference_showers)
        modes_to_run.append(
            (
                "matched",
                fs_custom_inc,
                fs_custom_len,
                fs_custom_inc_cond,
                fs_custom_n_hits_cond,
                fs_ref_ak,
            )
        )

        # 2. Add Specific Targets (if requested)
        if self.generate_target_inc_energies:
            # Windowed Matched Generation for specific targets
            targets = [20.0, 50.0, 80.0]
            for t in targets:
                # Filter for window +- 5 GeV
                mask = None
                if ref_inc is not None:
                    flat_inc = ref_inc.view(-1)
                    mask = (flat_inc >= (t - 5.0)) & (flat_inc <= (t + 5.0))

                # Limit by n_val_gen_showers (max to generate)
                (
                    custom_inc,
                    custom_len,
                    custom_inc_cond,
                    custom_n_hits_cond,
                    ref_ak,
                ) = get_ref_slice(mask=mask, limit=self.n_val_gen_showers)

                if custom_inc is not None and len(custom_inc) > 0:
                    modes_to_run.append(
                        (
                            float(t),
                            custom_inc,
                            custom_len,
                            custom_inc_cond,
                            custom_n_hits_cond,
                            ref_ak,
                        )
                    )

        # 3. Add Configured Incidents (if requested)
        if self.incident_energies_cfg:
            # Explicit config: Run fixed points with full (sliced) ref context
            _, cfg_len, cfg_inc_cond, cfg_n_hits_cond, cfg_ref_ak = get_ref_slice(
                limit=self.n_val_reference_showers
            )
            for e in self.incident_energies_cfg:
                modes_to_run.append(
                    (float(e), None, cfg_len, cfg_inc_cond, cfg_n_hits_cond, cfg_ref_ak)
                )

        sampling_args = self._resolve_sampling_args(pl_module)
        out_dir = self._get_output_directory(trainer)

        for (
            target_e,
            custom_inc,
            custom_len,
            custom_inc_cond,
            custom_n_hits_cond,
            ref_ak,
        ) in modes_to_run:
            # Logging Counts
            n_gen = len(custom_inc) if custom_inc is not None else self.n_val_gen_showers
            n_ref = len(ref_ak[0]) if (ref_ak is not None and ref_ak[0] is not None) else 0
            self.logger.info(
                f"[GenerativeEval] Mode: Target={target_e} | To Generate: {n_gen} | Reference Available: {n_ref}"
            )

            self._run_single_validation(
                trainer,
                pl_module,
                epoch,
                target_e,
                sampling_args,
                out_dir,
                ref_ak,
                custom_incident_energies=custom_inc,
                custom_n_hits=custom_len,
                custom_incident_cond=custom_inc_cond,
                custom_n_hits_cond=custom_n_hits_cond,
            )

    def _run_single_validation(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        epoch: int,
        target_e: Union[float, str],
        sampling_args: Dict[str, Any],
        out_dir: str,
        ref_inputs_ak: Tuple[Optional[ak.Array], Optional[ak.Array], Optional[ak.Array]],
        custom_incident_energies: Optional[torch.Tensor] = None,
        custom_n_hits: Optional[torch.Tensor] = None,
        custom_incident_cond: Optional[torch.Tensor] = None,
        custom_n_hits_cond: Optional[torch.Tensor] = None,
    ) -> None:
        # 1. Prepare Incident Energy
        world = int(getattr(trainer, "world_size", 1) or 1)
        rank = int(getattr(trainer, "global_rank", 0) or 0)

        # Calculate budget per rank
        n_global_target = (
            int(custom_incident_energies.shape[0])
            if custom_incident_energies is not None
            else self.n_val_gen_showers
        )
        n_local_budget = n_global_target // world
        if rank < (n_global_target % world):
            n_local_budget += 1
        start_idx = rank * (n_global_target // world) + min(rank, n_global_target % world)

        local_len = None
        local_inc_cond = None
        local_n_hits_cond = None
        if custom_incident_energies is not None:
            # Matched Generation: Use local reference energies found
            end_idx = start_idx + n_local_budget
            local_inc = custom_incident_energies[start_idx:end_idx]
            local_len = custom_n_hits[start_idx:end_idx] if custom_n_hits is not None else None
            local_inc_cond = (
                custom_incident_cond[start_idx:end_idx]
                if custom_incident_cond is not None
                else None
            )
            local_n_hits_cond = (
                custom_n_hits_cond[start_idx:end_idx] if custom_n_hits_cond is not None else None
            )
            n_local = int(local_inc.shape[0])
            local_inc = local_inc.to(pl_module.device)
            if local_len is not None:
                local_len = local_len.to(pl_module.device)
            if local_inc_cond is not None:
                local_inc_cond = local_inc_cond.to(pl_module.device)
            if local_n_hits_cond is not None:
                local_n_hits_cond = local_n_hits_cond.to(pl_module.device)
        else:
            # Standard Generation: Split global budget
            n_local = n_local_budget
            local_len = custom_n_hits.to(pl_module.device) if custom_n_hits is not None else None
            if local_len is not None and local_len.shape[0] > n_local:
                local_len = local_len[:n_local]
            if custom_incident_cond is not None:
                local_inc_cond = custom_incident_cond[:n_local].to(pl_module.device)
            if custom_n_hits_cond is not None:
                local_n_hits_cond = custom_n_hits_cond[:n_local].to(pl_module.device)

            if isinstance(target_e, (int, float)) and target_e > 0:
                local_inc = torch.full(
                    (n_local, 1), float(target_e), device=pl_module.device, dtype=torch.float32
                )
            else:
                # Fallback: [10..100]
                local_inc = torch.rand((n_local, 1), device=pl_module.device) * 90.0 + 10.0

        if n_local == 0:
            self._gather_generations({}, local_inc[:0].clone(), trainer, pl_module)
            return

        gen_sampling_args = dict(sampling_args)
        if local_inc_cond is not None:
            gen_sampling_args["incident_energy_cond"] = local_inc_cond
        if local_n_hits_cond is not None:
            gen_sampling_args["n_hits_cond"] = local_n_hits_cond

        # 2. Generate
        gen_local, x_shower, total_time = self._generate_chunks(
            pl_module, local_inc, gen_sampling_args, trainer, local_len
        )

        compute_time_ms = (total_time * 1000.0) / n_local if n_local > 0 else 0.0
        self.log(  # type: ignore
            "gen/time-per-shower_ms",
            compute_time_ms,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        if self.adapter:
            token_sources = {
                k: v
                for k, v in gen_local.items()
                if torch.is_tensor(v) and (k in {"token_ids"} or k.endswith("_token"))
            }
            token_mask = None
            for mkey in ("hit_mask", "attention_mask"):
                if mkey in gen_local and torch.is_tensor(gen_local[mkey]):
                    token_mask = gen_local[mkey]
                    break
            decode_input: Dict[str, torch.Tensor] = {}
            for k, v in gen_local.items():
                if torch.is_tensor(v):
                    decode_input[k] = v
            decode_input["incident_energy"] = x_shower
            decoded = self.adapter.decode(decode_input)
            gen_local["hit_features"] = decoded["hit_features"]
            gen_local["hit_mask"] = decoded["hit_mask"]
            for k in list(gen_local.keys()):
                if k in {"token_ids", "raw_energy", "energy_token"} or k.endswith("_token"):
                    del gen_local[k]

        # 4. Gather
        # Gathers hit_features/hit_mask + incident
        gen_data, x_shower_global = self._gather_generations(
            gen_local, x_shower, trainer, pl_module
        )

        # 5. Process & Plot (Rank 0 only)
        if is_main_process(trainer):
            # Convert Gen to AK
            if "hit_features" in gen_data and "hit_mask" in gen_data:
                phys_gen, glob_gen, feat_gen = physical_to_ak(
                    gen_data["hit_features"], gen_data["hit_mask"], x_shower_global
                )

                # Unpack Reference
                phys_ref, glob_ref, feat_ref = ref_inputs_ak

                # Make PlotInputs
                inputs = PlotInputs(
                    features_gen=feat_gen,
                    features_ref=feat_ref,
                    phys_gen=phys_gen,
                    phys_ref=phys_ref,
                    glob_gen=glob_gen,
                    glob_ref=glob_ref,
                )

                # Setup Subdir
                if target_e == "matched":
                    label = "energy_matched"
                elif isinstance(target_e, str) and target_e.startswith("matched"):
                    label = f"energy_{target_e}"
                elif isinstance(target_e, (int, float)) and target_e > 0:
                    label = f"energy_{int(target_e)}GeV"
                else:
                    label = "energy_mixed"

                sub_dir = os.path.join(out_dir, label)
                os.makedirs(sub_dir, exist_ok=True)

                # Save Bundle
                if self.should_save_bundle:
                    self._save_generation_artifacts(sub_dir, gen_data, x_shower_global, epoch)

                # Plot
                figures = plot_all(
                    inputs,
                    sub_dir,
                    epoch,
                    self.spatial_nbins,
                    should_save=self.should_save_plots,
                    bins_overrides=self.bins_overrides,
                    plot_individual_showers=self.plot_individual_showers,
                )

                # Log
                if self.should_save_plots:
                    images = [os.path.join(sub_dir, fname) for _, fname in figures]
                    self._log_images_to_experiment(trainer, figures, images, prefix=f"{label}_")

                # Cleanup
                for fig, _ in figures:
                    try:
                        plt.close(fig)  # type: ignore
                    except:
                        pass

    def _prefill_reference_cache(self, trainer: L.Trainer) -> None:
        if self._did_prefill_ref_cache or self._ref_cache_full:
            return
        if dist.is_available() and dist.is_initialized():
            self._did_prefill_ref_cache = True
            return
        if self.n_val_reference_showers <= 0:
            self._did_prefill_ref_cache = True
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if not self.dist_gather and not is_main_process(trainer):
            return

        dataloaders = getattr(trainer, "val_dataloaders", None)
        if isinstance(dataloaders, (list, tuple)):
            dataloader = dataloaders[0] if dataloaders else None
        else:
            dataloader = dataloaders

        if dataloader is None:
            dm = getattr(trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "val_dataloader"):
                dataloader = dm.val_dataloader()

        if dataloader is None:
            self._did_prefill_ref_cache = True
            return

        max_prefill_batches = int(self.n_val_reference_showers)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self._ref_cache_full or batch_idx >= max_prefill_batches:
                    break
                self._cache_reference_from_batch(batch)

        self._did_prefill_ref_cache = True

    def _cache_reference_from_batch(self, batch: Any) -> None:
        if self._ref_cache_full or self._current_ref_count >= self.n_val_reference_showers:
            self._ref_cache_full = True
            return
        if not isinstance(batch, dict):
            return

        inc_key = None
        for k in ["incident_energy", "shower_energy", "cond_info"]:
            if k in batch:
                inc_key = k
                break

        if inc_key is None:
            self.logger.debug(
                "GenerativeEvalCallback: No incident energy key found in validation batch."
            )
            return

        feats = None
        mask = None
        inc = None
        inc_cond = None
        n_hits_cond = None

        if self.adapter:
            try:
                decoded = self.adapter.decode(batch)
                if "hit_features" in decoded and "hit_mask" in decoded:
                    feats = decoded["hit_features"].detach().cpu()
                    mask = decoded["hit_mask"].detach().cpu().to(torch.bool)

                    inc_raw = decoded.get(inc_key)
                    if isinstance(inc_raw, torch.Tensor):
                        inc_raw_t = cast(torch.Tensor, inc_raw)
                        inc = inc_raw_t.detach().cpu()
                    else:
                        inc_raw = batch.get(inc_key)
                        if isinstance(inc_raw, torch.Tensor):
                            inc_raw_t = cast(torch.Tensor, inc_raw)
                            inc = inc_raw_t.detach().cpu()
            except Exception as e:
                self.logger.debug(f"Failed to decode validation batch: {e}")

        if "incident_energy_cond" in batch and torch.is_tensor(batch["incident_energy_cond"]):
            inc_cond = batch["incident_energy_cond"].detach().cpu()
        if "n_hits_cond" in batch and torch.is_tensor(batch["n_hits_cond"]):
            n_hits_cond = batch["n_hits_cond"].detach().cpu()

        if feats is None or mask is None or inc is None:
            return
        if inc_cond is None or n_hits_cond is None:
            return

        remaining = self.n_val_reference_showers - self._current_ref_count
        if remaining <= 0:
            self._ref_cache_full = True
            return
        if feats.shape[0] > remaining:
            feats = feats[:remaining]
            mask = mask[:remaining]
            inc = inc[:remaining]
            inc_cond = inc_cond[:remaining]
            n_hits_cond = n_hits_cond[:remaining]

        self._cached_ref_features.append(feats)
        self._cached_ref_mask.append(mask)
        self._cached_ref_incident.append(inc)
        self._cached_ref_incident_cond.append(inc_cond)
        self._cached_ref_n_hits_cond.append(n_hits_cond)
        self._current_ref_count += int(feats.shape[0])
        if self._current_ref_count >= self.n_val_reference_showers:
            self._ref_cache_full = True

    def _resolve_sampling_args(self, pl_module: L.LightningModule) -> Dict[str, Any]:
        sampling_args = dict(self.sampling)
        if "temperature" not in sampling_args:
            sampling_args["temperature"] = 1.0
        if "greedy" not in sampling_args:
            sampling_args["greedy"] = False
        if "use_kv_cache" not in sampling_args:
            sampling_args["use_kv_cache"] = True
        return sampling_args

    def _should_run_epoch(self, trainer: L.Trainer) -> bool:
        epoch = int(trainer.current_epoch)
        if self.run_test_at_epoch_zero and epoch == 0:
            return True
        if epoch < self.starting_at_epoch:
            return False
        return (epoch - self.starting_at_epoch) % max(self.every_n_epochs, 1) == 0

    def _generate_chunks(
        self,
        pl_module: L.LightningModule,
        local_inc: torch.Tensor,
        sampling_args: Dict[str, Any],
        trainer: Optional[L.Trainer] = None,
        local_n_hits: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
        n = int(local_inc.shape[0])
        generation_batch_size = int(self.sampling.get("batch_size", 64) or 64)
        generation_batch_size = int(max(1, min(generation_batch_size, max(n, 1))))
        sampling_args_without_batch_size = dict(sampling_args)
        sampling_args_without_batch_size.pop("batch_size", None)
        if local_n_hits is not None:
            sampling_args_without_batch_size.pop("n_hits", None)
        chunk_outputs: dict[str, list[torch.Tensor]] = {}
        incident_chunks: list[torch.Tensor] = []
        length_chunks: list[torch.Tensor] = []
        total_forward_time = 0.0
        if n == 0:
            return {}, local_inc[:0].clone(), total_forward_time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.inference_mode():
            for chunk_start in range(0, n, generation_batch_size):
                chunk_end = min(chunk_start + generation_batch_size, n)
                chunk_inc_cpu = local_inc[chunk_start:chunk_end].detach().clone()
                chunk_inc = chunk_inc_cpu.to(pl_module.device)
                chunk_len = None
                if local_n_hits is not None:
                    chunk_len_cpu = local_n_hits[chunk_start:chunk_end].detach().clone()
                    chunk_len = chunk_len_cpu.to(pl_module.device)
                chunk_sampling_args = dict(sampling_args_without_batch_size)
                incident_cond = chunk_sampling_args.get("incident_energy_cond")
                if torch.is_tensor(incident_cond):
                    if incident_cond.shape[0] == n:
                        chunk_sampling_args["incident_energy_cond"] = incident_cond[
                            chunk_start:chunk_end
                        ]
                    elif incident_cond.shape[0] == int(chunk_inc.shape[0]):
                        chunk_sampling_args["incident_energy_cond"] = incident_cond
                n_hits_cond = chunk_sampling_args.get("n_hits_cond")
                if torch.is_tensor(n_hits_cond):
                    if n_hits_cond.shape[0] == n:
                        chunk_sampling_args["n_hits_cond"] = n_hits_cond[chunk_start:chunk_end]
                    elif n_hits_cond.shape[0] == int(chunk_inc.shape[0]):
                        chunk_sampling_args["n_hits_cond"] = n_hits_cond
                # Diagnostic: force exact per-sample n_hits and bypass stop head.
                force_diag = bool(chunk_sampling_args.pop("force_n_hits_for_diagnostic", False))
                if force_diag and chunk_len is not None:
                    chunk_sampling_args["force_n_hits"] = chunk_len
                    # Ensure max_len covers the longest forced sample plus grace steps.
                    needed_max_len = int(chunk_len.max().item()) + 5
                    existing_max_len = int(chunk_sampling_args.get("max_len", 0) or 0)
                    if needed_max_len > existing_max_len:
                        chunk_sampling_args["max_len"] = needed_max_len
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                chunk_wall = time.time()
                chunk_gen = pl_module.generate_physical_showers(
                    batch_size=int(chunk_inc.shape[0]),
                    adapter=self.adapter,
                    x_shower=chunk_inc,
                    n_hits=chunk_len,
                    **chunk_sampling_args,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_forward_time += time.time() - chunk_wall
                incident_chunks.append(chunk_inc_cpu)
                if local_n_hits is not None and chunk_len is not None:
                    length_chunks.append(chunk_len_cpu)
                for key, value in chunk_gen.items():
                    if torch.is_tensor(value):
                        chunk_outputs.setdefault(key, []).append(value.detach().cpu())
                del chunk_gen
                del chunk_inc
        gen_local: Dict[str, torch.Tensor] = {}
        for key, tensors in chunk_outputs.items():
            if not tensors:
                continue
            gen_local[key] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

        x_shower = torch.cat(incident_chunks, dim=0) if incident_chunks else local_inc[:0].clone()
        if length_chunks:
            gen_local["n_hits"] = torch.cat(length_chunks, dim=0)
        return gen_local, x_shower, total_forward_time

    def _gather_generations(
        self,
        gen_local: Dict[str, torch.Tensor],
        x_shower: torch.Tensor,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        world = int(getattr(trainer, "world_size", 1) or 1)
        if not (self.dist_gather and world > 1 and dist.is_available() and dist.is_initialized()):
            return gen_local, x_shower
        payload_local: Dict[str, Any] = {"incident_energy": x_shower.detach().cpu()}
        # Only gather physical features and mask if present, plus stopping diagnostics
        keys_to_gather = [
            "hit_features",
            "hit_mask",
            "energy_token",
            "stop_score",
            "stop_step",
            "stop_pred",
        ]

        for key in keys_to_gather:
            if key in gen_local and torch.is_tensor(gen_local[key]):
                payload_local[key] = gen_local[key].detach().cpu()

        gather_list: list[Any] = [None for _ in range(world)]
        dist.all_gather_object(gather_list, payload_local)

        if not is_main_process(trainer):
            return {}, torch.empty(0)

        combined: Dict[str, list[torch.Tensor]] = {}
        for payload in gather_list:
            if not isinstance(payload, dict):
                continue
            for key, val in payload.items():
                if torch.is_tensor(val):
                    combined.setdefault(key, []).append(val)
        gen: Dict[str, torch.Tensor] = {}
        for key, tensors in combined.items():
            gen[key] = torch.cat(tensors, dim=0)
        x_shower_global = torch.cat(combined.get("incident_energy", []), dim=0)
        return gen, x_shower_global

    def _resolve_config_dict(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Helper to safely resolve OmegaConf or dict configs."""
        if cfg is None:
            return {}
        if isinstance(cfg, MappingABC):
            if isinstance(cfg, dict):
                return dict(cfg)
            resolved = OmegaConf.to_container(cfg, resolve=True)
            return dict(resolved) if isinstance(resolved, dict) else {}
        return {}

    def _get_output_directory(self, trainer: L.Trainer) -> str:
        if self.output_dir:
            return self.output_dir
        return os.path.join(trainer.default_root_dir, "plots")

    def _save_generation_artifacts(
        self,
        out_dir: str,
        gen_data: Dict[str, torch.Tensor],
        incident_energy: torch.Tensor,
        epoch: int,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if self.should_save_bundle:
            save_path = os.path.join(out_dir, f"val_gen_epoch{epoch}.pt")
            cpu_data = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in gen_data.items()
            }
            cpu_data["incident_energy"] = incident_energy.detach().cpu()
            try:
                torch.save(cpu_data, save_path)
            except Exception as exc:
                self.logger.warning("Failed to save generation bundle %s: %s", save_path, exc)
                try:
                    os.remove(save_path)
                except OSError:
                    pass

    def _log_images_to_experiment(self, trainer, figures, images, prefix: str = ""):
        comet = None
        for lg in trainer.loggers:
            exp = getattr(lg, "experiment", None)
            if exp is None:
                continue
            if hasattr(exp, "log_image"):
                comet = exp
                break

        if comet is not None:
            step = trainer.global_step
            if images and hasattr(comet, "log_image"):
                for image_path in images:
                    name = os.path.basename(image_path)
                    if prefix:
                        name = f"{prefix}{name}"
                    comet.log_image(image_path, step=step, name=name)
