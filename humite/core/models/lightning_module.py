from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim

from ..data.preprocessing.adapters import BaseAdapter
from ..data.preprocessing.utils import decompose_combined_spatial_tokens
from ..utils.pylogger import get_pylogger
from .heads import mixture_expected_value, mixture_gaussian_nll
from .physics_stop import PhysicsStopModel, build_physics_stop_targets, compute_physics_stop_loss


def _build_10_80_10_targets(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Soft targets for cross-entropy: 0.8 on true bin, 0.1 on each adjacent bin.

    Smoothing is strictly across the vocabulary dimension (no sequence-dim
    leakage). Boundary bins collapse to [0.9, 0.1] / [0.1, 0.9] via the
    additive accumulation (duplicated index falls back on the center bin).
    """
    flat = tokens.reshape(-1).long()
    num = int(flat.numel())
    device = tokens.device
    soft = torch.zeros((num, vocab_size), dtype=torch.float32, device=device)
    row = torch.arange(num, device=device)
    center = flat.clamp(0, vocab_size - 1)
    left = (center - 1).clamp(min=0)
    right = (center + 1).clamp(max=vocab_size - 1)
    soft[row, center] = 0.8
    soft[row, left] += 0.1  # if left == center (at 0), this re-adds 0.1 to center
    soft[row, right] += 0.1  # likewise at the right boundary
    # Renormalize so PAD targets (tokens == 0) and any edge case sum to 1.
    soft = soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return soft


class BackboneLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        token_pad_id: int = 0,
        energy_loss_weight: float = 1.0,
        energy_loss_min_weight: float = 0.0,
        energy_loss_warmup_steps: int = 0,
        optimizer_name: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lookahead_alpha: float = 0.5,
        lookahead_k: int = 6,
        lr_scheduler_name: str = "none",
        lr_scheduler_warmup_steps: int = 0,
        lr_scheduler_total_steps: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        backbone_lr_scale: float = 1.0,
        backbone_unfreeze_steps: int = 0,
        sampling: Optional[Dict[str, Any]] = None,
        axis_mode: Optional[str] = None,
        use_z_causal_mask: bool = False,
        physics_stop_model: Optional[PhysicsStopModel] = None,
        physics_stop_lr: float = 3e-4,
        physics_stop_weight_decay: float = 0.01,
        physics_stop_loss_weight: float = 1.0,
        physics_stop_pos_weight: float = 0.0,
        physics_stop_min_hits: int = 5,
        physics_stop_auto_freeze: bool = True,
        physics_stop_freeze_min_epochs: int = 2,
        physics_stop_freeze_patience: int = 4,
        physics_stop_freeze_min_delta: float = 1e-4,
        physics_stop_freeze_min_recall: float = 0.0,
        physics_stop_freeze_min_precision: float = 0.0,
        physics_stop_energy_sum_noise_sigma: float = 0.0,
        physics_stop_sampling_fraction: float = 1.0,
        training_mode: str = "all",
        energy_label_smoothing: str = "none",
    ) -> None:
        super().__init__()
        self.model = model
        self._z_mask_enabled = bool(use_z_causal_mask)
        self.save_hyperparameters(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "token_pad_id": token_pad_id,
                "energy_loss_weight": energy_loss_weight,
                "energy_loss_min_weight": energy_loss_min_weight,
                "energy_loss_warmup_steps": int(max(0, energy_loss_warmup_steps)),
                "optimizer_name": optimizer_name.lower(),
                "betas": betas,
                "eps": eps,
                "lookahead_alpha": lookahead_alpha,
                "lookahead_k": lookahead_k,
                "lr_scheduler_name": lr_scheduler_name,
                "lr_scheduler_warmup_steps": int(max(0, lr_scheduler_warmup_steps)),
                "lr_scheduler_total_steps": int(max(0, lr_scheduler_total_steps)),
                "lr_scheduler_min_lr": lr_scheduler_min_lr,
                "backbone_lr_scale": float(backbone_lr_scale),
                "backbone_unfreeze_steps": int(max(0, backbone_unfreeze_steps)),
                "sampling": sampling,
                "axis_mode": axis_mode,
                "use_z_causal_mask": use_z_causal_mask,
                "physics_stop_lr": float(physics_stop_lr),
                "physics_stop_weight_decay": float(physics_stop_weight_decay),
                "physics_stop_loss_weight": float(physics_stop_loss_weight),
                "physics_stop_pos_weight": float(physics_stop_pos_weight),
                "physics_stop_min_hits": int(max(1, physics_stop_min_hits)),
                "physics_stop_auto_freeze": bool(physics_stop_auto_freeze),
                "physics_stop_freeze_min_epochs": int(max(1, physics_stop_freeze_min_epochs)),
                "physics_stop_freeze_patience": int(max(1, physics_stop_freeze_patience)),
                "physics_stop_freeze_min_delta": float(max(0.0, physics_stop_freeze_min_delta)),
                "physics_stop_freeze_min_recall": float(
                    max(0.0, min(1.0, physics_stop_freeze_min_recall))
                ),
                "physics_stop_freeze_min_precision": float(
                    max(0.0, min(1.0, physics_stop_freeze_min_precision))
                ),
                "physics_stop_energy_sum_noise_sigma": float(
                    max(0.0, physics_stop_energy_sum_noise_sigma)
                ),
                "physics_stop_sampling_fraction": float(max(1e-8, physics_stop_sampling_fraction)),
                "training_mode": str(training_mode),
            }
        )
        self.training_mode = str(training_mode)
        self.energy_label_smoothing = str(energy_label_smoothing)
        self.sampling_cfg = sampling or {}
        self.axis_mode = axis_mode
        self.msg_logger = get_pylogger(__name__)
        self._epoch_wallclock_start: Optional[float] = None
        self._throughput_samples: int = 0
        self._energy_transform_params: Optional[Dict[str, Any]] = None
        self._energy_inverse_multiply: float = 1.0
        self._energy_inverse_subtract: float = 0.0
        self._energy_inverse_callable: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._energy_debug_logs: int = 0
        self._energy_debug_log_limit: int = 3
        self.log_scale_min: float = -4.0
        self.log_scale_max: float = 4.0
        self.scale_epsilon: float = 1e-4
        self._backbone_frozen: bool = False

        self.physics_stop_model: Optional[PhysicsStopModel] = physics_stop_model
        self._physics_stop_frozen: bool = False
        self._physics_stop_best_val_loss: float = float("inf")
        self._physics_stop_bad_epochs: int = 0
        self._manual_gradient_clip_val: Optional[float] = None
        self._manual_gradient_clip_algorithm: str = "norm"
        self._manual_accum_grad_batches: int = 1

        # Apply mode-specific parameter freezing before DDP wrapping.
        self._apply_training_mode_parameter_freezing()

        if self.physics_stop_model is not None and self.training_mode in (
            "all",
            "physics_stop_only",
        ):
            self.automatic_optimization = False

    def _apply_training_mode_parameter_freezing(self) -> None:
        if self.training_mode == "physics_stop_only":
            if self.physics_stop_model is None:
                raise ValueError(
                    "training_mode='physics_stop_only' requires use_physics_stop=true in config"
                )
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        return self.model(*args, **kwargs)

    def _determine_global_rank(self) -> int:
        try:
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        try:
            tr = getattr(self, "trainer", None)
            if tr is not None:
                if hasattr(tr, "global_rank"):
                    return int(tr.global_rank)
                strategy = getattr(tr, "strategy", None)
                if strategy is not None and hasattr(strategy, "global_rank"):
                    return int(strategy.global_rank)
        except Exception:
            pass
        for env_key in ("RANK", "WORLD_RANK"):
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    return int(env_val)
                except Exception:
                    continue
        return 0

    def _current_energy_loss_weight(self) -> float:
        target = float(self.hparams.energy_loss_weight)  # type: ignore[attr-defined]
        start = float(self.hparams.energy_loss_min_weight)  # type: ignore[attr-defined]
        warmup = int(self.hparams.energy_loss_warmup_steps)  # type: ignore[attr-defined]
        if warmup <= 0:
            return target
        step = float(getattr(self, "global_step", 0))
        progress = min(max(step, 0.0) / float(max(warmup, 1)), 1.0)
        return start + (target - start) * progress

    def _metric_scalar(self, metric_name: str) -> Optional[float]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        callback_metrics = getattr(trainer, "callback_metrics", None)
        if callback_metrics is None:
            return None
        metric_value = callback_metrics.get(metric_name)
        if metric_value is None:
            return None
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() == 0:
                return None
            metric_value = metric_value.detach().to(torch.float32).mean().cpu().item()
        try:
            scalar = float(metric_value)
        except Exception:
            return None
        if not math.isfinite(scalar):
            return None
        return scalar

    def _freeze_physics_stop_head(self) -> None:
        if self.physics_stop_model is None or self._physics_stop_frozen:
            return
        for parameter in self.physics_stop_model.parameters():
            parameter.requires_grad = False
        self.physics_stop_model.eval()
        self._physics_stop_frozen = True
        if self._determine_global_rank() == 0:
            self.msg_logger.info(
                {
                    "event": "physics_stop_auto_frozen",
                    "epoch": int(self.current_epoch),
                    "best_val_physics_stop_loss": float(self._physics_stop_best_val_loss),
                    "bad_epochs": int(self._physics_stop_bad_epochs),
                }
            )

    def _maybe_auto_freeze_physics_stop_head(self) -> None:
        if self.physics_stop_model is None or self._physics_stop_frozen:
            return
        if not bool(getattr(self.hparams, "physics_stop_auto_freeze", True)):
            return
        min_epochs = int(getattr(self.hparams, "physics_stop_freeze_min_epochs", 5))
        if int(self.current_epoch) + 1 < max(1, min_epochs):
            return

        current_val_loss = self._metric_scalar("val/physics_stop_loss")
        if current_val_loss is None:
            return

        min_delta = float(getattr(self.hparams, "physics_stop_freeze_min_delta", 1e-4))
        if current_val_loss < (self._physics_stop_best_val_loss - max(0.0, min_delta)):
            self._physics_stop_best_val_loss = current_val_loss
            self._physics_stop_bad_epochs = 0
            return

        self._physics_stop_bad_epochs += 1
        patience = int(getattr(self.hparams, "physics_stop_freeze_patience", 4))
        if self._physics_stop_bad_epochs < max(1, patience):
            return

        min_recall = float(getattr(self.hparams, "physics_stop_freeze_min_recall", 0.0))
        min_precision = float(getattr(self.hparams, "physics_stop_freeze_min_precision", 0.0))
        recall_value = self._metric_scalar("val/physics_stop_recall")
        precision_value = self._metric_scalar("val/physics_stop_precision")
        has_required_recall = (min_recall <= 0.0) or (
            recall_value is not None and recall_value >= min_recall
        )
        has_required_precision = (min_precision <= 0.0) or (
            precision_value is not None and precision_value >= min_precision
        )
        if has_required_recall and has_required_precision:
            self._freeze_physics_stop_head()

    def _resolve_scheduler_total_steps(self) -> int:
        # Explicit override wins
        total_steps = int(getattr(self.hparams, "lr_scheduler_total_steps", 0))
        if total_steps <= 0:
            total_steps = int(getattr(self.hparams, "scheduler_total_steps", 0))
        if total_steps > 0:
            return total_steps

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0

        max_steps = int(getattr(trainer, "max_steps", 0))
        return max_steps if max_steps > 0 else 0

    def _backbone_params(self) -> List[nn.Parameter]:
        core = getattr(self.model, "core", None)
        if core is None:
            return []
        params_fn = getattr(core, "parameters", None)
        if not callable(params_fn):
            return []
        params = params_fn()
        return list(cast(Iterable[nn.Parameter], params))

    def _split_optimizer_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        backbone_params = self._backbone_params()
        if not backbone_params:
            return [], list(self.parameters())
        backbone_ids = {id(param) for param in backbone_params}
        head_params = [param for param in self.parameters() if id(param) not in backbone_ids]
        return backbone_params, head_params

    def _set_backbone_trainable(self, enabled: bool) -> None:
        backbone_params = self._backbone_params()
        if not backbone_params:
            return
        for param in backbone_params:
            param.requires_grad = bool(enabled)

    def _maybe_freeze_backbone(self) -> None:
        freeze_steps = int(getattr(self.hparams, "backbone_unfreeze_steps", 0))
        if freeze_steps <= 0 or int(getattr(self, "global_step", 0)) >= freeze_steps:
            return
        self._set_backbone_trainable(False)
        self._backbone_frozen = True
        if self._determine_global_rank() == 0:
            self.msg_logger.info(
                {
                    "backbone_freeze_until_step": int(freeze_steps),
                    "backbone_lr_scale": float(getattr(self.hparams, "backbone_lr_scale", 1.0)),
                }
            )

    def _maybe_unfreeze_backbone(self) -> None:
        if self.training_mode == "physics_stop_only":
            return
        if not self._backbone_frozen:
            return
        freeze_steps = int(getattr(self.hparams, "backbone_unfreeze_steps", 0))
        if freeze_steps <= 0 or int(getattr(self, "global_step", 0)) < freeze_steps:
            return
        self._set_backbone_trainable(True)
        self._backbone_frozen = False
        if self._determine_global_rank() == 0:
            self.msg_logger.info({"backbone_unfrozen_step": int(self.global_step)})

    def _build_cosine_warmup_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LambdaLR:
        warmup_steps = int(getattr(self.hparams, "lr_scheduler_warmup_steps", 0))
        if warmup_steps <= 0:
            warmup_steps = int(getattr(self.hparams, "scheduler_warmup_steps", 0))
        warmup_steps = max(0, warmup_steps)
        total_steps = self._resolve_scheduler_total_steps()

        # In your setup this should always be > 0; fail loudly if misconfigured.
        if total_steps <= 0:
            raise ValueError("scheduler_total_steps must be > 0 for cosine_warmup scheduler")

        # Ensure there is at least 1 decay step after warmup.
        if warmup_steps >= total_steps:
            total_steps = warmup_steps + 1

        base_lr = float(getattr(self.hparams, "lr", 0.0))
        min_lr = float(getattr(self.hparams, "lr_scheduler_min_lr", 0.0))
        if min_lr <= 0.0:
            min_lr = float(getattr(self.hparams, "scheduler_min_lr", 0.0))
        min_ratio = (max(0.0, min_lr) / base_lr) if base_lr > 0.0 else 0.0

        # Precompute reciprocals for speed/clarity
        inv_warmup = 1.0 / warmup_steps if warmup_steps > 0 else None
        decay_steps = total_steps - warmup_steps
        inv_decay = 1.0 / decay_steps  # decay_steps guaranteed > 0 by guard above

        def lr_lambda(step: int) -> float:
            # Linear warmup: 0 -> 1 over warmup_steps
            if warmup_steps > 0 and step < warmup_steps:
                return step * inv_warmup  # type: ignore[operator]

            # Cosine decay from 1 -> min_ratio
            p = (step - warmup_steps) * inv_decay  # progress in [0, 1] (clamped)
            if p <= 0.0:
                p = 0.0
            elif p >= 1.0:
                p = 1.0

            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            return min_ratio + (1.0 - min_ratio) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _get_sampling_int(self, sampling_cfg: Dict[str, Any], key: str, default: int) -> int:
        value = sampling_cfg.get(key, default)
        if value is None:
            return int(default)
        return int(value)

    def _get_sampling_float(self, sampling_cfg: Dict[str, Any], key: str, default: float) -> float:
        value = sampling_cfg.get(key, default)
        if value is None:
            return float(default)
        return float(value)

    def _filter_logits_for_sampling(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        suppress_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adjusted = logits / temperature if temperature > 0 else logits
        if suppress_token_ids is not None and suppress_token_ids.numel() > 0:
            adjusted.index_fill_(-1, suppress_token_ids, float("-inf"))
        if top_k is not None and top_k > 0:
            k = min(top_k, adjusted.size(-1))
            topk_vals, topk_idx = torch.topk(adjusted, k, dim=-1)
            filtered = torch.full_like(adjusted, float("-inf"))
            filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            adjusted = filtered
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(adjusted, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            adjusted = torch.full_like(adjusted, float("-inf"))
            adjusted.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return adjusted

    def _sample_tokens_with_probs(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        suppress_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        filtered = self._filter_logits_for_sampling(
            logits, temperature, top_k, top_p, suppress_token_ids
        )
        probs = torch.softmax(filtered, dim=-1)
        # Sanitize before multinomial: when top-p/top-k/suppression combine to mask every
        # token (or when earlier NaN propagates), softmax rows become NaN or all-zero.
        # torch.multinomial on CUDA silently returns index = vocab_size (OOB) for such rows;
        # that token is then fed to nn.Embedding on the next step and triggers an
        # asynchronous indexSelectSmallIndex assert.  Fall back to a uniform distribution
        # over the whole vocab (rather than leaving the masking in place) so generation
        # keeps making progress; hard-clamp the sampled index to [0, vocab-1] as a second
        # line of defense against the known CUDA multinomial edge case.
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        vocab = max(probs.shape[-1], 1)
        uniform = torch.full_like(probs, 1.0 / float(vocab))
        probs = torch.where(probs_sum > 0, probs / probs_sum.clamp_min(1e-8), uniform)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token = next_token.clamp(0, probs.shape[-1] - 1)
        return next_token, probs

    def _select_tokens_and_mask(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if "token_ids" not in batch:
            raise RuntimeError("combined mode requires batch['token_ids']")
        if "attention_mask" not in batch:
            raise RuntimeError("combined mode requires batch['attention_mask']")

        tokens = batch["token_ids"].long()
        mask = batch["attention_mask"].to(torch.bool)

        if tokens.shape != mask.shape:
            raise RuntimeError(
                f"token_ids and attention_mask shape mismatch: {tokens.shape} vs {mask.shape}"
            )

        return tokens, mask

    def _token_ce_loss(
        self, logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Shift for next-token prediction
        pred = logits[:, :-1, :]
        tgt = tokens[:, 1:]
        msk = mask[:, 1:]
        vocab = pred.shape[-1]
        loss_per_pos = F.cross_entropy(pred.reshape(-1, vocab), tgt.reshape(-1), reduction="none")
        loss_per_pos = loss_per_pos.view_as(msk).to(pred.dtype)
        denom = msk.sum().clamp_min(1)
        return (loss_per_pos * msk).sum() / denom

    def _apply_z_causal_mask(
        self,
        logits: torch.Tensor,
        z_current: torch.Tensor,
        pad_id: int,
        missing_id: int,
    ) -> torch.Tensor:
        pred_logits = logits[:, :-1, :]
        vocab = pred_logits.shape[-1]
        bin_idx = torch.arange(vocab, device=logits.device, dtype=torch.long)
        mask_bins = bin_idx.unsqueeze(0).unsqueeze(0) < z_current.unsqueeze(-1)
        valid_z = (z_current.ne(pad_id)) & (z_current.ne(missing_id))
        apply_mask = valid_z.unsqueeze(-1) & mask_bins
        masked_pred = pred_logits.masked_fill(apply_mask, -1e9)
        return torch.cat([masked_pred, logits[:, -1:, :]], dim=1)

    def _apply_z_causal_mask_step(
        self,
        logits: torch.Tensor,
        z_current: torch.Tensor,
        pad_id: int,
        missing_id: int,
    ) -> torch.Tensor:
        vocab = logits.shape[-1]
        bin_idx = torch.arange(vocab, device=logits.device, dtype=torch.long)
        mask_bins = bin_idx.unsqueeze(0) < z_current.unsqueeze(-1)
        valid_z = (z_current.ne(pad_id)) & (z_current.ne(missing_id))
        apply_mask = valid_z.unsqueeze(-1) & mask_bins
        return logits.masked_fill(apply_mask, -1e9)

    def _energy_loss(
        self, out: dict, batch: dict, mask: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        target = batch.get("energy", None)
        if target is None:
            return None

        # Pure classification path (HCal-resolution binning): the head emits
        # `energy_logits` and no MOG keys; loss is CE (optionally 10-80-10
        # smoothed across the vocab dim only).
        energy_logits = out.get("energy_logits")
        if getattr(self.model, "use_energy_tokenization", False) and energy_logits is not None:
            energy_tokens = batch.get("energy_token")
            if energy_tokens is None:
                return None
            logits_shift = energy_logits[:, :-1, :]
            tokens_shift = energy_tokens[:, 1:].long()
            mask_shift = mask[:, 1:]
            V = int(logits_shift.shape[-1])

            if self.energy_label_smoothing == "10-80-10":
                soft = _build_10_80_10_targets(tokens_shift, V)
                log_probs = F.log_softmax(logits_shift.reshape(-1, V), dim=-1)
                ce = -(soft * log_probs).sum(dim=-1).view_as(mask_shift)
            else:
                ce = F.cross_entropy(
                    logits_shift.reshape(-1, V),
                    tokens_shift.reshape(-1),
                    reduction="none",
                ).view_as(mask_shift)

            mask_f = mask_shift.float()
            denom = mask_f.sum().clamp_min(1.0)
            energy_ce = (ce * mask_f).sum() / denom
            return energy_ce, {"energy_ce": energy_ce}

        # Energy Tokenization (Bin Classification + Regression)
        if getattr(self.model, "use_energy_tokenization", False):
            energy_tokens = batch.get("energy_token")
            logits = out.get("mog_logits")
            means = out.get("mog_means")
            log_scales = out.get("mog_log_scales")

            if logits is None or means is None or log_scales is None or energy_tokens is None:
                return None

            # Shift for AR: Predict t+1 from t
            logits_shift = logits[:, :-1, :]
            means_shift = means[:, :-1, :]
            log_scales_shift = log_scales[:, :-1, :]

            # Target and IDs at t+1
            target_shift = target[:, 1:]
            if target_shift.dim() == 2:
                target_shift = target_shift.unsqueeze(-1)

            tokens_shift = energy_tokens[:, 1:].long()
            mask_shift = mask[:, 1:]

            # 1. Classification Loss (Cross Entropy)
            vocab_size = logits_shift.shape[-1]
            ce_loss = F.cross_entropy(
                logits_shift.reshape(-1, vocab_size),
                tokens_shift.reshape(-1),
                reduction="none",
            )
            ce_loss = ce_loss.view_as(mask_shift)

            # 2. Regression Loss (Gathered Gaussian NLL)
            safe_tokens = tokens_shift.clamp(0, vocab_size - 1)
            gather_idx = safe_tokens.unsqueeze(-1)  # [B, T-1, 1]

            mu = torch.gather(means_shift, -1, gather_idx)
            log_sigma = torch.gather(log_scales_shift, -1, gather_idx)

            # Apply constraints
            log_sigma = torch.clamp(log_sigma, min=float(self.log_scale_min))
            log_sigma = torch.clamp(log_sigma, max=float(self.log_scale_max))

            sigma = torch.exp(log_sigma)
            if self.scale_epsilon > 0:
                sigma = sigma + self.scale_epsilon
                log_sigma = torch.log(sigma)

            var = sigma.pow(2)

            # Gaussian NLL
            mse_term = 0.5 * ((target_shift - mu).pow(2)) / var
            const_term = 0.5 * torch.log(
                torch.tensor(2 * 3.141592653589793, device=target.device, dtype=target.dtype)
            )
            nll_term = mse_term + log_sigma + const_term
            nll_term = nll_term.squeeze(-1)

            total_loss = ce_loss + nll_term

            masked_loss = total_loss * mask_shift.float()
            mask_float = mask_shift.float()
            denom = mask_shift.sum().clamp_min(1.0)

            avg_ce = (ce_loss * mask_float).sum() / denom
            avg_nll = (nll_term * mask_float).sum() / denom
            return masked_loss.sum() / denom, {"energy_ce": avg_ce, "energy_nll": avg_nll}

        pred = out.get("raw_energy")  # [B,T,1]
        if pred is None:
            return None
        if pred.dim() >= 3:
            pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
            pred = pred[..., :1]
        elif pred.dim() == 2:
            pred = pred.unsqueeze(-1)
        else:
            raise RuntimeError("energy_prediction_dim_invalid")
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        mask_f = mask.to(dtype=pred.dtype).unsqueeze(-1)

        if out.get("energy_head_type") == "mog":
            logits = out["mog_logits"][:, :-1, :]
            means = out["mog_means"][:, :-1, :]
            log_scales = out["mog_log_scales"][:, :-1, :]

            # Shift targets and mask to align with AR prediction (t -> t+1)
            target_shift = target[:, 1:]
            mask_shift = mask[:, 1:]

            nll = mixture_gaussian_nll(
                target_shift,
                logits,
                means,
                log_scales,
                mask=mask_shift,
                log_scale_min=self.log_scale_min,
                log_scale_max=self.log_scale_max,
                scale_epsilon=self.scale_epsilon,
            )
            return nll, {"energy_mog_nll": nll.detach()}
        # Fallback to MSE
        # Shift predictions and targets
        pred_shift = pred[:, :-1, :]
        target_shift = target[:, 1:]
        mask_shift = mask[:, 1:]

        if target_shift.dim() == 2:
            target_shift = target_shift.unsqueeze(-1)
        mask_f = mask_shift.to(dtype=pred.dtype).unsqueeze(-1)

        mse = ((pred_shift - target_shift) ** 2) * mask_f
        denom = mask_f.sum().clamp_min(1)
        return mse.sum() / denom, {}

    def _extract_step_energy(
        self,
        out: dict,
        step_index: int,
        fallback: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = int(fallback.shape[0])
        dtype = fallback.dtype

        # Pure classification path (HCal-resolution binning): sample token from
        # logits, map to `_energy_bin_centers[token]` (raw GeV) for continuous.
        energy_logits = out.get("energy_logits")
        if energy_logits is not None:
            if energy_logits.dim() == 2:
                step_logits = energy_logits
            else:
                step_logits = energy_logits.select(dim=1, index=step_index)
            step_logits = step_logits.reshape(batch_size, -1)
            V = int(step_logits.shape[-1])
            # Mask PAD (0) and MISSING (V-1) so we never sample reserved ids.
            forbid = torch.zeros((1, V), dtype=torch.bool, device=step_logits.device)
            forbid[0, 0] = True
            if V >= 2:
                forbid[0, V - 1] = True
            safe_logits = step_logits.masked_fill(forbid, -1e9)
            energy_T = float(getattr(self, "_energy_temperature", 1.0))
            if energy_T != 1.0:
                safe_logits = safe_logits / max(energy_T, 1e-6)
            probs = torch.softmax(safe_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp(min=0.0)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            if V > 2:
                uniform = torch.zeros_like(probs)
                uniform[..., 1 : V - 1] = 1.0 / float(V - 2)
            else:
                uniform = torch.full_like(probs, 1.0 / float(max(V, 1)))
            probs = torch.where(probs_sum > 0, probs / probs_sum.clamp_min(1e-8), uniform)
            # Always sample a discrete token — needed for the AR energy embedding
            # at the next step regardless of the continuous decode mode.
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if V > 2:
                token = token.clamp(1, V - 2)
            else:
                token = token.clamp(0, max(V - 1, 0))

            centers = getattr(self.model, "_energy_bin_centers", None)
            decode_mode = getattr(self, "_energy_decode_mode", "center")

            if decode_mode == "logit_weighted" and centers is not None:
                # Deterministic expected-value over all valid bins.
                # probs already has -1e9 on PAD(0) and MISSING(V-1), so
                # softmax effectively zeroes them; slice [1 : 1+N] extracts
                # the N data-token probabilities aligned with centers.
                centers_t = centers.to(device=step_logits.device, dtype=dtype)
                N = centers_t.numel()
                valid_probs = probs[:, 1 : 1 + N]  # [B, N]
                renorm = valid_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                value = ((valid_probs / renorm) * centers_t.unsqueeze(0)).sum(dim=-1, keepdim=True)
            elif decode_mode == "uniform_dequant":
                # Sample uniformly within the predicted bin [lo, hi).
                # Bins are narrow (0.5 σ_E), so this fills histogram gaps
                # without any model change.
                edges = getattr(self.model, "_energy_bin_edges", None)
                if edges is not None:
                    edges_t = edges.to(device=token.device, dtype=dtype)
                    # token is 1-indexed; lower edge = edges_t[token-1],
                    # upper edge = edges_t[token].
                    safe_t = token.clamp(1, edges_t.numel() - 1)
                    lo = edges_t[safe_t - 1]  # [B]
                    hi = edges_t[safe_t]  # [B]
                    u = torch.rand_like(lo)
                    value = (lo + u * (hi - lo)).unsqueeze(-1)
                elif centers is not None:
                    # Edges not available; fall back to bin center.
                    centers_t = centers.to(device=token.device, dtype=dtype)
                    safe_token = (token - 1).clamp(0, centers_t.numel() - 1)
                    value = centers_t[safe_token].unsqueeze(-1)
                else:
                    value = token.to(dtype=dtype).unsqueeze(-1)
            else:  # "center" (default)
                if centers is None:
                    # Centers not wired; downstream adapter token path handles decode.
                    value = token.to(dtype=dtype).unsqueeze(-1)
                else:
                    centers_t = centers.to(device=token.device, dtype=dtype)
                    safe_token = (token - 1).clamp(0, centers_t.numel() - 1)
                    value = centers_t[safe_token].unsqueeze(-1)

            return value, token

        logits = out.get("mog_logits")
        means = out.get("mog_means")
        log_scales = out.get("mog_log_scales")
        if logits is not None and means is not None and log_scales is not None:
            clamp_tuple = getattr(self.model, "energy_sample_clamp", None)
            if isinstance(clamp_tuple, (tuple, list)) and len(clamp_tuple) == 2:
                clamp_min = float(clamp_tuple[0])
                clamp_max = float(clamp_tuple[1])
            else:
                clamp_min, clamp_max = -5.0, 5.0
            nan_cfg = getattr(self.model, "energy_nan_replacement", None)
            nan_value = float(nan_cfg.get("nan", 0.0)) if isinstance(nan_cfg, dict) else 0.0
            posinf_value = (
                float(nan_cfg.get("posinf", clamp_max)) if isinstance(nan_cfg, dict) else clamp_max
            )
            neginf_value = (
                float(nan_cfg.get("neginf", clamp_min)) if isinstance(nan_cfg, dict) else clamp_min
            )
            if logits.dim() == 2:
                logits_slice = logits
            else:
                logits_slice = logits.select(dim=1, index=step_index)
            if means.dim() == 2:
                means_slice = means
            else:
                means_slice = means.select(dim=1, index=step_index)
            if log_scales.dim() == 2:
                log_scales_slice = log_scales
            else:
                log_scales_slice = log_scales.select(dim=1, index=step_index)
            if self.log_scale_min is not None:
                log_scales_slice = torch.clamp(log_scales_slice, min=float(self.log_scale_min))
            elif self.log_scale_max is not None:
                log_scales_slice = torch.clamp(log_scales_slice, max=float(self.log_scale_max))
            logits_view = logits_slice.reshape(batch_size, -1)
            means_view = means_slice.reshape(batch_size, -1)
            log_scales_view = log_scales_slice.reshape(batch_size, -1)
            energy_T = float(getattr(self, "_energy_temperature", 1.0))
            if energy_T != 1.0:
                logits_view = logits_view / max(energy_T, 1e-6)
            weight_logits = torch.softmax(logits_view, dim=-1)
            weight_logits = torch.nan_to_num(weight_logits, nan=0.0)
            weight_sum = weight_logits.sum(dim=-1, keepdim=True)
            normalized_weights = weight_logits / weight_sum.clamp_min(1e-8)
            components = max(weight_logits.shape[-1], 1)
            uniform = torch.full_like(weight_logits, 1.0 / float(components))
            invalid_mask = weight_sum <= 0
            mixture_weights = torch.where(invalid_mask, uniform, normalized_weights)
            # Clamp to strictly non-negative before multinomial: floating-point
            # division (weight_logits / weight_sum) can produce tiny negatives
            # that cause multinomial to return K (out-of-bounds) on CUDA.
            mixture_weights = mixture_weights.clamp(min=0.0)
            component_index = torch.multinomial(mixture_weights, num_samples=1).squeeze(-1)
            # Hard safety clamp: guard against the rare CUDA multinomial edge-case
            # where it returns K instead of K-1 for near-zero weight inputs.
            component_index = component_index.clamp(0, means_view.shape[1] - 1)
            gather_index = component_index.unsqueeze(-1)
            selected_means = means_view.gather(dim=1, index=gather_index)
            scales_view = F.softplus(log_scales_view) + self.scale_epsilon
            selected_scales = scales_view.gather(dim=1, index=gather_index)
            dist = torch.distributions.Normal(selected_means, selected_scales)
            sampled = dist.sample()
            sampled = torch.nan_to_num(
                sampled, nan=nan_value, posinf=posinf_value, neginf=neginf_value
            )
            sampled = torch.clamp(sampled, min=clamp_min, max=clamp_max)
            return sampled.to(dtype=dtype), component_index
        raw = out.get("raw_energy")
        if raw is None:
            return fallback, None
        step = raw.select(dim=1, index=step_index)
        step_view = step.reshape(batch_size, -1)
        if step_view.shape[1] == 0:
            return fallback, None
        if step_view.shape[1] == 1:
            return step_view.to(dtype=dtype), None
        return step_view[:, :1].to(dtype=dtype), None

    def _energy_tensor_stats(self, tensor: torch.Tensor) -> Tuple[float, float, float]:
        if tensor is None or tensor.numel() == 0:
            return 0.0, 0.0, 0.0
        clean = torch.nan_to_num(tensor.detach())
        flattened = clean.to(torch.float32).reshape(-1)
        return (
            float(flattened.min().cpu().item()),
            float(flattened.max().cpu().item()),
            float(flattened.mean().cpu().item()),
        )

    def _log_mask_info_once(
        self, mask: torch.Tensor, source: str, token_shape: Tuple[int, ...]
    ) -> None:
        if hasattr(self, "_has_logged_mask_info"):
            return
        try:
            per_item_true = mask.sum(dim=1).detach().cpu().tolist()
            self.msg_logger.debug(
                {
                    "mask/source": source,
                    "mask/shape": tuple(mask.shape),
                    "mask/true_total": int(mask.sum().detach().cpu().item()),
                    "mask/true_first5": per_item_true[:5],
                    "tokens/shape": token_shape,
                }
            )
        except Exception:
            pass
        self._has_logged_mask_info = True

    def _resolve_axis_names(self, batch: Optional[dict]) -> Tuple[str, ...]:
        if batch is not None:
            candidate = batch.get("spatial_axis_names")
            if isinstance(candidate, (list, tuple)):
                resolved = tuple(str(name) for name in candidate)
                if resolved:
                    return resolved
        model_axes = getattr(self.model, "axis_names", None)
        if isinstance(model_axes, (list, tuple)):
            resolved = tuple(str(name) for name in model_axes)
            if resolved:
                return resolved
        return ("x", "y", "z")

    def _gather_axis_tokens(self, batch: dict, axis_names: Tuple[str, ...]) -> Dict[str, Any]:
        tokens: Dict[str, torch.Tensor] = {}
        for axis in axis_names:
            key = f"{axis}_token"
            if key not in batch:
                raise RuntimeError(f"Missing {key} in batch for axis-aware training")
            tokens[axis] = batch[key].long()
        return tokens

    def _gather_axis_logits(
        self, out: Dict[str, Any], axis_names: Tuple[str, ...]
    ) -> Dict[str, Any]:
        logits: Dict[str, torch.Tensor] = {}
        for axis in axis_names:
            key = f"token_logits_{axis}"
            axis_logits = out.get(key)
            if axis_logits is None:
                raise RuntimeError(f"Missing {key} in model output")
            logits[axis] = axis_logits
        return logits

    def _compute_physics_stop_loss(
        self, batch: dict, mode: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.physics_stop_model is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {}

        physical_z = batch.get("physical_z_token")
        physical_x = batch.get("physical_x_token")
        physical_y = batch.get("physical_y_token")
        physical_energy = batch.get("physical_energy")
        n_hits = batch.get("n_hits")
        incident_energy = batch.get("incident_energy")

        if any(t is None for t in [physical_z, physical_x, physical_y]) and (
            batch.get("representation") == "hierarchical" and "token_ids" in batch
        ):
            combined_tokens = batch["token_ids"]
            combined_tokens_phys = self._unshift_sequence_to_physical(combined_tokens, shift=1)
            axis_nbins = batch.get("axis_nbins")
            axis_names = batch.get("spatial_axis_names", ("x", "y", "z"))
            if not isinstance(axis_nbins, dict):
                axis_nbins = {"x": 30, "y": 30, "z": 30}
            axis_tokens, _ = decompose_combined_spatial_tokens(
                combined_tokens=combined_tokens_phys,
                axis_nbins=axis_nbins,
                axis_names=tuple(axis_names),
            )
            physical_z = axis_tokens.get("z")
            physical_x = axis_tokens.get("x")
            physical_y = axis_tokens.get("y")
            if physical_energy is None and "energy" in batch:
                physical_energy = self._unshift_sequence_to_physical(batch["energy"], shift=2)

        if any(
            t is None
            for t in [physical_z, physical_x, physical_y, physical_energy, n_hits, incident_energy]
        ):
            zero = torch.tensor(0.0, device=self.device)
            return zero, {}

        physical_z = physical_z.to(self.device).long()
        physical_x = physical_x.to(self.device).long()
        physical_y = physical_y.to(self.device).long()
        physical_energy = physical_energy.to(self.device).float()
        physical_energy = torch.nan_to_num(physical_energy, nan=0.0)
        n_hits = n_hits.to(self.device)
        incident_energy = incident_energy.to(self.device).float()
        if incident_energy.dim() > 1:
            incident_energy = incident_energy.squeeze(-1)

        seq_len = physical_z.size(1)
        position_ratio = None
        if bool(getattr(self.physics_stop_model, "use_position_feature", False)):
            position_denominator = float(max(seq_len - 1, 1))
            position_indices = torch.arange(
                seq_len, device=self.device, dtype=physical_energy.dtype
            )
            position_ratio = (
                (position_indices / position_denominator)
                .unsqueeze(0)
                .expand(physical_z.size(0), -1)
            )

        # Raw hit counter: absolute position index, consistent with generation loop's t.
        # The model can learn shower-length thresholds per energy bin without normalisation.
        hit_counter = None
        if bool(getattr(self.physics_stop_model, "use_hit_counter", False)):
            hit_counter = (
                torch.arange(seq_len, device=self.device, dtype=physical_energy.dtype)
                .unsqueeze(0)
                .expand(physical_z.size(0), -1)
            )  # [B, T]

        # History: last k z-tokens and energies before each position.
        # Built via left-padded unfold so position t sees [z_{t-k}, ..., z_{t-1}].
        # Identical structure is maintained in generation via a rolling buffer.
        z_history = None
        energy_history = None
        if bool(getattr(self.physics_stop_model, "use_history", False)):
            k = int(getattr(self.physics_stop_model, "history_len", 10))
            z_padded = F.pad(physical_z, (k, 0), value=0)  # [B, T+k], long
            z_history = z_padded.unfold(1, k, 1)[:, :seq_len]  # [B, T, k]
            e_padded = F.pad(physical_energy, (k, 0), value=0.0)  # [B, T+k]
            energy_history = e_padded.unfold(1, k, 1)[:, :seq_len]  # [B, T, k]

        energy_sum = None
        if bool(getattr(self.physics_stop_model, "use_energy_sum", False)):
            physical_energy_gev = batch.get("physical_energy_gev")
            hit_mask = batch.get("hit_mask")
            if physical_energy_gev is not None:
                physical_energy_gev = physical_energy_gev.to(self.device).float()
                physical_energy_gev = torch.nan_to_num(physical_energy_gev, nan=0.0)
                noise_sigma = float(
                    getattr(self.hparams, "physics_stop_energy_sum_noise_sigma", 0.0)
                )
                if self.training and noise_sigma > 0.0:
                    # Apply noise to individual hit energies before cumsum so that
                    # the monotonic structure is preserved and adjacent positions
                    # remain correlated — unlike per-position noise on the cumsum
                    # itself, which creates spurious uniqueness that doesn't exist
                    # at validation time.
                    hit_energies_for_sum = torch.clamp(
                        physical_energy_gev
                        * (1.0 + torch.randn_like(physical_energy_gev) * noise_sigma),
                        min=0.0,
                    )
                else:
                    hit_energies_for_sum = physical_energy_gev
                if hit_mask is not None:
                    hit_energies_for_sum = hit_energies_for_sum * hit_mask.to(self.device).float()
                energy_sum_gev = torch.cumsum(hit_energies_for_sum, dim=1)
                sampling_fraction = float(
                    getattr(self.hparams, "physics_stop_sampling_fraction", 1.0)
                )
                energy_sum = (
                    energy_sum_gev
                    / incident_energy.unsqueeze(1).clamp(min=1e-8)
                    / sampling_fraction
                )

        stop_logits = self.physics_stop_model(
            physical_z,
            physical_x,
            physical_y,
            physical_energy,
            incident_energy,
            position_ratio=position_ratio,
            energy_sum=energy_sum,
            hit_counter=hit_counter,
            z_history=z_history,
            energy_history=energy_history,
        )

        stop_targets, stop_mask = build_physics_stop_targets(n_hits, seq_len, self.device)

        configured_pos_weight = float(getattr(self.hparams, "physics_stop_pos_weight", 0.0))
        if configured_pos_weight <= 0.0:
            effective_pos_weight = float(
                (stop_mask.sum() - stop_targets.sum()).clamp(min=1.0)
                / stop_targets.sum().clamp(min=1.0)
            )
        else:
            effective_pos_weight = configured_pos_weight

        loss = compute_physics_stop_loss(
            stop_logits, stop_targets, stop_mask, effective_pos_weight
        )

        loss_weight = float(getattr(self.hparams, "physics_stop_loss_weight", 1.0))
        weighted_loss = loss * loss_weight

        with torch.no_grad():
            predictions = (stop_logits > 0.0).float()
            stop_positions = stop_targets.eq(1.0) & stop_mask
            go_positions = stop_targets.eq(0.0) & stop_mask
            predicted_stop_positions = predictions.eq(1.0) & stop_mask
            stop_correct = (predictions.eq(1.0) & stop_positions).sum()
            stop_total = stop_positions.sum().clamp(min=1)
            go_correct = (predictions.eq(0.0) & go_positions).sum()
            go_total = go_positions.sum().clamp(min=1)
            predicted_stop_total = predicted_stop_positions.sum().clamp(min=1)
            valid_positions = stop_mask.sum().clamp(min=1)
            target_positive_rate = stop_positions.sum().to(torch.float32) / valid_positions.to(
                torch.float32
            )
            predicted_positive_rate = predicted_stop_positions.sum().to(
                torch.float32
            ) / valid_positions.to(torch.float32)
            precision = stop_correct.to(torch.float32) / predicted_stop_total.to(torch.float32)

            stop_logits_float = stop_logits.to(torch.float32)
            stop_logit_mean = torch.where(
                stop_positions,
                stop_logits_float,
                torch.zeros_like(stop_logits_float),
            ).sum() / stop_total.to(torch.float32)
            go_logit_mean = torch.where(
                go_positions,
                stop_logits_float,
                torch.zeros_like(stop_logits_float),
            ).sum() / go_total.to(torch.float32)

        metrics = {
            f"{mode}/physics_stop_loss": loss,
            f"{mode}/physics_stop_recall": stop_correct.float() / stop_total.float(),
            f"{mode}/physics_stop_go_accuracy": go_correct.float() / go_total.float(),
            f"{mode}/physics_stop_precision": precision,
            f"{mode}/physics_stop_target_positive_rate": target_positive_rate,
            f"{mode}/physics_stop_predicted_positive_rate": predicted_positive_rate,
            f"{mode}/physics_stop_stop_logit_mean": stop_logit_mean,
            f"{mode}/physics_stop_go_logit_mean": go_logit_mean,
            f"{mode}/physics_stop_pos_weight": torch.tensor(
                effective_pos_weight, device=stop_logits.device
            ),
        }

        return weighted_loss, metrics

    def _unshift_sequence_to_physical(self, sequence: torch.Tensor, shift: int) -> torch.Tensor:
        if sequence.dim() < 2 or shift <= 0:
            return sequence
        seq_len = int(sequence.shape[1])
        if shift >= seq_len:
            return torch.zeros_like(sequence)
        shifted = sequence[:, shift:]
        pad_shape = list(sequence.shape)
        pad_shape[1] = shift
        padding = torch.zeros(pad_shape, dtype=sequence.dtype, device=sequence.device)
        return torch.cat([shifted, padding], dim=1)

    def _step(
        self, batch: dict, mode: str
    ) -> Tuple[torch.Tensor, Dict[str, Any], int, Optional[str]]:
        """
        Shared logic for training and validation steps.
        Automatically switches between multi-axis and combined (single-stream) modes.
        """
        prepare = getattr(self.model, "ensure_split_axes", None)
        if callable(prepare):
            prepare(batch)

        axis_names = self._resolve_axis_names(batch)
        incident_energy = batch.get("incident_energy", None)

        # State Variables
        out: Dict[str, Any] = {}
        mask: Optional[torch.Tensor] = None
        mask_tensor = torch.empty((0, 0), dtype=torch.bool, device=self.device)
        axis_losses: Dict[str, torch.Tensor] = {}
        reference_axis: Optional[str] = None
        token_loss = torch.tensor(0.0, device=self.device)

        if self.axis_mode == "split":
            # --- SPLIT-AXIS PATH ---
            axis_tokens = self._gather_axis_tokens(batch, axis_names)
            reference_axis = axis_names[-1]
            reference_tokens = axis_tokens[reference_axis]

            forward_mask = batch.get("attention_mask")

            sanitized_forward = None
            if isinstance(forward_mask, torch.Tensor):
                sanitized_forward = self.model.sanitize_autoregressive_mask(forward_mask)
                batch["attention_mask"] = sanitized_forward

            attn_mask = sanitized_forward if sanitized_forward is not None else forward_mask
            out = self.model(
                batch=batch,
                attention_mask=attn_mask,
                shower_energy=incident_energy,
                mode=mode,
            )

            mask = out.get("mask")
            if mask is None:
                raise RuntimeError(f"Missing attention mask for split-axes {mode}")
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            # Sanitize mask again for loss calculation
            mask = self.model.sanitize_autoregressive_mask(mask)
            mask_tensor = cast(torch.Tensor, mask)

            axis_logits = self._gather_axis_logits(out, axis_names)
            first_axis = axis_names[0]
            self._log_mask_info_once(
                mask_tensor,
                "model.mask",
                tuple(axis_tokens[first_axis].shape),
            )

            axis_valid_masks = batch.get("axis_valid_mask")
            axis_valid_masks = axis_valid_masks if isinstance(axis_valid_masks, dict) else None
            stop_valid_mask = None
            if axis_valid_masks is not None and "z" in axis_valid_masks:
                stop_valid_mask = axis_valid_masks["z"].to(mask_tensor.device).to(torch.bool)

            for tokens in axis_tokens.values():
                if tokens.shape != mask_tensor.shape:
                    raise RuntimeError(f"Axis token shape mismatch for split-axes {mode}")

            for axis in axis_names:
                axis_mask = mask_tensor
                if axis_valid_masks is not None and axis in axis_valid_masks:
                    axis_mask = axis_valid_masks[axis].to(mask_tensor.device).to(torch.bool)
                logits_axis = axis_logits[axis]
                if axis == "z" and self._z_mask_enabled:
                    z_current = axis_tokens["z"][:, :-1]
                    pad_id = int(getattr(self.hparams, "token_pad_id", 0))
                    per_axis = getattr(self.model, "per_axis_embed", None)
                    missing_id = (
                        int(per_axis.vocab_sizes["z"] - 1)
                        if per_axis and "z" in per_axis.vocab_sizes
                        else int(logits_axis.shape[-1] - 1)
                    )
                    logits_axis = self._apply_z_causal_mask(
                        logits_axis, z_current, pad_id, missing_id
                    )
                axis_losses[axis] = self._token_ce_loss(logits_axis, axis_tokens[axis], axis_mask)
            token_loss = torch.stack(list(axis_losses.values())).sum()

        elif self.axis_mode == "combined":
            # --- COMBINED-STREAM PATH ---
            tokens, mask = self._select_tokens_and_mask(batch)
            forward_mask = batch.get("attention_mask")
            if not isinstance(forward_mask, torch.Tensor):
                forward_mask = mask

            sanitized_forward = self.model.sanitize_autoregressive_mask(forward_mask)
            batch["attention_mask"] = sanitized_forward

            out = self.model(
                batch=batch,
                attention_mask=sanitized_forward,
                shower_energy=incident_energy,
                mode=mode,
            )

            # Always derive tokens from batch; shifting is handled in the loss
            mask = out.get("mask", sanitized_forward)
            mask = self.model.sanitize_autoregressive_mask(mask)
            mask_tensor = cast(torch.Tensor, mask)

            self._log_mask_info_once(
                mask_tensor,
                "model.mask" if out.get("mask") is not None else "fallback.selector",
                tuple(tokens.shape),
            )

            axis_valid_masks = batch.get("axis_valid_mask")
            axis_valid_masks = axis_valid_masks if isinstance(axis_valid_masks, dict) else None
            stop_valid_mask = None
            if axis_valid_masks is not None and "token_ids" in axis_valid_masks:
                stop_valid_mask = (
                    axis_valid_masks["token_ids"].to(mask_tensor.device).to(torch.bool)
                )

            if mode == "train" and not hasattr(self, "_has_logged_parallel_training"):
                used_positions_per_item = mask_tensor[:, 1:].sum(dim=1).detach().cpu().tolist()
                logits = out["token_logits"]
                self.print(
                    {
                        "train_debug/branch": "single_stream",
                        "tokens_shape": tuple(tokens.shape),
                        "logits_shape": tuple(logits.shape),
                        "used_positions_per_item": used_positions_per_item,
                        "used_positions_total": int(
                            mask_tensor[:, 1:].sum().detach().cpu().item()
                        ),
                    }
                )
                self._has_logged_parallel_training = True

            loss_mask = mask_tensor
            token_loss = self._token_ce_loss(out["token_logits"], tokens, loss_mask)
        else:
            raise RuntimeError(f"Invalid axis_mode '{self.axis_mode}' in LightningModule")

        # --- ENERGY LOSS & AGGREGATION ---
        energy_out = {
            "hidden_states": out.get("hidden_states"),
            "energy_logits": out.get("energy_logits"),
            "raw_energy": out.get("raw_energy"),
            "mog_logits": out.get("mog_logits"),
            "mog_means": out.get("mog_means"),
            "mog_log_scales": out.get("mog_log_scales"),
            "energy_head_type": "mog" if out.get("mog_logits") is not None else "linear",
        }

        energy_result = self._energy_loss(energy_out, batch, mask_tensor)
        energy_loss: Optional[torch.Tensor] = None
        energy_sub_metrics: Dict[str, torch.Tensor] = {}
        if energy_result is not None:
            energy_loss, energy_sub_metrics = energy_result
        energy_weight = self._current_energy_loss_weight()
        weighted_energy_loss = None
        if energy_loss is not None:
            weighted_energy_loss = energy_loss * energy_weight

        total = token_loss + (weighted_energy_loss if weighted_energy_loss is not None else 0.0)

        # --- METRICS ---
        bs = int(mask_tensor.size(0))
        metrics = {}

        # Logging per-axis if available
        if reference_axis is not None:
            for axis, val in axis_losses.items():
                metrics[f"{mode}/token_ce_{axis}"] = val

        metrics[f"{mode}/token_ce"] = token_loss

        if energy_loss is not None:
            metrics[f"{mode}/energy_loss"] = energy_loss
            for sub_key, sub_val in energy_sub_metrics.items():
                metrics[f"{mode}/{sub_key}"] = sub_val

        metrics[f"{mode}/loss"] = total

        return total, metrics, bs, reference_axis

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:  # type: ignore[override]
        self._maybe_unfreeze_backbone()

    def training_step(self, batch: dict, batch_idx: int) -> Optional[torch.Tensor]:  # type: ignore[override]
        self._log_current_lr()
        # --- PHYSICS_STOP_ONLY: skip backbone forward pass entirely ---
        if self.training_mode == "physics_stop_only":
            accum = max(1, self._manual_accum_grad_batches)
            is_last = getattr(self.trainer, "is_last_batch", False)
            should_step = ((batch_idx + 1) % accum == 0) or is_last

            optimizers = self.optimizers()
            ps_opt = optimizers[0] if isinstance(optimizers, list) else optimizers
            if batch_idx % accum == 0:
                ps_opt.zero_grad()
            ps_loss, ps_metrics = self._compute_physics_stop_loss(batch, "train")
            self.manual_backward(ps_loss / float(accum))
            if should_step:
                if self._manual_gradient_clip_val is not None:
                    self.clip_gradients(
                        ps_opt,
                        gradient_clip_val=self._manual_gradient_clip_val,
                        gradient_clip_algorithm=self._manual_gradient_clip_algorithm,
                    )
                ps_opt.step()
            n_hits = batch.get("n_hits")
            bs = int(n_hits.shape[0]) if isinstance(n_hits, torch.Tensor) else 1
            for key, val in ps_metrics.items():
                if isinstance(val, torch.Tensor):
                    val = val.to(self.device)
                is_loss = key.endswith("loss")
                self.log(
                    key,
                    val,
                    prog_bar=key == "train/physics_stop_loss",
                    on_step=not is_loss,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )
            self._throughput_samples += bs
            return None

        loss, metrics, bs, ref_axis = self._step(batch, "train")

        if self.physics_stop_model is not None and self.training_mode == "all":
            accum = max(1, self._manual_accum_grad_batches)
            is_last = getattr(self.trainer, "is_last_batch", False)
            should_step = ((batch_idx + 1) % accum == 0) or is_last

            optimizers = self.optimizers()
            schedulers = self.lr_schedulers()
            if isinstance(optimizers, list):
                backbone_opt, physics_stop_opt = optimizers[0], optimizers[1]
            else:
                backbone_opt, physics_stop_opt = optimizers, None
            backbone_sched = None
            if isinstance(schedulers, list) and len(schedulers) > 0:
                backbone_sched = schedulers[0]
            elif schedulers is not None and not isinstance(schedulers, list):
                backbone_sched = schedulers

            if batch_idx % accum == 0:
                backbone_opt.zero_grad()
            self.manual_backward(loss / float(accum))
            if should_step:
                if self._manual_gradient_clip_val is not None:
                    self.clip_gradients(
                        backbone_opt,
                        gradient_clip_val=self._manual_gradient_clip_val,
                        gradient_clip_algorithm=self._manual_gradient_clip_algorithm,
                    )
                backbone_opt.step()
                if backbone_sched is not None:
                    backbone_sched.step()

            if physics_stop_opt is not None and not self._physics_stop_frozen:
                if batch_idx % accum == 0:
                    physics_stop_opt.zero_grad()
                ps_loss, ps_metrics = self._compute_physics_stop_loss(batch, "train")
                self.manual_backward(ps_loss / float(accum))
                if should_step:
                    if self._manual_gradient_clip_val is not None:
                        self.clip_gradients(
                            physics_stop_opt,
                            gradient_clip_val=self._manual_gradient_clip_val,
                            gradient_clip_algorithm=self._manual_gradient_clip_algorithm,
                        )
                    physics_stop_opt.step()
                metrics.update(ps_metrics)

            metrics["train/physics_stop_frozen"] = torch.tensor(
                1.0 if self._physics_stop_frozen else 0.0, device=self.device
            )

            for key, val in metrics.items():
                prog_bar = key in ("train/loss", "train/token_ce", "train/physics_stop_loss")
                if not prog_bar and ref_axis and key == f"train/token_ce_{ref_axis}":
                    prog_bar = True
                if isinstance(val, torch.Tensor):
                    val = val.to(self.device)
                is_loss = key.endswith("loss")
                self.log(
                    key,
                    val,
                    prog_bar=prog_bar,
                    on_step=not is_loss,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )
            self._throughput_samples += bs
            return None

        for key, val in metrics.items():
            prog_bar = False
            if key in ("train/loss", "train/token_ce"):
                prog_bar = True
            elif ref_axis and key == f"train/token_ce_{ref_axis}":
                prog_bar = True
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)
            is_loss = key.endswith("loss")

            self.log(
                key,
                val,
                prog_bar=prog_bar,
                on_step=not is_loss,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
        self._throughput_samples += bs
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:  # type: ignore[override]
        if self.training_mode == "physics_stop_only":
            if self.physics_stop_model is None:
                raise ValueError(
                    "training_mode='physics_stop_only' requires use_physics_stop=true in config"
                )
            ps_loss, ps_metrics = self._compute_physics_stop_loss(batch, "val")
            n_hits = batch.get("n_hits")
            bs = int(n_hits.shape[0]) if isinstance(n_hits, torch.Tensor) else 1
            # Keep a generic val/loss metric for checkpoint monitors while validating
            # only the physics-stop branch in this mode.
            metrics: Dict[str, Any] = {"val/loss": ps_loss}
            metrics.update(ps_metrics)
            for key, val in metrics.items():
                if isinstance(val, torch.Tensor):
                    val = val.to(self.device)
                self.log(
                    key,
                    val,
                    prog_bar=key in ("val/loss", "val/physics_stop_loss"),
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )
            return

        loss, metrics, bs, ref_axis = self._step(batch, "val")

        if self.physics_stop_model is not None:
            ps_loss, ps_metrics = self._compute_physics_stop_loss(batch, "val")
            metrics.update(ps_metrics)

        for key, val in metrics.items():
            prog_bar = False
            if key in ("val/loss", "val/token_ce"):
                prog_bar = True
            elif ref_axis and key == f"val/token_ce_{ref_axis}":
                prog_bar = True
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)

            self.log(
                key,
                val,
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )

    def configure_optimizers(self):  # type: ignore[override]
        name = str(getattr(self.hparams, "optimizer_name", "adamw")).lower()
        lr = float(self.hparams.lr)  # type: ignore[attr-defined]
        wd = float(self.hparams.weight_decay)  # type: ignore[attr-defined]
        betas = tuple(self.hparams.betas)  # type: ignore[attr-defined]
        eps = float(self.hparams.eps)  # type: ignore[attr-defined]

        # --- INDIVIDUAL TRAINING MODE EARLY RETURNS ---
        if self.training_mode == "physics_stop_only":
            if self.physics_stop_model is None:
                raise ValueError(
                    "training_mode='physics_stop_only' requires use_physics_stop=true in config"
                )
            ps_lr = float(getattr(self.hparams, "physics_stop_lr", 3e-4))
            ps_wd = float(getattr(self.hparams, "physics_stop_weight_decay", 0.01))
            ps_opt = optim.AdamW(
                list(self.physics_stop_model.parameters()), lr=ps_lr, weight_decay=ps_wd
            )
            return [ps_opt], []

        backbone_lr_scale = max(0.0, float(getattr(self.hparams, "backbone_lr_scale", 1.0)))
        backbone_unfreeze_steps = int(getattr(self.hparams, "backbone_unfreeze_steps", 0))
        backbone_params, head_params = self._split_optimizer_params()

        physics_stop_param_ids: set = set()
        if self.physics_stop_model is not None:
            physics_stop_param_ids = {id(p) for p in self.physics_stop_model.parameters()}

        use_backbone_groups = bool(backbone_params) and (
            backbone_unfreeze_steps > 0 or abs(backbone_lr_scale - 1.0) > 1e-8
        )
        if use_backbone_groups:
            params: Any = []
            filtered_head = [p for p in head_params if id(p) not in physics_stop_param_ids]
            filtered_backbone = [p for p in backbone_params if id(p) not in physics_stop_param_ids]
            if filtered_head:
                params.append({"params": filtered_head, "lr": lr})
            if filtered_backbone:
                params.append({"params": filtered_backbone, "lr": lr * backbone_lr_scale})
        else:
            params = [
                p
                for p in self.parameters()
                if p.requires_grad and id(p) not in physics_stop_param_ids
            ]
        opt: optim.Optimizer
        if name == "radam":
            from ..utils.optimizer.radam import RAdam

            opt = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=float(wd))
        elif name == "ranger":
            try:
                from ..utils.optimizer.ranger import Ranger

                opt = Ranger(
                    params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=float(wd),
                    alpha=float(self.hparams.lookahead_alpha),  # type: ignore[attr-defined]
                    k=int(self.hparams.lookahead_k),  # type: ignore[attr-defined]
                )
            except Exception:
                opt = optim.AdamW(params, lr=lr, weight_decay=wd, eps=eps, betas=betas)
        elif name == "plain_radam":
            from ..utils.optimizer.radam import PlainRAdam

            opt = PlainRAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=float(wd))
        else:
            opt = optim.AdamW(params, lr=lr, weight_decay=wd, eps=eps, betas=betas)

        scheduler_name = str(getattr(self.hparams, "lr_scheduler_name", "none")).lower()
        scheduler = None
        if scheduler_name in {"cosine_warmup", "cosine"}:
            scheduler = self._build_cosine_warmup_scheduler(opt)

        if self.physics_stop_model is not None and self.training_mode == "all":
            ps_lr = float(getattr(self.hparams, "physics_stop_lr", 3e-4))
            ps_wd = float(getattr(self.hparams, "physics_stop_weight_decay", 0.01))
            ps_params = [p for p in self.physics_stop_model.parameters() if p.requires_grad]
            ps_opt = optim.AdamW(ps_params, lr=ps_lr, weight_decay=ps_wd)
            optimizers_list = [opt, ps_opt]
            scheduler_configs = []
            if scheduler is not None:
                scheduler_configs.append(
                    {"scheduler": scheduler, "interval": "step", "frequency": 1}
                )
            return optimizers_list, scheduler_configs

        if scheduler is None:
            return opt
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._epoch_wallclock_start = time.perf_counter()
        self._throughput_samples = 0
        if self._determine_global_rank() != 0:
            return
        self.msg_logger.info(f"=== Starting training epoch {self.current_epoch} ===")

    def on_train_epoch_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = getattr(self, "_epoch_wallclock_start", None)
        if start is None:
            return
        elapsed = time.perf_counter() - float(start)
        minutes = elapsed / 60.0
        if self._determine_global_rank() == 0:
            self.msg_logger.info(
                f"=== Finished training epoch {self.current_epoch} in {minutes:.3f} minutes ==="
            )
            self.print({"train/epoch_minutes": round(minutes, 3)})
        self.log(
            "train/epoch_minutes",
            torch.tensor(minutes, device=self.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        samples_per_sec = float(self._throughput_samples) / max(elapsed, 1e-9)
        self.log(
            "vitals/samples_per_sec",
            torch.tensor(samples_per_sec, device=self.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    def on_validation_epoch_start(self) -> None:
        if self._determine_global_rank() != 0:
            return
        self.msg_logger.info(f"=== Starting validation epoch {self.current_epoch} ===")

    def _log_current_lr(self) -> None:
        opts = self.optimizers()
        if opts is None:
            return
        opt = opts[0] if isinstance(opts, list) else opts
        try:
            lr_value = float(opt.param_groups[0]["lr"])
        except Exception:
            return
        self.log(
            "lr",
            lr_value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

    def on_validation_epoch_end(self) -> None:
        self._maybe_auto_freeze_physics_stop_head()

    def on_fit_start(self) -> None:  # type: ignore[override]
        # Re-apply as a safety guard (e.g. after external state loads).
        self._apply_training_mode_parameter_freezing()

        self._maybe_freeze_backbone()
        try:
            total = 0
            trainable = 0
            for p in self.parameters():
                n = int(p.numel())
                total += n
                if p.requires_grad:
                    trainable += n
            if self._determine_global_rank() == 0:
                self.msg_logger.info({"params_total": total, "params_trainable": trainable})
        except Exception:
            pass

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        suppress_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sampled, _ = self._sample_tokens_with_probs(
            logits, temperature, top_k, top_p, suppress_token_ids
        )
        return sampled

    @torch.inference_mode()
    def generate_n_showers_batched(
        self,
        batch_size: int,
        x_shower: Optional[torch.Tensor] = None,
        *,
        adapter: Optional[BaseAdapter] = None,
        max_len: int = 1700,
        start_token_id: int = 0,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        device: Optional[torch.device] = None,
        min_len: int = 0,
        use_kv_cache: bool = True,
        gt_z_tokens: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        dev = device or self.device
        pad_id = int(self.hparams.token_pad_id if pad_token_id is None else pad_token_id)  # type: ignore

        # Merge sampling config with kwargs to support dynamic overrides
        effective_sampling_cfg = dict(self.sampling_cfg) if self.sampling_cfg else {}
        effective_sampling_cfg.update(kwargs)

        # Energy decode mode for the classification (HCal-resolution) path:
        #   "center"          — return bin center (default, deterministic)
        #   "uniform_dequant" — sample uniformly within [lower_edge, upper_edge]
        #                       of the predicted bin; fills histogram gaps
        #   "logit_weighted"  — deterministic expected value over all valid bins;
        #                       sub-bin precision, uses full transformer output
        self._energy_decode_mode = str(effective_sampling_cfg.get("energy_decode_mode", "center"))
        # T<1 sharpens MoG mixture / bin logits; per-component variance is unaffected.
        self._energy_temperature = float(effective_sampling_cfg.get("energy_temperature", 1.0))

        # Diagnostic mode: force exact per-sample n_hits and bypass physics stop.
        # Used to distinguish "stop head fires early" vs "backbone under-produces
        # late tokens" as the cause of shortened generated showers.
        force_n_hits = effective_sampling_cfg.pop("force_n_hits", None)
        if force_n_hits is not None:
            force_n_hits = torch.as_tensor(force_n_hits, device=dev).long().view(-1)
            if force_n_hits.numel() != batch_size:
                raise ValueError(
                    f"force_n_hits must have {batch_size} entries, got {force_n_hits.numel()}"
                )
            force_n_hits = force_n_hits.clamp(min=1, max=max_len)
            # Disable physics_stop path so generation runs past stop-head decisions.
            effective_sampling_cfg["stop_source"] = "length"

        # 1. Prepare Incident Energy
        incident_energy = x_shower.to(dev) if x_shower is not None else None
        if incident_energy is not None and incident_energy.dim() == 1:
            incident_energy = incident_energy.view(-1, 1)

        # Reset logging state
        self._has_logged_generation_debug = False
        self._energy_debug_logs = 0

        # 2. Initialize Model Generation State (Structural)
        init_kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "max_len": max_len,
            "pad_id": pad_id,
            "start_token_id": start_token_id,
            "device": dev,
            "sampling_config": effective_sampling_cfg,
        }

        state = self.model.init_generation_state(**init_kwargs)

        # 2b. Pre-compute constant embedding features (incident energy, n_hits
        # projections are identical at every decode step).
        _init_batch = self.model._build_decode_batch(state, incident_energy)
        self.model.cache_generation_features(_init_batch)

        # 3. Initialize Shared Physics/Energy Buffers (Lightning Managed)
        energies = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=dev)
        stop_step = torch.full((batch_size,), -1, dtype=torch.long, device=dev)
        active = torch.ones(batch_size, dtype=torch.bool, device=dev)
        running_energy_sum_gev = torch.zeros(batch_size, dtype=torch.float32, device=dev)
        consecutive_stop_votes = torch.zeros(batch_size, dtype=torch.long, device=dev)

        # Rolling buffers for physics stop history feature (last k physical hits).
        # Maintained as a circular window: oldest entry at index 0, newest at index k-1.
        _ps_use_history = self.physics_stop_model is not None and bool(
            getattr(self.physics_stop_model, "use_history", False)
        )
        _ps_history_len = (
            int(getattr(self.physics_stop_model, "history_len", 10)) if _ps_use_history else 0
        )
        ps_z_buffer = torch.zeros(batch_size, _ps_history_len, dtype=torch.long, device=dev)
        ps_e_buffer = torch.zeros(batch_size, _ps_history_len, dtype=torch.float32, device=dev)
        start_token_suppress_ids = torch.tensor(
            [int(start_token_id)], device=dev, dtype=torch.long
        )
        schedule = state.get("schedule")
        shift_z = None
        post_stop_extra_steps = 4
        post_stop_allowed_non_missing = {"z": 0, "x": 1, "y": 2, "energy": 3}
        logged_physics_stop_notice = False
        # Lazily resolved once on first use inside the combined-mode physics stop
        # path, then reused for the rest of the loop (avoids repeated dict/attr lookups).
        _ps_axis_nbins: Optional[Dict[str, int]] = None
        _ps_axis_names: Optional[Tuple[str, ...]] = None
        is_physics_stop_generation = (
            self.physics_stop_model is not None
            and str(effective_sampling_cfg.get("stop_source", "head")) == "physics_stop"
        )
        if is_physics_stop_generation:
            post_stop_extra_steps = 3
        if schedule is not None:
            if "z_token" in schedule.axis_shifts:
                shift_z = int(schedule.axis_shifts["z_token"])
            elif "token_ids" in schedule.axis_shifts:
                shift_z = int(schedule.axis_shifts["token_ids"])

        if gt_z_tokens is not None:
            if "axis_tokens" not in state or "z" not in state.get("axis_tokens", {}):
                raise NotImplementedError(
                    "gt_z_tokens override only supports split-axis hierarchical models "
                    "(state must contain axis_tokens['z'])."
                )
            if tuple(gt_z_tokens.shape) != (batch_size, max_len):
                raise ValueError(
                    f"gt_z_tokens must have shape ({batch_size}, {max_len}), "
                    f"got {tuple(gt_z_tokens.shape)}"
                )
            gt_z_tokens = gt_z_tokens.to(device=dev, dtype=torch.long)
        if "axis_nbins" in effective_sampling_cfg and isinstance(
            effective_sampling_cfg["axis_nbins"], dict
        ):
            state["axis_nbins"] = effective_sampling_cfg["axis_nbins"]
        if "spatial_axis_names" in effective_sampling_cfg:
            state["spatial_axis_names"] = tuple(effective_sampling_cfg["spatial_axis_names"])

        # 4. Generation Loop
        # Pre-allocate static KV buffers for the full sequence length so that the
        # attention forward uses in-place writes + slice views instead of torch.cat.
        # Layers with attention_window fall back to the dynamic path automatically.
        # When CUDA graphs are possible, buffers are zero-initialized and tracked by
        # GPU-scalar filled_tensor, making the transformer forward capturable.
        _has_windowed_layers = any(
            layer.attn.attention_window is not None for layer in self.model.core.decoder.layers
        )
        _use_cuda_graph = (
            use_kv_cache
            and dev.type == "cuda"
            and not _has_windowed_layers
            and max_len >= 5  # need t=0 prefill, t=1 warmup, t=2 capture, t>=3 replay
        )
        if use_kv_cache:
            param_dtype = next(self.model.parameters()).dtype
            past_key_values = self.model.preallocate_kv_cache(
                batch_size=batch_size,
                max_len=max_len + self.model.n_registers,
                device=dev,
                dtype=param_dtype,
                cuda_graph=_use_cuda_graph,
            )
        else:
            past_key_values = None

        # Pre-allocate static buffers for CUDA graph capture/replay.
        _graph_emb: Optional[torch.Tensor] = None
        _graph_mask: Optional[torch.Tensor] = None
        _graph_hidden: Optional[torch.Tensor] = None
        _cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        _graph_phase = 0  # 0 = warmup pending, 1 = capture pending, 2 = replay ready
        if _use_cuda_graph:
            _n_regs = self.model.n_registers
            _graph_emb = torch.zeros(
                batch_size, 1, self.model.embedding_dim, device=dev, dtype=param_dtype
            )
            _graph_mask = torch.zeros(batch_size, max_len + _n_regs, device=dev, dtype=torch.bool)
            _graph_mask[:, :_n_regs] = True  # registers are always visible

        # ── Optional torch.compile for peripheral decode operations ───────
        # The transformer core is captured in a manual CUDA graph above.
        # These compile the remaining per-step work (projection heads, energy
        # extraction, token sampling) so their many tiny kernels are fused
        # and wrapped in CUDA-graph replay, eliminating per-kernel launch
        # overhead across ~max_len iterations.
        #
        # Requires the inductor backend (needs a C compiler).  The cudagraphs
        # torch.compile on the peripheral ops was tested but is net negative
        # in PyTorch 2.2: stochastic ops (torch.multinomial, MoG sampling)
        # prevent CUDA graph capture inside reduce-overhead, so every call
        # pays guard-checking overhead without any graph-replay benefit.
        # The manual CUDA graph on the transformer core, RoPE cache, SDPA,
        # inline decomposition, and constant feature caching provide the
        # speedup. Keeping plain references for clarity.
        _compiled_proj_tok = self.model.project_tokens
        _compiled_proj_energy = self.model.project_energy
        _compiled_sample = self._sample_next_tokens
        _compiled_extract = self._extract_step_energy
        # Re-enable torch.compile for peripheral decode ops on all CUDA models,
        # regardless of whether the manual CUDA graph is active (e.g. windowed-
        # attention models with _use_cuda_graph=False also benefit).
        #
        # Mode split prevents the original overflow:
        #   - project_tokens / project_energy: fully deterministic → reduce-overhead
        #     (inductor captures them in a CUDA graph for launch-overhead elimination).
        #   - _sample_next_tokens / _extract_step_energy: contain torch.multinomial /
        #     torch.rand_like → mode="default" (kernel fusion without CUDA-graph
        #     capture, so no guard-check overhead on the random branches).
        #
        # backend="cudagraphs" is never used: it has a SymInt overflow in PyTorch ≤2.2
        # when tracing real model shapes.
        if (
            dev.type == "cuda"
            and hasattr(torch, "compile")
            and not bool(effective_sampling_cfg.get("disable_compile", False))
        ):
            try:
                _probe = torch.compile(lambda x: x + 1, mode="default", fullgraph=False)
                _probe(torch.zeros(1, device=dev, dtype=param_dtype))
                _compiled_proj_tok = torch.compile(
                    self.model.project_tokens, mode="reduce-overhead", fullgraph=False
                )
                _compiled_proj_energy = torch.compile(
                    self.model.project_energy, mode="reduce-overhead", fullgraph=False
                )
                _compiled_sample = torch.compile(
                    self._sample_next_tokens, mode="default", fullgraph=False
                )
                _compiled_extract = torch.compile(
                    self._extract_step_energy, mode="default", fullgraph=False
                )
            except Exception:
                pass  # inductor unavailable (no C compiler) — plain references stay

        # Helper to get logits at timestep t, handling both cached (len=1) and full (len>=t) returns
        def _get_step_tensor(tensor: torch.Tensor, t_index: int) -> torch.Tensor:
            if tensor.shape[1] == 1:
                return tensor[:, 0, :]
            return tensor[:, t_index, :]

        def _run_forward(t_step: int, cache: bool):
            nonlocal past_key_values

            if use_kv_cache:
                # Ensure current step is visible before forward
                # Even for t=0, although init handled it, this is safe/consistent
                state["attention_mask"][:, t_step] |= active

            step_out = self.model.decode_step(
                state, t_step, past_key_values, cache, incident_energy
            )
            pkv = step_out.get("present_key_values")
            if pkv is not None:
                past_key_values = pkv
            return step_out

        # We iterate t from 0 to max_len - 2.
        # At step t, we feed context up to t, and predict position t+1.
        for t in range(0, max_len - 1):
            if not active.any():
                break
            step_index = t + 1

            # A. Decode at t (context) -> predicts t+1
            # CUDA graph path: t=0 is prefill (normal), t=1 warmup, t=2 capture, t>=3 replay.
            # Embedding and projections run outside the graph; only the transformer
            # core is captured, eliminating per-layer kernel launch overhead.
            if _use_cuda_graph and t >= 1:
                # Update masks: state mask for stopping logic, graph mask for core
                state["attention_mask"][:, t] |= active
                if _graph_phase == 0:
                    # First graph step: sync graph mask from state mask (includes t=0 from prefill)
                    _graph_mask[:, _n_regs : _n_regs + max_len] = state["attention_mask"].to(
                        torch.bool
                    )
                else:
                    _graph_mask[:, _n_regs + t] |= active

                # Compute single-token embedding outside graph
                step_emb = self.model.compute_step_embedding(state, t, incident_energy)
                _graph_emb.copy_(step_emb)

                if _graph_phase == 0:
                    # Warmup: run core to populate CUDA lazy caches (cuBLAS handles, etc.)
                    torch.cuda.synchronize()
                    _graph_hidden = self.model._graph_decode_core(
                        _graph_emb, _graph_mask, past_key_values
                    )
                    _graph_phase = 1
                elif _graph_phase == 1:
                    # Capture the transformer core into a CUDA graph.
                    # Capture records but does NOT execute, so replay immediately
                    # to actually process this step's embedding.
                    _cuda_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_cuda_graph):
                        _graph_hidden = self.model._graph_decode_core(
                            _graph_emb, _graph_mask, past_key_values
                        )
                    _cuda_graph.replay()
                    _graph_phase = 2
                else:
                    # Replay the captured graph — same kernels, new tensor values
                    _cuda_graph.replay()

                # Project tokens and energy outside graph
                hidden = _graph_hidden
                batch_for_proj = (
                    self.model._build_decode_batch(state, incident_energy)
                    if hasattr(self.model, "_build_decode_batch")
                    else None
                )
                step_out: Dict[str, Any] = {}
                # Clone tensors returned by each reduce-overhead compiled head
                # immediately, before calling the next one. The two compiled
                # functions may share a CUDA-graph memory pool, so invoking
                # proj_energy would overwrite proj_tok's output buffer in place.
                # Cloning after both calls — or building a tuple of both returns
                # first — is unsafe for the same reason.
                _proj_tok_out = _compiled_proj_tok(hidden, batch_for_proj)
                for _k, _v in _proj_tok_out.items():
                    step_out[_k] = _v.clone() if isinstance(_v, torch.Tensor) else _v
                _proj_energy_out = _compiled_proj_energy(hidden, batch_for_proj)
                for _k, _v in _proj_energy_out.items():
                    step_out[_k] = _v.clone() if isinstance(_v, torch.Tensor) else _v
                step_out["present_key_values"] = past_key_values
            else:
                step_out = _run_forward(t, use_kv_cache)

            # B. Extract Energy (Prediction for t+1)
            energy_logits_out = step_out.get("energy_logits")
            energy_ref = step_out.get("mog_logits")
            if energy_ref is None:
                energy_ref = step_out.get("raw_energy")
            if energy_ref is None:
                energy_ref = energy_logits_out
            if energy_ref is None:
                raise RuntimeError(
                    "Generation step produced no energy output "
                    "(mog_logits / raw_energy / energy_logits all missing)."
                )
            uses_tokenized_energy = energy_ref is energy_logits_out

            # If step_out has len 1, we use index 0. If full sequence, we use index t.
            step_idx_for_extraction = t if energy_ref.shape[1] > 1 else 0

            step_energy, step_energy_token = _compiled_extract(
                step_out, step_idx_for_extraction, energies[:, t + 1, :]
            )

            # C. Sample Tokens for t+1
            new_tokens = {}
            logits_map = {}
            if "token_logits" in step_out:
                # Map combined output to canonical "token_ids" axis name for consistency
                logits_map["token_ids"] = step_out["token_logits"]
            else:
                for k, v in step_out.items():
                    if k.startswith("token_logits_"):
                        axis = k.replace("token_logits_", "")
                        logits_map[axis] = v

            for axis, tensor in logits_map.items():
                if axis == "z" and gt_z_tokens is not None:
                    new_tokens[axis] = gt_z_tokens[:, t + 1]
                    continue

                s_logits = _get_step_tensor(tensor, t)

                if axis == "z" and self._z_mask_enabled:
                    z_current = state["z_token"][:, t]
                    miss_id = int(schedule.missing_ids["z_token"])
                    s_logits = self._apply_z_causal_mask_step(s_logits, z_current, pad_id, miss_id)

                # Always suppress PAD and this axis' MISSING id
                if schedule is None:
                    raise RuntimeError("Expected schedule in hierarchical split mode")

                miss_id = int(schedule.missing_ids[f"{axis}_token"])
                suppress_ids = torch.tensor([pad_id, miss_id], device=dev, dtype=torch.long)
                force_missing = None
                if shift_z is not None and stop_step.ge(0).any():
                    allowed_non_missing_steps = int(post_stop_allowed_non_missing.get(axis, 0))
                    steps_since_stop = step_index - stop_step
                    force_missing = (
                        stop_step.ge(0)
                        & (steps_since_stop >= 1)
                        & (steps_since_stop > allowed_non_missing_steps)
                    )

                if greedy:
                    s_logits2 = s_logits.clone()
                    s_logits2.index_fill_(-1, suppress_ids, float("-inf"))
                    token_vals = s_logits2.argmax(dim=-1)
                else:
                    axis_temps = effective_sampling_cfg.get("axis_temperatures") or {}
                    axis_temp = float(
                        axis_temps.get(
                            axis, effective_sampling_cfg.get(f"{axis}_temperature", temperature)
                        )
                    )
                    token_vals = _compiled_sample(
                        s_logits, axis_temp, top_k, top_p, suppress_token_ids=suppress_ids
                    )
                if force_missing is not None:
                    miss_vals = torch.full_like(token_vals, miss_id)
                    token_vals = torch.where(force_missing, miss_vals, token_vals)
                new_tokens[axis] = token_vals

            # D. Update State & Buffers at t+1
            self.model.update_generation_state(
                state, t + 1, new_tokens, step_energy, step_energy_token, active
            )

            energies[:, t + 1, :] = torch.where(
                active.view(-1, 1), step_energy, energies[:, t + 1, :]
            )

            if (
                self.physics_stop_model is not None
                and bool(getattr(self.physics_stop_model, "use_energy_sum", False))
                and adapter is not None
            ):
                if uses_tokenized_energy:
                    new_energy_gev = step_energy
                else:
                    new_energy_gev = adapter.inverse_transform_energy(step_energy)
                if new_energy_gev.dim() > 1:
                    new_energy_gev = new_energy_gev.squeeze(-1)
                running_energy_sum_gev = torch.where(
                    active, running_energy_sum_gev + new_energy_gev, running_energy_sum_gev
                )

            if (
                self.physics_stop_model is not None
                and shift_z is not None
                and t >= 3
                and stop_step.lt(0).any()
                and is_physics_stop_generation
            ):
                min_hits_threshold = int(
                    effective_sampling_cfg.get(
                        "physics_stop_min_hits",
                        getattr(self.hparams, "physics_stop_min_hits", 5),
                    )
                )
                min_hits_threshold = max(1, min_hits_threshold)
                # Extract hit features aligned to the same physical hit.
                # energies[:,t,0] holds the energy generated at step t-1 (AR position t).
                # Split mode: energy shift=4 → AR pos t → hit t-4.
                # Combined mode: energy shift=2 → AR pos t → hit t-2.
                # Spatial offsets are chosen so z/x/y reference the identical hit.
                _stop_features_ready = False
                completed_z = completed_x = completed_y = None
                stop_reference_len = 0
                if "axis_tokens" in state and isinstance(state["axis_tokens"], dict):
                    # Split: z/x/y/energy shifts 1/2/3/4 → hit index at step t is t-4.
                    if t - 4 >= min_hits_threshold:
                        completed_z = state["axis_tokens"]["z"][:, t - 3]
                        completed_x = state["axis_tokens"]["x"][:, t - 2]
                        completed_y = state["axis_tokens"]["y"][:, t - 1]
                        stop_reference_len = int(state["axis_tokens"]["z"].size(1))
                        _stop_features_ready = True
                elif "token_ids" in state:
                    # Combined: spatial/energy shifts 1/2 → hit index at step t is t-2.
                    combined_hit_index = t - 2
                    if combined_hit_index >= min_hits_threshold:
                        combined_tokens = state["token_ids"]
                        # Resolve axis config once (axis_nbins / axis_names are
                        # constant across the loop — hoisted to _ps_axis_* below).
                        if _ps_axis_nbins is None:
                            _ps_axis_nbins = state.get("axis_nbins")
                            if (
                                not isinstance(_ps_axis_nbins, dict)
                                and adapter is not None
                                and hasattr(adapter, "axis_nbins")
                            ):
                                _ps_axis_nbins = adapter.axis_nbins
                            if not isinstance(_ps_axis_nbins, dict):
                                _ps_axis_nbins = {"x": 30, "y": 30, "z": 30}
                        if _ps_axis_names is None:
                            _ps_axis_names = state.get("spatial_axis_names")
                            if (
                                _ps_axis_names is None
                                and adapter is not None
                                and hasattr(adapter, "spatial_axis_names")
                            ):
                                _ps_axis_names = adapter.spatial_axis_names
                            if _ps_axis_names is None:
                                _ps_axis_names = ("x", "y", "z")
                            _ps_axis_names = tuple(_ps_axis_names)

                        # Inline single-token decomposition — O(1) instead of
                        # decomposing the full [B, seq_len] tensor every step.
                        # _unshift_sequence_to_physical(shift=1) maps index i
                        # to combined_tokens[:, i + 1].
                        _single_tok = combined_tokens[:, combined_hit_index + 1].long()
                        _is_missing = _single_tok.eq(0)
                        # Match decompose_combined_spatial_tokens: substitute 1
                        # for missing so that (valid - 1) % nb gives all axes = 1,
                        # matching the distribution the physics stop model was
                        # trained on.  Using 0 here would give axes = nb instead.
                        _rv = (
                            torch.where(_is_missing, torch.ones_like(_single_tok), _single_tok) - 1
                        )
                        _decomposed: Dict[str, torch.Tensor] = {}
                        for _ax_name in _ps_axis_names:
                            _nb = int(_ps_axis_nbins[_ax_name])
                            _decomposed[_ax_name] = _rv % _nb + 1
                            _rv = torch.div(_rv, _nb, rounding_mode="floor")

                        completed_z = _decomposed["z"]
                        completed_x = _decomposed["x"]
                        completed_y = _decomposed["y"]
                        stop_reference_len = int(combined_tokens.size(1))
                        _stop_features_ready = True
                else:
                    raise RuntimeError("Missing generation tokens for physics_stop source")
                if _stop_features_ready:
                    completed_energy = energies[:, t, 0]
                    physics_position_ratio = None
                    if bool(getattr(self.physics_stop_model, "use_position_feature", False)):
                        max_sequence_len = stop_reference_len
                        position_denominator = float(max(max_sequence_len - 1, 1))
                        physics_position_ratio = torch.full(
                            (batch_size,),
                            float(t) / position_denominator,
                            device=dev,
                            dtype=completed_energy.dtype,
                        )
                    ps_incident = (
                        incident_energy.squeeze(-1)
                        if incident_energy is not None
                        else torch.zeros(batch_size, device=dev)
                    )
                    ps_energy_sum = None
                    if bool(getattr(self.physics_stop_model, "use_energy_sum", False)):
                        sampling_fraction = float(
                            effective_sampling_cfg.get(
                                "physics_stop_sampling_fraction",
                                getattr(self.hparams, "physics_stop_sampling_fraction", 1.0),
                            )
                        )
                        ps_energy_sum = (
                            running_energy_sum_gev
                            / ps_incident.clamp(min=1e-8)
                            / sampling_fraction
                        )

                    # Hit counter: raw step index t, same quantity as arange(seq_len) in training.
                    ps_hit_counter = None
                    if bool(getattr(self.physics_stop_model, "use_hit_counter", False)):
                        ps_hit_counter = torch.full(
                            (batch_size,),
                            float(t),
                            device=dev,
                            dtype=completed_energy.dtype,
                        )

                    # History: snapshot the rolling buffer BEFORE updating it with current hit,
                    # so the model sees the k hits that preceded the current one — identical
                    # semantics to the unfold in _compute_physics_stop_loss.
                    ps_z_history = None
                    ps_e_history = None
                    if _ps_use_history:
                        ps_z_history = ps_z_buffer.clone()  # [B, k]
                        ps_e_history = ps_e_buffer.clone()  # [B, k]
                        # Shift buffer left and insert current hit at the end
                        ps_z_buffer = torch.roll(ps_z_buffer, -1, dims=1)
                        ps_z_buffer[:, -1] = completed_z
                        ps_e_buffer = torch.roll(ps_e_buffer, -1, dims=1)
                        ps_e_buffer[:, -1] = completed_energy

                    ps_logit = self.physics_stop_model(
                        completed_z,
                        completed_x,
                        completed_y,
                        completed_energy,
                        ps_incident,
                        position_ratio=physics_position_ratio,
                        energy_sum=ps_energy_sum,
                        hit_counter=ps_hit_counter,
                        z_history=ps_z_history,
                        energy_history=ps_e_history,
                    )
                    ps_prob = torch.sigmoid(ps_logit.squeeze(-1))
                    ps_tau_prob = effective_sampling_cfg.get("stop_probability_threshold", None)
                    if ps_tau_prob is None:
                        ps_tau_prob = 0.5
                    ps_tau_prob_value = float(ps_tau_prob)
                    if not logged_physics_stop_notice and self._determine_global_rank() == 0:
                        self.msg_logger.info(
                            {
                                "gen/stop_source": "physics_stop",
                                "gen/physics_stop_threshold": ps_tau_prob_value,
                                "gen/physics_stop_min_hits": int(min_hits_threshold),
                                "gen/physics_stop_mode": str(
                                    effective_sampling_cfg.get("stop_mode", "")
                                ),
                            }
                        )
                        logged_physics_stop_notice = True
                    ps_is_stop = ps_prob >= ps_tau_prob_value
                    still_undecided = active & stop_step.lt(0)
                    consecutive_stop_votes = torch.where(
                        still_undecided & ps_is_stop,
                        consecutive_stop_votes + 1,
                        torch.zeros_like(consecutive_stop_votes),
                    )
                    ps_consecutive_required = int(
                        effective_sampling_cfg.get("stop_consecutive_required", 1)
                    )
                    ps_just_triggered = still_undecided & (
                        consecutive_stop_votes >= ps_consecutive_required
                    )
                    stop_step = torch.where(
                        ps_just_triggered,
                        torch.full_like(stop_step, step_index + 1),
                        stop_step,
                    )

            # E. Check Stopping
            if force_n_hits is not None and shift_z is not None:
                # Mark stop as soon as (step_index + 1) reaches the forced target,
                # so hit_mask covers exactly force_n_hits positions. Piggyback on
                # the existing stop_step / tail_done mechanism for post-stop x/y
                # token completion.
                still_undecided = active & stop_step.lt(0)
                force_trigger = still_undecided & ((step_index + 1) >= force_n_hits)
                stop_step = torch.where(force_trigger, force_n_hits, stop_step)

            if shift_z is None:
                should_stop = self.model.check_stopping_condition(
                    state, new_tokens, step_index, effective_sampling_cfg
                )
            else:
                length_stop = self.model._length_stop_mask(  # type: ignore[attr-defined]
                    state, step_index, effective_sampling_cfg
                )
                if length_stop is None:
                    length_stop = torch.zeros_like(active)
                tail_done = torch.zeros_like(active)
                steps_since_stop = step_index - stop_step
                tail_done = stop_step.ge(0) & (steps_since_stop >= int(post_stop_extra_steps))
                should_stop = length_stop | tail_done

            just_stopped = should_stop & active

            unseen = stop_step.lt(0)
            stop_step = torch.where(
                unseen & just_stopped, torch.full_like(stop_step, step_index), stop_step
            )
            active = active & (~just_stopped)

        # 5. Finalize — release generation caches
        self.model.clear_generation_cache()

        result = {}
        if "axis_tokens" in state:
            for ax, tensor in state["axis_tokens"].items():
                result[f"{ax}_token"] = tensor
        elif "token_ids" in state:
            result["token_ids"] = state["token_ids"]
        if "energy_token" in state:
            result["energy_token"] = state["energy_token"]

        # Build mask from stop_step correctly
        steps = torch.arange(max_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        if force_n_hits is not None:
            # In diagnostic mode, the hit_mask is pinned to the forced length so
            # truncation matches the target exactly regardless of loop exit timing.
            final_stops = force_n_hits
            stop_step = force_n_hits
        else:
            final_stops = torch.where(stop_step < 0, torch.tensor(max_len, device=dev), stop_step)
        # hit_mask: 1 for 0...stop-1. 0 at stop.
        hit_mask_result = steps < final_stops.unsqueeze(1)
        result["hit_mask"] = hit_mask_result

        result["energy"] = energies
        result["stop_step"] = stop_step
        if "axis_names" in state:
            result["spatial_axis_names"] = state["axis_names"]

        return result

    @torch.inference_mode()
    def generate_physical_showers(
        self,
        batch_size: int,
        adapter: Optional[BaseAdapter] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates showers and (optionally) decodes them to physical space.

        If an adapter is provided, `adapter.decode()` is responsible for:
        - undoing hierarchical shifts (realignment)
        - removing PAD/MISSING (missing_mask)
        - integrating stop-derived hit_mask (from stop_step) in a physically aligned way
        """
        raw_output = self.generate_n_showers_batched(batch_size, adapter=adapter, **kwargs)

        if adapter is None:
            return raw_output

        decoded = adapter.decode(raw_output)

        raw_output.update(decoded)  # contains hit_features + hit_mask (+ conditioning passthrough)
        return raw_output
