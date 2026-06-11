from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

_LOGGER = logging.getLogger(__name__)


class EMAWeights(Callback):
    """Maintains an exponential moving average of model weights.

    During validation the live weights are swapped for the EMA weights so that
    all val metrics (and therefore ModelCheckpoint monitoring) reflect EMA
    quality.  The live weights are restored immediately after.

    The shadow dict is persisted through callback state_dict so it survives
    checkpoint resumption.

    Args:
        decay:               EMA decay factor (e.g. 0.9999).  Higher = slower
                             update = smoother but slower to track the model.
        start_step:          Global step at which EMA updates begin.  Set this
                             to your LR-warmup length so early noisy weights
                             do not pollute the average.
        update_every_n_steps: Apply EMA update every N optimizer steps.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        start_step: int = 0,
        update_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.start_step = int(max(0, start_step))
        self.update_every_n_steps = int(max(1, update_every_n_steps))

        self._shadow: Optional[Dict[str, torch.Tensor]] = None
        self._live_backup: Dict[str, torch.Tensor] = {}
        self._using_ema: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_shadow(self, pl_module: L.LightningModule) -> None:
        # Store on CPU pinned memory: keeps GPU footprint at 1x params during training.
        # The update still runs on GPU via a temporary cast; only the shadow storage is CPU.
        self._shadow = {
            name: param.data.detach().float().cpu().pin_memory()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }

    def _update_shadow(self, pl_module: L.LightningModule) -> None:
        if self._shadow is None:
            self._init_shadow(pl_module)
            return
        decay = self.decay
        for name, param in pl_module.named_parameters():
            if not param.requires_grad or name not in self._shadow:
                continue
            # Pull shadow to GPU, update, write back to CPU — avoids keeping
            # a permanent GPU copy of the shadow.
            shadow_gpu = self._shadow[name].to(param.device, non_blocking=True)
            shadow_gpu.mul_(decay).add_(param.data.detach().float(), alpha=1.0 - decay)
            self._shadow[name].copy_(shadow_gpu, non_blocking=True)

    def _swap_to_ema(self, pl_module: L.LightningModule) -> None:
        if self._shadow is None or self._using_ema:
            return
        # live_backup: keep on CPU to avoid the 3x GPU peak during validation
        self._live_backup = {}
        for name, param in pl_module.named_parameters():
            if name in self._shadow:
                self._live_backup[name] = param.data.float().cpu()
                param.data.copy_(
                    self._shadow[name].to(param.device, non_blocking=True).to(param.data.dtype)
                )
        self._using_ema = True

    def _swap_to_live(self, pl_module: L.LightningModule) -> None:
        if not self._using_ema:
            return
        for name, param in pl_module.named_parameters():
            if name in self._live_backup:
                param.data.copy_(
                    self._live_backup[name]
                    .to(param.device, non_blocking=True)
                    .to(param.data.dtype)
                )
        self._live_backup = {}
        self._using_ema = False

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step < self.start_step:
            return
        if trainer.global_step % self.update_every_n_steps != 0:
            return
        self._update_shadow(pl_module)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Do not swap during sanity check or before EMA is initialised
        if trainer.sanity_checking or self._shadow is None:
            return
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_to_live(pl_module)

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self._shadow is None:
            return {"shadow": None, "decay": self.decay, "start_step": self.start_step}
        return {
            "shadow": {k: v.cpu() for k, v in self._shadow.items()},
            "decay": self.decay,
            "start_step": self.start_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        shadow = state_dict.get("shadow")
        if shadow is not None:
            self._shadow = {k: v.float() for k, v in shadow.items()}
        # Restore scalar config from checkpoint in case the callback was
        # constructed with different values when resuming.
        if "decay" in state_dict:
            self.decay = float(state_dict["decay"])
        if "start_step" in state_dict:
            self.start_step = int(state_dict["start_step"])
