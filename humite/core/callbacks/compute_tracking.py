import time
from typing import Any, Dict, Optional

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

try:
    from torch.utils.flop_counter import FlopCounterMode

    HAS_FLOP_COUNTER = True
except Exception:
    FlopCounterMode = None  # type: ignore[assignment]
    HAS_FLOP_COUNTER = False

from humite.core.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ComputeTrackingCallback(Callback):
    """Track training-phase compute metrics (time, steps, optional FLOPs)."""

    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.total_flops = 0.0
        self.total_training_time = 0.0
        self.total_steps = 0
        self.step_start_time: Optional[float] = None
        self.flops_per_step: Optional[float] = None
        self._profiler: Optional[Any] = None
        self._enable_flop_profiling = True

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.total_flops = 0.0
        self.total_training_time = 0.0
        self.total_steps = 0
        self.step_start_time = None
        self.flops_per_step = None
        self._profiler = None
        self._enable_flop_profiling = bool(getattr(pl_module, "automatic_optimization", True))
        if not self._enable_flop_profiling:
            rank_zero_only(log.info)(
                "Disabling FLOP profiling because manual optimization is enabled."
            )
        if not HAS_FLOP_COUNTER:
            rank_zero_only(log.warning)(
                "FLOP counter not available in this torch build; logging time and steps only."
            )

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.step_start_time = time.time()
        if (
            self._enable_flop_profiling
            and HAS_FLOP_COUNTER
            and FlopCounterMode is not None
            and self.flops_per_step is None
            and self._profiler is None
        ):
            try:
                profiler = FlopCounterMode(pl_module, display=False)
                profiler.__enter__()
                self._profiler = profiler
            except Exception as exc:
                rank_zero_only(log.warning)(f"Failed to start FLOP profiling: {exc}")
                self._profiler = None
                self.flops_per_step = 0.0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.step_start_time is not None:
            batch_time = time.time() - self.step_start_time
            self.total_training_time += batch_time
            self.step_start_time = None
        self.total_steps += 1

        if self._profiler is not None:
            try:
                self._profiler.__exit__(None, None, None)
                first_step_flops = float(self._profiler.get_total_flops())
                self.flops_per_step = first_step_flops
                rank_zero_only(log.info)(
                    f"Estimated FLOPs per training step: {self.flops_per_step:.2e}"
                )
            except Exception as exc:
                rank_zero_only(log.warning)(f"Failed to finalize FLOP profiling: {exc}")
                self.flops_per_step = 0.0
            finally:
                self._profiler = None

        if self.flops_per_step is not None:
            self.total_flops += self.flops_per_step

        if self.total_steps % self.log_interval == 0:
            self.log_metrics(pl_module)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.log_metrics(pl_module, force=True)

    def log_metrics(self, pl_module: L.LightningModule, force: bool = False) -> None:
        metrics = {
            "compute/total_flops": self.total_flops,
            "compute/training_time_sec": self.total_training_time,
            "compute/total_steps": float(self.total_steps),
            "compute/flops_per_sec": (
                self.total_flops / self.total_training_time
                if self.total_training_time > 0
                else 0.0
            ),
        }
        on_step_flag = not force
        on_epoch_flag = force
        pl_module.log_dict(
            metrics, on_step=on_step_flag, on_epoch=on_epoch_flag, prog_bar=False, logger=True
        )
