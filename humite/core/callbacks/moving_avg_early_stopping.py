from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

_LOGGER = logging.getLogger(__name__)


class MovingAverageEarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val/loss",
        ema_alpha: float = 0.2,
        min_epochs_before_stop: int = 10,
        patience: int = 3,
        min_delta: float = 0.0,
        strict: bool = False,
        verbose: bool = True,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1].")
        if int(min_epochs_before_stop) < 0:
            raise ValueError("min_epochs_before_stop must be >= 0.")
        if int(patience) < 1:
            raise ValueError("patience must be >= 1.")
        if float(min_delta) < 0.0:
            raise ValueError("min_delta must be >= 0.")

        self.monitor = str(monitor)
        self.ema_alpha = float(ema_alpha)
        self.min_epochs_before_stop = int(min_epochs_before_stop)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.strict = bool(strict)
        self.verbose = bool(verbose)
        self.check_finite = bool(check_finite)

        self._ema_value: Optional[float] = None
        self._best_ema_value: float = float("inf")
        self._bad_epochs: int = 0
        self._validation_epochs_seen: int = 0
        self._stop_triggered: bool = False

    @property
    def state_key(self) -> str:
        return f"{self.__class__.__qualname__}.{self.monitor}"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_value": self._ema_value,
            "best_ema_value": self._best_ema_value,
            "bad_epochs": self._bad_epochs,
            "validation_epochs_seen": self._validation_epochs_seen,
            "stop_triggered": self._stop_triggered,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._ema_value = state_dict.get("ema_value")
        self._best_ema_value = float(state_dict.get("best_ema_value", float("inf")))
        self._bad_epochs = int(state_dict.get("bad_epochs", 0))
        self._validation_epochs_seen = int(state_dict.get("validation_epochs_seen", 0))
        self._stop_triggered = bool(state_dict.get("stop_triggered", False))

    def _metric_to_float(self, metric_value: Any) -> Optional[float]:
        if metric_value is None:
            return None
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() != 1:
                return None
            metric_scalar = float(metric_value.detach().cpu().item())
        else:
            try:
                metric_scalar = float(metric_value)
            except (TypeError, ValueError):
                return None
        if not math.isfinite(metric_scalar):
            return None
        return metric_scalar

    def _update_ema(self, current_value: float) -> float:
        if self._ema_value is None:
            self._ema_value = current_value
        else:
            self._ema_value = (self.ema_alpha * current_value) + (
                (1.0 - self.ema_alpha) * self._ema_value
            )
        return float(self._ema_value)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        callback_metrics = getattr(trainer, "callback_metrics", None)
        metric_raw = None if callback_metrics is None else callback_metrics.get(self.monitor)
        metric_value = self._metric_to_float(metric_raw)
        if metric_value is None:
            if self.strict:
                raise RuntimeError(f"Metric '{self.monitor}' is not available as a finite scalar.")
            return

        self._validation_epochs_seen += 1
        ema_value = self._update_ema(metric_value)

        if self.check_finite and not math.isfinite(ema_value):
            trainer.should_stop = True
            self._stop_triggered = True
            if trainer.is_global_zero and self.verbose:
                module_logger = getattr(pl_module, "msg_logger", _LOGGER)
                module_logger.warning(
                    "Stopping training: EMA for '%s' is non-finite.", self.monitor
                )
            return

        has_improved = ema_value < (self._best_ema_value - self.min_delta)
        is_diverged = ema_value > (self._best_ema_value + self.min_delta)

        if has_improved:
            self._best_ema_value = ema_value
            self._bad_epochs = 0
        elif is_diverged:
            self._bad_epochs += 1
        else:
            self._bad_epochs = 0

        pl_module.log(
            f"{self.monitor}_ema",
            float(ema_value),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            f"{self.monitor}_ema_best",
            float(self._best_ema_value),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            f"{self.monitor}_ema_bad_epochs",
            float(self._bad_epochs),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )

        is_armed = self._validation_epochs_seen >= self.min_epochs_before_stop
        if not is_armed or self._bad_epochs < self.patience or self._stop_triggered:
            return

        trainer.should_stop = True
        self._stop_triggered = True
        if trainer.is_global_zero and self.verbose:
            module_logger = getattr(pl_module, "msg_logger", _LOGGER)
            module_logger.info(
                "Stopping training with EMA divergence on '%s': epoch_count=%d ema=%.6f best_ema=%.6f bad_epochs=%d patience=%d min_delta=%.6f",
                self.monitor,
                self._validation_epochs_seen,
                ema_value,
                self._best_ema_value,
                self._bad_epochs,
                self.patience,
                self.min_delta,
            )
