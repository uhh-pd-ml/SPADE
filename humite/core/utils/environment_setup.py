from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Optional

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from .pylogger import get_pylogger


def disable_sentry_integrations() -> None:
    try:
        for key in (
            "SENTRY_DSN",
            "SENTRY_TRACES_SAMPLE_RATE",
            "SENTRY_PROFILES_SAMPLE_RATE",
        ):
            try:
                os.environ.pop(key, None)
            except Exception:
                pass
        os.environ["SENTRY_DSN"] = ""
        os.environ.setdefault("COMET_ERROR_TRACKING_ENABLE", "0")
        os.environ.setdefault("COMET_INTERNAL_SENTRY_DSN", "")
        os.environ.setdefault("COMET_INTERNAL_REPORTING", "0")
        try:
            import sentry_sdk  # type: ignore

            try:
                sentry_sdk.Hub.current.client = None  # type: ignore[attr-defined]
            except Exception:
                try:
                    sentry_sdk.init(dsn=None, default_integrations=False)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


def configure_logging(desired_level: Optional[str] = None) -> None:
    level_raw = desired_level or os.getenv("HUMITE_LOG_LEVEL", "INFO")
    try:
        level = getattr(logging, str(level_raw).upper(), logging.INFO)
    except Exception:
        level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    get_pylogger(__name__).setLevel(level)
    if level <= logging.DEBUG:
        for name in (
            "lightning",
            "pytorch_lightning",
            "lightning.pytorch",
            "torch",
            "torch.distributed",
            "urllib3",
            "requests",
            "matplotlib",
            "numba",
            "sentry_sdk",
            "comet_ml",
            "filelock",
            "asyncio",
        ):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass


def register_omegaconf_resolvers() -> None:
    def get_nodename_bigram() -> str:
        nodename = os.uname().nodename.split(".")[0]
        nodename_with_time = f"{nodename}_{int(time.time())}"
        hashed_name = hashlib.sha256(nodename_with_time.encode()).hexdigest()
        return f"{nodename}_{hashed_name[:6]}"

    def get_gpu_name() -> str:
        try:
            import torch as torch_local

            if torch_local.cuda.is_available():
                raw = str(torch_local.cuda.get_device_name(0))
            else:
                return "cpu"
        except Exception:
            return "cpu"
        value = raw.upper()
        if "H100" in value:
            return "H100"
        if "A100" in value:
            return "A100"
        if "V100" in value:
            return "V100"
        if "RTX 4090" in value or "4090" in value:
            return "RTX4090"
        if "L40" in value:
            return "L40"
        return "".join(ch for ch in raw if ch.isalnum())[:12]

    def coalesce(*args: object) -> str:
        for candidate in args:
            if candidate is None:
                continue
            text = str(candidate)
            if text.strip() and text.lower() not in {"none", "null"}:
                return text
        return ""

    try:
        has_resolver = getattr(OmegaConf, "has_resolver", None)
        if callable(has_resolver):
            if not OmegaConf.has_resolver("nodename_bigram"):
                OmegaConf.register_new_resolver(
                    "nodename_bigram", get_nodename_bigram, use_cache=True
                )
            if not OmegaConf.has_resolver("gpu_name"):
                OmegaConf.register_new_resolver("gpu_name", get_gpu_name, use_cache=True)
            if not OmegaConf.has_resolver("coalesce"):
                OmegaConf.register_new_resolver("coalesce", coalesce, use_cache=False)
            if not OmegaConf.has_resolver("eval"):
                OmegaConf.register_new_resolver("eval", eval)
        else:
            for name, func, cache in (
                ("nodename_bigram", get_nodename_bigram, True),
                ("gpu_name", get_gpu_name, True),
                ("coalesce", coalesce, False),
                ("eval", eval, True),
            ):
                try:
                    OmegaConf.register_new_resolver(name, func, use_cache=cache)
                except ValueError:
                    pass
    except Exception:
        for name, func, cache in (
            ("nodename_bigram", get_nodename_bigram, True),
            ("gpu_name", get_gpu_name, True),
            ("coalesce", coalesce, False),
            ("eval", eval, True),
        ):
            try:
                OmegaConf.register_new_resolver(name, func, use_cache=cache)
            except ValueError:
                pass


def access_config_value(source: Any, key: str) -> Any:
    if isinstance(source, DictConfig):
        return source.get(key, None)
    if isinstance(source, dict):
        return source.get(key, None)
    return getattr(source, key, None)


def derive_device_count(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 1
    if text.lower() == "auto":
        return 1
    try:
        return int(text)
    except Exception:
        return 1


def disable_dynamo_optimizer_if_required(cfg: DictConfig) -> None:
    should_disable = False
    world_size_raw = os.getenv("WORLD_SIZE")
    if world_size_raw is not None:
        try:
            should_disable = int(world_size_raw) > 1
        except Exception:
            should_disable = False
    trainer_cfg = getattr(cfg, "trainer", None)
    if not should_disable and trainer_cfg is not None:
        strategy_value = access_config_value(trainer_cfg, "strategy")
        if isinstance(strategy_value, str) and "ddp" in strategy_value.lower():
            should_disable = True
    if not should_disable and trainer_cfg is not None:
        devices_value = access_config_value(trainer_cfg, "devices")
        device_count = derive_device_count(devices_value)
        if device_count == -1 or device_count > 1:
            should_disable = True
    if should_disable:
        try:
            torch._dynamo.config.optimize_ddp = False
        except Exception:
            pass


def initialize_environment(cfg: DictConfig) -> None:
    disable_sentry_integrations()
    try:
        level = None
        if hasattr(cfg, "logging") and isinstance(cfg.logging, DictConfig):
            level = cfg.logging.get("level", None)
        configure_logging(str(level) if level else None)
    except Exception:
        configure_logging(None)
    register_omegaconf_resolvers()
    disable_dynamo_optimizer_if_required(cfg)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    torch.set_float32_matmul_precision("medium")
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.get("seed")), workers=True)


register_omegaconf_resolvers()
