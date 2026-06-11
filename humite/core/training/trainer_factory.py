from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..callbacks.generative_eval import GenerativeEvalCallback


def instantiate_logger_objects(obj: Any) -> Any:
    try:
        if isinstance(obj, DictConfig):
            if "_target_" in obj:
                return instantiate(obj)
            inner_logger: Any = obj.get("logger")  # type: ignore[arg-type]
            if isinstance(inner_logger, DictConfig) and "_target_" in inner_logger:
                return instantiate(inner_logger)
        if isinstance(obj, ListConfig):
            return [instantiate_logger_objects(item) for item in obj]
        if isinstance(obj, dict):
            if "_target_" in obj:
                return instantiate(obj)
            inner_logger_dict = obj.get("logger")
            if isinstance(inner_logger_dict, dict) and "_target_" in inner_logger_dict:
                return instantiate(inner_logger_dict)
        if isinstance(obj, list):
            return [instantiate_logger_objects(item) for item in obj]
    except Exception:
        pass
    return obj


def instantiate_callback_objects(callbacks: Any) -> Any:
    try:
        if isinstance(callbacks, DictConfig):
            if "_target_" in callbacks:
                return instantiate(callbacks)
            return {key: instantiate_callback_objects(value) for key, value in callbacks.items()}
        if isinstance(callbacks, ListConfig):
            return [instantiate_callback_objects(item) for item in callbacks]
        if isinstance(callbacks, list):
            resolved_callbacks = []
            for callback in callbacks:
                if isinstance(callback, DictConfig) and "_target_" in callback:
                    resolved_callbacks.append(instantiate(callback))
                elif isinstance(callback, dict) and "_target_" in callback:
                    resolved_callbacks.append(instantiate(callback))
                else:
                    resolved_callbacks.append(callback)
            return resolved_callbacks
        if isinstance(callbacks, dict):
            if "_target_" in callbacks:
                return instantiate(callbacks)
            return {key: instantiate_callback_objects(value) for key, value in callbacks.items()}
    except Exception:
        pass
    return callbacks


def instantiate_lightning_trainer(cfg_trainer: Any) -> L.Trainer:
    try:
        if isinstance(cfg_trainer, DictConfig) and "_target_" in cfg_trainer:
            return cast(L.Trainer, instantiate(cfg_trainer))
    except Exception:
        pass

    def access_config_value(source: Any, key: str) -> Any:
        if isinstance(source, DictConfig):
            if key in source:
                return source[key]
            return None
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    allowed_keys = {
        "max_steps",
        "min_steps",
        "accelerator",
        "devices",
        "num_sanity_val_steps",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "precision",
        "logger",
        "callbacks",
        "log_every_n_steps",
        "enable_checkpointing",
        "enable_progress_bar",
        "enable_model_summary",
        "gradient_clip_val",
        "accumulate_grad_batches",
        "deterministic",
        "default_root_dir",
        "strategy",
        "num_nodes",
    }
    trainer_kwargs: Dict[str, Any] = {}
    for key in allowed_keys:
        try:
            value = access_config_value(cfg_trainer, key)
        except Exception:
            value = None
        if value is not None:
            trainer_kwargs[key] = value
    if "logger" in trainer_kwargs:
        trainer_kwargs["logger"] = instantiate_logger_objects(trainer_kwargs["logger"])
    if "callbacks" in trainer_kwargs:
        trainer_kwargs["callbacks"] = instantiate_callback_objects(trainer_kwargs["callbacks"])
    try:
        if (
            not bool(trainer_kwargs.get("enable_progress_bar", True))
            and "callbacks" in trainer_kwargs
        ):
            cleaned_callbacks = []
            progress_bar_class = None
            try:
                from lightning.pytorch.callbacks.progress import (
                    TQDMProgressBar as progress_cls,  # type: ignore
                )

                progress_bar_class = progress_cls
            except Exception:
                try:
                    from lightning.pytorch.callbacks import (
                        TQDMProgressBar as progress_cls,  # type: ignore
                    )

                    progress_bar_class = progress_cls
                except Exception:
                    progress_bar_class = None
            for callback in cast(list[Any], trainer_kwargs.get("callbacks", [])):
                keep = True
                if progress_bar_class is not None:
                    try:
                        if isinstance(callback, progress_bar_class):
                            keep = False
                    except Exception:
                        pass
                if keep:
                    name = (
                        type(callback).__name__
                        if hasattr(callback, "__class__")
                        else str(callback)
                    )
                    if name == "TQDMProgressBar":
                        keep = False
                if keep:
                    cleaned_callbacks.append(callback)
            trainer_kwargs["callbacks"] = cleaned_callbacks
    except Exception:
        pass
    for key in (
        "max_steps",
        "min_steps",
        "num_sanity_val_steps",
        "log_every_n_steps",
        "num_nodes",
    ):
        if key in trainer_kwargs:
            trainer_kwargs[key] = int(trainer_kwargs[key])
    for key in (
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "gradient_clip_val",
    ):
        if key in trainer_kwargs:
            trainer_kwargs[key] = float(trainer_kwargs[key])
    return L.Trainer(**trainer_kwargs)


def build_generative_eval_callback(
    gen_cfg: Dict[str, Any], cfg_dataset: Any
) -> Optional[GenerativeEvalCallback]:
    try:
        kwargs: Dict[str, Any] = {
            "output_dir": gen_cfg.get("output_dir", None),
            "n_val_gen_showers": int(gen_cfg.get("n_val_gen_showers", 8)),
            "starting_at_epoch": int(gen_cfg.get("starting_at_epoch", 0)),
            "every_n_epochs": int(gen_cfg.get("every_n_epochs", 1)),
            "show_inline": bool(gen_cfg.get("show_inline", False)),
        }
        if "dist_gather" in gen_cfg:
            kwargs["dist_gather"] = bool(gen_cfg.get("dist_gather"))

        if "SanityCheckCallback" in gen_cfg:
            kwargs["SanityCheckCallback"] = bool(gen_cfg.get("SanityCheckCallback"))

        if "plot_individual_showers" in gen_cfg:
            kwargs["plot_individual_showers"] = bool(gen_cfg.get("plot_individual_showers"))

        if "generate_target_inc_energies" in gen_cfg:
            kwargs["generate_target_inc_energies"] = bool(
                gen_cfg.get("generate_target_inc_energies")
            )

        incident_energies = gen_cfg.get("incident_energies", None)
        if isinstance(incident_energies, (list, tuple)):
            kwargs["incident_energies"] = list(incident_energies)
        if "split_mode" in gen_cfg:
            kwargs["split_mode"] = str(gen_cfg.get("split_mode"))
        dataset_cfg: Dict[str, Any] = {}
        try:
            if isinstance(cfg_dataset, DictConfig):
                dataset_cfg = dict(OmegaConf.to_container(cfg_dataset, resolve=True) or {})
            elif isinstance(cfg_dataset, dict):
                dataset_cfg = dict(cfg_dataset)
        except Exception:
            dataset_cfg = dict(cfg_dataset) if isinstance(cfg_dataset, dict) else {}
        sampling_cfg = gen_cfg.get("sampling")
        sampling: Dict[str, Any] = {}
        if isinstance(sampling_cfg, dict):
            sampling = dict(sampling_cfg)
        if sampling.get("max_len") is None:
            pad_length = dataset_cfg.get("pad_length")
            if isinstance(pad_length, (int, float)):
                sampling["max_len"] = int(pad_length)
        if sampling:
            kwargs["sampling"] = sampling
        artifacts_cfg = gen_cfg.get("artifacts")
        if isinstance(artifacts_cfg, (list, tuple)):
            kwargs["artifacts"] = list(artifacts_cfg)
        spatial_bins_cfg = gen_cfg.get("spatial_nbins")
        if isinstance(spatial_bins_cfg, (list, tuple)) and len(spatial_bins_cfg) == 3:
            kwargs["spatial_nbins"] = (
                int(spatial_bins_cfg[0]),
                int(spatial_bins_cfg[1]),
                int(spatial_bins_cfg[2]),
            )
        elif all(
            isinstance(dataset_cfg.get(key), (int, float))
            for key in ("nbins_x", "nbins_y", "nbins_z")
        ):
            kwargs["spatial_nbins"] = (
                int(dataset_cfg["nbins_x"]),
                int(dataset_cfg["nbins_y"]),
                int(dataset_cfg["nbins_z"]),
            )
        energy_params = gen_cfg.get("energy_transform_params")
        if energy_params is None:
            features_cfg = dataset_cfg.get("features")
            if isinstance(features_cfg, dict):
                if isinstance(features_cfg.get("energy"), dict):
                    energy_params = dict(features_cfg["energy"])
                else:
                    energy_params = dict(features_cfg)
        if energy_params is not None:
            kwargs["energy_transform_params"] = energy_params
        if dataset_cfg:
            kwargs["dataset_config"] = dataset_cfg
        bins_cfg = gen_cfg.get("bins")
        if isinstance(bins_cfg, dict):
            kwargs["bins"] = dict(bins_cfg)
        return GenerativeEvalCallback(**kwargs)
    except Exception:
        return None


def derive_geometry_from_dataset(dataset: Any) -> Optional[Tuple[int, int, int]]:
    try:
        return (
            int(getattr(dataset, "nbins_x")),
            int(getattr(dataset, "nbins_y")),
            int(getattr(dataset, "nbins_z")),
        )
    except Exception:
        try:
            return (
                int(dataset.get("nbins_x")),
                int(dataset.get("nbins_y")),
                int(dataset.get("nbins_z")),
            )
        except Exception:
            return None


def iterate_comet_loggers(trainer: Any):
    try:
        loggers = []
        if hasattr(trainer, "loggers") and isinstance(trainer.loggers, list):
            loggers = trainer.loggers
        elif hasattr(trainer, "logger") and trainer.logger is not None:
            loggers = [trainer.logger]
        for logger in loggers:
            if hasattr(logger, "experiment") and getattr(logger, "experiment") is not None:
                yield logger
    except Exception:
        return
