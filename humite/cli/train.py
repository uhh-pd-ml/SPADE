from __future__ import annotations

# Load a repo-root .env (HUMITE_DATA_DIR, HUMITE_OUTPUT_DIR, COMET_API_TOKEN, ...)
# before anything reads the environment. usecwd=True searches up from the
# directory the command was launched in, which is the repo root in the docs.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ModuleNotFoundError:
    pass

try:
    import comet_ml
except ImportError:
    pass

import logging
import os
from pathlib import Path

os.environ["COMET_DISABLE_ANNOUNCEMENT"] = "1"
from typing import Any, Dict, cast

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from humite.core.callbacks.compute_tracking import ComputeTrackingCallback
from humite.core.data.datamodules.calo_datamodule import CaloDataModule, build_dm_cfg
from humite.core.training.trainer_factory import (
    build_generative_eval_callback,
    instantiate_lightning_trainer,
    iterate_comet_loggers,
)
from humite.core.utils.checkpoints import (
    apply_initial_weights_if_available,
)
from humite.core.utils.environment_setup import initialize_environment
from humite.core.utils.git_utils import collect_git_metadata, write_git_diff_file
from humite.core.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

torch._dynamo.config.optimize_ddp = False


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Silence Everett/Comet debug chatter while keeping your own DEBUG logs
    logging.getLogger("everett").setLevel(logging.WARNING)
    logging.getLogger("comet_ml").setLevel(logging.WARNING)
    logging.getLogger("comet").setLevel(logging.WARNING)
    initialize_environment(cfg)
    rank_zero_only(log.info)("Resolved Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    if not (isinstance(cfg.dataset, DictConfig) and "_target_" in cfg.dataset):
        raise RuntimeError("cfg.dataset must specify a _target_ for Hydra instantiation.")
    dataset_spec = instantiate(cfg.dataset)

    if hasattr(cfg.dataset, "features") and not dataset_spec.features:
        dataset_spec.features = OmegaConf.to_container(cfg.dataset.features, resolve=True)

    if not (isinstance(cfg.model, DictConfig) and "_target_" in cfg.model):
        raise RuntimeError("cfg.model must specify a _target_ for Hydra instantiation.")
    model = instantiate(cfg.model, spec=dataset_spec)

    init_weights_cfg = getattr(cfg, "init_weights", None)
    if init_weights_cfg is not None:
        try:
            apply_initial_weights_if_available(model, init_weights_cfg)
        except Exception as exc:
            log.error({"init_weights/error": str(exc), "init_weights/cfg": str(init_weights_cfg)})

    trainer = instantiate_lightning_trainer(cfg.trainer)
    if not (isinstance(trainer, L.Trainer) and isinstance(model, L.LightningModule)):
        raise RuntimeError(
            "Trainer or model is not a Lightning class; CaloDataModule requires Lightning."
        )

    run_dir_path = Path(
        cast(str, getattr(trainer, "default_root_dir", os.getcwd()) or os.getcwd())
    )
    run_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        resolved_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        resolved_cfg_path = run_dir_path / "hydra_resolved_config.yaml"
        resolved_cfg_path.write_text(resolved_cfg_yaml)
        resolved_cfg_container = OmegaConf.to_container(cfg, resolve=True)
        resolved_cfg_dict: Dict[str, Any]
        if isinstance(resolved_cfg_container, dict):
            resolved_cfg_dict = cast(Dict[str, Any], resolved_cfg_container)
        else:
            resolved_cfg_dict = {"hydra_config": resolved_cfg_container}

        for logger in iterate_comet_loggers(trainer):
            try:
                log_hyperparams_fn = getattr(logger, "log_hyperparams", None)
                if callable(log_hyperparams_fn):
                    log_hyperparams_fn(resolved_cfg_dict)
            except Exception:
                pass
            try:
                experiment = getattr(logger, "experiment")
                if experiment is not None:
                    experiment.log_asset(str(resolved_cfg_path), file_name=resolved_cfg_path.name)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    trainer_cfg_obj = cfg.trainer
    generative_config_raw: Any = None
    if isinstance(trainer_cfg_obj, DictConfig):
        generative_config_raw = trainer_cfg_obj.get("generative_eval", None)
    elif isinstance(trainer_cfg_obj, dict):
        generative_config_raw = trainer_cfg_obj.get("generative_eval", None)
    else:
        generative_config_raw = getattr(trainer_cfg_obj, "generative_eval", None)

    attach_generative_callback = False
    generative_eval_cfg: Dict[str, Any] = {}
    if generative_config_raw is None:
        attach_generative_callback = True
    elif isinstance(generative_config_raw, bool):
        attach_generative_callback = bool(generative_config_raw)
    else:
        try:
            if isinstance(generative_config_raw, DictConfig):
                attach_generative_callback = True
                generative_eval_cfg = dict(
                    OmegaConf.to_container(generative_config_raw, resolve=True) or {}
                )
            elif isinstance(generative_config_raw, dict):
                attach_generative_callback = True
                generative_eval_cfg = dict(generative_config_raw)
            else:
                generative_eval_cfg = dict(generative_config_raw)  # type: ignore[arg-type]
                attach_generative_callback = True
        except Exception:
            attach_generative_callback = False

    # off-switch: n_val_gen_showers <= 0 means no generative callback.
    if attach_generative_callback:
        try:
            n_val_gen_showers = int(generative_eval_cfg.get("n_val_gen_showers", 8))
            if n_val_gen_showers <= 0:
                attach_generative_callback = False
                log.info(
                    "GenerativeEvalCallback disabled because trainer.generative_eval.n_val_gen_showers <= 0."
                )
        except Exception:
            pass

    if attach_generative_callback:
        callback = build_generative_eval_callback(generative_eval_cfg, cfg.dataset)
        if callback is not None:
            cast(L.Trainer, trainer).callbacks.append(callback)
            try:
                log.debug(
                    {
                        "callback": "GenerativeEvalCallback",
                        "output_dir": getattr(callback, "output_dir", None),
                        "n_val_gen_showers": getattr(callback, "n_val_gen_showers", None),
                        "starting_at_epoch": getattr(callback, "starting_at_epoch", None),
                        "every_n_epochs": getattr(callback, "every_n_epochs", None),
                        "sampling": getattr(callback, "sampling", {}),
                        "spatial_nbins": getattr(callback, "spatial_nbins", None),
                    }
                )
            except Exception:
                pass
        else:
            try:
                log.warning("GenerativeEvalCallback could not be constructed; skipping attach.")
            except Exception:
                pass

    # Always attach ComputeTrackingCallback to monitor FLOPs and training time
    try:
        compute_callback = ComputeTrackingCallback()
        cast(L.Trainer, trainer).callbacks.append(compute_callback)
        log.info("Attached ComputeTrackingCallback for FLOPs and training time monitoring.")
    except Exception as e:
        log.warning(f"Failed to attach ComputeTrackingCallback: {e}")

    try:
        dm_cfg = build_dm_cfg(cfg)
        datamodule = CaloDataModule(
            dataset=dataset_spec,
            batch_size=int(dm_cfg.batch_size),
            num_workers=int(dm_cfg.num_workers),
        )
        git_metadata = collect_git_metadata()
        if git_metadata.get("commit"):
            try:
                log.info(f"git/commit: {git_metadata['commit']}")
                branch_name = git_metadata.get("branch")
                if branch_name:
                    log.info(f"git/branch: {branch_name}")
                diff_path, diff_preview = write_git_diff_file(git_metadata, str(run_dir_path))
                for logger in iterate_comet_loggers(trainer):
                    try:
                        experiment = getattr(logger, "experiment")
                        if experiment is not None:
                            experiment.log_parameter("git_commit", git_metadata["commit"])  # type: ignore[attr-defined]
                            if branch_name:
                                experiment.log_parameter("git_branch", branch_name)  # type: ignore[attr-defined]
                            if diff_path:
                                experiment.log_asset(
                                    diff_path, file_name=os.path.basename(diff_path)
                                )  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
        resume_path = None
        if isinstance(trainer_cfg_obj, DictConfig):
            resume_candidate = trainer_cfg_obj.get("resume_ckpt_path")
            if resume_candidate is not None and str(resume_candidate):
                resume_path = str(resume_candidate)
        trainer_obj = cast(L.Trainer, trainer)
        if not model.automatic_optimization and trainer_obj.gradient_clip_val is not None:
            model._manual_gradient_clip_val = float(trainer_obj.gradient_clip_val)
            model._manual_gradient_clip_algorithm = trainer_obj.gradient_clip_algorithm or "norm"
            trainer_obj.gradient_clip_val = None
            trainer_obj.gradient_clip_algorithm = None
        if not model.automatic_optimization:
            manual_accum = int(getattr(trainer_obj, "accumulate_grad_batches", 1))
            if manual_accum > 1:
                model._manual_accum_grad_batches = manual_accum
                trainer_obj.accumulate_grad_batches = 1
        trainer_obj.fit(model, datamodule=datamodule, ckpt_path=resume_path)
    except Exception as exc:
        log.error(f"Datamodule setup or training failed: {exc}")
        raise


if __name__ == "__main__":
    main()  # type: ignore[misc]
