from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from ..models.factory import build_model_core
from ..models.gpt_decoder import TransformerConfig
from ..models.lightning_module import BackboneLightningModule
from ..models.physics_stop import PhysicsStopModel


def build_model(
    cfg: Optional[Any] = None,
    spec: Optional[Any] = None,
    name: Optional[str] = None,
    transformer: Optional[Any] = None,
    kwargs: Optional[Any] = None,
    optim: Optional[Any] = None,
) -> BackboneLightningModule:
    resolved_kwargs = _dict_from_any(
        kwargs if kwargs is not None else _resolve_attr(cfg, "kwargs")
    )
    axis_mode = resolved_kwargs.get("axis_mode", None)
    transformer_source = transformer or _resolve_attr(cfg, "transformer")
    if transformer_source is None:
        raise ValueError("Missing transformer configuration for model builder")
    if isinstance(transformer_source, TransformerConfig):
        transformer_cfg = transformer_source
    else:
        transformer_cfg = TransformerConfig(**_dict_from_any(transformer_source))
    resolved_optim = _dict_from_any(optim if optim is not None else _resolve_attr(cfg, "optim"))
    register_count = int(resolved_kwargs.get("n_registers", 0) or 0)
    if register_count > 0 and getattr(transformer_cfg, "num_registers", 0) < register_count:
        transformer_cfg.num_registers = register_count
    model = build_model_core(transformer=transformer_cfg, kwargs=resolved_kwargs)
    lr = float(resolved_optim.get("lr", 1e-4))
    wd = float(resolved_optim.get("weight_decay", 0.01))
    opt_name_raw = resolved_optim.get("optimizer_name", resolved_optim.get("name", "adamw"))
    opt_name = str(opt_name_raw).lower()
    betas = _resolve_betas(resolved_optim.get("betas", (0.9, 0.999)))
    opt_eps = float(resolved_optim.get("eps", 1e-8))
    la_alpha = float(resolved_optim.get("lookahead_alpha", 0.5))
    la_k = int(resolved_optim.get("lookahead_k", 6))
    scheduler_name = str(resolved_optim.get("scheduler_name", "none"))
    scheduler_warmup_steps = int(resolved_optim.get("scheduler_warmup_steps", 0))
    scheduler_total_steps = int(
        resolved_optim.get("scheduler_total_steps", resolved_optim.get("scheduler_max_steps", 0))
    )
    scheduler_min_lr = float(resolved_optim.get("scheduler_min_lr", 0.0))
    backbone_lr_scale = float(resolved_optim.get("backbone_lr_scale", 1.0))
    backbone_unfreeze_steps = int(resolved_optim.get("backbone_unfreeze_steps", 0))
    token_pad_id = int(
        resolved_kwargs.get(
            "token_pad_id",
            getattr(getattr(spec, "vocab", None), "pad_id", 0) if spec is not None else 0,
        )
    )
    energy_loss_weight = float(resolved_kwargs.get("energy_loss_weight", 1.0))
    energy_loss_min_weight = float(resolved_kwargs.get("energy_loss_min_weight", 0.0))
    energy_loss_warmup_steps = int(resolved_kwargs.get("energy_loss_warmup_steps", 0))
    use_z_causal_mask = bool(resolved_kwargs.get("use_z_causal_mask", False))

    physics_stop_model = None
    if bool(resolved_kwargs.get("use_physics_stop", False)):
        per_axis_vocab_sizes = resolved_kwargs.get("per_axis_vocab_sizes", {})
        physics_stop_model = PhysicsStopModel(
            vocab_size_z=int(per_axis_vocab_sizes.get("z", 64)),
            vocab_size_x=int(per_axis_vocab_sizes.get("x", 64)),
            vocab_size_y=int(per_axis_vocab_sizes.get("y", 64)),
            spatial_embedding_dim=int(resolved_kwargs.get("physics_stop_embedding_dim", 16)),
            energy_projection_dim=int(resolved_kwargs.get("physics_stop_energy_dim", 16)),
            hidden_dim=int(resolved_kwargs.get("physics_stop_hidden_dim", 128)),
            dropout_p=float(resolved_kwargs.get("physics_stop_dropout_p", 0.1)),
            num_hidden_layers=int(resolved_kwargs.get("physics_stop_num_layers", 2)),
            use_position_feature=bool(
                resolved_kwargs.get("physics_stop_use_position_feature", False)
            ),
            position_projection_dim=int(resolved_kwargs.get("physics_stop_position_dim", 16)),
            use_energy_sum=bool(resolved_kwargs.get("physics_stop_use_energy_sum", False)),
            use_hit_counter=bool(resolved_kwargs.get("physics_stop_use_hit_counter", False)),
            hit_counter_log_scale=bool(
                resolved_kwargs.get("physics_stop_hit_counter_log_scale", True)
            ),
            hit_counter_projection_dim=int(
                resolved_kwargs.get("physics_stop_hit_counter_dim", 16)
            ),
            use_history=bool(resolved_kwargs.get("physics_stop_use_history", False)),
            history_len=int(resolved_kwargs.get("physics_stop_history_len", 10)),
            history_projection_dim=int(resolved_kwargs.get("physics_stop_history_dim", 32)),
        )
    physics_stop_lr = float(resolved_kwargs.get("physics_stop_lr", 3e-4))
    physics_stop_weight_decay = float(resolved_kwargs.get("physics_stop_weight_decay", 0.01))
    physics_stop_loss_weight = float(resolved_kwargs.get("physics_stop_loss_weight", 1.0))
    physics_stop_pos_weight = float(resolved_kwargs.get("physics_stop_pos_weight", 0.0))
    physics_stop_min_hits = int(resolved_kwargs.get("physics_stop_min_hits", 5))
    physics_stop_auto_freeze = bool(resolved_kwargs.get("physics_stop_auto_freeze", True))
    physics_stop_freeze_min_epochs = int(resolved_kwargs.get("physics_stop_freeze_min_epochs", 5))
    physics_stop_freeze_patience = int(resolved_kwargs.get("physics_stop_freeze_patience", 4))
    physics_stop_freeze_min_delta = float(
        resolved_kwargs.get("physics_stop_freeze_min_delta", 1e-4)
    )
    physics_stop_freeze_min_recall = float(
        resolved_kwargs.get("physics_stop_freeze_min_recall", 0.0)
    )
    physics_stop_freeze_min_precision = float(
        resolved_kwargs.get("physics_stop_freeze_min_precision", 0.0)
    )
    physics_stop_energy_sum_noise_sigma = float(
        resolved_kwargs.get("physics_stop_energy_sum_noise_sigma", 0.0)
    )
    physics_stop_sampling_fraction = float(
        resolved_kwargs.get("physics_stop_sampling_fraction", 1.0)
    )
    training_mode = str(resolved_kwargs.get("training_mode", "all"))
    energy_label_smoothing = str(resolved_kwargs.get("energy_label_smoothing", "none"))

    return BackboneLightningModule(
        model=model,
        lr=lr,
        weight_decay=wd,
        token_pad_id=token_pad_id,
        energy_loss_weight=energy_loss_weight,
        energy_loss_min_weight=energy_loss_min_weight,
        energy_loss_warmup_steps=energy_loss_warmup_steps,
        optimizer_name=opt_name,
        betas=betas,
        eps=opt_eps,
        lookahead_alpha=la_alpha,
        lookahead_k=la_k,
        lr_scheduler_name=scheduler_name,
        lr_scheduler_warmup_steps=scheduler_warmup_steps,
        lr_scheduler_total_steps=scheduler_total_steps,
        lr_scheduler_min_lr=scheduler_min_lr,
        backbone_lr_scale=backbone_lr_scale,
        backbone_unfreeze_steps=backbone_unfreeze_steps,
        axis_mode=axis_mode,
        use_z_causal_mask=use_z_causal_mask,
        physics_stop_model=physics_stop_model,
        physics_stop_lr=physics_stop_lr,
        physics_stop_weight_decay=physics_stop_weight_decay,
        physics_stop_loss_weight=physics_stop_loss_weight,
        physics_stop_pos_weight=physics_stop_pos_weight,
        physics_stop_min_hits=physics_stop_min_hits,
        physics_stop_auto_freeze=physics_stop_auto_freeze,
        physics_stop_freeze_min_epochs=physics_stop_freeze_min_epochs,
        physics_stop_freeze_patience=physics_stop_freeze_patience,
        physics_stop_freeze_min_delta=physics_stop_freeze_min_delta,
        physics_stop_freeze_min_recall=physics_stop_freeze_min_recall,
        physics_stop_freeze_min_precision=physics_stop_freeze_min_precision,
        physics_stop_energy_sum_noise_sigma=physics_stop_energy_sum_noise_sigma,
        physics_stop_sampling_fraction=physics_stop_sampling_fraction,
        training_mode=training_mode,
        energy_label_smoothing=energy_label_smoothing,
    )


def _resolve_attr(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    if isinstance(source, DictConfig):
        return source.get(key)
    return getattr(source, key, None)


def _dict_from_any(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        materialized = OmegaConf.to_container(value, resolve=True)
        return dict(materialized) if isinstance(materialized, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {k: getattr(value, k) for k in vars(value) if not k.startswith("_")}
    return {}


def _resolve_betas(betas: Any) -> Tuple[float, float]:
    try:
        if isinstance(betas, (list, tuple)) and len(betas) == 2:
            return float(betas[0]), float(betas[1])
    except Exception:
        pass
    return 0.9, 0.999
