from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

# Maps a `subset` name to the normalized key prefix it restricts loading to.
# "all" (and any unlisted subset) loads everything not caught by exclude_patterns.
_SUBSET_PREFIXES: Dict[str, str] = {
    "core": "core.",
    "gpt": "core.",
    "decoder": "core.",
    "physics_stop": "physics_stop_model.",
}

import torch
from omegaconf import DictConfig

try:
    from hydra.utils import to_absolute_path as _to_abs
except Exception:  # pragma: no cover
    _to_abs = None

from .pylogger import get_pylogger

log = get_pylogger(__name__)


def _load_checkpoint_contents(checkpoint_path: str) -> Dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosec
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")  # nosec


def load_pretrained_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    subset: str = "all",
    strict: bool = False,
    exclude_patterns: list | None = None,
) -> Tuple[list[str], list[str]]:
    checkpoint = _load_checkpoint_contents(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered: Dict[str, torch.Tensor] = {}
    target_keys = set(model.state_dict().keys())
    # Ignore empty patterns so values like [""] do not exclude every key.
    exclude_patterns = [str(pat).strip() for pat in (exclude_patterns or []) if str(pat).strip()]
    required_prefix = _SUBSET_PREFIXES.get(subset)
    for key, value in state_dict.items():
        normalized = key[6:] if key.startswith("model.") else key
        if required_prefix is not None and not normalized.startswith(required_prefix):
            continue
        if any(pat in normalized for pat in exclude_patterns):
            continue
        candidate_keys = [key]
        if key.startswith("model."):
            candidate_keys.append(normalized)

        selected_key = None
        for candidate in candidate_keys:
            if candidate in target_keys:
                selected_key = candidate
                break

        if selected_key is not None:
            filtered[selected_key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return list(missing), list(unexpected)


def apply_initial_weights_if_available(model: torch.nn.Module, init_cfg: Any) -> None:
    checkpoint_path_value = None
    subset = "all"
    strict = False
    exclude_patterns: list = []
    try:
        if isinstance(init_cfg, DictConfig):
            checkpoint_path_value = init_cfg.get("ckpt_path", init_cfg.get("path", ""))
            subset = str(init_cfg.get("subset", "all"))
            strict = bool(init_cfg.get("strict", False))
            exclude_patterns = [
                str(pat).strip()
                for pat in list(init_cfg.get("exclude_patterns", []))
                if str(pat).strip()
            ]
        elif isinstance(init_cfg, dict):
            checkpoint_path_value = init_cfg.get("ckpt_path", init_cfg.get("path", ""))
            subset = str(init_cfg.get("subset", "all"))
            strict = bool(init_cfg.get("strict", False))
            exclude_patterns = [
                str(pat).strip()
                for pat in list(init_cfg.get("exclude_patterns", []))
                if str(pat).strip()
            ]
        else:
            checkpoint_path_value = getattr(init_cfg, "ckpt_path", getattr(init_cfg, "path", ""))
            subset = str(getattr(init_cfg, "subset", "all"))
            strict = bool(getattr(init_cfg, "strict", False))
            exclude_patterns = [
                str(pat).strip()
                for pat in list(getattr(init_cfg, "exclude_patterns", []))
                if str(pat).strip()
            ]
    except Exception:
        checkpoint_path_value = getattr(init_cfg, "ckpt_path", getattr(init_cfg, "path", ""))
    checkpoint_raw = str(checkpoint_path_value or "")
    if not checkpoint_raw.strip():
        log.debug({"init_weights": "skipped", "reason": "no ckpt_path set"})
        return
    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.is_absolute() and _to_abs is not None:
        try:
            checkpoint_path = Path(_to_abs(checkpoint_raw))
        except Exception:
            checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.is_file():
        log.warning({"init_weights/warning": "checkpoint_not_found", "path": str(checkpoint_path)})
    target = model
    missing, unexpected = load_pretrained_weights_from_checkpoint(
        target,
        str(checkpoint_path),
        subset=subset,
        strict=strict,
        exclude_patterns=exclude_patterns,
    )

    log.info(
        {
            "init_weights/loaded_from": str(checkpoint_path),
            "init_weights/subset": subset,
            "init_weights/strict": bool(strict),
            "init_weights/exclude_patterns": exclude_patterns,
            "init_weights/missing": missing[:10],
            "init_weights/missing_count": len(missing),
            "init_weights/unexpected": unexpected[:10],
            "init_weights/unexpected_count": len(unexpected),
        }
    )
