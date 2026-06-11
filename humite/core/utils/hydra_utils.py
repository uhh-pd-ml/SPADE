from __future__ import annotations

from typing import Any


def to_dict(cfg: Any) -> dict:
    if hasattr(cfg, "to_container"):
        return cfg.to_container(resolve=True)
    if isinstance(cfg, dict):
        return cfg
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
