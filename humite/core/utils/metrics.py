from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()
