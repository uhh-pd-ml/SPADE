from __future__ import annotations

import torch


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask
