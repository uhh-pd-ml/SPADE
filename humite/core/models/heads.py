from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _clamp_optional(
    tensor: torch.Tensor, min_value: float | None, max_value: float | None
) -> torch.Tensor:
    if min_value is not None and max_value is not None:
        return torch.clamp(tensor, min=min_value, max=max_value)
    if min_value is not None:
        return torch.clamp(tensor, min=min_value)
    if max_value is not None:
        return torch.clamp(tensor, max=max_value)
    return tensor


class TokenHead(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StopHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_dim: int = 0,
        hidden_dim: int = 128,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditioning_dim = int(max(0, conditioning_dim))
        input_dim = int(embedding_dim) + self.conditioning_dim
        hidden = int(max(2, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout_p)),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout_p)),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditioning_dim > 0:
            if conditioning is None:
                if x.dim() == 3:
                    bsz, seq_len = x.shape[0], x.shape[1]
                    conditioning = x.new_zeros((bsz, seq_len, self.conditioning_dim))
                elif x.dim() == 2:
                    bsz = x.shape[0]
                    conditioning = x.new_zeros((bsz, self.conditioning_dim))
                else:
                    raise RuntimeError(f"Unsupported StopHead input rank: {x.dim()}")
            conditioning = conditioning.to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, conditioning], dim=-1)
        return self.mlp(x)


class EnergyHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MixtureOfGaussiansEnergyHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_components: int,
        *,
        log_scale_min: float | None = None,
        log_scale_max: float | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_components = int(num_components)
        self.log_scale_min = float(log_scale_min) if log_scale_min is not None else None
        self.log_scale_max = float(log_scale_max) if log_scale_max is not None else None

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.logit_head = nn.Linear(hidden_dim, self.num_components)
        self.mean_head = nn.Linear(hidden_dim, self.num_components)
        self.log_scale_head = nn.Linear(hidden_dim, self.num_components)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.mlp(x)
        logits = self.logit_head(x)
        means = self.mean_head(x)
        raw_log_scales = self.log_scale_head(x)
        log_scales = _clamp_optional(raw_log_scales, self.log_scale_min, self.log_scale_max)
        return {"logits": logits, "means": means, "log_scales": log_scales}


def mixture_expected_value(logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logits, dim=-1)
    expected = (weights * means).sum(dim=-1, keepdim=True)
    return expected


def mixture_gaussian_nll(
    target: torch.Tensor,
    logits: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    log_scale_min: float | None = None,
    log_scale_max: float | None = None,
    scale_epsilon: float = 0.0,
) -> torch.Tensor:
    target_expanded = target.expand_as(means)
    bounded_log_scales = _clamp_optional(log_scales, log_scale_min, log_scale_max)
    scales = F.softplus(bounded_log_scales) + float(scale_epsilon)
    log_weights = torch.log_softmax(logits, dim=-1)
    log_probs = (
        -0.5 * (((target_expanded - means) / scales) ** 2)
        - torch.log(scales)
        - 0.5
        * torch.log(torch.tensor(2 * 3.141592653589793, device=means.device, dtype=means.dtype))
    )
    log_mix = torch.logsumexp(log_weights + log_probs, dim=-1)
    nll = -log_mix
    if mask is not None:
        nll = nll * mask.to(nll.dtype)
        if reduction == "mean":
            denom = mask.sum().clamp_min(1)
            return nll.sum() / denom
        if reduction == "sum":
            return nll.sum()
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
