from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PhysicsStopModel(nn.Module):
    def __init__(
        self,
        vocab_size_z: int,
        vocab_size_x: int,
        vocab_size_y: int,
        spatial_embedding_dim: int = 16,
        energy_projection_dim: int = 16,
        hidden_dim: int = 128,
        dropout_p: float = 0.1,
        num_hidden_layers: int = 2,
        use_position_feature: bool = False,
        position_projection_dim: int = 16,
        use_energy_sum: bool = False,
        use_hit_counter: bool = False,
        hit_counter_log_scale: bool = True,
        hit_counter_projection_dim: int = 16,
        use_history: bool = False,
        history_len: int = 10,
        history_projection_dim: int = 32,
    ) -> None:
        super().__init__()
        self.use_position_feature = bool(use_position_feature)
        self.use_energy_sum = bool(use_energy_sum)
        self.use_hit_counter = bool(use_hit_counter)
        self.hit_counter_log_scale = bool(hit_counter_log_scale)
        self.use_history = bool(use_history)
        self.history_len = int(max(1, history_len))
        self.position_projection_dim = int(max(1, position_projection_dim))

        self.z_embedding = nn.Embedding(vocab_size_z, spatial_embedding_dim, padding_idx=0)
        self.x_embedding = nn.Embedding(vocab_size_x, spatial_embedding_dim, padding_idx=0)
        self.y_embedding = nn.Embedding(vocab_size_y, spatial_embedding_dim, padding_idx=0)
        self.hit_energy_projection = nn.Linear(1, energy_projection_dim)
        self.incident_energy_projection = nn.Linear(1, energy_projection_dim)

        if self.use_position_feature:
            self.position_projection = nn.Linear(1, self.position_projection_dim)
        if self.use_energy_sum:
            self.energy_sum_projection = nn.Linear(1, energy_projection_dim)
        if self.use_hit_counter:
            self.hit_counter_projection = nn.Linear(1, hit_counter_projection_dim)
        if self.use_history:
            # For each of the k history steps: embed z (spatial_embedding_dim) +
            # project energy (energy_projection_dim), then flatten and project down.
            history_raw_dim = self.history_len * (spatial_embedding_dim + energy_projection_dim)
            self.history_projection = nn.Linear(history_raw_dim, history_projection_dim)

        classifier_input_dim = 3 * spatial_embedding_dim + 2 * energy_projection_dim
        if self.use_position_feature:
            classifier_input_dim += self.position_projection_dim
        if self.use_energy_sum:
            classifier_input_dim += energy_projection_dim
        if self.use_hit_counter:
            classifier_input_dim += hit_counter_projection_dim
        if self.use_history:
            classifier_input_dim += history_projection_dim

        layers = []
        current_dim = classifier_input_dim
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        z_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        hit_energy: torch.Tensor,
        incident_energy: torch.Tensor,
        position_ratio: Optional[torch.Tensor] = None,
        energy_sum: Optional[torch.Tensor] = None,
        hit_counter: Optional[torch.Tensor] = None,
        z_history: Optional[torch.Tensor] = None,
        energy_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_embedded = self.z_embedding(z_tokens)
        x_embedded = self.x_embedding(x_tokens)
        y_embedded = self.y_embedding(y_tokens)

        # is_batched_sequence: True for training ([B, T, D]), False for generation ([B, D])
        is_batched_sequence = z_embedded.dim() == 3

        if hit_energy.dim() == z_embedded.dim() - 1:
            hit_energy = hit_energy.unsqueeze(-1)
        hit_energy_features = self.hit_energy_projection(hit_energy)

        if incident_energy.dim() == 1:
            incident_energy = incident_energy.unsqueeze(-1)
        incident_energy_features = self.incident_energy_projection(incident_energy)
        if is_batched_sequence:
            incident_energy_features = incident_energy_features.unsqueeze(1).expand(
                -1, z_embedded.size(1), -1
            )

        position_features = None
        if self.use_position_feature:
            if position_ratio is None:
                raise ValueError(
                    "position_ratio must be provided when use_position_feature is enabled"
                )
            position_ratio = position_ratio.to(
                dtype=hit_energy_features.dtype, device=z_embedded.device
            )
            if is_batched_sequence:
                if position_ratio.dim() != 2:
                    raise ValueError("position_ratio must have shape [B, T] for sequence input")
                if position_ratio.shape != z_tokens.shape:
                    raise ValueError("position_ratio shape must match token shape [B, T]")
            else:
                if position_ratio.dim() != 1:
                    raise ValueError("position_ratio must have shape [B] for step input")
                if position_ratio.shape != z_tokens.shape:
                    raise ValueError("position_ratio shape must match token shape [B]")
            position_ratio = position_ratio.unsqueeze(-1)
            position_features = self.position_projection(position_ratio)

        energy_sum_features = None
        if self.use_energy_sum:
            if energy_sum is None:
                raise ValueError("energy_sum must be provided when use_energy_sum is enabled")
            if energy_sum.dim() == z_embedded.dim() - 1:
                energy_sum = energy_sum.unsqueeze(-1)
            energy_sum_features = self.energy_sum_projection(energy_sum)
            if is_batched_sequence and energy_sum_features.dim() == 2:
                energy_sum_features = energy_sum_features.unsqueeze(1).expand(
                    -1, z_embedded.size(1), -1
                )

        hit_counter_features = None
        if self.use_hit_counter:
            if hit_counter is None:
                raise ValueError("hit_counter must be provided when use_hit_counter is enabled")
            hc = hit_counter.to(dtype=hit_energy_features.dtype, device=z_embedded.device)
            if self.hit_counter_log_scale:
                hc = torch.log1p(hc.clamp(min=0.0))
            # [B] → [B, 1] or [B, T] → [B, T, 1]
            hit_counter_features = self.hit_counter_projection(hc.unsqueeze(-1))

        history_features = None
        if self.use_history:
            if z_history is None or energy_history is None:
                raise ValueError(
                    "z_history and energy_history must be provided when use_history is enabled"
                )
            # z_history: [B, k] (gen) or [B, T, k] (train), long
            # energy_history: [B, k] (gen) or [B, T, k] (train), float
            zh = z_history.to(device=z_embedded.device)
            eh = energy_history.to(dtype=hit_energy_features.dtype, device=z_embedded.device)
            # Embed history z tokens — reuse z_embedding (same vocab, same scale)
            zh_embedded = self.z_embedding(zh)  # [..., k, spatial_embedding_dim]
            # Project history energies — reuse hit_energy_projection (same energy scale)
            eh_projected = self.hit_energy_projection(
                eh.unsqueeze(-1)
            )  # [..., k, energy_projection_dim]
            # Concat along feature dim and flatten k
            hist_cat = torch.cat([zh_embedded, eh_projected], dim=-1)  # [..., k, D]
            *leading, k, d = hist_cat.shape
            hist_flat = hist_cat.reshape(*leading, k * d)  # [B, k*D] or [B, T, k*D]
            history_features = self.history_projection(hist_flat)

        feature_parts = [
            z_embedded,
            x_embedded,
            y_embedded,
            hit_energy_features,
            incident_energy_features,
        ]
        if position_features is not None:
            feature_parts.append(position_features)
        if energy_sum_features is not None:
            feature_parts.append(energy_sum_features)
        if hit_counter_features is not None:
            feature_parts.append(hit_counter_features)
        if history_features is not None:
            feature_parts.append(history_features)

        concatenated_features = torch.cat(feature_parts, dim=-1)
        return self.classifier(concatenated_features).squeeze(-1)


def compute_physics_stop_loss(
    stop_logits: torch.Tensor,
    stop_targets: torch.Tensor,
    stop_mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], device=stop_logits.device, dtype=stop_logits.dtype)
    per_position_loss = F.binary_cross_entropy_with_logits(
        stop_logits, stop_targets.to(stop_logits.dtype), reduction="none", pos_weight=pw
    )
    masked_loss = per_position_loss * stop_mask.to(per_position_loss.dtype)
    denominator = stop_mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denominator


def build_physics_stop_targets(
    n_hits: torch.Tensor,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = n_hits.size(0)
    position_indices = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    is_last_hit = position_indices.eq(n_hits.unsqueeze(1) - 1)
    is_valid_position = position_indices.lt(n_hits.unsqueeze(1))
    stop_targets = is_last_hit.to(torch.float32)
    return stop_targets, is_valid_position
