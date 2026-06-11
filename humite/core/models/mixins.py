from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class EnergyNoiseConfig:
    base_std_fraction: float = 0.01
    schedule_power: float = 0.5


class EnergyNoiseMixin:
    def add_energy_noise(
        self,
        energy_values: torch.Tensor,
        global_step: int,
        max_steps: int,
        config: EnergyNoiseConfig,
    ) -> torch.Tensor:
        step_fraction = torch.tensor(
            float(global_step) / float(max_steps), device=energy_values.device
        )
        std = (
            config.base_std_fraction
            * torch.clamp(step_fraction, min=1e-6) ** config.schedule_power
        )
        noise = torch.randn_like(energy_values) * std
        return torch.clamp(energy_values + noise, min=0.0)


class CumulativeEnergyMixin:
    def calculate_cumulative_energy(
        self,
        raw_energies: torch.Tensor,
        spatial_tokens: torch.Tensor,
        shower_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cumulative = torch.cumsum(torch.relu(raw_energies), dim=1)
        if shower_energy is not None:
            normalization = torch.clamp(shower_energy, min=1e-6)
            cumulative = cumulative / normalization.unsqueeze(-1)
        return cumulative
