from __future__ import annotations

from torch import nn


class Identity(nn.Module):
    def forward(self, x):  # type: ignore[no-untyped-def]
        return x
