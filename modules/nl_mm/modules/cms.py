"""Continuum Memory System (CMS) implementation."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class ContinuumMLP(nn.Module):
    """A stack of per-level feed-forward networks used by the CMS."""

    def __init__(self, d_model: int, mult: int, levels: Iterable[dict]):
        super().__init__()
        self.levels: List[dict] = list(levels)
        self.blocks = nn.ModuleDict()
        for level in self.levels:
            hidden = mult * d_model
            self.blocks[level["name"]] = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.blocks[level["name"]](x) for level in self.levels]
        if not outputs:
            return torch.zeros_like(x)
        update = outputs[0]
        for contribution in outputs[1:]:
            update = update + contribution
        return update
