"""Continuum Memory System (CMS) implementation."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class ContinuumMLP(nn.Module):
    """A residual stack of MLPs with level annotations."""

    def __init__(self, d_model: int, mult: int, levels: Iterable[dict]):
        super().__init__()
        self.levels: List[dict] = list(levels)
        self.blocks = nn.ModuleDict()
        for level in self.levels:
            hidden = mult * d_model
            self.blocks[level["name"]] = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for level in self.levels:
            block = self.blocks[level["name"]]
            x = x + block(x)
        return x
