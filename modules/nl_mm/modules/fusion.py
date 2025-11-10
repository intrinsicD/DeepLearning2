"""Central latent memory fusion utilities."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn

from .fast_weights import FastWeightLinearAttention, FastWeightState


class CLMMemory(nn.Module):
    def __init__(self, d_model: int, length: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.length = length
        self.latent = nn.Parameter(torch.randn(length, d_model) * 0.02)
        self.fast_attn = FastWeightLinearAttention(d_model, n_heads)

    def forward(self, batch_size: int, state: FastWeightState | None = None) -> Tuple[torch.Tensor, FastWeightState]:
        latent = self.latent.unsqueeze(0).expand(batch_size, -1, -1)
        latents, state = self.fast_attn(latent, state)
        return latents, state


class ModalityStream:
    def __init__(self, tokens: torch.Tensor, mask: torch.Tensor | None = None):
        self.tokens = tokens
        self.mask = mask


def fuse_modalities(streams: Sequence[ModalityStream], clm: CLMMemory, state: FastWeightState | None = None) -> Tuple[torch.Tensor, FastWeightState]:
    assert streams, "At least one modality stream is required"
    batch = streams[0].tokens.size(0)
    latent, state = clm(batch, state)
    fused = latent
    for stream in streams:
        attn = torch.einsum("bld,btd->blt", fused, stream.tokens)
        if stream.mask is not None:
            attn = attn.masked_fill(stream.mask[:, None, :].logical_not(), float("-inf"))
        weights = attn.softmax(dim=-1)
        fused = fused + torch.einsum("blt,btd->bld", weights, stream.tokens)
    return fused, state


def broadcast_to_modalities(latent: torch.Tensor, streams: Sequence[ModalityStream]) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    for stream in streams:
        weights = torch.einsum("btd,bld->btl", stream.tokens, latent).softmax(dim=-1)
        outputs.append(stream.tokens + torch.einsum("btl,bld->btd", weights, latent))
    return outputs
