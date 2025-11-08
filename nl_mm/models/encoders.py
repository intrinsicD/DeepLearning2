"""Encoders for the NL-MM architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..modules.cms import ContinuumMLP
from ..modules.fast_weights import FastWeightLinearAttention, FastWeightState
from ..modules.ttt import TTTAdapter


@dataclass
class NLBlockState:
    fast: Optional[FastWeightState] = None


class NLBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cms: ContinuumMLP, ttt: Optional[TTTAdapter] = None):
        super().__init__()
        self.fast_attn = FastWeightLinearAttention(d_model, n_heads)
        self.cms = cms
        self.ttt = ttt

    def forward(self, x: torch.Tensor, state: NLBlockState, *, slow_state: Optional[torch.Tensor] = None, enable_ttt: bool = False) -> tuple[torch.Tensor, NLBlockState]:
        y, fast_state = self.fast_attn(x, state.fast, slow_state=slow_state)
        z = self.cms(y)
        if enable_ttt and self.ttt is not None:
            z = self.ttt(z)
        return z, NLBlockState(fast=fast_state)


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        vocab = cfg.get("vocab_size", 32000)
        depth = cfg["depth"]["text"]
        self.token = nn.Embedding(vocab, d_model)
        max_pos = cfg.get("max_position_embeddings", 1024)
        self.pos = nn.Parameter(torch.randn(max_pos, d_model) * 0.01)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            cms = ContinuumMLP(d_model, cfg["ffn_mult"], cfg["cms_levels"])
            ttt = (
                TTTAdapter(
                    d_model,
                    rank=cfg["ttt"]["adapter_rank"],
                    eta=cfg["ttt"]["eta"],
                    max_steps=cfg["ttt"].get("max_steps", 2),
                )
                if cfg["ttt"]["enable"]
                else None
            )
            self.blocks.append(NLBlock(d_model, cfg["n_heads"], cms=cms, ttt=ttt))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, state: Optional[list[NLBlockState]] = None, *, enable_ttt: bool = False) -> tuple[torch.Tensor, list[NLBlockState]]:
        if state is None:
            state = [NLBlockState() for _ in range(len(self.blocks))]
        seq_len = tokens.size(1)
        if seq_len > self.pos.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings {self.pos.size(0)}"
            )
        x = self.token(tokens) + self.pos[:seq_len]
        new_state: list[NLBlockState] = []
        slow_state = x.mean(dim=1)
        for block, block_state in zip(self.blocks, state):
            x, next_state = block(x, block_state, slow_state=slow_state, enable_ttt=enable_ttt)
            slow_state = x.mean(dim=1)
            new_state.append(next_state)
        return self.norm(x), new_state


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        depth = cfg["depth"]["image"]
        patch = cfg.get("patch_size", 8)
        in_chans = cfg.get("image_channels", 3)
        self.patch = nn.Conv2d(in_chans, d_model, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            cms = ContinuumMLP(d_model, cfg["ffn_mult"], cfg["cms_levels"])
            ttt = (
                TTTAdapter(
                    d_model,
                    rank=cfg["ttt"]["adapter_rank"],
                    eta=cfg["ttt"]["eta"],
                    max_steps=cfg["ttt"].get("max_steps", 2),
                )
                if cfg["ttt"]["enable"]
                else None
            )
            self.blocks.append(NLBlock(d_model, cfg["n_heads"], cms=cms, ttt=ttt))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor, state: Optional[list[NLBlockState]] = None, *, enable_ttt: bool = False) -> tuple[torch.Tensor, list[NLBlockState]]:
        if state is None:
            state = [NLBlockState() for _ in range(len(self.blocks))]
        x = self.patch(images)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        slow_state = x.mean(dim=1)
        new_state: list[NLBlockState] = []
        for block, block_state in zip(self.blocks, state):
            x, next_state = block(x, block_state, slow_state=slow_state, enable_ttt=enable_ttt)
            slow_state = x.mean(dim=1)
            new_state.append(next_state)
        return self.norm(x), new_state


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        depth = cfg["depth"]["audio"]
        in_chans = cfg.get("audio_channels", 1)
        self.frontend = nn.Sequential(nn.Conv1d(in_chans, d_model // 2, kernel_size=3, padding=1), nn.GELU(), nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            cms = ContinuumMLP(d_model, cfg["ffn_mult"], cfg["cms_levels"])
            ttt = (
                TTTAdapter(
                    d_model,
                    rank=cfg["ttt"]["adapter_rank"],
                    eta=cfg["ttt"]["eta"],
                    max_steps=cfg["ttt"].get("max_steps", 2),
                )
                if cfg["ttt"]["enable"]
                else None
            )
            self.blocks.append(NLBlock(d_model, cfg["n_heads"], cms=cms, ttt=ttt))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio: torch.Tensor, state: Optional[list[NLBlockState]] = None, *, enable_ttt: bool = False) -> tuple[torch.Tensor, list[NLBlockState]]:
        if state is None:
            state = [NLBlockState() for _ in range(len(self.blocks))]
        x = self.frontend(audio).transpose(1, 2)
        slow_state = x.mean(dim=1)
        new_state: list[NLBlockState] = []
        for block, block_state in zip(self.blocks, state):
            x, next_state = block(x, block_state, slow_state=slow_state, enable_ttt=enable_ttt)
            slow_state = x.mean(dim=1)
            new_state.append(next_state)
        return self.norm(x), new_state
