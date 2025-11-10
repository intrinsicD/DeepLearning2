"""Decoders for NL-MM."""
from __future__ import annotations

import torch
from torch import nn


class TextDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.get("vocab_size", 32000)
        self.lm_head = nn.Linear(cfg["d_model"], self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        logits = self.lm_head(tokens)
        if "text_target" in batch:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["text_target"].view(-1), ignore_index=0)
            return loss
        return logits


class ImageDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        self.channels = cfg.get("image_channels", 3)
        self.resolution = cfg.get("image_resolution", 64)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.channels * self.resolution * self.resolution))

    def forward(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        x = tokens.mean(dim=1)
        recon = self.out(x).view(x.size(0), self.channels, self.resolution, self.resolution)
        if "image_target" in batch:
            return nn.functional.mse_loss(recon, batch["image_target"])
        return recon


class AudioDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        self.mel_bins = cfg.get("audio_mel_bins", 80)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.mel_bins))

    def forward(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        x = tokens.mean(dim=1)
        mel = self.proj(x)
        if "audio_target" in batch:
            return nn.functional.l1_loss(mel, batch["audio_target"])
        return mel
