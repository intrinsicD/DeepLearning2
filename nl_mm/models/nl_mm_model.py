"""Top-level NL-MM model wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from ..modules.cms import ContinuumMLP
from ..modules.fast_weights import FastWeightState
from ..modules.fusion import CLMMemory, ModalityStream, broadcast_to_modalities, fuse_modalities
from ..modules.nl_core import LevelSpec, NLScheduler, build_level_states
from ..modules.optim.routing import build_optimizer_factories
from .encoders import AudioEncoder, TextEncoder, VisionEncoder
from .decoders import AudioDecoder, ImageDecoder, TextDecoder


@dataclass
class NLMMState:
    text: Optional[list] = None
    image: Optional[list] = None
    audio: Optional[list] = None
    clm: Optional[FastWeightState] = None


class NLMM(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.txt_enc = TextEncoder(cfg)
        self.img_enc = VisionEncoder(cfg)
        self.aud_enc = AudioEncoder(cfg)

        self.clm = CLMMemory(cfg["d_model"], cfg["L_mem"], cfg["n_heads"])
        self.cms = ContinuumMLP(cfg["d_model"], cfg["ffn_mult"], cfg["cms_levels"])

        self.txt_dec = TextDecoder(cfg)
        self.img_dec = ImageDecoder(cfg)
        self.aud_dec = AudioDecoder(cfg)
        self.nl_scheduler: Optional[NLScheduler] = None

    def _gather_level_parameters(self, level_name: str) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.modules():
            if isinstance(module, ContinuumMLP) and level_name in module.blocks:
                params.extend(list(module.blocks[level_name].parameters()))
        return params

    def configure_scheduler(self, cfg: Dict) -> NLScheduler:
        optimizer_factories = build_optimizer_factories(cfg)
        assigned: set[int] = set()
        specs: list[LevelSpec] = []
        level_param_map: Dict[str, list[nn.Parameter]] = {}
        for level in cfg["cms_levels"]:
            params = self._gather_level_parameters(level["name"])
            level_param_map[level["name"]] = params
            for param in params:
                assigned.add(id(param))

        remaining = [p for p in self.parameters() if id(p) not in assigned]
        if remaining:
            fastest = min(cfg["cms_levels"], key=lambda spec: spec["chunk_size"])["name"]
            level_param_map.setdefault(fastest, []).extend(remaining)

        for level in cfg["cms_levels"]:
            params = level_param_map.get(level["name"], [])
            if not params:
                continue
            specs.append(LevelSpec(**level, params=params))

        if not specs:
            raise ValueError("No parameters were assigned to NL scheduler levels")

        level_states = build_level_states(specs, optimizer_factories)
        self.nl_scheduler = NLScheduler(level_states)
        return self.nl_scheduler

    def forward(self, batch: Dict[str, torch.Tensor], state: Optional[NLMMState] = None, *, enable_ttt: bool = False) -> tuple[Dict[str, torch.Tensor], NLMMState]:
        state = state or NLMMState()
        text_tokens, state.text = self.txt_enc(batch["text"], state.text, enable_ttt=enable_ttt) if "text" in batch else (None, None)
        image_tokens, state.image = self.img_enc(batch["image"], state.image, enable_ttt=enable_ttt) if "image" in batch else (None, None)
        audio_tokens, state.audio = self.aud_enc(batch["audio"], state.audio, enable_ttt=enable_ttt) if "audio" in batch else (None, None)

        streams = []
        ordering = []
        if text_tokens is not None:
            stream = ModalityStream(text_tokens)
            streams.append(stream)
            ordering.append(("text", stream))
        if image_tokens is not None:
            stream = ModalityStream(image_tokens)
            streams.append(stream)
            ordering.append(("image", stream))
        if audio_tokens is not None:
            stream = ModalityStream(audio_tokens)
            streams.append(stream)
            ordering.append(("audio", stream))
        if not streams:
            raise ValueError("Batch contains no supported modalities for fusion")
        latent, clm_state = fuse_modalities(streams, self.clm, state.clm)
        latent = self.cms(latent)
        state.clm = clm_state

        broadcast = broadcast_to_modalities(latent, streams)
        outputs: Dict[str, torch.Tensor] = {}
        for (name, _), representation in zip(ordering, broadcast):
            if name == "text":
                outputs[name] = self.txt_dec(representation, batch)
            elif name == "image":
                outputs[name] = self.img_dec(representation, batch)
            elif name == "audio":
                outputs[name] = self.aud_dec(representation, batch)
        return outputs, state
