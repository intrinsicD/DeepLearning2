# multimodal_brain_v2.py
#
# Canonical shared-latent multimodal architecture with:
# - ModalityInterface: wraps arbitrary encoder/decoder via residual Up/Down adapters
# - Shared latent "language" (d_shared)
# - Tiny thinking core over a few tokens: [S] (state), one token per modality, optional [M] memory
# - ThinkControl to modulate how much "thinking" happens (K steps, mode)
#
# This file is intentionally HF/diffusers-agnostic. You plug in whatever encoders/decoders
# you like (CLIP, ImageBind, Whisper, T5, GPT, VAE, HiFi-GAN, ...), and just implement
# small projection/adapters as the glue.
#
# Inspired by:
# - CLIP / LiT-style joint embedding spaces: dual encoders aligned in a shared space. :contentReference[oaicite:1]{index=1}
# - ImageBind: one embedding space for many modalities. :contentReference[oaicite:2]{index=2}
# - BLIP-2 Q-Former & residual adapters for parameter-efficient modularity. :contentReference[oaicite:3]{index=3}

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def freeze_module(m: nn.Module, trainable: bool = False):
    """Freeze / unfreeze a module in one call."""
    for p in m.parameters():
        p.requires_grad = trainable
    m.eval() if not trainable else m.train()
    return m


class ReZero(nn.Module):
    """Zero-init scalar gate for residuals (stabilizes deep stacks)."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.alpha * residual


class PreNorm(nn.Module):
    """LayerNorm + wrapped fn (for pre-LN transformer blocks)."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class SelfAttentionBlock(nn.Module):
    """Standard transformer block: self-attn + FFN, both with ReZero."""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.g1 = ReZero()
        self.g2 = ReZero()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        sa_out, _ = self.sa(x, x, x, need_weights=False)
        x = x + self.g1(sa_out)
        # FFN
        x = x + self.g2(self.ff(x))
        return x


# ------------------------------------------------------------
# Adapters: Up (encoder -> shared) and Down (shared -> decoder)
# ------------------------------------------------------------

class UpAdapter(nn.Module):
    """
    Glue from encoder's native space -> shared latent space.

    This is where you "teach" a new encoder to speak the canonical latent language.
    You can swap encoders by swapping this adapter (and retraining only it).
    """
    def __init__(self, d_in: int, d_shared: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_shared) if d_in != d_shared else nn.Identity()
        d_hid = max(d_shared, hidden_mult * d_shared)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_shared),
        )
        self.gate = ReZero()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D_in) or (B, T, D_in). If sequence, we'll mean-pool over T.

        Returns: (B, d_shared) pooled representation in shared space.
        """
        if h.dim() == 3:
            h = h.mean(dim=1)  # simple pooling; you can customize
        x = self.proj(h)
        return x + self.gate(self.ff(x))


class DownAdapter(nn.Module):
    """
    Glue from shared latent space -> decoder's native space.

    This is the symmetric twin of UpAdapter. It lets you swap decoders while keeping
    the thinking core frozen.
    """
    def __init__(self, d_shared: int, d_out: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, hidden_mult * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mult * d_shared, d_out),
        )

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        """
        z_shared: (B, d_shared) vector (e.g. [S] token).
        Returns: (B, d_out) decoder input representation.
        """
        return self.ff(z_shared)


# ------------------------------------------------------------
# ModalityInterface: enc/dec + adapters
# ------------------------------------------------------------

class ModalityInterface(nn.Module):
    """
    Wraps:
      - encoder: arbitrary backbone for a modality (text, image, audio, ...)
      - up_adapter: encoder space -> shared latent space (d_shared)
      - decoder: optional backbone to turn decoded representations into actual outputs
      - down_adapter: shared latent -> decoder space

    This is the "glue" object. The thinking core only sees the shared latent;
    you can hot-swap encoders/decoders by swapping/adapting this interface.
    """

    def __init__(
        self,
        *,
        name: str,
        encoder: nn.Module,
        up_adapter: UpAdapter,
        decoder: Optional[nn.Module] = None,
        down_adapter: Optional[DownAdapter] = None,
        freeze_encoder: bool = True,
        freeze_decoder: bool = True,
    ):
        super().__init__()
        self.name = name
        self.encoder = freeze_module(encoder, trainable=not freeze_encoder)
        self.up = up_adapter
        self.decoder = freeze_module(decoder, trainable=not freeze_decoder) if decoder is not None else None
        self.down = down_adapter

    def encode_to_shared(self, x: Any) -> torch.Tensor:
        """
        x: whatever the encoder expects (tensor, dict, etc.)
        Returns: (B, d_shared) in canonical latent space.
        """
        h = self.encoder(x)
        # support encoders that return (feats, mask) or dicts
        if isinstance(h, tuple):
            h, _ = h
        if isinstance(h, dict):
            # try common keys
            for key in ("last_hidden_state", "hidden_states", "embeddings"):
                if key in h:
                    h = h[key]
                    break
        return self.up(h)

    def decode_from_shared(self, z_shared: torch.Tensor) -> Any:
        """
        z_shared: (B, d_shared) vector from thinking core, e.g. [S] token.
        Returns whatever the decoder emits (text, image tensor, audio waveform, etc.).
        """
        if self.decoder is None or self.down is None:
            raise RuntimeError(f"Modality '{self.name}' has no decoder/down_adapter configured.")
        h_dec = self.down(z_shared)
        return self.decoder(h_dec)


# ------------------------------------------------------------
# ThinkControl: how much & how we think
# ------------------------------------------------------------

@dataclass
class ThinkControl:
    steps: int = 2
    mode: Literal["default", "fast", "deep", "off"] = "default"
    # Optional: per-modality weights for how strongly they should influence thinking
    modality_weights: Dict[str, float] = field(default_factory=dict)

    def effective_steps(self) -> int:
        if self.mode == "off":
            return 0
        if self.mode == "fast":
            return max(0, min(1, self.steps))
        if self.mode == "deep":
            return self.steps + 2
        return self.steps


# ------------------------------------------------------------
# Thinking core: operate on a tiny set of tokens in shared space
# ------------------------------------------------------------

class ThinkingCore(nn.Module):
    """
    Tiny transformer over a few tokens:
      [S]  : global state token (aggregates all modalities)
      [mod]: one token per modality (text/image/audio/...)
      [M]  : optional memory token (for future extension)

    The core ONLY lives in the shared space. Encoders/decoders are plugged in via adapters.
    """

    def __init__(
        self,
        d_shared: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_memory_token: bool = False,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.use_memory_token = use_memory_token

        # These will be created dynamically per modality set:
        # we keep a registry of learned base embeddings for [S], [M], and per-modality tags.
        self.global_token = nn.Parameter(torch.randn(1, 1, d_shared) / (d_shared ** 0.5))
        self.memory_token = nn.Parameter(torch.randn(1, 1, d_shared) / (d_shared ** 0.5))
        self.modality_tokens = nn.ParameterDict()  # name -> (1,1,d_shared)

        self.blocks = nn.ModuleList([SelfAttentionBlock(d_shared, n_heads, dropout) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_shared)

    def ensure_modality(self, name: str):
        if name not in self.modality_tokens:
            # Create parameter with the same device and dtype as an existing parameter
            # to ensure it's on the correct device from the start
            param = nn.Parameter(torch.randn(1, 1, self.d_shared) / (self.d_shared ** 0.5))
            # Register it properly so it moves with the module
            self.register_parameter(f"modality_token_{name}", param)
            self.modality_tokens[name] = param

    def forward(
        self,
        z_by_mod: Dict[str, torch.Tensor],
        control: ThinkControl,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        z_by_mod: dict of {modality_name: (B, d_shared)} from UpAdapters
        control: ThinkControl to decide number of passes
        memory_state: optional previous [M] token (B, d_shared) for persistence

        Returns:
          - tokens: (B, N_tokens, d_shared) final tokens
          - z_global: (B, d_shared) final [S] token
          - z_by_mod_out: dict of updated tokens per modality name
        """
        device = next(iter(z_by_mod.values())).device
        B = next(iter(z_by_mod.values())).size(0)

        # --- Build initial tokens
        tokens: List[torch.Tensor] = []
        names: List[str] = []

        # [S] global token: start with learned base + weighted sum of modalities
        s0 = self.global_token.to(device).expand(B, -1, -1)  # (B,1,D)
        if len(z_by_mod) > 0:
            stacked = torch.stack(list(z_by_mod.values()), dim=1)  # (B, M, D)
            weights = []
            for name in z_by_mod.keys():
                w = control.modality_weights.get(name, 1.0)
                weights.append(w)
            w = torch.tensor(weights, device=device, dtype=stacked.dtype).view(1, -1, 1)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            agg = (stacked * w).sum(dim=1, keepdim=True)  # (B,1,D)
            s0 = s0 + agg
        tokens.append(s0)
        names.append("[S]")

        # modality tokens
        for name, z in z_by_mod.items():
            self.ensure_modality(name)
            base = self.modality_tokens[name].to(z.device).expand(B, 1, -1)
            tokens.append(base + z.unsqueeze(1))  # (B,1,D)
            names.append(name)

        # optional memory token
        if self.use_memory_token:
            if memory_state is None:
                m0 = self.memory_token.to(device).expand(B, 1, -1)
            else:
                m0 = memory_state.unsqueeze(1)  # (B,1,D)
            tokens.append(m0)
            names.append("[M]")

        x = torch.cat(tokens, dim=1)  # (B, N_tokens, D)

        # --- Apply K transformer passes
        K = control.effective_steps()
        for _ in range(K):
            for blk in self.blocks:
                x = blk(x)

        x = self.final_ln(x)

        # parse outputs
        idx_s = names.index("[S]")
        z_global = x[:, idx_s, :]  # (B,D)

        z_by_mod_out: Dict[str, torch.Tensor] = {}
        for name in z_by_mod.keys():
            idx = names.index(name)
            z_by_mod_out[name] = x[:, idx, :]

        # updated memory
        if self.use_memory_token:
            idx_m = names.index("[M]")
            memory_state_out = x[:, idx_m, :]
        else:
            memory_state_out = torch.zeros(B, self.d_shared, device=device)

        return x, z_global, z_by_mod_out, memory_state_out


# ------------------------------------------------------------
# MultimodalBrain: orchestrates modalities + thinking core
# ------------------------------------------------------------

class MultimodalBrain(nn.Module):
    """
    High-level model:

      inputs (per modality) --(encoders+UpAdapters)--> shared embeddings z_by_mod
                                               |
                                            ThinkingCore
                                               |
                                shared global latent (and per-modality latents)
                                               |
                           (DownAdapters+decoders) -> outputs per requested modality

    You can:
      - Swap encoders/decoders by swapping ModalityInterface + adapters.
      - Train only adapters + thinking core.
      - Keep this as the stable "brain" while I/O improves over time.
    """

    def __init__(
        self,
        *,
        d_shared: int = 512,
        modalities: Dict[str, ModalityInterface],
        thinking_core: Optional[ThinkingCore] = None,
        use_memory: bool = False,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.modalities = nn.ModuleDict(modalities)
        self.core = thinking_core or ThinkingCore(d_shared=d_shared, n_layers=3, n_heads=8, dropout=0.0, use_memory_token=use_memory)
        self.use_memory = use_memory
        self._memory_state: Optional[torch.Tensor] = None  # (B,D) last [M] token

    def reset_memory(self):
        self._memory_state = None

    def encode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        inputs: {modality_name: whatever its encoder expects}
        Returns: {modality_name: (B, d_shared)} in canonical latent space.
        """
        z_by_mod: Dict[str, torch.Tensor] = {}
        for name, iface in self.modalities.items():
            if name in inputs:
                z_by_mod[name] = iface.encode_to_shared(inputs[name])
        if not z_by_mod:
            raise ValueError("No known modalities present in inputs.")
        # ensure same batch size
        B_sizes = {z.size(0) for z in z_by_mod.values()}
        if len(B_sizes) != 1:
            raise ValueError(f"Inconsistent batch sizes across modalities: {B_sizes}")
        return z_by_mod

    def think(self, z_by_mod: Dict[str, torch.Tensor], control: ThinkControl) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run the thinking core, possibly updating memory.
        """
        tokens, z_global, z_by_mod_out, mem_out = self.core(z_by_mod, control, self._memory_state)
        if self.use_memory:
            self._memory_state = mem_out.detach()
        return tokens, z_global, z_by_mod_out

    def decode_outputs(
        self,
        z_global: torch.Tensor,
        z_by_mod_out: Dict[str, torch.Tensor],
        request_outputs: List[str],
        use_global: bool = True,
    ) -> Dict[str, Any]:
        """
        Decode requested modalities from shared latent.

        If use_global=True, use z_global as source; else use per-modality tokens.
        """
        outputs: Dict[str, Any] = {}
        for name in request_outputs:
            if name not in self.modalities:
                continue
            iface = self.modalities[name]
            z_src = z_global if use_global or name not in z_by_mod_out else z_by_mod_out[name]
            outputs[name] = iface.decode_from_shared(z_src)
        return outputs

    def forward(
        self,
        inputs: Dict[str, Any],
        request_outputs: Optional[List[str]] = None,
        control: Optional[ThinkControl] = None,
    ) -> Dict[str, Any]:
        """
        High-level forward:

          1) Encode inputs of available modalities into shared latent space.
          2) Think (run core K steps).
          3) Decode requested modalities from the shared space.

        returns:
          dict with keys:
            - each requested modality
            - "_z_global": final global latent
            - "_tokens": full token matrix (for analysis)
        """
        control = control or ThinkControl()
        request_outputs = request_outputs or []

        z_by_mod = self.encode_inputs(inputs)
        tokens, z_global, z_by_mod_out = self.think(z_by_mod, control)

        outs = {}
        if request_outputs:
            outs.update(self.decode_outputs(z_global, z_by_mod_out, request_outputs))
        outs["_z_global"] = z_global
        outs["_tokens"] = tokens
        outs["_z_by_mod"] = z_by_mod_out
        return outs


# ------------------------------------------------------------
# Example usage with dummy enc/dec (so you can test the wiring)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Dummy encoders that just project inputs to some native space
    class DummyTextEnc(nn.Module):
        def __init__(self, d_in: int = 64, d_hid: int = 384):
            super().__init__()
            self.proj = nn.Linear(d_in, d_hid)
        def forward(self, x: torch.Tensor):
            # x: (B,T,d_in) -> (B,T,d_hid)
            return self.proj(x)

    class DummyImageEnc(nn.Module):
        def __init__(self, d_in: int = 128, d_hid: int = 768):
            super().__init__()
            self.proj = nn.Linear(d_in, d_hid)
        def forward(self, x: torch.Tensor):
            # x: (B,H,W,d_in) flattened -> (B,H*W,d_hid) in practice; here just (B,T,d_in)
            return self.proj(x)

    class DummyAudioEnc(nn.Module):
        def __init__(self, d_in: int = 80, d_hid: int = 384):
            super().__init__()
            self.proj = nn.Linear(d_in, d_hid)
        def forward(self, x: torch.Tensor):
            # x: (B,T,d_in)
            return self.proj(x)

    # Dummy decoders that map from down-adapted vectors to some space
    class DummyTextDec(nn.Module):
        def __init__(self, d_in: int = 256, vocab_size: int = 1000):
            super().__init__()
            self.ff = nn.Linear(d_in, vocab_size)
        def forward(self, x: torch.Tensor):
            # x: (B,d_in) -> logits over vocab
            return self.ff(x)

    class DummyImageDec(nn.Module):
        def __init__(self, d_in: int = 256, img_dim: int = 32*32*3):
            super().__init__()
            self.ff = nn.Linear(d_in, img_dim)
        def forward(self, x: torch.Tensor):
            B = x.size(0)
            img = self.ff(x).view(B, 3, 32, 32)
            return img

    class DummyAudioDec(nn.Module):
        def __init__(self, d_in: int = 256, wav_len: int = 16000):
            super().__init__()
            self.ff = nn.Linear(d_in, wav_len)
        def forward(self, x: torch.Tensor):
            return self.ff(x)  # (B, wav_len)

    d_shared = 512

    # build modality interfaces
    text_iface = ModalityInterface(
        name="text",
        encoder=DummyTextEnc(),
        up_adapter=UpAdapter(d_in=384, d_shared=d_shared),
        decoder=DummyTextDec(d_in=256),
        down_adapter=DownAdapter(d_shared=d_shared, d_out=256),
        freeze_encoder=True,
        freeze_decoder=True,
    )
    image_iface = ModalityInterface(
        name="image",
        encoder=DummyImageEnc(),
        up_adapter=UpAdapter(d_in=768, d_shared=d_shared),
        decoder=DummyImageDec(d_in=256),
        down_adapter=DownAdapter(d_shared=d_shared, d_out=256),
        freeze_encoder=True,
        freeze_decoder=True,
    )
    audio_iface = ModalityInterface(
        name="audio",
        encoder=DummyAudioEnc(),
        up_adapter=UpAdapter(d_in=384, d_shared=d_shared),
        decoder=DummyAudioDec(d_in=256),
        down_adapter=DownAdapter(d_shared=d_shared, d_out=256),
        freeze_encoder=True,
        freeze_decoder=True,
    )

    brain = MultimodalBrain(
        d_shared=d_shared,
        modalities={"text": text_iface, "image": image_iface, "audio": audio_iface},
        thinking_core=ThinkingCore(d_shared=d_shared, n_layers=2, n_heads=8, dropout=0.0, use_memory_token=True),
        use_memory=True,
    )

    B, T = 2, 16
    # Fake inputs (shapes just for sanity)
    inputs = {
        "text": torch.randn(B, T, 64),
        "image": torch.randn(B, T, 128),
        "audio": torch.randn(B, T, 80),
    }

    ctrl = ThinkControl(steps=2, mode="default", modality_weights={"text": 1.0, "image": 1.0, "audio": 1.0})
    outs = brain(inputs, request_outputs=["text", "image", "audio"], control=ctrl)

    print("z_global:", outs["_z_global"].shape)
    print("tokens:", outs["_tokens"].shape)
    print("text logits:", outs["text"].shape)
    print("image:", outs["image"].shape)
    print("audio:", outs["audio"].shape)
