# multimodal_thinking_fixed.py
# PyTorch 2.x — modular multimodal architecture with controllable "thinking"
# Fixed version with:
# - Decoder adapters (text/image/audio) to bridge latent slots Z -> real decoders
# - Conditioning parity across walkers (FiLM for all)
# - Optional global Z-memory + per-session modal memories (safe blending)
# - Masks plumbed through cross-attention (variable-length tokens)
# - Removed @torch.no_grad()/forced eval from encoders to allow fine-tuning
# - Robust shape/device/dtype handling
# - Slightly deeper Perceiver-style aggregator (x-attn + self-attn blocks)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------
# Utilities
# -----------------------------

def freeze_module(m: nn.Module, trainable: bool = False):
    for p in m.parameters():
        p.requires_grad = trainable
    if trainable:
        m.train()
    else:
        m.eval()
    return m


def as_tokens(x: torch.Tensor) -> torch.Tensor:
    """Ensure (B, T, D). Accepts (B, D) and expands to T=1."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3, f"Expected (B, T, D), got {tuple(x.shape)}"
    return x


def ensure_device_dtype(ref: torch.Tensor, *tensors: Optional[torch.Tensor]):
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        else:
            out.append(t.to(device=ref.device, dtype=ref.dtype))
    return out if len(out) > 1 else out[0]


class PreNorm(nn.Module):
    def __init__(self, d: int, fn: nn.Module):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fn = fn

    def forward(self, x, *args, **kw):
        return self.fn(self.ln(x), *args, **kw)


class ReZero(nn.Module):
    """Zero-init residual gate for stable deep residual learning.
    Starts as identity skip-only; learns to scale residual path.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.alpha * residual


# -----------------------------
# Conditioning (FiLM)
# -----------------------------
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: x * (1+gamma) + beta (broadcast over tokens)."""

    def __init__(self, d: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, 2 * d))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]):
        if cond is None:
            return x
        gb = self.mlp(cond)  # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


# -----------------------------
# Attention helpers
# -----------------------------
class CrossAttention(nn.Module):
    """Multi-head cross-attn in (B, T, D) convention with key_padding_mask support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, bias=bias, batch_first=True
        )

    def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # key_padding_mask: (B, T_kv) with True = pad (ignored)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class SelfAttentionBlock(nn.Module):
    """Transformer-style self-attention + FF block (pre-norm)."""

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.sa = PreNorm(d, nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True))
        self.ff = PreNorm(
            d, nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        )
        self.g1 = ReZero()
        self.g2 = ReZero()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # PyTorch MHA uses key_padding_mask at call-time; for self-attn on x we pass it as attn_mask via wrapper
        sa_out, _ = self.sa.fn(x, x, x, key_padding_mask=attn_mask, need_weights=False)
        x = x + self.g1(sa_out)
        x = x + self.g2(self.ff(x))
        return x


# -----------------------------
# Residual connectors: modal <-> thinking
# -----------------------------
class ModalToThink(nn.Module):
    """Residual adapter mapping a modal latent (d_m) -> thinking latent (D)."""

    def __init__(self, d_modal: int, d_think: int, hidden_mult: int = 4, p: float = 0.0):
        super().__init__()
        self.in_proj = nn.Identity() if d_modal == d_think else nn.Linear(d_modal, d_think)
        h = max(d_think, hidden_mult * d_think)
        self.ff = nn.Sequential(nn.Linear(d_think, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, d_think))
        self.gate = ReZero()

    def forward(self, m_tokens: torch.Tensor) -> torch.Tensor:
        x = as_tokens(m_tokens)
        x = self.in_proj(x)
        return x + self.gate(self.ff(x))


class ThinkToModal(nn.Module):
    """Residual adapter mapping the thinking latent (D) -> modal latent (d_m)."""

    def __init__(self, d_think: int, d_modal: int, hidden_mult: int = 4, p: float = 0.0):
        super().__init__()
        self.out_proj = nn.Identity() if d_think == d_modal else nn.Linear(d_think, d_modal)
        h = max(d_modal, hidden_mult * d_modal)
        self.ff = nn.Sequential(nn.Linear(d_modal, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, d_modal))
        self.gate = ReZero()

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        x = self.out_proj(Z)
        return x + self.gate(self.ff(x))


# -----------------------------
# Ports wrapping *pretrained* enc/dec into modal latent spaces
# -----------------------------
TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]


class ModalEncoderPort(nn.Module):
    """
    Wraps a (frozen or trainable) pretrained encoder to produce *modal latent* tokens.
    Accepts input either as tensor X or tuple (X, pad_mask) where pad_mask is (B, T) with True = pad.
    If the encoder returns a tuple (feats, mask), we propagate that mask.
    """

    def __init__(self, encoder: nn.Module, d_feats: int, d_modal: int, freeze: bool = True):
        super().__init__()
        self.encoder = freeze_module(encoder, trainable=not freeze)
        self.proj = nn.Identity() if d_feats == d_modal else nn.Linear(d_feats, d_modal)

    def forward(self, x: TensorOrTuple) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(x, tuple):
            X, pad = x
        else:
            X, pad = x, None
        out = self.encoder(X)
        if isinstance(out, tuple):
            feats, mask = out
        else:
            feats, mask = out, pad
        feats = as_tokens(self.proj(feats))  # (B, Tm, d_modal)
        return feats, mask


class ModalDecoderPort(nn.Module):
    """
    Generic decoder port for simple decoders that consume modal latent tokens directly.
    For sophisticated decoders, use DecoderAdapters below.
    """

    def __init__(self, decoder: nn.Module, d_modal: int, freeze: bool = True):
        super().__init__()
        self.decoder = freeze_module(decoder, trainable=not freeze)
        self.ln = nn.LayerNorm(d_modal)

    def forward(self, modal_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.ln(modal_latent))


# -----------------------------
# Perceiver-style Aggregator: build initial Z0 from modal->thinking tokens
# -----------------------------
class PerceiverAggregator(nn.Module):
    def __init__(self, d_think: int, n_slots: int = 128, heads: int = 8, p: float = 0.0, depth: int = 1):
        super().__init__()
        self.n_slots = n_slots
        self.d_think = d_think
        self.slots = nn.Parameter(torch.randn(n_slots, d_think) / (d_think ** 0.5))
        self.xattn = PreNorm(d_think, CrossAttention(d_think, heads, p))
        self.ff = PreNorm(d_think,
                          nn.Sequential(nn.Linear(d_think, 4 * d_think), nn.GELU(), nn.Linear(4 * d_think, d_think)))
        self.g1, self.g2 = ReZero(), ReZero()
        self.depth = depth
        self.self_blocks = nn.ModuleList([SelfAttentionBlock(d_think, heads, p) for _ in range(depth)])

    def forward(self, modal_tokens: List[torch.Tensor],
                modal_masks: List[Optional[torch.Tensor]] = None) -> torch.Tensor:
        kv = torch.cat([as_tokens(z) for z in modal_tokens], dim=1)  # (B, sum_T, D)
        if modal_masks and any(m is not None for m in modal_masks):
            masks = [m if m is not None else torch.zeros(z.size(0), z.size(1), dtype=torch.bool, device=z.device) for
                     z, m in zip(modal_tokens, modal_masks)]
            kv_mask = torch.cat(masks, dim=1)
        else:
            kv_mask = None
        B = kv.size(0)
        q = self.slots.unsqueeze(0).expand(B, self.n_slots, self.d_think)
        x = q + self.g1(self.xattn(q, kv, key_padding_mask=kv_mask))
        x = x + self.g2(self.ff(x))
        for blk in self.self_blocks:
            x = blk(x)  # self-attn over slots
        return x  # Z0: (B, N_slots, D)


# -----------------------------
# Decoder Adapters (solve the decoder input problem)
# -----------------------------
class TokenDistiller(nn.Module):
    """Learned queries cross-attend to Z and distill to a target number of tokens and dimension."""

    def __init__(self, d_in: int, d_out: int, out_tokens: int, heads: int = 8, p: float = 0.0):
        super().__init__()
        # Ensure heads divides d_out for MultiheadAttention
        while d_out % heads != 0 and heads > 1:
            heads = heads // 2
        heads = max(1, heads)

        self.queries = nn.Parameter(torch.randn(out_tokens, d_out) / (d_out ** 0.5))
        self.kv_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.xattn = PreNorm(d_out, CrossAttention(d_out, heads, p))
        self.ff = PreNorm(d_out, nn.Sequential(nn.Linear(d_out, 4 * d_out), nn.GELU(), nn.Linear(4 * d_out, d_out)))
        self.g1, self.g2 = ReZero(), ReZero()

    def forward(self, Z: torch.Tensor, Z_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = Z.size(0)
        kv = self.kv_proj(Z)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        x = q + self.g1(self.xattn(q, kv, key_padding_mask=Z_mask))
        x = x + self.g2(self.ff(x))
        return x  # (B, out_tokens, d_out)


class TextDecoderAdapter(nn.Module):
    """
    Distill Z -> text-side conditioning.
    Modes:
      - "prefix": return soft prefix embeddings for decoder-only LMs (B, P, d_txt)
      - "encdec": return (encoder_hidden_states, attention_mask) for seq2seq decoders (e.g., T5)
    Real decoders should accept these via kwargs.
    """

    def __init__(self, d_think: int, d_text: int, prefix_len: int = 16, mode: Literal["prefix", "encdec"] = "prefix"):
        super().__init__()
        self.mode = mode
        self.distill = TokenDistiller(d_think, d_text, out_tokens=prefix_len)

    def forward(self, Z: torch.Tensor, Z_mask: Optional[torch.Tensor] = None):
        tokens = self.distill(Z, Z_mask)
        if self.mode == "prefix":
            return {"prefix_embeds": tokens}
        else:  # encdec
            attn_mask = torch.zeros(tokens.size(0), tokens.size(1), dtype=torch.bool, device=tokens.device)
            return {"encoder_hidden_states": tokens, "encoder_attention_mask": attn_mask}


class ImageDecoderAdapter(nn.Module):
    """
    Distill Z -> spatial latent grid for VAE/LDM decoders.
    Configure (H, W, C_lat) for the target decoder latent space.
    """

    def __init__(self, d_think: int, C_lat: int = 4, H: int = 32, W: int = 32, heads: int = 8):
        super().__init__()
        self.H, self.W, self.C = H, W, C_lat
        self.distill = TokenDistiller(d_think, d_out=C_lat, out_tokens=H * W, heads=heads)

    def forward(self, Z: torch.Tensor, Z_mask: Optional[torch.Tensor] = None):
        tokens = self.distill(Z, Z_mask)  # (B, H*W, C)
        B = tokens.size(0)
        grid = tokens.transpose(1, 2).reshape(B, self.C, self.H, self.W)
        return {"latent_grid": grid}


class AudioDecoderAdapter(nn.Module):
    """Distill Z -> mel-spectrogram frames (T_mel, F)."""

    def __init__(self, d_think: int, F: int = 80, T: int = 200, heads: int = 8):
        super().__init__()
        self.T, self.F = T, F
        self.distill = TokenDistiller(d_think, d_out=F, out_tokens=T, heads=heads)

    def forward(self, Z: torch.Tensor, Z_mask: Optional[torch.Tensor] = None):
        mels = self.distill(Z, Z_mask)  # (B, T, F)
        return {"mel": mels}


# -----------------------------
# Thinking module (walkers) — all support conditioning
# -----------------------------
class Walker(nn.Module):
    def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError


class TransformerRefiner(Walker):
    def __init__(self, d: int, heads: int = 8, depth: int = 4, p: float = 0.0, cond_dims: Dict[str, int] = None):
        super().__init__()
        self.xattn = PreNorm(d, CrossAttention(d, heads, p))
        enc_layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.g = ReZero()
        self.film = nn.ModuleDict({m: FiLM(d, cd) for m, cd in (cond_dims or {}).items()})

    def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        kvs, masks, conds = ctx["modal_kv"], ctx["modal_masks"], ctx["conds"]
        kv_list, mask_list = [], []
        for m, kv in kvs.items():
            cond = conds.get(m, None)
            if m in self.film and cond is not None:
                kv = self.film[m](kv, cond)
            kv_list.append(kv)
            mask_list.append(masks.get(m, None))
        kv = torch.cat(kv_list, dim=1)
        kv_mask = None
        if any(msk is not None for msk in mask_list):
            # Build a concatenated key_padding_mask
            filled_masks = [
                msk if msk is not None else torch.zeros(kv_list[i].size(0), kv_list[i].size(1), dtype=torch.bool,
                                                        device=kv_list[i].device) for i, msk in enumerate(mask_list)]
            kv_mask = torch.cat(filled_masks, dim=1)
        Z = Z + self.g(self.xattn(Z, kv, key_padding_mask=kv_mask))
        Z = self.enc(Z)
        return Z, {"kv_mask": kv_mask}


class DiffusionWalker(Walker):
    """
    Latent denoising update in thinking space with real conditioning:
    - sinusoidal step embeddings
    - FiLM-like modulation from pooled cond vectors
    """

    def __init__(self, d: int, steps_max: int = 12, cond_dim_total: int = 0):
        super().__init__()
        self.d = d
        self.eps = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        sigmas = torch.linspace(1.0, 0.05, steps_max)
        self.register_buffer("sigmas", sigmas)
        self.t_proj = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.cond_film = nn.Sequential(nn.Linear(cond_dim_total, 2 * d)) if cond_dim_total > 0 else None

    def _timestep_embed(self, step_idx: int, B: int, device, dtype):
        t = torch.full((B, 1), float(step_idx), device=device, dtype=dtype)
        i = torch.arange(self.d, device=device, dtype=dtype).unsqueeze(0)
        div = torch.exp(-i * (torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(1, self.d - 1)))
        ang = t * div
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, 2d)
        return self.t_proj(emb).unsqueeze(1)  # (B, 1, d)

    def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        B, N, D = Z.shape
        step_idx = int(ctx.get("step", 0))
        sigma = self.sigmas[min(step_idx, self.sigmas.numel() - 1)]
        t_emb = self._timestep_embed(step_idx, B, Z.device, Z.dtype)
        X = Z + t_emb
        if self.cond_film is not None and "conds" in ctx and any(ctx["conds"].values()):
            pooled = torch.cat([c for c in ctx["conds"].values() if c is not None], dim=-1)
            gamma, beta = self.cond_film(pooled).chunk(2, dim=-1)
            X = X * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        e = self.eps(X)
        return Z - sigma * e, {"sigma": sigma}


class RetentionLikeWalker(Walker):


    class RetentionLikeWalker(Walker):
        """A lightweight retention-style update along slot dimension (not a full RetNet).
        Applies an exponential moving aggregation over slots; linear-time in tokens.
        """

        def __init__(self, d: int, decay: float = 0.9):
            super().__init__()
            self.decay = nn.Parameter(torch.tensor(decay))
            self.out = nn.Linear(d, d)
            self.ln = nn.LayerNorm(d)

        def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
            B, N, D = Z.shape
            s = torch.zeros(B, D, device=Z.device, dtype=Z.dtype)
            outs = []
            for t in range(N):
                s = torch.sigmoid(self.decay) * s + Z[:, t, :]
                outs.append(self.out(s).unsqueeze(1))
            Y = torch.cat(outs, dim=1)
            return self.ln(Y), {}


class WorldModelWalker(Walker):
    """GRU-like latent dynamics with cross-attn messages + FiLM conditioning."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.msg = PreNorm(d, CrossAttention(d, heads))
        self.gru = nn.GRUCell(d, d)
        self.ln = nn.LayerNorm(d)
        self.g = ReZero()
        self.film = None  # could plug FiLM on kv if needed

    def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        kv = torch.cat(list(ctx["modal_kv"].values()), dim=1)
        kv_mask = None
        masks = list(ctx["modal_masks"].values())
        if any(m is not None for m in masks):
            B = Z.size(0)
            sizes = [t.size(1) for t in ctx["modal_kv"].values()]
            filled = []
            i = 0
            for m, sz in zip(masks, sizes):
                if m is None:
                    filled.append(torch.zeros(B, sz, dtype=torch.bool, device=Z.device))
                else:
                    filled.append(m)
                i += 1
            kv_mask = torch.cat(filled, dim=1)
        msg = self.g(self.msg(Z, kv, key_padding_mask=kv_mask))
        B, N, D = Z.shape
        Zf, mf = Z.reshape(B * N, D), msg.reshape(B * N, D)
        Zn = self.gru(self.ln(Zf + mf), Zf).reshape(B, N, D)
        return Zn, {}


class TemporalRetentionWalker(Walker):
    """
    Retention-style update along a provided temporal stream (e.g., one modality's tokens).
    Use when a clear time axis exists; complements the slot-based retention block.
    """

    def __init__(self, d: int, decay: float = 0.9, source_modality: str = "text"):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(decay))
        self.source = source_modality
        self.proj_in = nn.Linear(d, d)
        self.proj_out = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def step(self, Z: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        kv = ctx["modal_kv"].get(self.source, None)
        if kv is None:
            return Z, {}
        B, T, D = kv.shape
        s = torch.zeros(B, D, device=Z.device, dtype=Z.dtype)
        out = []
        K = torch.sigmoid(self.decay)
        for t in range(T):
            s = K * s + self.proj_in(kv[:, t, :])
            out.append(s.unsqueeze(1))
        summary = self.proj_out(torch.cat(out, dim=1)).mean(dim=1, keepdim=True)
        return self.ln(Z + summary), {}


WALKERS = {
    "transformer": TransformerRefiner,
    "diffusion": DiffusionWalker,
    "retention": RetentionLikeWalker,
    "temporal": TemporalRetentionWalker,
    "world": WorldModelWalker,
}


# -----------------------------
# Control & Memory
# -----------------------------
@dataclass
class ThinkControl:
    steps: int = 3
    mode: Literal["default", "fast", "deep", "off"] = "default"
    keep_in_mind: Dict[str, bool] = field(default_factory=dict)  # per-modality modal-memory
    keep_in_mind_global: bool = False  # Z-memory EMA
    memory_injection_strength: float = 0.5  # blend factor for injecting EMA(Z) into current Z


# -----------------------------
# The Multimodal Thinking Model
# -----------------------------
class MultiModalThinking(nn.Module):
    """
    - text/image/audio each have: Encoder -> modal latent (d_m); Decoder consumes modal latent or
      decoder-adapter outputs to produce real outputs.
    - residual adapters map modal <-> thinking latent (D)
    - aggregator builds Z0 from modal->thinking tokens
    - Walker runs K 'thinking' steps (controllable), with per-modality conditioning
    - decode only requested modalities
    - optional per-session modal memories and global Z-memory
    """

    def __init__(
            self,
            d_think: int,
            # per-modality enc/dec specs
            encoders: Dict[str, Tuple[nn.Module, int, int]],  # name -> (encoder, d_feats, d_modal)
            decoders: Dict[str, Tuple[nn.Module, int]],  # name -> (decoder, d_modal_for_simple_port)
            freeze_io: bool = True,
            n_slots: int = 128,
            aggregator_depth: int = 1,
            walker_kind: str = "transformer",
            walker_kwargs: Dict = None,
            cond_dims: Dict[str, int] = None,  # conditioning vector dim per mod (optional)
            # optional decoder adapters (solve decoder-input mismatch)
            decoder_adapters: Dict[str, nn.Module] = None,  # e.g., {"text": TextDecoderAdapter(...), ...}
    ):
        super().__init__()
        self.d_think = d_think
        self.modal_names: List[str] = sorted(list(set(list(encoders.keys()) + list(decoders.keys()))))

        # Build modal enc/dec ports + adapters
        self.enc_ports = nn.ModuleDict()
        self.modal2think = nn.ModuleDict()
        self.think2modal = nn.ModuleDict()
        self.dec_ports = nn.ModuleDict()
        self.d_modal: Dict[str, int] = {}

        for m in self.modal_names:
            if m in encoders:
                enc, d_feats, d_modal = encoders[m]
                self.enc_ports[m] = ModalEncoderPort(enc, d_feats, d_modal, freeze=freeze_io)
                self.modal2think[m] = ModalToThink(d_modal, d_think)
                self.d_modal[m] = d_modal
            if m in decoders:
                dec, d_modal_out = decoders[m]
                self.think2modal[m] = ThinkToModal(d_think, d_modal_out)
                self.dec_ports[m] = ModalDecoderPort(dec, d_modal_out, freeze=freeze_io)

        self.aggregator = PerceiverAggregator(d_think, n_slots=n_slots, depth=aggregator_depth)

        # Walker
        walker_kwargs = walker_kwargs or {}
        if walker_kind == "transformer":
            walker_kwargs = {**walker_kwargs, "cond_dims": cond_dims or {}}
        self.walker: Walker = WALKERS[walker_kind](d_think, **walker_kwargs)

        # Short-term per-session per-modality memories (modal latent space)
        self._mem_bank: Dict[str, Dict[str, torch.Tensor]] = {}
        self.mem_gate = nn.ParameterDict({m: nn.Parameter(torch.tensor(0.5)) for m in self.modal_names})
        # Global Z memory (EMA)
        self._z_memory_bank: Dict[str, torch.Tensor] = {}
        self.z_ema = 0.9

        # Optional decoder adapters
        self.decoder_adapters = nn.ModuleDict(decoder_adapters or {})

        # Optional: weight initialization policy (comment out if you prefer defaults)
        self.apply(init_weights_)

    # ---- Memory helpers ----
    @torch.no_grad()
    def reset_session(self, session_id: str):
        self._mem_bank.pop(session_id, None)
        self._z_memory_bank.pop(session_id, None)

    def _get_session_mem(self, session_id: Optional[str]) -> Dict[str, torch.Tensor]:
        if session_id is None:
            return {}
        return self._mem_bank.setdefault(session_id, {})

    # ---- Forward ----
    def forward(
            self,
            inputs: Dict[str, TensorOrTuple],
            request_outputs: List[str],
            control: ThinkControl = ThinkControl(),
            cond_vectors: Dict[str, Optional[torch.Tensor]] = None,  # B×C per mod (optional)
            session_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        cond_vectors = cond_vectors or {}
        # Get batch size from first available input (handle tensor, tuple, or dict)
        first_val = next(iter(inputs.values()))
        if isinstance(first_val, tuple):
            B_ref = first_val[0].size(0)
        elif isinstance(first_val, dict):
            # If it's a dict (e.g., from preprocessing), get the first tensor value
            B_ref = next(v for v in first_val.values() if torch.is_tensor(v)).size(0)
        else:
            B_ref = first_val.size(0)

        # 1) Encode -> modal latents (with masks)
        modal_latents: Dict[str, torch.Tensor] = {}
        modal_masks: Dict[str, Optional[torch.Tensor]] = {}
        sess_mem = self._get_session_mem(session_id)
        for m, port in self.enc_ports.items():
            if m not in inputs:
                continue
            z_m, m_mask = port(inputs[m])  # (B, Tm, d_m), (B, Tm) with True=pad
            # apply per-modality memory blending if requested
            keep = bool(control.keep_in_mind.get(m, False)) if control.keep_in_mind else False
            if keep:
                prev = sess_mem.get(m, None)
                if prev is None or prev.shape != z_m.shape:
                    sess_mem[m] = z_m.detach()
                else:
                    g = torch.sigmoid(self.mem_gate[m])
                    blended = g * prev + (1 - g) * z_m
                    sess_mem[m] = blended.detach()
                    z_m = blended
            modal_latents[m] = z_m
            modal_masks[m] = m_mask

        assert len(modal_latents) > 0, "No inputs provided for available encoders."

        # 2) Map modal -> thinking tokens + KV dict
        think_tokens: List[torch.Tensor] = []
        think_masks: List[Optional[torch.Tensor]] = []
        modal_kv: Dict[str, torch.Tensor] = {}
        for m, z_m in modal_latents.items():
            to_think = self.modal2think[m](z_m)
            think_tokens.append(to_think)
            think_masks.append(modal_masks[m])
            modal_kv[m] = to_think

        # 3) Aggregate to initial Z0
        Z = self.aggregator(think_tokens, think_masks)  # (B, N_slots, D)

        # 4) Optional global Z-memory EMA
        if control.keep_in_mind_global and session_id is not None:
            prevZ = self._z_memory_bank.get(session_id, None)
            if prevZ is None or prevZ.shape != Z.shape:
                self._z_memory_bank[session_id] = Z.detach()
            else:
                alpha = float(max(0.0, min(1.0, control.memory_injection_strength)))
                Z = (1 - alpha) * Z + alpha * prevZ
                self._z_memory_bank[session_id] = (self.z_ema * prevZ + (1 - self.z_ema) * Z).detach()

        # 5) Thinking (controllable)
        steps = control.steps
        if control.mode == "off":
            steps = 0
        elif control.mode == "fast":
            steps = max(0, min(steps, 1))
        elif control.mode == "deep":
            steps = steps + 2

        ctx = {
            "modal_kv": modal_kv,
            "modal_masks": modal_masks,
            "conds": {m: cond_vectors.get(m, None) for m in modal_kv},
        }

        prev_pred = None
        for step_idx in range(steps):
            ctx["step"] = step_idx
            if prev_pred is not None:
                ctx["prev_pred"] = prev_pred
            Z, _aux = self.walker.step(Z, ctx)
            prev_pred = Z.detach()

        # 6) Decode requested outputs
        outs: Dict[str, torch.Tensor] = {}
        # Build a concatenated mask for Z if needed by adapters
        Z_mask = None  # slots have no padding by construction
        for m in request_outputs:
            if m in self.decoder_adapters:
                # Use adapter + user-provided decoder that accepts **kwargs
                adapter = self.decoder_adapters[m]
                kwargs = adapter(Z, Z_mask)
                dec = self.dec_ports[m].decoder if m in self.dec_ports else None
                if dec is None:
                    raise RuntimeError(f"Decoder adapter provided for '{m}', but no decoder port is registered.")
                outs[m] = dec(**kwargs)  # type: ignore[arg-type]
            else:
                # Fallback: map think->modal, then simple decoder port
                if m not in self.dec_ports:
                    continue
                z_m_out = self.think2modal[m](Z)  # (B, N_slots, d_m_out)
                y = self.dec_ports[m](z_m_out)  # shape depends on decoder
                outs[m] = y

        outs["_Z"] = Z
        return outs


# -----------------------------
# Initialization policy
# -----------------------------

def init_weights_(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


# -----------------------------
# Minimal dummy wiring (for quick sanity check)
# -----------------------------
if __name__ == "__main__":
    # Dummy backbones (replace with real enc/dec like ViT / wav2vec2 / T5 / VAE-LDM / vocoder)
    class DummyEnc(nn.Module):
        def __init__(self, d_in, d_feats):
            super().__init__()
            self.proj = nn.Linear(d_in, d_feats)

        def forward(self, x):  # x: (B, T, d_in)
            return torch.tanh(self.proj(x))  # (B, T, d_feats)


    class DummyDec(nn.Module):
        def __init__(self, d_modal):
            super().__init__()
            self.proj = nn.Linear(d_modal, d_modal)

        def forward(self, tokens):
            # pretend we consume a sequence and produce a pooled vector
            return self.proj(tokens).mean(dim=1)  # (B, d_modal)


    B, Tt, Ti, Ta = 2, 12, 16, 20
    encoders = {
        "text": (DummyEnc(64, 512), 512, 512),
        "image": (DummyEnc(128, 768), 768, 768),
        "audio": (DummyEnc(80, 512), 512, 512),
    }
    decoders = {
        "text": (DummyDec(512), 512),
        "image": (DummyDec(768), 768),
        "audio": (DummyDec(512), 512),
    }

    # Example adapters (these would normally target real decoders)
    decoder_adapters = nn.ModuleDict({
        "text": TextDecoderAdapter(d_think=768, d_text=512, prefix_len=8, mode="prefix"),
        "image": ImageDecoderAdapter(d_think=768, C_lat=4, H=8, W=8),
        "audio": AudioDecoderAdapter(d_think=768, F=80, T=32),
    })

    model = MultiModalThinking(
        d_think=768,
        encoders=encoders,
        decoders=decoders,
        freeze_io=True,
        n_slots=64,
        aggregator_depth=1,
        walker_kind="transformer",
        walker_kwargs={"depth": 2, "heads": 8},
        cond_dims={"text": 32, "image": 32, "audio": 32},
        decoder_adapters=decoder_adapters,
    )

    x = {
        "text": torch.randn(B, Tt, 64),
        "image": torch.randn(B, Ti, 128),
        "audio": torch.randn(B, Ta, 80),
    }
    ctrl = ThinkControl(steps=3, keep_in_mind={"text": True, "audio": True}, keep_in_mind_global=True)
    cond = {"text": torch.randn(B, 32), "image": torch.randn(B, 32), "audio": torch.randn(B, 32)}

    y = model(x, request_outputs=["text", "image", "audio"], control=ctrl, cond_vectors=cond, session_id="demo")
    print({k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v) for k, v in y.items()})
