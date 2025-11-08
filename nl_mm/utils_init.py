"""Initialization helpers for the NL-MM architecture."""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from torch import nn

__all__ = [
    "RMSNorm",
    "QKNorm",
    "apply_deepnorm_scaling",
    "convert_layernorm_to_rmsnorm",
    "deepnorm_constants",
    "downscale_ffn_and_v_out",
    "init_conv_kaiming",
    "init_linear_glorot",
    "init_trunc_normal_002",
    "init_ttt_adapters_identity",
    "insert_qk_norm_in_attention_modules",
    "is_conv",
    "is_linear",
    "iter_nl_blocks",
    "replace_module",
    "scale_module_weights_",
    "set_global_seed",
    "tie_or_init_output_heads",
    "zero_fast_weight_memory",
]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms
        return normed * self.weight


class QKNorm(nn.Module):
    """Per-head RMS normalization for query/key projections."""

    def __init__(self, d_head: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        shape = (1, 1, 1, d_head)
        self.s_q = nn.Parameter(torch.ones(shape))
        self.s_k = nn.Parameter(torch.ones(shape))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        rms_q = torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        rms_k = torch.sqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        q = q / rms_q * self.s_q
        k = k / rms_k * self.s_k
        return q, k


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def is_conv(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    components = dotted_name.split(".")
    if not components:
        raise ValueError("Module name cannot be empty")
    for name in components[:-1]:
        parent = getattr(parent, name)
    setattr(parent, components[-1], new_module)


def convert_layernorm_to_rmsnorm(root: nn.Module, eps: float = 1e-6) -> None:
    for name, module in list(root.named_modules())[::-1]:
        if not name:
            continue
        if isinstance(module, nn.LayerNorm):
            normalized_shape = module.normalized_shape
            if isinstance(normalized_shape, torch.Size):
                dim = normalized_shape[-1]
            elif isinstance(normalized_shape, (tuple, list)):
                dim = int(normalized_shape[-1])
            else:
                dim = int(normalized_shape)
            rms = RMSNorm(dim, eps=eps)
            rms.weight.data.copy_(module.weight.data)
            replace_module(root, name, rms)


def insert_qk_norm_in_attention_modules(root: nn.Module, eps: float = 1e-6) -> None:
    for module in root.modules():
        if hasattr(module, "wq") and hasattr(module, "wk") and hasattr(module, "n_heads"):
            d_model = module.wq.out_features
            d_head = d_model // module.n_heads
            qk_norm = QKNorm(d_head, eps=eps)
            module.qk_norm = qk_norm  # type: ignore[attr-defined]


def deepnorm_constants(depth: int, arch_kind: str) -> tuple[float, float]:
    if depth <= 0:
        raise ValueError("depth must be positive for DeepNorm")
    if arch_kind == "encoder":
        alpha = (2 * depth) ** 0.25
        beta = (8 * depth) ** -0.25
    elif arch_kind == "decoder":
        alpha = (2 * depth) ** 0.25
        beta = (8 * depth) ** -0.25
    elif arch_kind == "encdec":
        alpha = (2 * depth) ** 0.25
        beta = (8 * depth) ** -0.25
    else:
        raise ValueError("arch_kind must be one of {'encoder', 'decoder', 'encdec'}")
    return alpha, beta


def iter_nl_blocks(stack: nn.Module) -> Iterator[nn.Module]:
    for module in stack.modules():
        if hasattr(module, "fast_attn") and hasattr(module, "cms"):
            yield module


def apply_deepnorm_scaling(stack: nn.Module, depth: int, arch_kind: str) -> None:
    alpha, _ = deepnorm_constants(depth, arch_kind)
    for block in iter_nl_blocks(stack):
        setattr(block, "residual_alpha", alpha)


def scale_module_weights_(module: nn.Module | None, gain: float) -> None:
    if module is None or not hasattr(module, "weight"):
        return
    with torch.no_grad():
        module.weight.mul_(gain)


def downscale_ffn_and_v_out(stack: nn.Module, depth: int, arch_kind: str) -> None:
    _, beta = deepnorm_constants(depth, arch_kind)
    for block in iter_nl_blocks(stack):
        cms = getattr(block, "cms", None)
        if cms is not None:
            for level in getattr(cms, "blocks", {}).values():
                for submodule in level.modules():
                    if isinstance(submodule, nn.Linear):
                        scale_module_weights_(submodule, beta)
        attn = getattr(block, "fast_attn", None)
        if attn is not None:
            scale_module_weights_(getattr(attn, "wv", None), beta)
            scale_module_weights_(getattr(attn, "wo", None), beta)


def init_linear_glorot(linear: nn.Linear) -> None:
    nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_conv_kaiming(conv: nn.Module) -> None:
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if getattr(conv, "bias", None) is not None:
        nn.init.zeros_(conv.bias)


def init_trunc_normal_002(tensor: torch.Tensor) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=0.02)


def init_ttt_adapters_identity(root: nn.Module, sigma: float = 1e-3) -> None:
    from .modules.ttt import TTTAdapter  # local import to avoid cycles

    for module in root.modules():
        if isinstance(module, TTTAdapter):
            nn.init.normal_(module.inp.weight, mean=0.0, std=sigma)
            nn.init.zeros_(module.out.weight)


def zero_fast_weight_memory(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "fast_state_init"):
            module.fast_state_init(lambda memory: memory.zero_())


def tie_or_init_output_heads(model: nn.Module) -> None:
    token_emb = getattr(getattr(model, "txt_enc", None), "token", None)
    if token_emb is not None and hasattr(model, "txt_dec") and hasattr(model.txt_dec, "lm_head"):
        model.txt_dec.lm_head.weight = token_emb.weight


def insert_mu_parametrization_scaling(module: nn.Module, width: int) -> None:
    """Placeholder for μP integration (hook for future extension)."""
    _ = (module, width)

