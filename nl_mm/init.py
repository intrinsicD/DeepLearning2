"""Initialization entry point for NL-MM models."""
from __future__ import annotations

from typing import Dict

from torch import nn

from .utils_init import (
    apply_deepnorm_scaling,
    convert_layernorm_to_rmsnorm,
    downscale_ffn_and_v_out,
    init_conv_kaiming,
    init_linear_glorot,
    init_trunc_normal_002,
    init_ttt_adapters_identity,
    insert_qk_norm_in_attention_modules,
    is_conv,
    is_linear,
    set_global_seed,
    tie_or_init_output_heads,
    zero_fast_weight_memory,
)


def _resolve_stack(model: nn.Module, name: str) -> nn.Module | None:
    direct = getattr(model, f"{name}_enc", None)
    if direct is not None:
        return direct
    aliases = {"text": "txt_enc", "image": "img_enc", "audio": "aud_enc"}
    alias = aliases.get(name)
    if alias is not None:
        return getattr(model, alias, None)
    return None


def apply_nlmm_init(model: nn.Module, arch_depths: Dict[str, int], arch_kind: str) -> nn.Module:
    """Apply the NL-MM initialization recipe in-place."""

    set_global_seed(1337)
    convert_layernorm_to_rmsnorm(model)
    insert_qk_norm_in_attention_modules(model)

    for name, depth in arch_depths.items():
        stack = _resolve_stack(model, name)
        if stack is None:
            continue
        apply_deepnorm_scaling(stack, depth, arch_kind)

    for module in model.modules():
        if is_linear(module):
            init_linear_glorot(module)
        elif is_conv(module):
            init_conv_kaiming(module)

    for name, depth in arch_depths.items():
        stack = _resolve_stack(model, name)
        if stack is None:
            continue
        downscale_ffn_and_v_out(stack, depth, arch_kind)

    txt_enc = getattr(model, "txt_enc", None)
    if txt_enc is not None:
        init_trunc_normal_002(txt_enc.token.weight)
        if hasattr(txt_enc, "pos"):
            init_trunc_normal_002(txt_enc.pos)

    for attr in ("positional_emb", "positional_embedding", "position_embedding"):
        if hasattr(model, attr):
            init_trunc_normal_002(getattr(model, attr))

    clm = getattr(model, "clm", None)
    if clm is not None and hasattr(clm, "latent"):
        init_trunc_normal_002(clm.latent)

    zero_fast_weight_memory(model)
    init_ttt_adapters_identity(model)
    tie_or_init_output_heads(model)
    return model


__all__ = ["apply_nlmm_init"]

