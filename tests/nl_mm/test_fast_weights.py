from __future__ import annotations

import torch

from modules.nl_mm.modules.fast_weights import FastWeightLinearAttention, FastWeightState


def test_fast_weight_state_reinitializes_for_batch_mismatch():
    attn = FastWeightLinearAttention(d_model=8, n_heads=2)
    x_big = torch.randn(2, 4, 8)
    _, state = attn(x_big, None)

    x_small = torch.randn(1, 4, 8)
    out, new_state = attn(x_small, state)

    assert out.shape[0] == x_small.size(0)
    assert new_state.memory.shape[0] == x_small.size(0)


def test_fast_weight_state_reinitializes_for_dtype_mismatch():
    attn = FastWeightLinearAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 3, 8)
    _, state = attn(x, None)

    mismatched_state = FastWeightState(memory=state.memory.to(torch.float16))
    out, new_state = attn(x, mismatched_state)

    assert out.dtype == x.dtype
    assert new_state.memory.dtype == x.dtype
