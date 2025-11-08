"""Fast weight linear attention with HOPE-style modulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class FastWeightState:
    memory: torch.Tensor

    @staticmethod
    def init(batch: int, n_heads: int, head_dim: int, *, device: torch.device, dtype: torch.dtype) -> "FastWeightState":
        memory = torch.zeros(batch, n_heads, head_dim, head_dim, device=device, dtype=dtype)
        return FastWeightState(memory=memory)


class HOPEProjection(nn.Module):
    """Tiny MLP that generates per-head affine modulation parameters."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        hidden = max(d_model // 4, 16)
        self.mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 3 * n_heads))

    def forward(self, slow_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.mlp(slow_state)
        params = params.view(slow_state.size(0), self.n_heads, 3)
        gamma_q, gamma_k, gamma_v = params.unbind(-1)
        return gamma_q.tanh() + 1.0, gamma_k.tanh() + 1.0, gamma_v.tanh() + 1.0


class FastWeightLinearAttention(nn.Module):
    """Implements Eq. (12–16) using batched outer-products."""

    def __init__(self, d_model: int, n_heads: int, *, normalize: bool = True, decay: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.normalize = normalize
        self.decay = decay

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.hope_projection = HOPEProjection(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def init_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> FastWeightState:
        return FastWeightState.init(batch, self.n_heads, self.head_dim, device=device, dtype=dtype)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = tensor.shape
        return tensor.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, state: Optional[FastWeightState], *, slow_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, FastWeightState]:
        if state is None:
            state = self.init_state(x.size(0), device=x.device, dtype=x.dtype)
        memory = state.memory.detach()

        q = self._reshape(self.wq(x))
        k = self._reshape(self.wk(x))
        v = self._reshape(self.wv(x))

        if slow_state is not None:
            gamma_q, gamma_k, gamma_v = self.hope_projection(slow_state)
            q = q * gamma_q[:, :, None, None]
            k = k * gamma_k[:, :, None, None]
            v = v * gamma_v[:, :, None, None]

        if self.normalize:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        outputs = []
        updated_memory = memory.clone()
        for t in range(x.size(1)):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            if self.decay > 0:
                updated_memory.mul_(1.0 - self.decay)
            outer = torch.einsum("bhd,bhe->bhde", v_t, k_t)
            updated_memory = updated_memory + outer
            y_t = torch.einsum("bhde,bhe->bhd", updated_memory, q_t)
            outputs.append(y_t)
        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        out = self.out_proj(out)
        return out, FastWeightState(memory=updated_memory.detach())
