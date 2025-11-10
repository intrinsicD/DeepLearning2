"""Test-time training adapters implementing the L2 inner objective."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TTTAdapter(nn.Module):
    """LoRA-style adapter updated via Eq. (28–29)."""

    def __init__(self, d_model: int, rank: int = 16, eta: float = 1e-3, max_steps: int = 2):
        super().__init__()
        self.inp = nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(rank, d_model, bias=False)
        self.eta = eta
        self.max_steps = max_steps
        self.register_buffer("_steps", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out(F.gelu(self.inp(x)))

    @torch.no_grad()
    def reset(self) -> None:
        self._steps.zero_()

    @torch.no_grad()
    def ttt_step(self, x: torch.Tensor, dL_dy: torch.Tensor) -> None:
        if self._steps.item() >= self.max_steps:
            return
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        grad_out = dL_dy.reshape(-1, D)

        proj = self.inp(x_flat)
        hidden_grad = grad_out @ self.out.weight
        gelu_grad = hidden_grad * torch.sigmoid(1.702 * proj)  # derivative approximation
        dW_out = torch.einsum("bo,bi->oi", grad_out, F.gelu(proj)) / max(1, x_flat.size(0))
        dW_in = gelu_grad.t() @ x_flat / max(1, x_flat.size(0))

        gram = x_flat.t() @ x_flat
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        contraction = identity - gram / (gram.diagonal().mean() + 1e-6)

        self.inp.weight.copy_(self.inp.weight @ contraction)
        self.out.weight.add_(-self.eta * dW_out)
        self.inp.weight.add_(-self.eta * dW_in)
        self._steps += 1
