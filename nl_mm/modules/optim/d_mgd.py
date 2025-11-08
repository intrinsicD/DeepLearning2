"""Deep Momentum Gradient Descent (DMGD)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch


class DMGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, beta: float = 0.9, nonlinearity: str = "none"):
        defaults = dict(lr=lr, beta=beta, nonlinearity=nonlinearity)
        super().__init__(params, defaults)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        for param in self.mlp.parameters():
            param.requires_grad_(False)
        self._mlp_device: torch.device | None = None

    def _apply_nonlinearity(self, m: torch.Tensor, nonlinearity: str) -> torch.Tensor:
        if nonlinearity == "nsqrt":
            return torch.sign(m) * torch.sqrt(m.abs() + 1e-6)
        return m

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            nl = group["nonlinearity"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if self._mlp_device != grad.device:
                    self.mlp.to(grad.device)
                    self._mlp_device = grad.device
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]
                grad_m = grad.mean().view(1, 1)
                deep = self.mlp(grad_m).view(1).item()
                momentum.mul_(beta).add_(grad * deep)
                if nl != "none":
                    momentum.copy_(self._apply_nonlinearity(momentum, nl))
                p.add_(-lr * momentum)
        return loss
