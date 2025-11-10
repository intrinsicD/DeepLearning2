"""Deep Momentum Gradient Descent (DMGD)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch


class DMGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.9,
        nonlinearity: str = "none",
        learnable_modulation: bool = True,
        mlp_lr: float = 1e-2,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            nonlinearity=nonlinearity,
            learnable_modulation=learnable_modulation,
        )
        super().__init__(params, defaults)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        self.learnable_modulation = learnable_modulation
        self.mlp_lr = mlp_lr
        if not self.learnable_modulation:
            for param in self.mlp.parameters():
                param.requires_grad_(False)
        self._mlp_device: torch.device | None = None

    def _apply_nonlinearity(self, m: torch.Tensor, nonlinearity: str) -> torch.Tensor:
        if nonlinearity == "nsqrt":
            return torch.sign(m) * torch.sqrt(m.abs() + 1e-6)
        return m

    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            nl = group["nonlinearity"]
            learnable = group["learnable_modulation"]
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
                if learnable:
                    with torch.enable_grad():
                        deep = self.mlp(grad_m).view(1)
                        target = grad.abs().mean().detach().view_as(deep)
                        mlp_loss = (deep - target).pow(2).mean()
                        grads = torch.autograd.grad(mlp_loss, self.mlp.parameters())
                    with torch.no_grad():
                        for param, g in zip(self.mlp.parameters(), grads):
                            param.add_(-self.mlp_lr * g)
                    deep_value = deep.detach().to(grad.dtype)
                else:
                    with torch.no_grad():
                        deep_value = self.mlp(grad_m).view(1)
                    deep_value = deep_value.to(grad.dtype)
                update = beta * momentum + grad * deep_value
                if nl != "none":
                    update = self._apply_nonlinearity(update, nl)
                with torch.no_grad():
                    momentum.copy_(update)
                    p.add_(-lr * update)
        return loss

    def modulation_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return trainable parameters controlling the deep modulation."""
        return self.mlp.parameters()
