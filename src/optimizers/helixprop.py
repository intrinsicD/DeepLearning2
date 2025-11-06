"""HelixProp optimizer."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class HelixProp(Optimizer):
    r"""HelixProp optimizer.

    HelixProp couples three moving statistics of the gradient: the running
    average itself, the average gradient change, and the alignment between
    consecutive gradients.  The update direction twists these three axes in a
    helix-like manner: the momentum term drives progress, the gradient-change
    term absorbs local curvature, and the alignment modulates the final step.

    Key ideas that differ from standard optimizers:

    * The optimizer explicitly maintains an exponential moving average of the
      *gradient differences*.  This term (`diff_avg`) acts as a lightweight
      estimate of how the landscape bends locally.
    * A second accumulator tracks the *energy* of the gradient differences
      (`energy`), which serves as a curvature-aware scale rather than the
      magnitude of gradients themselves.
    * A bounded alignment factor measures the directional agreement of the
      current gradient with the previous one, using the curvature energy as a
      normaliser.  The alignment directly modulates the step, shrinking updates
      when the direction reverses and amplifying them when successive gradients
      agree.

    Together these ingredients yield an update of the form

    .. math::

       \Delta \theta_t = -\eta\, (m_t - d_t) \odot \frac{1 + a_t}{\sqrt{E_t} + \varepsilon},

    where :math:`m_t` is the momentum term, :math:`d_t` the averaged gradient
    change, :math:`E_t` the curvature energy, and :math:`a_t` the alignment
    modulator.  The subtraction removes spurious oscillations while the
    alignment keeps the dynamics stable even on problems with rapidly changing
    curvature.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.9,
        rho: float = 0.95,
        theta: float = 0.99,
        kappa: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 < rho < 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 < theta < 1.0:
            raise ValueError(f"Invalid theta value: {theta}")
        if not 0.0 < kappa < 1.0:
            raise ValueError(f"Invalid kappa value: {kappa}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta=beta,
            rho=rho,
            theta=theta,
            kappa=kappa,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            rho = group["rho"]
            theta = group["theta"]
            kappa = group["kappa"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled_wd = group["decoupled_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("HelixProp does not support sparse gradients")

                if weight_decay != 0.0 and not decoupled_wd:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["diff_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["energy"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["alignment"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["prev_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                diff_avg = state["diff_avg"]
                energy = state["energy"]
                alignment = state["alignment"]
                prev_grad = state["prev_grad"]

                state["step"] += 1

                grad_delta = grad - prev_grad
                diff_avg.mul_(rho).add_(grad_delta, alpha=1.0 - rho)
                exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)
                energy.mul_(theta).addcmul_(grad_delta, grad_delta, value=1.0 - theta)

                denom = torch.sqrt(energy).add(eps)
                alignment_update = torch.tanh((grad * prev_grad) / denom)
                alignment.mul_(kappa).add_(alignment_update, alpha=1.0 - kappa)

                numerator = exp_avg - diff_avg
                modulator = torch.clamp(1.0 + alignment, min=0.1, max=2.0)
                update = numerator / denom * modulator

                p.add_(update, alpha=-lr)

                if decoupled_wd and weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                prev_grad.copy_(grad)

        return loss
