"""Gradient-Difference Aware Anderson (GDA²) optimizer implementation.

This module provides a drop-in PyTorch optimizer that combines an AdamW-style
preconditioner with a Type-II Anderson acceleration scheme.  The Anderson
correction is regularised using gradient-difference statistics, enabling the
optimizer to favour history segments that exhibit stable curvature while
down-weighting noisy updates.  The implementation mirrors the description
provided in the accompanying documentation and is designed to interoperate with
the existing optimizer registry in :mod:`src.optimizers`.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


TensorList = List[torch.Tensor]


def _flatten(tensors: TensorList) -> torch.Tensor:
    """Flatten a list of tensors into a contiguous 1-D tensor."""

    if not tensors:
        return torch.tensor([], device=tensors[0].device if tensors else "cpu")
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten_like(vec: torch.Tensor, like: TensorList) -> TensorList:
    """Reshape a flat tensor to match the shapes in ``like``."""

    out: TensorList = []
    offset = 0
    for t in like:
        num = t.numel()
        out.append(vec[offset : offset + num].view_as(t))
        offset += num
    return out


class GDA2(Optimizer):
    r"""Gradient-Difference Aware Anderson (GDA²) Optimizer.

    The optimizer wraps an AdamW-style preconditioner with a Type-II Anderson
    mixer whose regularisation is weighted by gradient-difference statistics
    ``y_j = g_{j+1} - g_j``.  See the module docstring for a detailed
    description of the algorithmic components.

    Args:
        params: Iterable of parameters to optimise.
        lr: Base learning rate for the preconditioned step (default ``1e-3``).
        betas: Adam momentum coefficients (default ``(0.9, 0.999)``).
        eps: Numerical stability term for Adam (default ``1e-8``).
        weight_decay: Decoupled weight decay coefficient (default ``0.0``).
        history_size: Anderson memory ``m`` (default ``5``).
        reg: Base Tikhonov regularisation :math:`\lambda_0` (default ``1e-2``).
        nu: Curvature-to-regularisation gain (default ``0.5``).
        tau: Trust cap for the correction norm relative to the base step
            (default ``1.5``).
        interval: Apply Anderson every ``interval`` steps (default ``1``).
        cos_shift: Shift ``\tau_c`` inside ``max(0, cos + \tau_c)`` (default
            ``0.1``).
        w_clip: Clip for the inverse-curvature factor
            (default ``(0.1, 10.0)``).
        solver_dtype: Data type used for the small linear solves (default
            :class:`torch.float32`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        history_size: int = 5,
        reg: float = 1e-2,
        nu: float = 0.5,
        tau: float = 1.5,
        interval: int = 1,
        cos_shift: float = 0.1,
        w_clip: Tuple[float, float] = (0.1, 10.0),
        solver_dtype: torch.dtype = torch.float32,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.m = history_size
        self.reg0 = reg
        self.nu = nu
        self.tau = tau
        self.interval = interval
        self.cos_shift = cos_shift
        self.wmin, self.wmax = w_clip
        self.solver_dtype = solver_dtype

        self._gstate: Dict[int, Dict[str, object]] = {}

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            gid = id(group)
            gstate = self._gstate.setdefault(
                gid,
                {
                    "step": 0,
                    "f_hist": deque(maxlen=self.m + 1),
                    "g_hist": deque(maxlen=self.m + 1),
                    "s_hist": deque(maxlen=self.m + 1),
                },
            )
            gstate["step"] = int(gstate["step"]) + 1

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            base_steps: TensorList = []
            grads: TensorList = []

            for p in params_with_grad:
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] = int(state["step"]) + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** state["step"]
                bias_c2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
                step_t = -lr * (exp_avg / bias_c1) / denom

                base_steps.append(step_t)
                grads.append(grad)

            f_k = _flatten(base_steps).detach()
            g_k = _flatten(grads).detach()

            do_anderson = (gstate["step"] > 1) and ((gstate["step"] % self.interval) == 0)
            if do_anderson:
                f_hist: Deque[torch.Tensor] = gstate["f_hist"]
                g_hist: Deque[torch.Tensor] = gstate["g_hist"]
                s_hist: Deque[torch.Tensor] = gstate["s_hist"]

                f_cols = list(f_hist) + [f_k]
                g_cols = list(g_hist) + [g_k]
                s_cols = list(s_hist)

                pairs = min(len(f_cols) - 1, self.m, len(s_cols))
                if pairs > 0:
                    df_cols: TensorList = []
                    dx_cols: TensorList = []
                    weights: List[float] = []

                    for j in range(-pairs - 1, -1):
                        df = (f_cols[j + 1] - f_cols[j]).to(self.solver_dtype)
                        s = s_cols[j].to(self.solver_dtype)
                        y = (g_cols[j + 1] - g_cols[j]).to(self.solver_dtype)

                        sy = torch.dot(s, y)
                        ns = torch.norm(s)
                        ny = torch.norm(y)
                        cos = sy / (ns * ny + 1e-16)
                        cos = torch.clamp(cos + self.cos_shift, min=0.0)
                        inv_curv = ns / (ny + 1e-16)
                        inv_curv = torch.clamp(inv_curv, self.wmin, self.wmax)
                        weight = float(max(1e-8, (cos * inv_curv).item()))

                        df_cols.append(df)
                        dx_cols.append(s)
                        weights.append(weight)

                    df_mat = torch.stack(df_cols, dim=1)
                    rhs = df_mat.T @ f_k.to(self.solver_dtype)
                    gram = df_mat.T @ df_mat

                    w = torch.tensor(weights, dtype=self.solver_dtype, device=df_mat.device)
                    w = w / (w.mean() + 1e-16)

                    curvature_samples: List[float] = []
                    for s_vec, g0, g1 in zip(dx_cols, g_cols[-pairs - 1 : -1], g_cols[-pairs:]):
                        y_vec = (g1 - g0).to(self.solver_dtype)
                        denom = torch.norm(s_vec) + 1e-16
                        curvature_samples.append(float(torch.norm(y_vec) / denom))

                    curvature = sum(curvature_samples) / len(curvature_samples) if curvature_samples else 0.0
                    lam = self.reg0 * (1.0 + self.nu * curvature)

                    reg_matrix = lam * torch.diag(1.0 / (w + 1e-16))

                    system = gram + reg_matrix

                    try:
                        theta = torch.linalg.solve(system, rhs)
                    except RuntimeError:
                        eye = torch.eye(system.shape[0], dtype=system.dtype, device=system.device)
                        theta = torch.linalg.solve(system + 1e-6 * eye, rhs)

                    correction = torch.zeros_like(f_k, dtype=self.solver_dtype)
                    for coeff, s_vec in zip(theta, dx_cols):
                        correction.add_(s_vec, alpha=float(coeff))

                    base_norm = torch.norm(f_k)
                    corr_norm = torch.norm(correction)
                    if corr_norm > self.tau * base_norm:
                        scale = (self.tau * base_norm / (corr_norm + 1e-16)).item()
                        correction.mul_(scale)

                    mixed = (f_k.to(self.solver_dtype) - correction).to(f_k.dtype)

                    if torch.dot(g_k, mixed) >= 0:
                        final_step = f_k
                    else:
                        final_step = mixed
                else:
                    final_step = f_k
            else:
                final_step = f_k

            step_tensors = _unflatten_like(final_step, params_with_grad)
            for p, delta in zip(params_with_grad, step_tensors):
                p.data.add_(delta)

            gstate["f_hist"].append(f_k.detach())
            gstate["g_hist"].append(g_k.detach())
            gstate["s_hist"].append(final_step.detach())

        return loss

