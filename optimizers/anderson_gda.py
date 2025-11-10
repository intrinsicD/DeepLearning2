"""
Anderson-accelerated optimizer with gradient difference awareness.

This module provides ``AndersonGDA`` (Gradient Difference Aware) – a PyTorch
compatible optimizer that augments vanilla gradient descent with two
acceleration mechanisms:

1. **Gradient difference correction**.  Similar in spirit to heavy-ball or
   Nesterov momentum, the update uses the difference between the current
   gradient and the previous gradient to anticipate curvature changes.  A
   positive ``beta`` value dampens oscillations and can speed up convergence
   when the gradient direction varies slowly.

2. **Anderson acceleration**.  After computing a candidate step, the
   optimizer performs a small fixed‑point extrapolation over a window of the
   most recent residuals (differences between the one‑step updates and the
   original parameters).  This linear combination attempts to cancel out
   stationary error components and has been shown to substantially
   accelerate root‑finding and optimization procedures.  See Walker and
   Ni (2011) for an overview of Anderson acceleration.

The implementation below operates on individual parameter tensors.  Each
parameter maintains its own history of residuals and candidate updates, so
stateful acceleration does not leak across unrelated tensors.  Because the
memory dimension ``m`` is typically small (≤5) and the residuals are never
materialised as full matrices simultaneously, the computational overhead is
modest.  A small diagonal regularization ``eps`` stabilises the least‑squares
solve when residual vectors become nearly linearly dependent.

Example usage::

    from optimizers.anderson_gda import AndersonGDA
    optimizer = AndersonGDA(model.parameters(), lr=1e-3, beta=0.5, m=3)

    for input, target in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        loss.backward()
        optimizer.step()

Notes
-----
While Anderson acceleration can provide significant speedups, it can also
instability for non‑convex objectives or when gradients are noisy.  We
recommend starting with a small history size (m=2 or 3) and monitoring
training curves to ensure stable behaviour.  Setting ``beta=0.0`` will
disable the gradient difference correction.  Setting ``m=0`` will
effectively reduce the optimizer to plain gradient descent with the
gradient difference term.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class AndersonGDA(Optimizer):
    """Implements an Anderson accelerated optimizer with gradient difference awareness.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-3
        Learning rate for the underlying gradient descent step.
    beta : float, default 0.0
        Coefficient for the gradient difference term.  A nonzero value
        introduces a correction proportional to the change in gradient
        ``(g_k - g_{k-1})``.  Setting ``beta=0`` disables this term.
    m : int, default 3
        Number of previous residuals to use for Anderson acceleration.  A
        value of zero disables Anderson acceleration altogether.  Larger
        values provide a richer subspace but increase computation and
        memory cost.
    eps : float, default 1e-8
        Diagonal regularisation added to the Gram matrix when solving the
        least squares problem.  Helps avoid singularity when residuals
        become collinear.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-3,
        beta: float = 0.0,
        m: int = 3,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if m < 0:
            raise ValueError("Memory m must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        defaults = dict(lr=lr, beta=beta, m=m, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:  # type: ignore[override]
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.

        Returns
        -------
        loss : float, optional
            The evaluated loss if ``closure`` is provided.
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["beta"]
            m: int = group["m"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad.detach()
                state = self.state[p]

                # Initialise state on first use
                if not state:
                    state["prev_grad"] = torch.zeros_like(grad)
                    state["history"]: list[Tuple[Tensor, Tensor]] = []

                prev_grad = state["prev_grad"]

                # Basic gradient descent step
                step_basic = -lr * grad

                # Gradient difference term (acts like momentum but uses grad diff)
                step_diff = torch.zeros_like(step_basic)
                if beta != 0.0:
                    step_diff = -beta * lr * (grad - prev_grad)

                # Candidate update (F(x_k))
                candidate = p.data + step_basic + step_diff
                residual = candidate - p.data

                # Update gradient history
                state["prev_grad"] = grad.clone()

                # Anderson acceleration: update history and solve for combination
                if m > 0:
                    history: list[Tuple[Tensor, Tensor]] = state["history"]
                    # Append current (candidate, residual)
                    history.append((candidate.clone(), residual.clone()))
                    # Keep last m entries (including current)
                    if len(history) > m:
                        history.pop(0)

                    # Use up to m-1 previous residuals (exclude current) for
                    # constructing the subspace.  If there are fewer than 2
                    # residuals, acceleration is skipped.
                    if len(history) > 1:
                        # Separate previous residuals (all but last)
                        prev_residuals = [res.clone() for (_, res) in history[:-1]]
                        current_residual = residual

                        k = len(prev_residuals)
                        # Build Gram matrix and cross vector
                        gram = torch.empty((k, k), dtype=current_residual.dtype, device=current_residual.device)
                        b = torch.empty(k, dtype=current_residual.dtype, device=current_residual.device)
                        # Flatten residuals to 1-D for dot products
                        cr_flat = current_residual.view(-1)
                        # Pre-flatten previous residuals for efficiency
                        prev_flats = [r.view(-1) for r in prev_residuals]

                        for i in range(k):
                            b[i] = torch.dot(prev_flats[i], cr_flat)
                            for j in range(k):
                                if j <= i:
                                    val = torch.dot(prev_flats[i], prev_flats[j])
                                    gram[i, j] = val
                                    gram[j, i] = val  # symmetric

                        # Regularise Gram matrix for stability
                        diag_indices = torch.arange(k, device=current_residual.device)
                        gram[diag_indices, diag_indices] += eps

                        # Solve least squares: gram * c = b
                        try:
                            c = torch.linalg.solve(gram, b)
                        except RuntimeError:
                            # Fall back to pseudo-inverse if solve fails
                            c = torch.linalg.lstsq(gram, b).solution

                        # Compute Anderson update: delta = sum_i c_i * r_i
                        delta = torch.zeros_like(current_residual)
                        for coeff, r in zip(c, prev_residuals):
                            delta += coeff * r

                        # Apply accelerated update: x_{k+1} = x_k + delta
                        p.data.add_(delta)
                    else:
                        # Not enough history: fallback to candidate
                        p.data.copy_(candidate)
                else:
                    # Anderson disabled: simply apply candidate
                    p.data.copy_(candidate)

        return loss

