r"""Reference implementation of the Muon optimizer with lightweight routing logic.

This module provides :class:`MuonFast`, a PyTorch-compatible optimizer that applies
Muon updates to matrix-shaped parameters (two-dimensional tensors) while routing any
other parameters to a fallback optimizer (AdamW by default).  The implementation is a
pure-PyTorch baseline that mirrors the semantics of the high-performance design
outlined in the MuonFast plan.  It keeps the public API stable so that accelerated
CUDA/Triton backends can be integrated later without affecting user code.

Compared with the initial reference draft, this version introduces three key
improvements inspired by production implementations:

* **Routing guardrails** – two-dimensional tensors whose smallest dimension falls
  below a configurable threshold are automatically routed to the fallback optimizer,
  mirroring Keras' advice to avoid Muon on very small matrices.
* **Pre-scaled Newton--Schulz iterations** – the inverse square root is estimated in
  float32 after scaling the Gram matrix by the mean trace, which accelerates
  convergence and improves numerical stability.
* **Adaptive stopping** – the orthogonalization loop exits early once the Newton--
  Schulz residual drops below a tolerance, saving iterations on well-conditioned
  updates.

The orthogonalization step uses a small number of Newton--Schulz iterations to
approximate ``(\Delta^\top \Delta)^{-1/2}`` in float32 for numerical robustness.  The
resulting orthogonalized update is applied to the parameter in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer


@dataclass
class _FallbackConfig:
    """Configuration for the fallback optimizer.

    Parameters
    ----------
    name:
        Identifier for the optimizer class.  Only ``"adamw"`` and ``"sgd"`` are
        currently supported.
    kwargs:
        Keyword arguments forwarded to the fallback optimizer constructor.
    """

    name: str = "adamw"
    kwargs: Optional[dict] = None

    def instantiate(self, params: Sequence[torch.nn.Parameter]) -> Optional[Optimizer]:
        if not params:
            return None

        opts = dict(self.kwargs or {})
        name = self.name.lower()
        if name == "adamw":
            cls = AdamW
        elif name == "sgd":
            from torch.optim import SGD

            cls = SGD
        else:
            raise ValueError(f"Unsupported fallback optimizer '{self.name}'.")

        return cls(params, **opts)


def _as_param_groups(params: Union[Iterable[Tensor], Iterable[dict]]) -> List[dict]:
    """Normalize arbitrary parameter inputs into a list of param group dictionaries."""
    if isinstance(params, torch.nn.Parameter):
        return [dict(params=[params])]

    try:
        params = list(params)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise TypeError("params argument given to MuonFast should be an iterable") from exc

    if not params:
        return []

    if isinstance(params[0], dict):
        return [dict(pg) for pg in params]  # shallow copy to avoid mutation issues

    return [dict(params=params)]


def _split_param_groups(
    param_groups: Sequence[dict],
    min_dim_muon: int,
    strict_small_matrices: bool,
) -> Tuple[List[dict], List[torch.nn.Parameter]]:
    """Split the user-provided parameter groups into Muon and fallback sets."""
    muon_groups: List[dict] = []
    fallback_params: List[torch.nn.Parameter] = []

    for group in param_groups:
        group_params = group.get("params", [])
        if isinstance(group_params, torch.nn.Parameter):
            group_params = [group_params]

        muon_params: List[torch.nn.Parameter] = []
        for param in group_params:
            if not isinstance(param, torch.nn.Parameter):  # pragma: no cover - sanity
                raise TypeError("Parameter groups must contain torch.nn.Parameter objects")
            if not param.requires_grad:
                continue

            if param.ndim == 2 and (
                strict_small_matrices or min(param.shape) >= min_dim_muon
            ):
                muon_params.append(param)
            else:
                fallback_params.append(param)

        if muon_params:
            new_group = dict(group)
            new_group["params"] = muon_params
            muon_groups.append(new_group)

    return muon_groups, fallback_params


class MuonFast(Optimizer):
    r"""Pure-PyTorch Muon optimizer with parameter routing.

    Parameters
    ----------
    params:
        Iterable of parameters or parameter groups following the PyTorch optimizer
        convention.  All two-dimensional tensors are optimized with Muon; all other
        tensors are routed to the fallback optimizer.
    lr:
        Learning rate for the Muon updates.
    momentum:
        Momentum coefficient :math:`\mu`.
    weight_decay:
        Decoupled weight decay applied to Muon parameters.
    ns_iters:
        Number of Newton--Schulz iterations used to approximate the inverse square
        root.  Values between 3 and 5 typically offer a good trade-off between
        accuracy and cost.
    eps:
        Diagonal regularization added to :math:`\Delta^\top \Delta` before the
        Newton--Schulz iterations.
    dtype:
        Placeholder for future mixed-precision control.  The reference implementation
        performs all computations in float32 regardless of this argument.
    backend:
        Placeholder for selecting alternative kernel backends.  Only the pure PyTorch
        path is implemented at present.
    graph_capture:
        Reserved for CUDA Graph support.  Unused in the reference implementation.
    fallback:
        Configuration for the fallback optimizer used for non-matrix parameters.
    fallback_options:
        Convenience alias for ``fallback.kwargs``.  When both are provided, explicit
        ``fallback_options`` entries override any keys set inside ``fallback``.
    min_dim_muon:
        Minimum dimension that a two-dimensional tensor must satisfy (with
        ``min(m, n) >= min_dim_muon``) in order to be optimized with Muon.  Smaller
        matrices are routed to the fallback optimizer by default because the fixed
        Newton--Schulz work and kernel launch overhead typically outweigh the gains of
        orthogonalization.  Set ``strict_small_matrices=True`` to override this
        behaviour.
    strict_small_matrices:
        When ``True``, all two-dimensional tensors are optimized with Muon regardless
        of their size.
    ns_tol:
        Frobenius-norm tolerance used to terminate the Newton--Schulz iterations
        early.  A value of ``0.0`` disables the early-stop heuristic.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        ns_iters: int = 3,
        eps: float = 1e-6,
        dtype: str = "bf16",
        backend: str = "cuda",
        graph_capture: bool = False,
        fallback: Optional[_FallbackConfig] = None,
        fallback_options: Optional[dict] = None,
        min_dim_muon: int = 64,
        strict_small_matrices: bool = False,
        ns_tol: float = 1e-4,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")
        if momentum < 0.0:
            raise ValueError("Momentum must be non-negative.")
        if ns_iters < 0:
            raise ValueError("ns_iters must be non-negative.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if min_dim_muon < 0:
            raise ValueError("min_dim_muon must be non-negative.")
        if ns_tol < 0.0:
            raise ValueError("ns_tol must be non-negative.")

        param_groups = _as_param_groups(params)
        muon_groups, fallback_params = _split_param_groups(
            param_groups,
            min_dim_muon=min_dim_muon,
            strict_small_matrices=strict_small_matrices,
        )

        if not muon_groups:
            raise ValueError(
                "MuonFast requires at least one two-dimensional parameter; "
                "all other parameters are routed to the fallback optimizer."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            eps=eps,
            dtype=dtype,
            backend=backend,
            graph_capture=graph_capture,
            ns_tol=ns_tol,
        )

        super().__init__(muon_groups, defaults)

        if fallback is None:
            fallback = _FallbackConfig()
        if fallback_options:
            opts = dict(fallback.kwargs or {})
            opts.update(fallback_options)
            fallback = _FallbackConfig(name=fallback.name, kwargs=opts)

        if fallback.kwargs is None:
            fallback.kwargs = {"lr": lr, "weight_decay": weight_decay}
        else:
            fallback.kwargs.setdefault("lr", lr)
            fallback.kwargs.setdefault("weight_decay", weight_decay)

        self._fallback_opt = fallback.instantiate(fallback_params)
        self._fallback_config = fallback

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._fallback_opt is not None:
            self._fallback_opt.step()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_iters = group["ns_iters"]
            eps = group["eps"]
            ns_tol = group["ns_tol"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonFast does not support sparse gradients.")

                state = self.state[param]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(param, dtype=param.dtype)
                    state["momentum_buffer"] = buf

                buf.mul_(momentum).add_(grad)
                update = -lr * buf

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                ortho_update = self._orthogonalize(
                    update,
                    ns_iters=ns_iters,
                    eps=eps,
                    tol=ns_tol,
                )
                param.add_(ortho_update)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)
        if self._fallback_opt is not None:
            self._fallback_opt.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _orthogonalize(update: Tensor, ns_iters: int, eps: float, tol: float) -> Tensor:
        """Project the update onto the closest orthogonal matrix using Newton--Schulz."""
        if update.ndim != 2:
            return update

        device = update.device
        dtype = torch.float32
        mat = update.to(dtype)

        m, n = mat.shape

        if mat.abs().max().item() == 0.0:
            return update

        # Use the smaller dimension for the Gram matrix to minimize computation
        # For m x n matrix:
        # - If m <= n: compute (mat @ mat.T)^{-1/2} @ mat (left multiplication)
        # - If m > n: compute mat @ (mat.T @ mat)^{-1/2} (right multiplication)
        if m <= n:
            # Compute mat @ mat.T (m x m matrix)
            gram = mat @ mat.transpose(0, 1)
            size = m
            left_multiply = True
        else:
            # Compute mat.T @ mat (n x n matrix)
            gram = mat.transpose(0, 1) @ mat
            size = n
            left_multiply = False

        identity = torch.eye(size, device=device, dtype=dtype)
        gram = gram + eps * identity

        trace = torch.trace(gram)
        if trace.item() <= 0.0:
            return update

        scale = torch.rsqrt(trace / float(size))
        if not torch.isfinite(scale).item():
            return update

        scaled_gram = gram * (scale * scale)
        y = scaled_gram
        z = identity.clone()
        zy = z @ y

        for _ in range(ns_iters):
            if tol > 0.0:
                residual = torch.linalg.norm(identity - zy, ord="fro")
                if residual <= tol:
                    break

            t = 0.5 * (3.0 * identity - zy)
            y = y @ t
            z = t @ z
            zy = z @ y

        inv_sqrt = z * scale

        if left_multiply:
            orthogonal_update = (inv_sqrt @ mat).to(update.dtype)
        else:
            orthogonal_update = (mat @ inv_sqrt).to(update.dtype)

        return orthogonal_update

    def state_dict(self):  # type: ignore[override]
        """Augment the default state dict with fallback optimizer metadata."""
        base_state = super().state_dict()
        if self._fallback_opt is not None:
            base_state["fallback_state"] = self._fallback_opt.state_dict()
            base_state["fallback_config"] = {
                "name": self._fallback_config.name,
                "kwargs": self._fallback_config.kwargs,
            }
        return base_state

    def load_state_dict(self, state_dict):  # type: ignore[override]
        fallback_state = state_dict.pop("fallback_state", None)
        fallback_config = state_dict.pop("fallback_config", None)
        super().load_state_dict(state_dict)

        if fallback_config is not None:
            self._fallback_config = _FallbackConfig(**fallback_config)
        if fallback_state is not None and self._fallback_opt is not None:
            self._fallback_opt.load_state_dict(fallback_state)
