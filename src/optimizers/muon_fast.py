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


# Polynomial coefficient tables for Newton--Schulz transforms.
# Each entry maps an identifier to a list of per-iteration coefficient lists.
# A coefficient list ``[c0, c1, ...]`` represents ``c0 * I + c1 * (ZY) + c2 * (ZY)^2 + ...``.
_NS_POLYNOMIALS = {
    "simple": [[1.5, -0.5]],
    "quintic": [[15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0]],
    # Polar-Express schedule: progressively higher-order minimax polynomials.
    # Coefficients follow the odd-polynomial sequence detailed in recent
    # Newton--Schulz acceleration work (cubic → quintic → nonic).
    "polar_express": [
        [1.5, -0.5],
        [15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0],
        [315.0 / 128.0, -420.0 / 128.0, 378.0 / 128.0, -180.0 / 128.0, 35.0 / 128.0],
    ],
}


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
    """Normalize arbitrary parameter inputs into a list of param group dictionaries.

    Also attempts to preserve parameter names for reliable exclude_fn filtering.
    """
    if isinstance(params, torch.nn.Parameter):
        return [dict(params=[params])]

    try:
        params_list = list(params)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise TypeError("params argument given to MuonFast should be an iterable") from exc

    if not params_list:
        return []

    if isinstance(params_list[0], dict):
        return [dict(pg) for pg in params_list]  # shallow copy to avoid mutation issues

    return [dict(params=params_list)]


def _attach_param_names(param_groups: List[dict], model: Optional[torch.nn.Module] = None) -> None:
    """Attach parameter names to tensors for reliable exclude_fn filtering.

    This enables exclude_fn to reliably identify parameters like 'classifier', 'lm_head', 'embed'
    which is critical for Keras' guidance to not apply Muon to embeddings and output layers.
    """
    if model is not None:
        # Build name->param mapping from model
        name_map = {id(param): name for name, param in model.named_parameters()}

        for group in param_groups:
            group_params = group.get("params", [])
            if isinstance(group_params, torch.nn.Parameter):
                group_params = [group_params]

            for param in group_params:
                if not hasattr(param, '_param_name'):
                    param._param_name = name_map.get(id(param), '')


def _split_param_groups(
    param_groups: Sequence[dict],
    min_dim_muon: int,
    strict_small_matrices: bool,
    exclude_fn: Optional[callable] = None,
) -> Tuple[List[dict], List[torch.nn.Parameter]]:
    """Split the user-provided parameter groups into Muon and fallback sets.

    Routing heuristics (in order):
    1. Apply exclude_fn if provided (now with reliable names: 'classifier', 'lm_head', 'embed')
    2. Must be 2D
    3. min(m, n) >= min_dim_muon (unless strict_small_matrices=True)
    4. Aspect ratio < 32:1 (very skinny matrices are inefficient for NS)

    Per Keras guidance: Don't use Muon for embeddings, final output FC, or {0,1}-D vars.
    Default exclusions if exclude_fn=None: parameters with names matching common patterns.
    """
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

            # Get parameter name (attached by _attach_param_names or empty)
            param_name = getattr(param, '_param_name', '')

            # Check exclude_fn first (for final layer, embeddings, etc.)
            if exclude_fn is not None:
                if exclude_fn(param_name, param):
                    fallback_params.append(param)
                    continue
            elif param_name:  # Default exclusions if no custom exclude_fn
                # Per Keras: exclude embeddings, classifier/output heads, layer norms
                lower_name = param_name.lower()
                if any(pattern in lower_name for pattern in [
                    'classifier', 'lm_head', 'head', 'embed',
                    'norm', 'ln', 'bias'  # Also exclude norms and biases by default
                ]):
                    fallback_params.append(param)
                    continue

            # Auto-exclude 0-D and 1-D parameters (biases, norms)
            if param.ndim < 2:
                fallback_params.append(param)
                continue

            # Check if parameter is suitable for Muon
            is_2d = param.ndim == 2
            if is_2d:
                m, n = param.shape
                meets_min_dim = strict_small_matrices or min(m, n) >= min_dim_muon

                # Avoid very skinny matrices (aspect ratio > 32:1)
                aspect_ratio = max(m, n) / max(min(m, n), 1)
                reasonable_aspect = aspect_ratio <= 32.0

                # Auto-exclude likely classifier heads (small output dimension)
                # This catches heads even without names (Keras guidance)
                likely_head = min(m, n) <= 10  # num_classes typically ≤ 10 for toy datasets

                if meets_min_dim and reasonable_aspect and not likely_head:
                    muon_params.append(param)
                else:
                    fallback_params.append(param)
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
    nesterov:
        Whether to use Nesterov momentum (default True, matching standard Muon practice).
        When True, uses lookahead gradient: g_hat = grad + momentum * velocity.
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
        Note: By default, fallback inherits ``lr`` and ``weight_decay`` from Muon.
        To use different values, explicitly pass ``{"lr": ..., "weight_decay": ...}``
        here, as Muon and AdamW often require different hyperparameters at scale.
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
    verbose:
        When ``True``, prints a routing audit on initialization showing which
        parameters use Muon vs. fallback optimizer. Useful for verifying setup.
    lr_fallback:
        Learning rate for fallback optimizer. If None, inherits ``lr`` from Muon.
        Recommended to set explicitly, as Muon and AdamW often need different LRs.
    wd_fallback:
        Weight decay for fallback optimizer. If None, inherits ``weight_decay``.
    exclude_fn:
        Optional callable ``(name: str, param: Tensor) -> bool`` that returns True
        for parameters to exclude from Muon (route to fallback instead).
        Recommended: exclude final output layer and embeddings per Keras guidance.
        Example: ``lambda name, p: 'fc2' in name or 'embed' in name``
    scale_mode:
        Update scaling mode: ``"spectral"`` (default, NeMo-style), ``"shape"``,
        ``"rms_to_rms"``, or ``None``. Applies per-parameter scaling to
        orthogonalized updates to match AdamW's RMS norm characteristics and
        improve LR transferability. ``"rms_to_rms"`` keeps the per-parameter
        RMS aligned with a target magnitude.
    scale_extra:
        Additional multiplicative scale factor applied to updates (default 1.0).
        Can be tuned to match specific model characteristics, or combined with
        ``rms_to_rms`` to stretch/shrink the RMS after matching the target.
    scale_rms_target:
        Optional RMS target used when ``scale_mode="rms_to_rms"``. When ``None``
        the unscaled update RMS is preserved, so setting ``scale_extra`` alone
        has no effect. Provide an explicit value (e.g., ``1e-3``) to normalize
        per-parameter RMS magnitudes across widths.
    ns_coefficients:
        Newton-Schulz coefficient type: ``"simple"`` (default 3I-ZY transform),
        ``"quintic"`` (faster-converging cubic variant), or ``"polar_express"``
        (optimized for polar decomposition). Quintic/polar often allow lower
        ns_iters with better quality.
    model:
        Optional model reference for reliable parameter name extraction. When provided,
        enables name-based exclusion (e.g., 'classifier', 'lm_head', 'embed') per
        Keras guidance. If None, only shape-based heuristics are used.
    matmul_precision:
        Precision hint forwarded to :func:`torch.set_float32_matmul_precision`.
        ``"high"`` (default) matches NVIDIA's guidance for Muon; ``"medium`` or
        ``"default"`` can be selected to trade accuracy for speed. Pass ``None`` to
        skip setting the global flag.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
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
        ns_tol: float = 1e-3,
        verbose: bool = False,
        lr_fallback: Optional[float] = None,
        wd_fallback: Optional[float] = None,
        exclude_fn: Optional[callable] = None,
        scale_mode: str = "spectral",
        scale_extra: float = 1.0,
        scale_rms_target: Optional[float] = None,
        ns_coefficients: str = "simple",
        model: Optional[torch.nn.Module] = None,
        matmul_precision: Optional[str] = "high",
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
        if scale_mode not in ["spectral", "shape", "rms_to_rms", None]:
            raise ValueError("scale_mode must be 'spectral', 'shape', 'rms_to_rms', or None")
        if ns_coefficients not in ["simple", "quintic", "polar_express"]:
            raise ValueError("ns_coefficients must be 'simple', 'quintic', or 'polar_express'")
        if matmul_precision not in {"high", "medium", "default", None}:
            raise ValueError("matmul_precision must be 'high', 'medium', 'default', or None")

        # Convert params to param groups and attach names if model provided
        param_groups = _as_param_groups(params)

        # Attach parameter names for reliable exclude_fn (Keras guidance)
        if model is not None:
            _attach_param_names(param_groups, model)

        muon_groups, fallback_params = _split_param_groups(
            param_groups,
            min_dim_muon=min_dim_muon,
            strict_small_matrices=strict_small_matrices,
            exclude_fn=exclude_fn,
        )

        if not muon_groups:
            raise ValueError(
                "MuonFast requires at least one two-dimensional parameter; "
                "all other parameters are routed to the fallback optimizer."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            eps=eps,
            dtype=dtype,
            backend=backend,
            graph_capture=graph_capture,
            ns_tol=ns_tol,
            scale_mode=scale_mode,
            scale_extra=scale_extra,
            scale_rms_target=scale_rms_target,
            ns_coefficients=ns_coefficients,
        )

        super().__init__(muon_groups, defaults)

        # Set matmul precision for FP32 accumulation in NS iterations
        # This ensures Gram matrix and NS multiplications don't silently downgrade
        if backend == "cuda" and matmul_precision is not None:
            try:
                torch.set_float32_matmul_precision(matmul_precision)
            except:
                pass  # Older PyTorch versions may not support this

        # Warn about placeholder flags
        if backend != "cuda":
            import warnings
            warnings.warn("MuonFast: 'backend' is a placeholder in the reference implementation.")
        if graph_capture:
            import warnings
            warnings.warn("MuonFast: 'graph_capture' is not implemented in the reference implementation.")

        if fallback is None:
            fallback = _FallbackConfig()
        if fallback_options:
            opts = dict(fallback.kwargs or {})
            opts.update(fallback_options)
            fallback = _FallbackConfig(name=fallback.name, kwargs=opts)

        # Handle separate LR/WD for fallback
        if fallback.kwargs is None:
            fallback.kwargs = {}

        # Use explicit lr_fallback/wd_fallback if provided, otherwise inherit from Muon
        fallback.kwargs.setdefault("lr", lr_fallback if lr_fallback is not None else lr)
        fallback.kwargs.setdefault("weight_decay", wd_fallback if wd_fallback is not None else weight_decay)

        self._fallback_opt = fallback.instantiate(fallback_params)
        self._fallback_config = fallback

        # Print routing audit if verbose
        if verbose:
            self.print_routing_audit()

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
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_iters = group["ns_iters"]
            eps = group["eps"]
            ns_tol = group["ns_tol"]
            scale_mode = group["scale_mode"]
            scale_extra = group["scale_extra"]
            scale_rms_target = group["scale_rms_target"]
            ns_coefficients = group["ns_coefficients"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonFast does not support sparse gradients.")

                state = self.state[param]

                # Use FP32 master buffer for momentum to prevent precision loss in bf16/fp16 training
                buf = state.get("momentum_buffer_fp32")
                if buf is None:
                    buf = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["momentum_buffer_fp32"] = buf

                # Always work in FP32 for momentum accumulation
                grad_fp32 = grad.detach().to(torch.float32)

                if nesterov:
                    # Nesterov momentum: use lookahead gradient
                    # g_hat = grad + momentum * velocity_{t-1}
                    g_hat = grad_fp32.add(buf, alpha=momentum)
                    # Update velocity: v_t = momentum * v_{t-1} + grad
                    buf.mul_(momentum).add_(grad_fp32)
                    # Use lookahead for the update
                    update_fp32 = -lr * g_hat
                else:
                    # Classical momentum
                    buf.mul_(momentum).add_(grad_fp32)
                    update_fp32 = -lr * buf

                # Keep update in FP32 to avoid double-casting in _orthogonalize
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                ortho_update = self._orthogonalize(
                    update_fp32,
                    ns_iters=ns_iters,
                    eps=eps,
                    tol=ns_tol,
                    state=state,
                    ns_coefficients=ns_coefficients,
                )

                # Apply per-parameter update scaling (NeMo-style)
                if scale_mode is not None:
                    m, n = param.shape
                    scale = self._compute_update_scale(
                        m,
                        n,
                        scale_mode,
                        scale_extra,
                        update=ortho_update,
                        target_rms=scale_rms_target,
                    )
                    ortho_update = ortho_update * scale

                # Cast to param dtype only once at the end
                param.add_(ortho_update.to(param.dtype))

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)
        if self._fallback_opt is not None:
            self._fallback_opt.zero_grad(set_to_none=set_to_none)

    def print_routing_audit(self) -> None:
        """Print a routing audit showing which parameters use Muon vs fallback optimizer.

        Useful for verifying that 2D matrices are correctly routed to Muon and other
        parameters to the fallback optimizer.
        """
        print("=" * 70)
        print("MuonFast Routing Audit")
        print("=" * 70)

        muon_count = 0
        muon_params = 0
        print("\nMuon-optimized parameters (2D matrices):")
        for group_idx, group in enumerate(self.param_groups):
            for param in group["params"]:
                muon_count += 1
                muon_params += param.numel()
                print(f"  [{group_idx}] shape={list(param.shape)}, numel={param.numel():,}")

        print(f"\nTotal Muon parameters: {muon_count} tensors, {muon_params:,} elements")

        if self._fallback_opt is not None:
            fallback_count = 0
            fallback_params = 0
            print(f"\nFallback-optimized parameters ({self._fallback_config.name.upper()}):")
            for group_idx, group in enumerate(self._fallback_opt.param_groups):
                for param in group["params"]:
                    fallback_count += 1
                    fallback_params += param.numel()
                    print(f"  [{group_idx}] shape={list(param.shape)}, numel={param.numel():,}")

            print(f"\nTotal fallback parameters: {fallback_count} tensors, {fallback_params:,} elements")
            print(f"\nMuon coverage: {muon_params / (muon_params + fallback_params) * 100:.1f}% of parameters")
        else:
            print("\nNo fallback optimizer (all parameters use Muon)")

        print("=" * 70)

    @staticmethod
    def _compute_update_scale(
        m: int,
        n: int,
        mode: str,
        extra: float,
        update: Optional[Tensor] = None,
        target_rms: Optional[float] = None,
    ) -> float:
        """Compute per-parameter update scaling factor (NeMo-style).

        Args:
            m, n: Matrix dimensions.
            mode: Scaling mode ('spectral', 'shape', or 'rms_to_rms').
            extra: Additional multiplicative factor.
            update: Orthogonalized update tensor (required for ``rms_to_rms``).
            target_rms: Desired RMS magnitude for ``rms_to_rms`` scaling.

        Returns
        -------
        float
            Scale factor to apply to the orthogonalized update.
        """

        if mode == "shape":
            return extra * (max(1.0, m / n) ** 0.5)
        if mode == "spectral":
            return extra * (max(m, n) ** 0.5)
        if mode == "rms_to_rms":
            if update is None:
                return extra
            rms = update.pow(2).mean().sqrt().item()
            if rms == 0.0:
                return 0.0
            target = target_rms if target_rms is not None else rms
            return extra * (target / rms)
        return extra

    @staticmethod
    def _orthogonalize(update: Tensor, ns_iters: int, eps: float, tol: float, state: dict, ns_coefficients: str = "simple") -> Tensor:
        """Project the update onto the closest orthogonal matrix using Newton--Schulz.

        Args:
            update: Update matrix (already in FP32)
            ns_iters: Number of Newton-Schulz iterations
            eps: Regularization epsilon
            tol: Early-stop tolerance (size-normalized)
            state: Optimizer state dict for caching
            ns_coefficients: "simple" (3I-ZY) or "quintic" (faster convergence)

        Returns:
            Orthogonalized update in FP32
        """
        if update.ndim != 2:
            return update

        device = update.device
        dtype = torch.float32
        mat = update
        m, n = mat.shape

        if mat.abs().max().item() == 0.0:
            return update

        # Use the smaller dimension for the Gram matrix
        if m <= n:
            gram = mat @ mat.transpose(0, 1)
            size = m
            left_multiply = True
        else:
            gram = mat.transpose(0, 1) @ mat
            size = n
            left_multiply = False

        # EXACT PATH for small matrices (≤64): Cholesky/eig provide exact inverse sqrt
        if size <= 64:
            identity_small = torch.eye(size, device=device, dtype=dtype)
            gram_reg = gram + eps * identity_small
            try:
                chol, info = torch.linalg.cholesky_ex(gram_reg, upper=False)
                if int(info.item()) == 0:
                    inv_lower = torch.linalg.solve_triangular(
                        chol,
                        identity_small,
                        upper=False,
                        left=True,
                    )
                    inv_sqrt = inv_lower.transpose(0, 1) @ inv_lower
                    if left_multiply:
                        return inv_sqrt @ mat
                    return mat @ inv_sqrt
            except RuntimeError:
                pass

            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(gram_reg)
                eigenvalues = torch.clamp(eigenvalues, min=eps)
                inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
                inv_sqrt = eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0)
                inv_sqrt = inv_sqrt @ eigenvectors.transpose(0, 1)
                if left_multiply:
                    return inv_sqrt @ mat
                return mat @ inv_sqrt
            except RuntimeError:
                pass

        # Standard NS path for larger matrices
        adaptive_ns_iters = ns_iters
        if size < 128:
            adaptive_ns_iters = min(3, ns_iters)

        # Cache identity matrix
        ns_cache = state.get("ns_cache")
        if ns_cache is None:
            ns_cache = state["ns_cache"] = {}

        cache_key = (device, dtype, size)
        identity = ns_cache.get(cache_key)
        if identity is None:
            identity = torch.eye(size, device=device, dtype=dtype)
            ns_cache[cache_key] = identity

        coeff_schedule = _NS_POLYNOMIALS[ns_coefficients]

        def _ns_transform(current_zy: Tensor, iteration: int) -> Tensor:
            coeffs = coeff_schedule[min(iteration, len(coeff_schedule) - 1)]
            result = coeffs[0] * identity
            if len(coeffs) == 1:
                return result
            power = current_zy
            last_idx = len(coeffs) - 1
            for idx, coeff in enumerate(coeffs[1:], start=1):
                result = result + coeff * power
                if idx < last_idx:
                    power = power @ current_zy
            return result

        gram = gram + eps * identity
        eps_bumped = False

        while True:
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

            prev_residual = float("inf")
            restart = False

            for it in range(adaptive_ns_iters):
                transform = _ns_transform(zy, it)
                y = y @ transform
                z = transform @ z
                zy = z @ y

                if it > 0 and tol > 0.0 and (it % 2 == 1):
                    residual = torch.linalg.norm(identity - zy, ord="fro") / (size ** 0.5)

                    if not eps_bumped and it > 2:
                        if abs(residual - prev_residual) < tol * 0.1:
                            gram = gram + 3.0 * eps * identity
                            eps_bumped = True
                            restart = True
                            break

                    prev_residual = residual

                    if residual <= tol:
                        break

            if restart:
                continue
            break

        inv_sqrt = z * scale

        if left_multiply:
            orthogonal_update = inv_sqrt @ mat
        else:
            orthogonal_update = mat @ inv_sqrt

        # Return in FP32; caller will cast to param.dtype
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
