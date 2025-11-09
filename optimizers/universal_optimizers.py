"""Universal MuonFast: Works on any architecture (CNNs, ViTs, RNNs, etc.)

Key improvements over original MuonFast:
1. Adam-style adaptive learning rates (second moment)
2. Adaptive orthogonalization (only when beneficial)
3. Magnitude-preserving orthogonalization option
4. Better scaling for small matrices
5. Automatic architecture detection and tuning
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class UniversalMuon(Optimizer):
    """Universal Muon optimizer that works on any architecture.
    
    Combines the best of Adam (adaptive LR) with Muon (orthogonalization)
    in a way that automatically adapts to the architecture.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: (beta1, beta2) for momentum and variance (default: (0.9, 0.999))
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        ortho_mode: "auto", "always", "never", or "preserve_magnitude" (default: "auto")
        ortho_threshold: Only orthogonalize if min_dim >= threshold (default: 128)
        ns_iters: Newton-Schulz iterations (default: 3)
        scale_mode: "spectral", "adaptive", or None (default: "adaptive")
    """
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ortho_mode: str = "auto",
        ortho_threshold: int = 128,
        ns_iters: int = 3,
        scale_mode: str = "adaptive",
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if ortho_mode not in ["auto", "always", "never", "preserve_magnitude"]:
            raise ValueError(f"Invalid ortho_mode: {ortho_mode}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ortho_mode=ortho_mode,
            ortho_threshold=ortho_threshold,
            ns_iters=ns_iters,
            scale_mode=scale_mode,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            ortho_mode = group["ortho_mode"]
            ortho_threshold = group["ortho_threshold"]
            ns_iters = group["ns_iters"]
            scale_mode = group["scale_mode"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Weight decay
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                
                # Update biased first moment (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment (variance)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Compute step direction (like Adam)
                step_direction = exp_avg / bias_correction1
                
                # Decide whether to orthogonalize
                should_orthogonalize = self._should_orthogonalize(
                    p, ortho_mode, ortho_threshold
                )
                
                if should_orthogonalize:
                    # Store original magnitude
                    original_norm = step_direction.norm().item()
                    
                    # Orthogonalize
                    step_direction = self._orthogonalize(step_direction, ns_iters, eps)
                    
                    # Preserve magnitude if requested
                    if ortho_mode == "preserve_magnitude":
                        new_norm = step_direction.norm().item()
                        if new_norm > 1e-8:
                            step_direction.mul_(original_norm / new_norm)
                
                # Compute adaptive step size (Adam-style)
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                step_size = lr / denom
                
                # Apply scaling if needed
                if scale_mode == "adaptive" and p.ndim == 2:
                    m, n = p.shape
                    min_dim = min(m, n)
                    # Adaptive scaling based on matrix size
                    if min_dim < 256:
                        # More conservative for small matrices
                        scale_factor = math.sqrt(min_dim / 256.0)
                        step_size = step_size * scale_factor
                elif scale_mode == "spectral" and p.ndim == 2:
                    m, n = p.shape
                    scale_factor = math.sqrt(max(m, n))
                    step_size = step_size * scale_factor
                
                # Apply update
                p.add_(step_direction * step_size, alpha=-1)
        
        return loss
    
    def _should_orthogonalize(
        self, 
        param: Tensor, 
        ortho_mode: str, 
        ortho_threshold: int
    ) -> bool:
        """Determine if parameter should be orthogonalized."""
        if ortho_mode == "never":
            return False
        if ortho_mode == "always":
            return param.ndim == 2
        
        # For preserve_magnitude and auto, only orthogonalize if meets threshold
        if param.ndim != 2:
            return False
        
        m, n = param.shape
        min_dim = min(m, n)
        
        # Only orthogonalize large enough matrices
        return min_dim >= ortho_threshold
    
    @staticmethod
    def _orthogonalize(update: Tensor, ns_iters: int, eps: float) -> Tensor:
        """Simple Newton-Schulz orthogonalization."""
        if update.ndim != 2:
            return update
        
        m, n = update.shape
        device, dtype = update.device, update.dtype
        
        # Use smaller dimension for Gram matrix
        if m <= n:
            gram = update @ update.T
            size = m
            left_multiply = True
        else:
            gram = update.T @ update
            size = n
            left_multiply = False
        
        # Add regularization
        identity = torch.eye(size, device=device, dtype=dtype)
        gram = gram + eps * identity
        
        # Normalize
        trace = torch.trace(gram)
        if trace <= 0:
            return update
        
        scale = torch.rsqrt(trace / size)
        scaled_gram = gram * (scale * scale)
        
        # Newton-Schulz iterations
        Y = scaled_gram
        Z = identity.clone()
        
        for _ in range(ns_iters):
            ZY = Z @ Y
            Y = 1.5 * Y - 0.5 * Y @ ZY
            Z = 1.5 * Z - 0.5 * ZY @ Z
        
        inv_sqrt = Z * scale
        
        # Apply to update
        if left_multiply:
            return inv_sqrt @ update
        else:
            return update @ inv_sqrt


class UniversalAndersonGDA(Optimizer):
    """Universal Anderson GDA that works on any architecture.
    
    Combines Adam with gradient-aware Anderson acceleration and safety checks.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: (beta1, beta2) for momentum and variance (default: (0.9, 0.999))
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        anderson_m: Anderson memory size (default: 3)
        anderson_reg: Base regularization for Anderson (default: 1e-3)
        trust_region: Trust region coefficient (default: 1.5)
        use_weighting: Use gradient-curvature weighting (default: True)
    """
    
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        anderson_m: int = 3,
        anderson_reg: float = 1e-3,
        trust_region: float = 1.5,
        use_weighting: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if anderson_m < 0:
            raise ValueError(f"Invalid anderson_m: {anderson_m}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            trust_region=trust_region,
            use_weighting=use_weighting,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            anderson_m = group["anderson_m"]
            anderson_reg = group["anderson_reg"]
            trust_region = group["trust_region"]
            use_weighting = group["use_weighting"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["grad_history"] = []
                    state["step_history"] = []
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Weight decay
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                
                # Update biased first moment (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment (variance)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Compute base Adam step
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                base_step = -lr * (exp_avg / bias_correction1) / denom
                
                # Apply Anderson acceleration if we have history
                if anderson_m > 0 and state["step"] > 1:
                    base_step = self._anderson_accelerate(
                        base_step=base_step,
                        grad=grad,
                        state=state,
                        anderson_m=anderson_m,
                        anderson_reg=anderson_reg,
                        trust_region=trust_region,
                        use_weighting=use_weighting,
                    )
                
                # Update history
                state["grad_history"].append(grad.clone())
                state["step_history"].append(base_step.clone())
                if len(state["grad_history"]) > anderson_m + 1:
                    state["grad_history"].pop(0)
                    state["step_history"].pop(0)
                
                # Apply update
                p.add_(base_step)
        
        return loss
    
    def _anderson_accelerate(
        self,
        base_step: Tensor,
        grad: Tensor,
        state: dict,
        anderson_m: int,
        anderson_reg: float,
        trust_region: float,
        use_weighting: bool,
    ) -> Tensor:
        """Apply Anderson acceleration with safety checks."""
        grad_history = state["grad_history"]
        step_history = state["step_history"]
        
        # Need at least 2 history entries
        if len(grad_history) < 2:
            return base_step
        
        # Flatten for linear algebra
        base_step_flat = base_step.flatten()
        grad_flat = grad.flatten()
        
        # Build residual matrix (differences)
        pairs = min(len(step_history) - 1, anderson_m)
        if pairs == 0:
            return base_step
        
        residuals = []
        grad_diffs = []
        weights = []
        
        for i in range(-pairs, 0):
            # Residual: difference in steps
            res = (step_history[i] - step_history[i-1]).flatten()
            residuals.append(res)
            
            # Gradient difference for weighting
            if use_weighting:
                g_diff = (grad_history[i] - grad_history[i-1]).flatten()
                grad_diffs.append(g_diff)
                
                # Compute weight based on gradient curvature
                s_norm = res.norm().item()
                y_norm = g_diff.norm().item()
                
                if s_norm > 1e-8 and y_norm > 1e-8:
                    # Cosine similarity
                    cos = torch.dot(res, g_diff).item() / (s_norm * y_norm)
                    cos = max(0.0, cos)
                    
                    # Inverse curvature estimate
                    inv_curv = s_norm / y_norm
                    inv_curv = max(0.1, min(10.0, inv_curv))
                    
                    weight = max(1e-8, cos * inv_curv)
                else:
                    weight = 1e-8
                
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Stack residuals into matrix
        R = torch.stack(residuals, dim=1)  # (n_params, pairs)
        
        # Build system: R^T R θ = R^T (current_step - last_step)
        current_res = (base_step_flat - step_history[-1].flatten())
        gram = R.T @ R
        rhs = R.T @ current_res
        
        # Apply weighting
        if use_weighting:
            w = torch.tensor(weights, dtype=gram.dtype, device=gram.device)
            w = w / (w.mean() + 1e-8)
            W = torch.diag(1.0 / (w + 1e-8))
            gram = gram + anderson_reg * W
        else:
            gram = gram + anderson_reg * torch.eye(pairs, device=gram.device, dtype=gram.dtype)
        
        # Solve for coefficients
        try:
            theta = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            # Fallback to base step if solve fails
            return base_step
        
        # Compute correction
        correction = (R @ theta).view_as(base_step)
        
        # Trust region: clip if too large
        base_norm = base_step.norm().item()
        corr_norm = correction.norm().item()
        
        if corr_norm > trust_region * base_norm and base_norm > 1e-8:
            correction.mul_(trust_region * base_norm / corr_norm)
        
        # Apply correction with descent check
        accelerated = base_step - correction
        
        # Safety: check if it's a descent direction
        if torch.dot(grad_flat, accelerated.flatten()) >= 0:
            # Not a descent direction, use base step
            return base_step
        
        return accelerated


# Backward compatibility aliases
MuonUniversal = UniversalMuon
AndersonGDAUniversal = UniversalAndersonGDA

