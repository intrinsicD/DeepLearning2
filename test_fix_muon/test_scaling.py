"""Test different scaling approaches for Muon orthogonalization."""

import torch
import math

def orthogonalize_original(update, ns_iters=5, eps=1e-6):
    """Original broken version - always uses mat.T @ mat."""
    mat = update.to(torch.float32)
    device = mat.device
    
    gram = mat.transpose(0, 1) @ mat
    n = gram.shape[0]
    gram = gram + eps * torch.eye(n, device=device)
    
    frob_norm = torch.linalg.norm(gram)
    identity = torch.eye(n, device=device)
    y = gram / frob_norm
    z = identity.clone()
    
    for _ in range(ns_iters):
        t = 0.5 * (3.0 * identity - z @ y)
        y = y @ t
        z = t @ z
    
    inv_sqrt = z / math.sqrt(frob_norm)
    return (mat @ inv_sqrt).to(update.dtype)

def orthogonalize_current(update, ns_iters=5, eps=1e-6):
    """Current version with dimension selection and trace scaling."""
    mat = update.to(torch.float32)
    device = mat.device
    m, n = mat.shape
    
    if m <= n:
        gram = mat @ mat.T
        size = m
        left_multiply = True
    else:
        gram = mat.T @ mat
        size = n
        left_multiply = False
    
    identity = torch.eye(size, device=device)
    gram = gram + eps * identity
    
    trace = torch.trace(gram)
    scale = torch.rsqrt(trace / float(size))
    scaled_gram = gram * (scale * scale)
    
    y = scaled_gram
    z = identity.clone()
    
    for _ in range(ns_iters):
        t = 0.5 * (3.0 * identity - z @ y)
        y = y @ t
        z = t @ z
    
    inv_sqrt = z * scale
    
    if left_multiply:
        result = (inv_sqrt @ mat).to(update.dtype)
    else:
        result = (mat @ inv_sqrt).to(update.dtype)
    
    return result

def test_scaling():
    """Test what scaling does to the updates."""
    
    # Test on fc1.weight shape
    update = torch.randn(128, 3136) * 0.001  # Typical scaled by lr
    
    print("Original update:")
    print(f"  Shape: {update.shape}")
    print(f"  Norm: {update.norm().item():.6f}")
    print(f"  Max abs: {update.abs().max().item():.6f}")
    print()
    
    ortho_current = orthogonalize_current(update)
    
    print("Current orthogonalization:")
    print(f"  Shape: {ortho_current.shape}")
    print(f"  Norm: {ortho_current.norm().item():.6f}")
    print(f"  Max abs: {ortho_current.abs().max().item():.6f}")
    print(f"  Norm ratio: {ortho_current.norm().item() / update.norm().item():.2f}")
    print()
    
    # What does the trace scaling do?
    mat = update.to(torch.float32)
    gram = mat @ mat.T
    trace = torch.trace(gram)
    size = 128
    scale = torch.rsqrt(trace / float(size))
    
    print(f"Trace: {trace.item():.6e}")
    print(f"Trace/size: {(trace/size).item():.6e}")
    print(f"Scale (rsqrt): {scale.item():.6f}")
    print()
    
    # Check if we need to scale by sqrt(n) or sqrt(m)
    print("Theoretical Frobenius norm of orthogonal matrix:")
    print(f"  For m×n orthogonal (m<=n): ||U|| = sqrt(m) = sqrt({min(update.shape)}) = {math.sqrt(min(update.shape)):.2f}")
    print(f"  Ortho norm / sqrt(m) = {ortho_current.norm().item() / math.sqrt(128):.6f}")

if __name__ == "__main__":
    test_scaling()

