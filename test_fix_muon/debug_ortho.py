"""Debug the orthogonalization to understand what's going wrong."""

import torch
import math

def debug_orthogonalization():
    """Debug step by step what happens in orthogonalization."""
    
    # Simple case: 3x5 matrix
    m, n = 3, 5
    update = torch.randn(m, n, dtype=torch.float32)
    
    print(f"Original update shape: {update.shape}")
    print(f"Update:\n{update}\n")
    
    # Current implementation
    if m <= n:
        gram = update @ update.T  # 3x3
        size = m
        left_multiply = True
        print(f"Using left multiplication (m <= n)")
    else:
        gram = update.T @ update  # nxn
        size = n
        left_multiply = False
        print(f"Using right multiplication (m > n)")
    
    print(f"Gram matrix shape: {gram.shape}")
    print(f"Gram:\n{gram}\n")
    
    eps = 1e-6
    gram = gram + eps * torch.eye(size)
    
    frob_norm = torch.linalg.norm(gram)
    print(f"Frobenius norm: {frob_norm:.6f}\n")
    
    identity = torch.eye(size)
    y = gram / frob_norm
    z = identity.clone()
    
    ns_iters = 5
    for i in range(ns_iters):
        t = 0.5 * (3.0 * identity - z @ y)
        y = y @ t
        z = t @ z
        print(f"Iteration {i+1}: ||z @ y - I|| = {torch.norm(z @ y - identity):.6f}")
    
    inv_sqrt = z / math.sqrt(frob_norm)
    print(f"\nInverse sqrt computed")
    print(f"inv_sqrt shape: {inv_sqrt.shape}\n")
    
    if left_multiply:
        result = inv_sqrt @ update
        print(f"Result = inv_sqrt @ update")
    else:
        result = update @ inv_sqrt
        print(f"Result = update @ inv_sqrt")
    
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}\n")
    
    # Check orthogonality
    if m <= n:
        product = result @ result.T
        identity_check = torch.eye(m)
        print(f"Checking result @ result.T ≈ I")
    else:
        product = result.T @ result
        identity_check = torch.eye(n)
        print(f"Checking result.T @ result ≈ I")
    
    print(f"Product:\n{product}\n")
    print(f"Expected identity:\n{identity_check}\n")
    error = torch.norm(product - identity_check)
    print(f"Orthogonality error: {error:.6e}\n")
    
    # What we actually want: normalize to unit norm in the appropriate direction
    # For Muon, we want the columns/rows to have unit norm
    # Let's check SVD-based orthogonalization
    print("=" * 60)
    print("Comparing with SVD-based orthogonalization:")
    U, S, Vt = torch.linalg.svd(update, full_matrices=False)
    svd_result = U @ Vt
    print(f"SVD result shape: {svd_result.shape}")
    
    if m <= n:
        svd_product = svd_result @ svd_result.T
        print(f"SVD: result @ result.T ≈ I")
    else:
        svd_product = svd_result.T @ svd_result
        print(f"SVD: result.T @ result ≈ I")
    
    svd_error = torch.norm(svd_product - identity_check)
    print(f"SVD orthogonality error: {svd_error:.6e}")

if __name__ == "__main__":
    debug_orthogonalization()

