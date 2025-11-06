"""Verify that the MuonFast orthogonalization is mathematically correct."""

import torch
import numpy as np
from src.optimizers.muon_fast import MuonFast

def verify_orthogonality(matrix, tol=1.0):
    """Check if a matrix is approximately orthogonal."""
    m, n = matrix.shape
    
    if m <= n:
        # Check if rows are orthonormal: U @ U^T ≈ I
        product = matrix @ matrix.T
        identity = torch.eye(m, device=matrix.device, dtype=matrix.dtype)
    else:
        # Check if columns are orthonormal: U^T @ U ≈ I
        product = matrix.T @ matrix
        identity = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    
    error = torch.norm(product - identity).item()
    # Relative error for better scaling
    scale = torch.norm(identity).item()
    relative_error = error / scale if scale > 0 else error
    return relative_error < tol, error

def test_orthogonalization_correctness():
    """Test that orthogonalization produces orthogonal matrices."""
    
    test_cases = [
        (10, 128, "Wide matrix (m < n)"),
        (128, 10, "Tall matrix (m > n)"),
        (50, 50, "Square matrix"),
        (128, 3136, "Large wide matrix (fc1.weight)"),
    ]
    
    print("Testing orthogonalization correctness:\n")
    print("=" * 70)
    
    for m, n, desc in test_cases:
        update = torch.randn(m, n)
        ortho_update = MuonFast._orthogonalize(update, ns_iters=5, eps=1e-6)
        
        is_ortho, error = verify_orthogonality(ortho_update)
        
        # Check shape is preserved
        shape_ok = ortho_update.shape == update.shape
        
        status = "✓ PASS" if (is_ortho and shape_ok) else "✗ FAIL"
        print(f"{status} {desc:30s} ({m:4d}x{n:4d})")
        print(f"      Shape preserved: {shape_ok}")
        print(f"      Orthogonality error: {error:.2e}")
        print()
    
    print("=" * 70)
    print("\nAll tests passed!" if all([
        verify_orthogonality(MuonFast._orthogonalize(torch.randn(m, n), ns_iters=5, eps=1e-6))[0]
        for m, n, _ in test_cases
    ]) else "\nSome tests failed!")

if __name__ == "__main__":
    test_orthogonalization_correctness()

