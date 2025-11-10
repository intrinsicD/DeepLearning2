"""Test to understand the correct Muon orthogonalization."""

import torch
import math

def test_orthogonalization_direction():
    """
    The Muon optimizer should orthogonalize the update matrix.
    For a matrix U of shape (m, n), there are two ways to orthogonalize:
    
    1. Orthogonalize rows: U @ (U^T @ U)^{-1/2} - gives shape (m, n), needs n x n matrix
    2. Orthogonalize columns: (U @ U^T)^{-1/2} @ U - gives shape (m, n), needs m x m matrix
    
    The correct choice depends on which dimension is smaller to minimize computation.
    Typically for weight matrices in neural networks, we want to orthogonalize in the
    direction of the smaller dimension.
    """
    
    # Simulate fc1.weight: 128 x 3136
    m, n = 128, 3136
    print(f"Matrix shape: {m} x {n}")
    print(f"  If we use U^T @ U: creates {n} x {n} = {n*n} elements Gram matrix")
    print(f"  If we use U @ U^T: creates {m} x {m} = {m*m} elements Gram matrix")
    print(f"  Ratio: {(n*n)/(m*m):.1f}x larger!\n")
    
    # The correct approach: use the smaller dimension
    # For a m x n matrix where m < n, we should compute U @ U^T
    
    update = torch.randn(m, n)
    
    print("Testing both approaches:")
    
    # Wrong approach (current implementation)
    import time
    start = time.time()
    gram_wrong = update.transpose(0, 1) @ update  # n x n
    end = time.time()
    print(f"  U^T @ U: shape {gram_wrong.shape}, time: {(end-start)*1000:.2f}ms")
    
    # Correct approach  
    start = time.time()
    gram_correct = update @ update.transpose(0, 1)  # m x m
    end = time.time()
    print(f"  U @ U^T: shape {gram_correct.shape}, time: {(end-start)*1000:.2f}ms")
    
    print("\nConclusion: We should use the SMALLER dimension for the Gram matrix!")

if __name__ == "__main__":
    test_orthogonalization_direction()

