"""Detailed profiling of MuonFast orthogonalization."""

import torch
import time
import numpy as np

# Test the orthogonalization on different matrix sizes
def test_orthogonalize():
    from src.optimizers.muon_fast import MuonFast
    
    # Test matrices of different sizes
    test_sizes = [
        (10, 128),    # fc2.weight
        (128, 3136),  # fc1.weight
    ]
    
    print("Testing orthogonalization performance:\n")
    
    for size in test_sizes:
        update = torch.randn(size)
        
        # Time the orthogonalization
        times = []
        for _ in range(10):
            start = time.time()
            result = MuonFast._orthogonalize(update, ns_iters=3, eps=1e-6)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Shape {str(size):15s}: {avg_time:7.2f} +/- {std_time:5.2f} ms")
        print(f"  Gram matrix size: {size[1]}x{size[1]}")
        print()

if __name__ == "__main__":
    test_orthogonalize()

