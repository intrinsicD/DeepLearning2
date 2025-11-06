"""
Device management utilities for GPU/CUDA support.
"""

import torch


def get_device(prefer_cuda=True):
    """
    Get the best available device (CUDA or CPU).
    
    Args:
        prefer_cuda (bool): If True, prefer CUDA when available
        
    Returns:
        torch.device: The selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def print_gpu_info():
    """Print detailed information about available GPU(s)."""
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Information:")
        print("=" * 60)
        print(f"CUDA Available: Yes")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
        
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("GPU Information:")
        print("=" * 60)
        print("CUDA Available: No")
        print("Running on CPU")
        print("=" * 60)


def get_memory_usage(device):
    """
    Get current memory usage for a device.
    
    Args:
        device (torch.device): The device to check
        
    Returns:
        dict: Dictionary with memory usage statistics (in MB)
    """
    if device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated(device) / 1024**2,
            'cached': torch.cuda.memory_reserved(device) / 1024**2,
            'total': torch.cuda.get_device_properties(device).total_memory / 1024**2
        }
    return {'allocated': 0, 'cached': 0, 'total': 0}


def clear_gpu_memory():
    """Clear cached GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
