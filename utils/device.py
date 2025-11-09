"""Device utilities (refactored)."""
import torch

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['get_device']

