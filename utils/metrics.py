"""Metric utilities."""
from __future__ import annotations
from typing import Dict, Any
import torch

def text_loss_metric(batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> float:
    loss = outputs.get('text') or outputs.get('loss')
    if torch.is_tensor(loss):
        return -float(loss.detach().cpu())  # Higher is better (lower loss)
    return 0.0

__all__ = ['text_loss_metric']

