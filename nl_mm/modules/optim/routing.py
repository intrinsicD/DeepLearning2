"""Optimizer routing utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from .d_mgd import DMGD


def split_parameters(module: nn.Module, min_dim: int) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    dmgd_params: List[nn.Parameter] = []
    adam_params: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim > 1 and min(param.shape) >= min_dim:
            dmgd_params.append(param)
        else:
            adam_params.append(param)
    return dmgd_params, adam_params


def build_optimizers(module: nn.Module, cfg: Dict) -> Dict[str, torch.optim.Optimizer]:
    dmgd_params, adam_params = split_parameters(module, cfg["routing"]["small_param_min_dim"])
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    if dmgd_params:
        dm_cfg = cfg["optimizer"]["dmgd"]
        optimizers["dmgd"] = DMGD(dmgd_params, lr=dm_cfg["lr"], beta=dm_cfg.get("beta", 0.9), nonlinearity=dm_cfg.get("nonlinearity", "none"))
    if adam_params:
        ad_cfg = cfg["optimizer"]["adamw"]
        optimizers["adamw"] = torch.optim.AdamW(adam_params, lr=ad_cfg["lr"], weight_decay=ad_cfg.get("weight_decay", 0.0))
    return optimizers
