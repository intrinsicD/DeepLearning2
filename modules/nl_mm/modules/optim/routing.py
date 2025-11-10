"""Optimizer routing utilities."""
from __future__ import annotations

from typing import Callable, Dict, Iterable

import torch
from torch import nn

from .d_mgd import DMGD

OptimizerFactory = Callable[[Iterable[nn.Parameter], float | None], torch.optim.Optimizer]


def build_optimizer_factories(cfg: Dict) -> Dict[str, OptimizerFactory]:
    """Return callables that instantiate optimizers on demand.

    Each factory creates a new optimizer instance scoped to a level so that
    slower levels can accumulate gradients without interference from faster
    updates.
    """

    factories: Dict[str, OptimizerFactory] = {}

    if "dmgd" in cfg.get("optimizer", {}):
        dm_cfg = cfg["optimizer"]["dmgd"]

        def make_dmgd(params: Iterable[nn.Parameter], override_lr: float | None = None) -> torch.optim.Optimizer:
            params = list(params)
            if not params:
                raise ValueError("DMGD optimizer requires at least one parameter")
            lr = override_lr if override_lr is not None else dm_cfg["lr"]
            return DMGD(
                params,
                lr=lr,
                beta=dm_cfg.get("beta", 0.9),
                nonlinearity=dm_cfg.get("nonlinearity", "none"),
                learnable_modulation=dm_cfg.get("learnable_modulation", True),
                mlp_lr=dm_cfg.get("mlp_lr", 1e-2),
            )

        factories["dmgd"] = make_dmgd

    if "adamw" in cfg.get("optimizer", {}):
        ad_cfg = cfg["optimizer"]["adamw"]

        def make_adamw(params: Iterable[nn.Parameter], override_lr: float | None = None) -> torch.optim.Optimizer:
            params = list(params)
            if not params:
                raise ValueError("AdamW optimizer requires at least one parameter")
            lr = override_lr if override_lr is not None else ad_cfg["lr"]
            return torch.optim.AdamW(params, lr=lr, weight_decay=ad_cfg.get("weight_decay", 0.0))

        factories["adamw"] = make_adamw

    return factories
