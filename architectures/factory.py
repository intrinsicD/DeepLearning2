"""Factory to build models by name to plug into NeuralNet."""
from __future__ import annotations
from typing import Dict

from nl_mm.models.nl_mm_model import NLMM
from nl_mm.utils import load_config
from nl_mm.init import apply_nlmm_init


def build_model(name: str, **kwargs):
    if name.lower() in {"nlmm", "nl-mm"}:
        cfg_path = kwargs.get('config', 'nl_mm/configs/nano_8gb.yaml')
        cfg: Dict = load_config(cfg_path)
        model = NLMM(cfg)
        apply_nlmm_init(model, cfg.get('depth', {}), cfg.get('arch_kind', 'encoder'))
        return model
    raise ValueError(f"Unknown model: {name}")

__all__ = ['build_model']

