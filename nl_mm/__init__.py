"""Nested Learning Multimodal (NL-MM) package.

This package exposes a reference implementation of the architecture described
in the "Nested Learning" whitepaper.  The code is organized so that each
component of the theoretical framework has a direct analogue in code.  The
modules are intentionally light-weight to keep the single-GPU training target
easy to satisfy while also being amenable to future scaling via DDP/ZeRO.
"""

from .models.nl_mm_model import NLMM
from .modules.nl_core import LevelSpec, NLScheduler

__all__ = ["NLMM", "LevelSpec", "NLScheduler"]
