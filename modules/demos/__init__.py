"""Interactive demos showcasing core architectures."""

from importlib import import_module
from types import ModuleType

__all__ = ['demo_multimodal_memory']

demo_multimodal_memory: ModuleType = import_module('.demo_multimodal_memory', __name__)
