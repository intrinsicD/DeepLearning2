"""Aggregate package for experiments, demos, and specialized modules."""

from importlib import import_module
from types import ModuleType
import sys

__all__ = ['experiments', 'demos', 'nl_mm']

# Provide convenient accessors for subpackages.
experiments: ModuleType = import_module('.experiments', __name__)
demos: ModuleType = import_module('.demos', __name__)
nl_mm: ModuleType = import_module('.nl_mm', __name__)

# Backward compatibility alias so legacy code importing ``nl_mm`` still works.
sys.modules.setdefault('nl_mm', nl_mm)
