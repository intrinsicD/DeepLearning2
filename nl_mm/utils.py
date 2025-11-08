"""Shared utilities for NL-MM tooling."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a configuration dictionary from YAML or JSON."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return json.loads(text)

    data = yaml.safe_load(text)
    if data is None:
        raise ValueError(f"Configuration file {config_path} is empty")
    return data
