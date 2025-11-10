"""Evaluation utilities for NL-MM."""
from __future__ import annotations

import argparse
import torch

from .models.nl_mm_model import NLMM
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the NL-MM model")
    parser.add_argument("--config", type=str, default="modules/nl_mm/configs/tiny_single_gpu.yaml")
    return parser.parse_args()


def evaluate(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLMM(cfg).to(device)
    model.eval()
    with torch.no_grad():
        dummy = {"text": torch.randint(0, cfg.get("vocab_size", 32000), (1, 16), device=device)}
        outputs, _ = model(dummy, enable_ttt=cfg["ttt"]["enable"])
        print({k: v.shape if torch.is_tensor(v) else v for k, v in outputs.items()})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
