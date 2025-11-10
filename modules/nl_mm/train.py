"""Training entry point for NL-MM."""
from __future__ import annotations

import argparse
import torch

from .init import apply_nlmm_init
from .models.nl_mm_model import NLMM
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NL-MM model")
    parser.add_argument("--config", type=str, default="modules/nl_mm/configs/tiny_single_gpu.yaml", help="Path to configuration file")
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLMM(cfg).to(device)
    apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))
    scheduler = model.configure_scheduler(cfg)
    model.train()

    for step in range(cfg.get("train_steps", 10)):
        tokens = torch.randint(0, cfg.get("vocab_size", 32000), (2, 16), device=device)
        targets = torch.randint(0, cfg.get("vocab_size", 32000), (2, 16), device=device)
        dummy = {"text": tokens, "text_target": targets}
        outputs, _ = model(dummy)
        loss = outputs["text"]
        loss.backward()
        scheduler.step_all(step)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["train_steps"] = args.steps
    train(cfg)


if __name__ == "__main__":
    main()
