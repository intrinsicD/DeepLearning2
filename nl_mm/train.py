"""Training entry point for NL-MM."""
from __future__ import annotations

import argparse
import torch

from .models.nl_mm_model import NLMM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NL-MM model")
    parser.add_argument("--config", type=str, default="nl_mm/configs/tiny_single_gpu.yaml", help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        import json

        return json.loads(text)


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLMM(cfg).to(device)
    scheduler = model.configure_scheduler(cfg)
    model.train()

    model.zero_grad(set_to_none=True)
    for step in range(cfg.get("train_steps", 10)):
        tokens = torch.randint(0, cfg.get("vocab_size", 32000), (2, 16), device=device)
        targets = torch.randint(0, cfg.get("vocab_size", 32000), (2, 16), device=device)
        dummy = {"text": tokens, "text_target": targets}
        outputs, _ = model(dummy)
        loss = outputs["text"]
        loss.backward()
        updates = scheduler.step_all(step)
        if any(updates.values()):
            model.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["train_steps"] = args.steps
    train(cfg)


if __name__ == "__main__":
    main()
