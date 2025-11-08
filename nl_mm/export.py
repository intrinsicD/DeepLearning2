"""TorchScript/ONNX export helpers."""
from __future__ import annotations

import argparse
import torch

from .models.nl_mm_model import NLMM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NL-MM model")
    parser.add_argument("--config", type=str, default="nl_mm/configs/tiny_single_gpu.yaml")
    parser.add_argument("--output", type=str, default="nl_mm_torchscript.pt")
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


def export_model(cfg: dict, output: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLMM(cfg).to(device)
    model.eval()
    dummy = {"text": torch.randint(0, cfg.get("vocab_size", 32000), (1, 16), device=device)}

    class Wrapper(torch.nn.Module):
        def __init__(self, model: NLMM):
            super().__init__()
            self.model = model

        def forward(self, text: torch.Tensor):
            outputs, _ = self.model({"text": text})
            return outputs["text"]

    wrapper = Wrapper(model)
    scripted = torch.jit.trace(wrapper, dummy["text"])
    scripted.save(output)
    print(f"Exported TorchScript model to {output}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    export_model(cfg, args.output)


if __name__ == "__main__":
    main()
