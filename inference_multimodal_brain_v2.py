"""Standalone inference script for Multimodal Brain v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from brain_v2_components import Preproc, build_brain
from multimodal_brain_v2 import ThinkControl


def _get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠ Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    return checkpoint if isinstance(checkpoint, dict) else {}


def _compute_pairwise(z_by_mod: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, list]]:
    modalities = list(z_by_mod.keys())
    results: Dict[str, Dict[str, list]] = {}
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i + 1 :]:
            z1 = F.normalize(z_by_mod[mod1], dim=-1)
            z2 = F.normalize(z_by_mod[mod2], dim=-1)
            sim = z1 @ z2.t()
            matrix = sim.cpu().tolist()
            diag = torch.diagonal(sim).cpu().tolist()
            results[f"{mod1}__{mod2}"] = {"matrix": matrix, "diag": diag}
    return results


def _compute_global_alignment(z_global: torch.Tensor, z_by_mod_out: Dict[str, torch.Tensor]) -> Dict[str, list]:
    alignments: Dict[str, list] = {}
    z_norm = F.normalize(z_global, dim=-1)
    for name, z in z_by_mod_out.items():
        align = F.cosine_similarity(z_norm, F.normalize(z, dim=-1), dim=-1)
        alignments[name] = align.cpu().tolist()
    return alignments


def run_inference(args: argparse.Namespace) -> Dict:
    device = _get_device(args.device)
    print(f"Using device: {device}")

    model = build_brain(
        d_shared=args.d_shared,
        device=device,
        freeze_text=not args.unfreeze_text,
        freeze_image=not args.unfreeze_image,
        train_audio_encoder=args.train_audio,
    )
    model.eval()

    checkpoint = _load_checkpoint(model, Path(args.checkpoint), device)
    print("Checkpoint loaded.")

    preproc = Preproc()
    inputs = preproc.prepare_user_inputs(
        texts=args.text,
        image_paths=args.image,
        audio_paths=args.audio,
        device=device,
    )

    encode_inputs = {k: v for k, v in inputs.items() if not k.startswith("_")}
    control = ThinkControl(steps=args.steps, mode=args.think_mode)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    if device.type == "cuda":
        use_autocast = args.precision in {"fp16", "bf16"}
        if args.precision == "fp16":
            dtype = torch.float16
        elif args.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        use_autocast = False
        dtype = torch.float32

    with torch.no_grad(), torch.amp.autocast(amp_device, enabled=use_autocast, dtype=dtype):
        z_by_mod = model.encode_inputs(encode_inputs)
        tokens, z_global, z_by_mod_out = model.think(z_by_mod, control)
        decoded = {}
        if args.request:
            decoded = model.decode_outputs(z_global, z_by_mod_out, args.request)

    pairwise_in = _compute_pairwise(z_by_mod)
    pairwise_out = _compute_pairwise(z_by_mod_out)
    global_align = _compute_global_alignment(z_global, z_by_mod_out)

    summary = {}
    for name, z in z_by_mod.items():
        summary[name] = {
            "mean": z.mean().item(),
            "std": z.std().item(),
            "norm": z.norm(dim=-1).mean().item(),
        }

    result = {
        "batch_size": next(iter(z_by_mod.values())).size(0),
        "modalities": list(z_by_mod.keys()),
        "pairwise_input_similarity": pairwise_in,
        "pairwise_output_similarity": pairwise_out,
        "global_alignment": global_align,
        "embedding_stats": summary,
        "raw_texts": inputs.get("_raw_texts"),
    }

    if decoded:
        result["decoded"] = {k: _to_serializable(v) for k, v in decoded.items()}
    if args.include_tokens:
        result["tokens"] = tokens.cpu().tolist()
    if args.include_latents:
        result["z_global"] = z_global.cpu().tolist()
        result["z_by_mod_out"] = {k: v.cpu().tolist() for k, v in z_by_mod_out.items()}

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(result, indent=2))
        print(f"Saved results to {args.save_json}")

    return result


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, (list, dict, str, float, int)) or value is None:
        return value
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference with Multimodal Brain v2")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--text", action="append", help="Text input. Repeat flag for multiple samples.")
    p.add_argument("--image", action="append", help="Path to image file. Repeat for multiple samples.")
    p.add_argument("--audio", action="append", help="Path to audio file. Repeat for multiple samples.")
    p.add_argument("--request", nargs="*", default=None, help="Modalities to decode if decoders are attached.")
    p.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, ...)")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Inference precision")
    p.add_argument("--d-shared", type=int, default=512, help="Shared latent dimensionality")
    p.add_argument("--steps", type=int, default=2, help="Number of thinking steps")
    p.add_argument("--think-mode", type=str, default="default", help="ThinkControl mode")
    p.add_argument("--unfreeze-text", action="store_true", help="Allow text encoder gradients (for experimentation)")
    p.add_argument("--unfreeze-image", action="store_true", help="Allow image encoder gradients")
    p.add_argument("--train-audio", action="store_true", help="Keep audio encoder trainable")
    p.add_argument("--include-tokens", action="store_true", help="Include intermediate tokens in JSON output")
    p.add_argument("--include-latents", action="store_true", help="Include latent tensors in JSON output")
    p.add_argument("--save-json", type=str, default=None, help="Optional path to save JSON results")
    return p


def main(argv: Optional[Iterable[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_inference(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
