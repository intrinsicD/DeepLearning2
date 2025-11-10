from __future__ import annotations

import torch

from modules.nl_mm.models.nl_mm_model import NLMM


def _tiny_cfg() -> dict:
    return {
        "d_model": 16,
        "n_heads": 2,
        "ffn_mult": 2,
        "L_mem": 4,
        "depth": {"text": 1, "image": 1, "audio": 1},
        "cms_levels": [
            {"name": "fast", "chunk_size": 1, "lr": 1e-3, "optimizer": "dmgd"},
            {"name": "slow", "chunk_size": 4, "lr": 5e-4, "optimizer": "adamw"},
        ],
        "optimizer": {
            "dmgd": {"lr": 1e-3, "beta": 0.9},
            "adamw": {"lr": 5e-4, "weight_decay": 0.0},
        },
        "ttt": {"enable": False, "adapter_rank": 4, "eta": 1e-3},
        "vocab_size": 64,
        "max_position_embeddings": 32,
        "image_channels": 3,
        "image_resolution": 8,
        "patch_size": 4,
        "audio_channels": 1,
        "audio_mel_bins": 8,
    }


def test_nlmm_forward_backward_all_modalities():
    cfg = _tiny_cfg()
    model = NLMM(cfg)
    scheduler = model.configure_scheduler(cfg)

    assigned_ids = []
    for state in scheduler._level_states.values():  # type: ignore[attr-defined]
        assigned_ids.extend(id(p) for p in state.spec.params)

    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids) == {id(p) for p in model.parameters()}

    batch_size = 2
    text = torch.randint(0, cfg["vocab_size"], (batch_size, 5))
    image = torch.randn(batch_size, cfg["image_channels"], cfg["image_resolution"], cfg["image_resolution"])
    audio = torch.randn(batch_size, cfg["audio_channels"], 16)

    batch = {
        "text": text,
        "text_target": torch.randint(0, cfg["vocab_size"], text.shape),
        "image": image,
        "image_target": torch.randn_like(image),
        "audio": audio,
        "audio_target": torch.randn(batch_size, cfg["audio_mel_bins"]),
    }

    outputs, _ = model(batch)
    loss = outputs["text"] + outputs["image"] + outputs["audio"]
    loss.backward()

    nonzero_grads = sum(1 for p in model.parameters() if p.grad is not None)
    assert nonzero_grads > 0

    stepped = scheduler.step_all(0)
    assert stepped["fast"] is True
    assert stepped["slow"] is False
