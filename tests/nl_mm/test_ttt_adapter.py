from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.nl_mm.modules.ttt import TTTAdapter
from modules.nl_mm.models.nl_mm_model import NLMM


def _minimal_config():
    return {
        "d_model": 32,
        "n_heads": 2,
        "ffn_mult": 2,
        "cms_levels": [
            {"name": "fast", "chunk_size": 1, "lr": 1e-3, "optimizer": "dmgd"},
        ],
        "depth": {"text": 1, "image": 0, "audio": 0},
        "ttt": {"enable": True, "adapter_rank": 4, "eta": 1e-3},
        "L_mem": 2,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "image_channels": 3,
        "image_resolution": 8,
        "patch_size": 4,
        "audio_channels": 1,
        "audio_mel_bins": 16,
    }


def test_ttt_adapter_step_updates_weights():
    adapter = TTTAdapter(d_model=32, rank=4, eta=1e-2)
    x = torch.randn(2, 3, 32)
    grad = torch.randn_like(x)

    inp_before = adapter.inp.weight.clone()
    out_before = adapter.out.weight.clone()

    adapter.ttt_step(x, grad)

    assert adapter.inp.weight.shape == inp_before.shape
    assert adapter.out.weight.shape == out_before.shape
    assert not torch.allclose(adapter.inp.weight, inp_before)
    assert not torch.allclose(adapter.out.weight, out_before)


def test_nlmm_forward_allows_ttt_in_eval_mode():
    cfg = _minimal_config()
    model = NLMM(cfg)
    model.eval()

    batch = {
        "text": torch.randint(0, cfg["vocab_size"], (2, 5)),
        "text_target": torch.randint(0, cfg["vocab_size"], (2, 5)),
    }

    with torch.no_grad():
        outputs, _ = model(batch, enable_ttt=True)

    assert "text" in outputs
    assert outputs["text"].ndim == 0
