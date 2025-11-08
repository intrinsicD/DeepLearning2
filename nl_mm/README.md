# NL-MM: Nested Learning Multimodal Reference Implementation

This package provides a compact, single-GPU friendly implementation of the
architecture described in the *Nested Learning* whitepaper.  The code mirrors
the conceptual decomposition of the paper:

- **Fast-weight linear attention** implements the associative memory updates of
  Eqs. (12–16).
- **Continuum Memory System (CMS)** exposes multiple residual MLP stacks with
  configurable update cadences as described by Eqs. (30–31).
- **Test-time training adapters** provide the on-line L2 regression updates
  (Eqs. (27–29)).
- **Deep Momentum Gradient Descent (DMGD)** treats the optimizer as a learnable
  memory, following Sec. 2.3.
- **Central latent memory** binds the modalities together via HOPE-style dynamic
  projections.

## Layout

```
nl_mm/
  configs/          # YAML configuration files
  data/             # Minimal multimodal dataset helpers
  modules/          # Core Nested Learning components
  models/           # Encoders, decoders, and the NL-MM wrapper
  train.py          # Training entry point
  evaluate.py       # Evaluation entry point
  export.py         # TorchScript export helper
```

## Quickstart

```bash
python -m nl_mm.train --config nl_mm/configs/tiny_single_gpu.yaml
python -m nl_mm.evaluate --config nl_mm/configs/tiny_single_gpu.yaml
python -m nl_mm.export --config nl_mm/configs/tiny_single_gpu.yaml --output nl_mm_torchscript.pt
```

All scripts default to CPU execution but automatically leverage CUDA if
available.  The configuration files are structured so that scaling to larger
models primarily involves increasing the model dimension, the memory length, and
adding more CMS levels.
