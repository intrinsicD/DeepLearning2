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
modules/nl_mm/
  configs/          # JSON configuration files
  modules/          # Core Nested Learning components
  models/           # Encoders, decoders, and the NL-MM wrapper
  train.py          # Training entry point
  evaluate.py       # Evaluation entry point
  export.py         # TorchScript export helper
```

## Quickstart

```bash
python -m modules.nl_mm.train --config modules/nl_mm/configs/tiny_single_gpu.yaml
python -m modules.nl_mm.evaluate --config modules/nl_mm/configs/tiny_single_gpu.yaml
python -m modules.nl_mm.export --config modules/nl_mm/configs/tiny_single_gpu.yaml --output nl_mm_torchscript.pt
```

All scripts default to CPU execution but automatically leverage CUDA if
available.  The configuration files are structured so that scaling to larger
models primarily involves increasing the model dimension, the memory length, and
adding more CMS levels.

## Training workflow

The training entrypoint is [`modules/nl_mm/train.py`](train.py).  It constructs an
`NLMM` instance from the configuration, applies the paper-specific fast weight
initialization, and then iterates a simple synthetic training loop while calling
the model-configured optimizer and scheduler stack.【F:modules/nl_mm/train.py†L1-L35】

1. **Install dependencies.**  Install PyTorch (CUDA build if you plan to use a
   GPU) together with the Python packages declared in [`requirements.txt`](../requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
   The configuration loader supports either JSON or YAML; installing `pyyaml`
   allows the sample YAML config to be parsed.【F:modules/nl_mm/utils.py†L1-L22】
2. **Select a configuration.**  The default
   [`configs/tiny_single_gpu.yaml`](configs/tiny_single_gpu.yaml) file specifies
   a 512-dimensional model with three CMS levels (fast, mid, slow) and enables
   AMP in bfloat16.  The JSON keys map directly to the keyword arguments of
   `NLMM`, e.g. `d_model`, `n_heads`, `ffn_mult`, memory length `L_mem`, per
   modality depths, and optimizer hyper-parameters.【F:modules/nl_mm/configs/tiny_single_gpu.yaml†L1-L24】
   - The `cms_levels` list defines the continuum memory update cadence and the
     optimizer used for each level.
   - The `ttt` block toggles test-time training adapters and their rank.
   - The `precision` block sets the automatic mixed precision policy and gradient clipping.
3. **Run training.**  Invoke the module with the chosen config and desired step
   count:
   ```bash
   python -m modules.nl_mm.train --config modules/nl_mm/configs/tiny_single_gpu.yaml --steps 200
   ```
   - The script automatically selects `cuda` when available; otherwise it runs on CPU.【F:modules/nl_mm/train.py†L18-L23】
   - `--steps` overrides `train_steps` inside the configuration, allowing quick
     experiments without modifying the file.【F:modules/nl_mm/train.py†L12-L34】
   - The reference implementation currently feeds randomly sampled token ids in
     place of a real dataloader; integrate your dataset by replacing the dummy
     batch creation (`torch.randint` calls) with modality-specific tensors.
4. **Monitor optimization.**  The `NLMM.configure_scheduler` helper builds the
   composite optimizer state (AdamW for slow weights + DMGD for fast weights) as
   defined in the configuration.  Each iteration backpropagates the cross-entropy
   loss returned in `outputs["text"]` and calls `scheduler.step_all(step)` to
   advance every optimizer in lockstep.【F:modules/nl_mm/train.py†L24-L33】

### Extending the training loop

- **Custom batches.**  Replace the placeholder token sampling with tensors drawn
  from your multimodal pipeline, respecting the dictionary structure expected by
  `NLMM.forward`.  For example, supply `{"text": text_tokens, "image": image_latents, "audio": spectrogram}` and optional `*_target`
  entries for supervised heads.
- **Longer runs.**  Adjust `train_steps`, batch size, and sequence length in the
  config file to match available hardware.  The default configuration uses short
  sequences (`(2, 16)` in the example loop) suitable for smoke tests.
- **Precision/AMP.**  Tweak the `precision` dictionary if your accelerator lacks
  bfloat16 support; valid values include `fp16`, `bf16`, or `null` to disable AMP.

## Evaluation and inference

Evaluation is orchestrated by [`modules/nl_mm/evaluate.py`](evaluate.py).  It instantiates
the same configuration, switches the model to evaluation mode, and performs a
forward pass with optionally enabled test-time training adapters.【F:modules/nl_mm/evaluate.py†L1-L33】

```bash
python -m modules.nl_mm.evaluate --config modules/nl_mm/configs/tiny_single_gpu.yaml
```

- Pass `--config` to reuse checkpoints trained with alternate hyper-parameters.
- The script prints the tensor shapes of the returned modalities so you can
  verify the expected logits.  Replace the dummy text tensor with real inputs to
  evaluate your trained model.
- To disable adapters during evaluation, set `"enable": false` inside the `ttt`
  block of your configuration.【F:modules/nl_mm/evaluate.py†L18-L25】

## Exporting for deployment

[`modules/nl_mm/export.py`](export.py) traces the model with TorchScript for deployment.
It wraps `NLMM` in a lightweight module that exposes a single `forward(text)`
method and saves the traced graph to the path supplied via `--output`.【F:modules/nl_mm/export.py†L1-L35】

```bash
python -m modules.nl_mm.export --config modules/nl_mm/configs/tiny_single_gpu.yaml --output nl_mm_torchscript.pt
```

- Ensure the configuration matches the weights of the trained model before
  tracing.  Load your checkpoint into the `NLMM` instance prior to wrapping if
  you have persisted weights.
- The dummy trace input uses a `(1, 16)` text sequence; adjust this to the shape
  you intend to serve (TorchScript captures tensor ranks and dtypes).
- The exported artifact can be loaded back with `torch.jit.load("nl_mm_torchscript.pt")`
  and called with a token tensor of the same shape/dtype as used during tracing.
