# Nested Learning Multimodal Architecture – Code Review

## Summary
This review covers the implementation under `modules/nl_mm/`, with emphasis on the fast-weight attention, fusion stack, schedulers, and test-time training adapter. I inspected the code paths exercised during standard training, evaluation, and TorchScript export.

## Critical Bugs
1. **Shape error in `TTTAdapter.ttt_step`** – The update for the low-rank input projection computes `dW_in` via `torch.einsum("bo,bi->ob", ...)`, which yields a tensor of shape `(rank, batch\*tokens)` instead of `(rank, d_model)`. When `batch * tokens != d_model`, the subsequent in-place update `self.inp.weight.add_(-self.eta * dW_in)` fails with a dimension mismatch. The einsum indices should mirror a matrix multiply over the batch dimension (e.g. `"bo,bi->oi"`). 【F:modules/nl_mm/modules/ttt.py†L33-L49】

## Additional Observations
- The DMGD optimizer hard-codes a detached MLP whose parameters never receive gradients. If the intent was to meta-learn the modulation, gradients must be enabled explicitly. As written, the modulation stays at initialization and merely rescales momentum by a learned constant.
- The scheduler assigns all parameters that are not part of a `ContinuumMLP` block to the fastest level. Verify that this matches the intended cadence for modality encoders/decoders and the CLM memory.

## Recommendations
- Fix the einsum indices in `TTTAdapter.ttt_step` and add a regression test that exercises `enable_ttt=True` during evaluation to surface tensor shape mistakes.
- Consider documenting or reworking the DMGD helper MLP if adaptive modulation is required; today it acts as a frozen random transform.
- Add assertions that every modality required for fusion provides a mask with shape `(batch, tokens)` when masking is enabled.
