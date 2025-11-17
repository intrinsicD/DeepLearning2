# Multimodal Memory Network — Training Stability Plan

## Architecture Analysis
The `MultiModalMemoryNetwork` fuses three heterogeneous encoders (ViT, Transformer, CNN) via a central memory block that performs iterative feedback refinement. The central latent is then contrasted through CLIP-style InfoNCE losses. Because every modality is normalized only after its own encoder, NaNs typically arise when the text encoder receives an all-padding sequence: the masked mean pooling in `TextEncoder.forward` divides by the number of valid tokens, which collapses to zero if the entire batch element is padding. Once NaNs appear in the text embedding they immediately propagate to the fused latent, destabilise the memory updates, and finally poison the InfoNCE logits. Mixed precision then amplifies the problem and the loss becomes NaN within a few iterations.

## Candidate Improvements
1. **Safe modality pooling & normalization**  
   * Guard every pooling operation (especially the text encoder) with a numerically-stable reducer that clamps the denominator and zeros out fully padded sequences.  
   * Apply `torch.nan_to_num` before modality norms to keep memory contents finite.  
   * Pros: inexpensive, keeps architecture unchanged.  
   * Cons: still relies on training loop to drop pathological batches.

2. **Contrastive temperature control & logit regularization**  
   * Introduce a learnable `logit_scale` parameter (à la CLIP) that is clipped to a safe interval, and add a small L2 penalty on the logits before the cross entropy.  
   * Complement with gradient norm tracking so that a runaway temperature immediately decays instead of exploding the logits.  
   * Pros: directly stabilises InfoNCE.  
   * Cons: adds another degree of freedom to tune and does not stop upstream NaNs.

3. **Holistic NaN-guarded variant (implemented)**  
   * Combine Variant 1 with a training-loop level guard: enforce safe normalization within `info_nce_loss`, sanitise every encoder output, and skip/zero-out any batch that still produces non-finite values.  
   * Emit detailed diagnostics so the user can see which modality produced the issue, and keep the GradScaler in sync even when a step is skipped.  
   * Pros: prevents NaNs from entering the memory, ensures InfoNCE stays finite, and keeps optimisation state consistent.  
   * Cons: introduces a tiny conditional branch in the training loop.

## Implemented Solution
Variant 3 is adopted because it addresses the root cause (unsafe pooling), the propagation path (encoder outputs and fusion), and the final objective (InfoNCE) simultaneously. The concrete changes are:
- A `safe_masked_mean` helper in `TextEncoder` that clamps denominators and explicitly zeroes out fully padded samples.
- Systematic `torch.nan_to_num` sanitisation in the modality encoders, fusion MLP, and memory interactions.
- A NaN-aware `info_nce_loss` (`temperature` clamped, logits sanitised).
- Training-loop guards that skip updates when any embedding or loss becomes non-finite while keeping AMP's `GradScaler` coherent.

This variant keeps throughput high while making training robust to adversarially bad batches and padding artefacts.
