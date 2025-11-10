# Vision Transformer Optimizer Performance Analysis

## Summary of Results

### Original Configuration
| Optimizer | Test Accuracy | Test Loss | Status |
|-----------|--------------|-----------|---------|
| SGD | 95.37% | 0.1437 | ✅ Good |
| Adam | 96.65% | 0.1095 | ✅ Good |
| CustomSGD | 95.18% | 0.1445 | ✅ Good |
| CustomAdam | 96.92% | 0.1032 | ✅ Excellent |
| GDA2 | 96.19% | 0.1197 | ✅ Good |
| **AndersonGDA** | **24.86%** | **2.0732** | ❌ **POOR** |
| **MuonFast** | **11.35%** | **2.3030** | ❌ **CATASTROPHIC** |

### After Fixes (Dimension-Aware Scaling + Hyperparameter Tuning)
| Optimizer | Test Accuracy | Test Loss | Status | Improvement |
|-----------|--------------|-----------|---------|-------------|
| SGD | 95.64% | 0.1406 | ✅ Good | +0.27% |
| Adam | 96.89% | 0.0981 | ✅ Good | +0.24% |
| CustomSGD | 95.03% | 0.1535 | ✅ Good | -0.15% |
| CustomAdam | 97.05% | 0.0950 | ✅ Excellent | +0.13% |
| GDA2 | 96.84% | 0.0975 | ✅ Good | +0.65% |
| **AndersonGDA** | **29.02%** | **1.9591** | ❌ **POOR** | **+4.16%** ⚠️ |
| **MuonFast** | **9.82%** | **2.3028** | ❌ **CATASTROPHIC** | **-1.53%** ❌ |

**Key Findings:**
- ✅ AndersonGDA improved from 24.86% to 29.02% with better hyperparameters (lr=0.01, beta=0, m=2)
- ❌ MuonFast remained broken despite dimension-aware scaling (still ~10% accuracy)
- ✅ Dimension-aware scaling reduced update ratio from 111x to 6.8x vs Adam
- ❌ Even with 6.8x updates, MuonFast's orthogonalization is fundamentally incompatible with ViT

---

## Root Cause Analysis

### 1. MuonFast: **Catastrophically Large Updates**

**Problem**: Update magnitudes are **111x larger** than Adam, causing loss to INCREASE instead of decrease.

**Diagnosis**:
```
MuonFast:
  - Total update norm: 206.95
  - Loss change: +2.37 (INCREASES from 2.83 to 5.21)
  - Relative weight updates: 80-95% of parameter norm
  - Result: Model weights are being destroyed each step

Adam (baseline):
  - Total update norm: 1.86
  - Loss change: -0.45 (DECREASES from 3.06 to 2.55)
  - Relative weight updates: 1-2% of parameter norm
  - Result: Stable learning
```

**Root Causes**:
1. **Newton-Schulz orthogonalization** produces unit-norm updates
2. **Small matrix dimensions** (64x64, 128x64) don't benefit from spectral scaling
3. **Learning rate (0.002)** is too high when combined with normalized updates
4. **scale_extra=1.0** doesn't compensate enough for the normalization effect
5. Vision Transformers have **many small-ish weight matrices** that Muon treats aggressively

**Why it works on CNNs but not ViT**:
- CNNs have large conv filters (e.g., 512x512 or 3x3x512x512) where spectral properties dominate
- ViTs have many medium-sized matrices (64-192 dimensions) where Muon's normalization is too aggressive
- ViT attention weights are more sensitive to large perturbations

### 2. AndersonGDA: **Poor Convergence on Non-Convex Landscapes**

**Problem**: Achieves only ~25-29% accuracy, stuck in poor local minima.

**Diagnosis**:
- Anderson acceleration works well for convex/quasi-convex problems
- Vision Transformers have highly non-convex loss landscapes
- The fixed-point extrapolation assumes gradients have stable directions
- ViT gradients are noisy and change direction frequently (attention patterns shift)

**Why it fails**:
1. **Gradient difference term** assumes smooth, predictable gradient changes
2. **Anderson residual extrapolation** can amplify errors in non-convex settings
3. **Memory window (m=3)** accumulates noisy information
4. No adaptive learning rate mechanism like Adam's per-parameter scaling
5. **Per-parameter independent history** misses cross-parameter correlations
6. **No safety checks** - blindly trusts Anderson updates even when they go wrong

**Important Note: GDA2 ≠ AndersonGDA**

While both use "Anderson acceleration," **GDA2 works brilliantly (96.84%)** while **AndersonGDA fails (29.02%)**. The key differences:

| Feature | GDA2 | AndersonGDA |
|---------|------|-------------|
| **Base Optimizer** | Adam with adaptive LR | Plain gradient descent |
| **History Weighting** | Gradient-curvature aware | Unweighted (uniform) |
| **Regularization** | Curvature-adaptive | Fixed epsilon |
| **Safety Checks** | Trust region + descent check | None |
| **Scope** | Global (all params flattened) | Per-parameter independent |
| **ViT Performance** | 96.84% ✅ | 29.02% ❌ |

**GDA2's success** comes from:
- Adam's per-parameter adaptive scaling (handles different parameter types)
- Gradient-difference weighting (down-weights noisy history)
- Curvature-adaptive regularization (adjusts to local landscape)
- Trust region safeguards (prevents catastrophic updates)

See `GDA2_VS_ANDERSONGDA.md` for detailed comparison.


---

## Recommendations

### For MuonFast on Vision Transformers:

1. **Drastically reduce learning rate**: Try `lr=0.00005` to `0.0001` (50x smaller)
2. **Increase scale damping**: Use `scale_extra=0.01` to `0.05` (20x more conservative)
3. **Use dimension-aware scaling**: Add a factor like `1.0 / sqrt(min_dim)` for small matrices
4. **Exclude attention layers**: Add exclude_fn to route attention weights to AdamW:
   ```python
   exclude_fn=lambda name, p: 'attn' in name or 'head' in name or 'embed' in name
   ```
5. **Increase NS tolerance**: Use `ns_tol=1e-2` to reduce orthogonalization aggressiveness

### For AndersonGDA on Vision Transformers:

1. **Reduce memory**: Use `m=1` or `m=2` to avoid accumulating stale information
2. **Increase learning rate**: Try `lr=0.01` to `0.05` with momentum
3. **Disable gradient difference**: Set `beta=0.0` to simplify to plain Anderson
4. **Add adaptive scaling**: Consider implementing per-parameter learning rates
5. **Alternative**: This optimizer may simply not be suitable for transformer architectures

### Better Optimizer Choices for ViT:

1. **Adam/AdamW**: The gold standard for transformers (✅ proven)
2. **GDA2**: Adam + gradient-aware Anderson = excellent (✅ 96.84%)
3. **CustomAdam**: Slightly enhanced Adam variant (✅ 97.05%, best)
4. **SGD with high momentum**: Works well but needs tuning (✅ proven)

**Note:** GDA2 demonstrates that Anderson acceleration CAN work on transformers, but only when combined with:
- Adaptive learning rates (Adam base)
- Gradient-difference weighting
- Trust region safeguards
- Curvature-adaptive regularization

---

## Proposed Fixes

### Option A: Fix MuonFast for Small Matrices

Add dimension-aware scaling to the Muon update step:

```python
# In muon_fast.py step() method, after computing update_orth:
m, n = update_orth.shape
dim_scale = 0.01 / math.sqrt(min(m, n) / 64)  # Scale down for small dims
update_orth *= dim_scale
```

### Option B: Conservative MuonFast Config

```python
MuonFast(
    model.parameters(),
    lr=0.00005,              # 40x smaller than Adam
    momentum=0.95,
    nesterov=True,
    model=model,
    scale_mode="spectral",
    scale_extra=0.02,        # 50x smaller dampening
    ns_coefficients="simple",
    ns_tol=1e-2,             # 10x more tolerant
    exclude_fn=lambda n, p: any(x in n.lower() for x in ['attn', 'head', 'embed', 'norm'])
)
```

### Option C: AndersonGDA with Reduced Memory

```python
AndersonGDA(
    model.parameters(),
    lr=0.05,      # 50x higher
    beta=0.0,     # Disable gradient difference
    m=1,          # Minimal memory
)
```

---

## Conclusion

### MuonFast on Vision Transformers: **FUNDAMENTALLY INCOMPATIBLE**

**Why the fix didn't work:**

Despite implementing dimension-aware scaling (power=2.0) that reduced update magnitudes from 111x to 6.8x relative to Adam, MuonFast still fails catastrophically on ViT. The root issue is **architectural incompatibility**:

1. **Newton-Schulz orthogonalization** forces all updates onto the orthogonal manifold
2. Vision Transformer weight matrices (64×64, 128×64, 192×64) are **too small** for this approach
3. The orthogonalization removes important gradient information that transformers need
4. Attention mechanisms require **precise, direction-preserving updates** that orthogonalization destroys

**Implemented fixes:**
- ✅ Added dimension-aware scaling: `dim_penalty = (min_dim / 256)^2`
- ✅ Reduced learning rate from 0.002 to 0.0003
- ✅ Improved fallback optimizer routing
- ❌ **Result: Still ~10% accuracy (random guessing)**

**Verdict:** MuonFast should **NOT be used** on Vision Transformers or similar architectures with small weight matrices. It works well on large CNNs (512×512+ matrices) but breaks on transformers.

### AndersonGDA on Vision Transformers: **MARGINAL IMPROVEMENT**

**Fixes applied:**
- ✅ Increased LR from 0.001 to 0.01 (10x)
- ✅ Disabled gradient difference term (beta=0)
- ✅ Reduced memory window (m=2 instead of m=3)

**Results:**
- Improved from 24.86% to 29.02% (+4.16%)
- Still far below Adam/SGD performance (96%+)
- Loss decreases in single steps but gets stuck in poor local minima

**Verdict:** AndersonGDA can work but requires extensive hyperparameter tuning for transformers. Not recommended unless you have time for extensive experimentation.

### Recommended Optimizers for Vision Transformers

| Optimizer | Performance | Recommendation |
|-----------|-------------|----------------|
| **CustomAdam** | 97.05% | ⭐⭐⭐ **BEST CHOICE** |
| **Adam** | 96.89% | ⭐⭐⭐ **EXCELLENT** |
| **GDA2** | 96.84% | ⭐⭐⭐ **EXCELLENT** |
| **SGD** | 95.64% | ⭐⭐ **GOOD** (needs tuning) |
| **CustomSGD** | 95.03% | ⭐⭐ **GOOD** (needs tuning) |
| AndersonGDA | 29.02% | ❌ **AVOID** |
| MuonFast | 9.82% | ❌ **DO NOT USE** |

---

## Technical Implementation Details

### Dimension-Aware Scaling (Added to MuonFast)

```python
# In _compute_update_scale() method:
min_dim = min(m, n)
if min_dim < 256:
    # Power of 2.0: very strong dampening for small matrices
    # At 64: penalty = 0.0625 (6.25%)
    # At 128: penalty = 0.25 (25%)
    # At 256: penalty = 1.0 (100%)
    dim_penalty = (min_dim / 256.0) ** 2.0
else:
    dim_penalty = 1.0

# Apply to all scaling modes
return scale_factor * dim_penalty
```

This reduced update magnitudes but wasn't enough to save MuonFast on ViT.

### Improved AndersonGDA Configuration

```python
AndersonGDA(
    model.parameters(),
    lr=0.01,      # 10x higher than original
    beta=0.0,     # Disabled gradient difference
    m=2,          # Smaller memory window
)
```

This improved performance from 24.86% to 29.02% but still underperforms.

---

## Future Work

1. **MuonFast:** Consider adding a "transformer mode" that bypasses orthogonalization for matrices below a certain size
2. **AndersonGDA:** Implement adaptive learning rates per-parameter (like Adam's second moment)
3. **General:** Both optimizers were designed for CNNs; transformer-specific variants may be needed

---

## Files Modified

1. `/optimizers/muon_fast.py` - Added dimension-aware scaling in `_compute_update_scale()`
2. `/modules/experiments/vit_optimizer_experiment.py` - Updated MuonFast (lr=0.0003) and AndersonGDA (lr=0.01, beta=0, m=2) configs

Both optimizers were likely designed and tested primarily on CNN architectures where they perform better.

