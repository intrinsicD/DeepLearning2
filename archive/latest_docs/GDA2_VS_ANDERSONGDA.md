# GDA2 vs AndersonGDA: Why One Works and One Fails on ViT

## Performance Comparison

| Optimizer | Test Accuracy | Test Loss | Status |
|-----------|--------------|-----------|--------|
| **GDA2** | **96.84%** | **0.0975** | ✅ **EXCELLENT** |
| **AndersonGDA** | **29.02%** | **1.9591** | ❌ **POOR** |

**The Question:** Both claim to use Anderson acceleration, so why does GDA2 work brilliantly while AndersonGDA fails miserably?

---

## Critical Differences

### 1. **Base Optimizer: Adam vs Plain Gradient Descent**

**GDA2:**
```python
# Uses AdamW-style preconditioner with adaptive learning rates
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # First moment (momentum)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # Second moment (variance)
denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
step_t = -lr * (exp_avg / bias_c1) / denom  # Per-parameter adaptive step
```

**AndersonGDA:**
```python
# Uses plain gradient descent with optional gradient difference
step_basic = -lr * grad  # Fixed learning rate, no adaptation
step_diff = -beta * lr * (grad - prev_grad)  # Optional momentum-like term
```

**Impact:** GDA2's Adam base provides:
- ✅ **Per-parameter adaptive learning rates** (critical for transformers)
- ✅ **Automatic scale normalization** via second moment
- ✅ **Stable base step** that rarely goes in the wrong direction

AndersonGDA's plain GD suffers:
- ❌ **Fixed learning rate** across all parameters
- ❌ **No adaptive scaling** for different parameter types
- ❌ **Prone to instability** on non-uniform loss landscapes

### 2. **Gradient-Difference Weighting vs Unweighted Mixing**

**GDA2:**
```python
# Weights each history element by gradient curvature information
s = s_cols[j]  # Previous step
y = (g_cols[j + 1] - g_cols[j])  # Gradient difference
sy = torch.dot(s, y)
cos = sy / (ns * ny + 1e-16)  # Cosine similarity (curvature alignment)
inv_curv = ns / (ny + 1e-16)  # Inverse curvature estimate
weight = max(1e-8, (cos * inv_curv).item())  # Combine for weight
```

**AndersonGDA:**
```python
# Treats all residuals equally, no weighting
c = torch.linalg.solve(gram, b)  # Simple least-squares, no weights
delta = sum(c_i * r_i)  # Uniform mixing
```

**Impact:** GDA2's weighting:
- ✅ **Down-weights noisy history** when gradients are unstable
- ✅ **Up-weights stable directions** with consistent curvature
- ✅ **Adapts to local landscape** properties

AndersonGDA's uniform mixing:
- ❌ **Amplifies noise** from bad history entries
- ❌ **No discrimination** between good and bad directions
- ❌ **Accumulates errors** in non-convex regions

### 3. **Regularization Strategy**

**GDA2:**
```python
# Curvature-adaptive regularization
curvature = sum(||y_j|| / ||s_j||) / pairs  # Estimate local curvature
lam = self.reg0 * (1.0 + self.nu * curvature)  # Scale with curvature
reg_matrix = lam * torch.diag(1.0 / (w + 1e-16))  # Weight-dependent
system = gram + reg_matrix  # Solve with adaptive regularization
```

**AndersonGDA:**
```python
# Fixed regularization
gram[diag_indices, diag_indices] += eps  # Constant epsilon
```

**Impact:** GDA2's adaptive regularization:
- ✅ **Increases damping** in high-curvature regions
- ✅ **Reduces damping** in flat regions
- ✅ **Prevents overfitting** to noisy history

AndersonGDA's fixed regularization:
- ❌ **Same damping everywhere** regardless of landscape
- ❌ **Can be too weak** or too strong
- ❌ **Doesn't adapt** to changing conditions

### 4. **Trust Region / Safety Checks**

**GDA2:**
```python
# Trust region on correction magnitude
base_norm = torch.norm(f_k)
corr_norm = torch.norm(correction)
if corr_norm > self.tau * base_norm:  # tau = 1.5 by default
    scale = (self.tau * base_norm / (corr_norm + 1e-16)).item()
    correction.mul_(scale)  # Clip to trust region

# Descent direction check
mixed = f_k - correction
if torch.dot(g_k, mixed) >= 0:  # Not a descent direction?
    final_step = f_k  # Fall back to base step
else:
    final_step = mixed  # Use corrected step
```

**AndersonGDA:**
```python
# No safety checks - always applies Anderson update
p.data.add_(delta)  # Trusts Anderson completely
```

**Impact:** GDA2's safety mechanisms:
- ✅ **Prevents catastrophic updates** when Anderson fails
- ✅ **Falls back to safe base step** if correction is bad
- ✅ **Guarantees descent direction** (mostly)

AndersonGDA's blind trust:
- ❌ **No fallback** when acceleration fails
- ❌ **Can move uphill** if residuals are misleading
- ❌ **Compounds errors** over multiple steps

### 5. **Per-Parameter vs Global History**

**GDA2:**
```python
# Global history across all parameters (flattened)
f_k = _flatten(base_steps).detach()  # All parameters together
g_k = _flatten(grads).detach()
# Anderson operates on entire parameter vector
```

**AndersonGDA:**
```python
# Per-parameter independent history
for p in group["params"]:
    state["history"]  # Separate history for each parameter
    # Anderson operates independently on each tensor
```

**Impact:** GDA2's global approach:
- ✅ **Captures cross-parameter correlations**
- ✅ **More stable** with many small parameters
- ✅ **Better for transformers** with many coupled matrices

AndersonGDA's per-parameter approach:
- ❌ **Ignores parameter interactions**
- ❌ **Less stable** with small matrices
- ❌ **Poor for coupled architectures** like transformers

---

## Why GDA2 Works on Vision Transformers

1. **Adam Base Provides Stability**: The adaptive learning rate ensures each parameter type (attention, MLP, norms) gets appropriate step sizes

2. **Gradient Weighting Handles Non-Convexity**: Down-weighting noisy history prevents error accumulation in non-convex regions

3. **Safety Checks Prevent Disasters**: Trust region and descent checks ensure Anderson never makes things worse

4. **Curvature-Adaptive Regularization**: Automatically adjusts to local landscape properties

5. **Global History for Coupled Parameters**: Transformer layers are highly coupled; global history captures this

## Why AndersonGDA Fails on Vision Transformers

1. **Plain GD Base is Inadequate**: Fixed learning rate can't handle the diverse parameter scales in transformers

2. **Unweighted Mixing Amplifies Noise**: Treats all history equally, accumulating errors in non-convex regions

3. **No Safety Net**: When Anderson goes wrong, there's no fallback

4. **Fixed Regularization**: Can't adapt to changing curvature across attention and MLP layers

5. **Per-Parameter Independence**: Misses the cross-layer correlations critical for transformers

---

## The Verdict

**GDA2 is NOT just "Anderson acceleration"** - it's a sophisticated hybrid that:
- Uses Adam as a preconditioner
- Adds gradient-difference aware weighting
- Implements curvature-adaptive regularization
- Includes trust region safeguards
- Operates globally across parameters

**AndersonGDA is a "pure" Anderson implementation** that:
- Uses simple gradient descent
- Applies textbook Anderson mixing
- Has minimal safety mechanisms
- Operates per-parameter independently

The name "GDA2" (Gradient-Difference Aware Anderson) highlights its key innovation: **weighting history by gradient curvature information**, which is exactly what makes it work on non-convex problems like ViT.

---

## Analogy

**AndersonGDA** is like following a GPS that averages your last 3 routes:
- Works great on a grid (convex)
- Fails in mountains with switchbacks (non-convex)

**GDA2** is like a GPS that:
- Checks road conditions (curvature)
- Ignores routes that were bumpy (weights)
- Uses traffic data to adjust speed (adaptive LR)
- Has manual override when GPS seems wrong (safety checks)

For Vision Transformers (the "mountains"), you need the smart GPS!

---

## Recommended Configuration

For ViT and other transformers, use **GDA2** with defaults:
```python
GDA2(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),  # Adam moments
    history_size=5,       # Anderson memory
    reg=1e-2,            # Base regularization
    nu=0.5,              # Curvature scaling
    tau=1.5,             # Trust region
)
```

**Do NOT use AndersonGDA** for transformers unless you:
1. Implement per-parameter adaptive learning rates
2. Add gradient-difference weighting
3. Add trust region safeguards
4. Switch to global history
...at which point you've reinvented GDA2!

