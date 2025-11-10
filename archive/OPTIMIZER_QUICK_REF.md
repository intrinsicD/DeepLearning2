# 🎯 Optimizer Test - Quick Commands

## Copy-Paste Commands

### ⚡ Quick Test (15 min) - RECOMMENDED
```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 5 \
    --batch_size 8 \
    --subset 5000 \
    --optimizers adam,adamw,muon,dmgd
```

### 📊 Medium Test (45 min) - More Accurate
```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 10 \
    --batch_size 8 \
    --subset 10000
```

### 🔬 Full Test (2-3 hours) - Most Accurate
```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 15 \
    --batch_size 8 \
    --subset 0
```

### 🚀 Skip Test & Train Directly
```bash
python train_nlmm_flickr8k.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 8
```

---

## View Results

### After test completes:
```bash
# View plot
xdg-open results/folder_per_model/nl_mm/outputs/optimizer_comparison/optimizer_comparison.png

# View JSON
cat results/folder_per_model/nl_mm/outputs/optimizer_comparison/optimizer_comparison.json | jq
```

---

## What Each Optimizer Means

| Optimizer | Speed | Quality | Memory | Best For |
|-----------|-------|---------|--------|----------|
| Adam | Medium | ⭐⭐⭐⭐ | High | Baseline |
| **AdamW** | Medium | ⭐⭐⭐⭐⭐ | High | **Best overall** |
| SGD | Fast | ⭐⭐⭐ | Low | Memory-limited |
| RMSprop | Medium | ⭐⭐⭐⭐ | Medium | Alternative |
| **Muon** | Medium | ⭐⭐⭐⭐⭐ | High | **Your custom optimizer** |
| **DMGD** | Fast | ⭐⭐⭐⭐ | Medium | **Your custom optimizer** |

**Your Custom Optimizers:**
- **Muon**: Universal adaptive optimizer with orthogonalization
- **DMGD**: Deep Momentum GD with learnable modulation

**Prediction:** AdamW or Muon will likely win! 🏆

---

**Just run the first command to get started!** 🚀

