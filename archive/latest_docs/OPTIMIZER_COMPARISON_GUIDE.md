# 🔬 NL-MM Optimizer Comparison Guide

This guide will help you find the best optimizer for the nl_mm model on your hardware.

---

## 🚀 Quick Start

### Run Quick Test (5 epochs, ~15 minutes):
```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 5 \
    --batch_size 8 \
    --subset 5000
```

### Run Full Test (10 epochs, ~45 minutes):
```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 10 \
    --batch_size 8 \
    --subset 10000
```

---

## 📊 What Gets Tested

The script compares these optimizers:
1. **Adam** - Adaptive learning rate
2. **AdamW** - Adam with weight decay
3. **SGD** - Stochastic Gradient Descent (with momentum)
4. **RMSprop** - Root Mean Square Propagation

### Metrics Compared:
- ✅ **Training Loss** - Lower is better
- ✅ **Validation R@1** - Higher is better (retrieval accuracy)
- ✅ **Training Speed** - Faster is better
- ✅ **Convergence** - How quickly loss decreases
- ✅ **Final Performance** - Best achieved R@1

---

## 🎯 Command Options

### Basic Options:
```bash
--config             # Model config (nano_8gb.yaml for 8GB GPU)
--data_dir           # Path to Flickr8k dataset
--epochs             # Epochs per optimizer (5-10 recommended)
--batch_size         # Batch size (4-8 for 8GB GPU)
--lr                 # Learning rate (default: 1e-3)
```

### Testing Options:
```bash
--subset 5000        # Use 5000 samples (faster testing)
--subset 10000       # Use 10000 samples (more accurate)
--subset 0           # Use full dataset (slowest, most accurate)
```

### Optimizer Selection:
```bash
--optimizers adam,adamw,sgd,rmsprop    # Test all (default)
--optimizers adam,adamw                # Test only Adam variants
--optimizers sgd                       # Test only SGD
```

---

## 📈 Output

### Files Created:
```
results/folder_per_model/nl_mm/outputs/optimizer_comparison/
├── optimizer_comparison.json     # Detailed metrics
└── optimizer_comparison.png      # Visual comparison
```

### Comparison Plot Shows:
1. **Training Loss Over Time** - How fast each optimizer learns
2. **Validation R@1 Over Time** - Retrieval accuracy improvement
3. **Average Epoch Time** - Speed comparison
4. **Final Training Loss** - Best final loss achieved
5. **Best Validation R@1** - Best accuracy achieved
6. **Total Training Time** - Overall time taken

### Terminal Output Example:
```
================================================================================
OPTIMIZER COMPARISON SUMMARY
================================================================================
Optimizer    Final Loss   Best R@1     Avg Time/Epoch  Total Time  
--------------------------------------------------------------------------------
adam         0.5234       42.50        45.2            3.8         
adamw        0.4987       45.30        46.1            3.9         
sgd          0.6123       38.20        42.3            3.5         
rmsprop      0.5456       41.80        44.8            3.7         
================================================================================

🏆 Best Results:
  Lowest Loss:  adamw (0.4987)
  Highest R@1:  adamw (45.30%)
  Fastest:      sgd (3.5 min)
```

---

## 🔍 How to Interpret Results

### Look for:

1. **Best Loss** → Most effective learning
2. **Best R@1** → Best for retrieval tasks
3. **Fast Convergence** → Steep loss curve early
4. **Stable Training** → Smooth curves, no spikes
5. **Speed vs Quality** → Balance between time and performance

### Typical Results:

| Optimizer | Speed | Quality | Memory | Best For |
|-----------|-------|---------|--------|----------|
| **AdamW** | Medium | ⭐⭐⭐⭐⭐ | High | Best overall |
| **Adam** | Medium | ⭐⭐⭐⭐ | High | Good baseline |
| **SGD** | Fast | ⭐⭐⭐ | Low | Memory-limited |
| **RMSprop** | Medium | ⭐⭐⭐⭐ | Medium | Alternative to Adam |

---

## 💡 Recommendations by GPU

### 8GB GPU (Your Case):
```bash
# Quick test (15 min):
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 5 \
    --batch_size 8 \
    --subset 5000 \
    --optimizers adam,adamw
```

### 12-16GB GPU:
```bash
# Full test (45 min):
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/tiny_single_gpu.yaml \
    --data_dir ./flickr8k \
    --epochs 10 \
    --batch_size 16 \
    --subset 10000
```

### 24GB+ GPU:
```bash
# Comprehensive test (2 hours):
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/tiny_single_gpu.yaml \
    --data_dir ./flickr8k \
    --epochs 20 \
    --batch_size 32 \
    --subset 0  # Full dataset
```

---

## 🎓 Understanding Optimizers

### Adam
**Pros:**
- Adaptive learning rates per parameter
- Works well out-of-the-box
- Handles sparse gradients well

**Cons:**
- Can overfit
- Higher memory usage

**Best for:** Quick experiments

### AdamW
**Pros:**
- Adam with proper weight decay
- Better generalization
- SOTA for transformers

**Cons:**
- Slightly slower than Adam
- Higher memory

**Best for:** Production training (RECOMMENDED)

### SGD with Momentum
**Pros:**
- Simple and stable
- Lower memory usage
- Can escape local minima

**Cons:**
- Requires careful tuning
- May need learning rate scheduling

**Best for:** Memory-constrained environments

### RMSprop
**Pros:**
- Good for non-stationary objectives
- Works well with RNNs
- Moderate memory

**Cons:**
- Less popular for transformers
- Can be unstable

**Best for:** Alternative to Adam

---

## 📝 Example Workflow

### Step 1: Quick Test
```bash
# Test Adam vs AdamW (15 min)
python test_nl_mm_optimizers.py \
    --epochs 5 \
    --subset 5000 \
    --optimizers adam,adamw
```

### Step 2: View Results
```bash
# Open the plot
xdg-open results/folder_per_model/nl_mm/outputs/optimizer_comparison/optimizer_comparison.png

# Check JSON for detailed metrics
cat results/folder_per_model/nl_mm/outputs/optimizer_comparison/optimizer_comparison.json
```

### Step 3: Train with Best Optimizer
```bash
# If AdamW wins, use it for full training:
python train_nlmm_flickr8k.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 8
```

---

## ⚡ Speed Up Testing

### Use Smaller Config:
```bash
# Create even smaller config for testing
--config modules/nl_mm/configs/nano_8gb.yaml  # Already small!
```

### Use Fewer Epochs:
```bash
--epochs 3  # Quick sanity check
```

### Use Smaller Subset:
```bash
--subset 2000  # Very fast test
```

### Test Fewer Optimizers:
```bash
--optimizers adam,adamw  # Skip SGD and RMSprop
```

---

## 🐛 Troubleshooting

### OOM Error:
```bash
# Reduce batch size
--batch_size 4

# Or use smaller config
--config modules/nl_mm/configs/nano_8gb.yaml
```

### Too Slow:
```bash
# Use smaller subset
--subset 2000

# Fewer epochs
--epochs 3

# Test fewer optimizers
--optimizers adamw
```

### Import Errors:
```bash
# Make sure you're in the right directory
cd /home/alex/Documents/DeepLearning2

# Check dataset exists
ls flickr8k/Flickr8k_Dataset/ | head
```

---

## 📊 Expected Results

### Quick Test (5 epochs, 5000 samples):
```
⏱️  Time: ~15 minutes
📈 Accuracy: 30-40% R@1
🎯 Purpose: Find best optimizer quickly
```

### Medium Test (10 epochs, 10000 samples):
```
⏱️  Time: ~45 minutes
📈 Accuracy: 40-50% R@1
🎯 Purpose: Reliable comparison
```

### Full Test (20 epochs, full dataset):
```
⏱️  Time: ~3 hours
📈 Accuracy: 50-60% R@1
🎯 Purpose: Production-quality comparison
```

---

## 🏆 Recommended Test

**For your 8GB GPU, run this:**

```bash
python test_nl_mm_optimizers.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 5 \
    --batch_size 8 \
    --subset 5000 \
    --optimizers adam,adamw \
    --lr 1e-3
```

**This will:**
- ✅ Take ~15 minutes
- ✅ Test the two most popular optimizers
- ✅ Give reliable results
- ✅ Fit comfortably in 8GB GPU
- ✅ Show which optimizer to use for full training

---

## 📖 After Testing

Once you know the best optimizer, update your training command:

```bash
# If AdamW wins (most likely):
python train_nlmm_flickr8k.py \
    --config modules/nl_mm/configs/nano_8gb.yaml \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 8
```

The nano_8gb.yaml config already uses AdamW by default! 🎉

---

**Created:** November 8, 2025  
**Purpose:** Find best optimizer for nl_mm model  
**Recommended:** Run quick test first, then full training  
**Action:** Copy and run the recommended command above! 🚀

