# 🚀 NL-MM Training Quick Reference Card

## ⚡ Quick Start (30 seconds)

```bash
# Download data
bash download_flickr8k.sh

# Train model
python train_flickr8k.py --data_dir ./flickr8k --epochs 30

# Done! Model saved to outputs/best_multimodal_model.pt
```

---

## 📚 Documentation Files

| File | Purpose | When to Read |
|------|---------|-------------|
| **NLMM_TRAINING_SUMMARY.md** | Complete overview | 📖 **Start here!** |
| **TRAINING_NLMM_GUIDE.md** | Detailed training guide | Need specifics |
| **MONITORING_GUIDE.md** | Monitoring & troubleshooting | During training |
| **train_nlmm_flickr8k.py** | Training script | Ready to use |

---

## 🎯 Three Ways to Train

### 1️⃣ Simple (Best for First Time)
```bash
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```
- Easiest to use
- Good defaults
- 2-3 hours on RTX 3090

### 2️⃣ Core NL-MM (Best for Research)
```bash
python train_nlmm_flickr8k.py \
    --config nl_mm/configs/tiny_single_gpu.yaml \
    --data_dir ./flickr8k --epochs 30
```
- Pure nl_mm architecture
- Full configurability
- Nested Learning features

### 3️⃣ Improved (Best Results)
```bash
python train_flickr8k_improved.py \
    --data_dir ./flickr8k --epochs 50 --batch_size 64
```
- Optimized hyperparameters
- Better performance
- Longer training time

---

## 📊 What to Watch

### Console Output
```
Epoch 10/30
loss: 1.234 ⬇ | i2t: 0.876 | i2a: 0.943

🔍 Evaluation:
   Image→Text R@1: 32.45% ⬆ (+3.2%)
   Text→Image R@1: 29.87% ⬆ (+2.8%)
   
   💾 Saved best model ✅
```

### Key Metrics

| Metric | Good Progress |
|--------|--------------|
| **Loss** | 5.0 → 2.0 → 1.0 → 0.5 |
| **R@1** | 2% → 15% → 35% → 55% |
| **R@5** | 5% → 35% → 60% → 80% |

### GPU Monitor
```bash
watch -n 1 nvidia-smi
```
**Target:** >90% utilization, <85°C temp

---

## 🔧 Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| 💥 **Out of memory** | `--batch_size 16` |
| 🐌 **Too slow** | `--num_workers 8` |
| 📈 **Loss not decreasing** | `--lr 5e-3` |
| 📉 **Poor R@1 (<20% at epoch 20)** | Train longer or increase model size |

---

## 📁 Data Format

Your dataset needs:

```python
batch = {
    "text": torch.Tensor,    # (B, seq_len) - Token IDs
    "images": torch.Tensor,  # (B, 3, H, W) - RGB images
    "audio": torch.Tensor,   # (B, 1, freq, time) - Spectrograms
}
```

**Flickr8k provides all three!** ✅

---

## ⏱️ Training Time

| Config | Batch | GPU | Time (30 epochs) |
|--------|-------|-----|------------------|
| Tiny | 64 | RTX 3090 | **1 hour** ⚡ |
| Small | 32 | RTX 3090 | **2.5 hours** ⭐ |
| Medium | 16 | RTX 3090 | 6 hours |
| Large | 8 | A100 | 8 hours |

**Recommended:** Start with Small config

---

## 🎓 Expected Results (Flickr8k, 30 epochs)

| Metric | Target |
|--------|--------|
| Image→Text R@1 | 50-60% |
| Text→Image R@1 | 48-58% |
| Image→Text R@5 | 75-85% |
| Training Loss | 0.5-1.0 |

---

## 🛠️ Useful Commands

### Monitor training
```bash
tail -f flickr8k_training.log
```

### Check progress
```bash
python show_flickr8k_results.py
```

### Test model
```bash
python test_multimodal_trained.py --checkpoint outputs/best_model.pt
```

### Demo inference
```bash
python demo_multimodal_memory.py --model outputs/best_model.pt
```

### Continue training
```bash
python continue_training_flickr8k.py \
    --checkpoint outputs/best_model.pt --epochs 20
```

---

## 🎨 Supported Datasets

| Dataset | Modalities | Size | Ready? |
|---------|-----------|------|--------|
| **Flickr8k + FACC** | Text+Image+Audio | 8K | ✅ **Yes** |
| COCO Captions | Text+Image | 118K | Can add |
| AudioCaps | Text+Audio | 50K | Can add |
| Common Voice | Text+Audio | 1000+ hrs | Can add |

---

## 💡 Pro Tips

1. **Start small** - Use tiny config first to verify everything works
2. **Monitor GPU** - Should be >90% utilized during training
3. **Save often** - Checkpoints every 5-10 epochs
4. **Compare optimizers** - Try Adam, SGD, UniversalAndersonGDA
5. **Use AMP** - Mixed precision speeds up training 2x
6. **Evaluate regularly** - Check R@1 every 5 epochs
7. **Plot curves** - Visualize progress to catch issues early

---

## 📖 Read More

1. **NLMM_TRAINING_SUMMARY.md** - Full overview
2. **TRAINING_NLMM_GUIDE.md** - Step-by-step guide
3. **MONITORING_GUIDE.md** - Monitoring reference
4. **nl_mm/README.md** - Architecture details

---

## 🆘 Getting Help

1. Check logs: `tail -100 flickr8k_training.log`
2. Verify data: `ls flickr8k/`
3. Test GPU: `nvidia-smi`
4. Run simple test: `python train_flickr8k.py --epochs 2`

---

## ✅ Pre-flight Checklist

Before training:
- [ ] Dataset downloaded (`bash download_flickr8k.sh`)
- [ ] GPU available (`nvidia-smi`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Config reviewed
- [ ] Output directory created

Start training:
```bash
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```

**Expected:** 2.5 hours → 50-60% R@1 → Working multimodal model! 🎉

---

**Created:** November 8, 2025  
**Version:** 1.0

For detailed information, see **NLMM_TRAINING_SUMMARY.md**

