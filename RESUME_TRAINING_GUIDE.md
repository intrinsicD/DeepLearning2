# Resume Training Guide

## ✅ Resume Functionality Added!

Your training script now supports resuming from checkpoints with the `--resume` argument.

## 🚀 How to Resume Training

### Basic Usage

```bash
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

### Full Example with All Options

```bash
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --precision fp16 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

## 📦 What Gets Saved in Checkpoints

### New Format (Comprehensive)
Starting now, checkpoints save:
- ✅ **Model weights** - All model parameters
- ✅ **Optimizer state** - Adam momentum, learning rate schedule, etc.
- ✅ **Training progress** - Current epoch, global step
- ✅ **Best validation loss** - For tracking improvement
- ✅ **Config** - Training hyperparameters for reference
- ✅ **GradScaler state** - Mixed precision training state (if using fp16)

Checkpoint structure:
```python
{
    'epoch': 42,
    'global_step': 158250,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'best_val': 0.2345,
    'val_loss': 0.2345,
    'scaler_state_dict': {...},
    'config': {
        'epochs': 50,
        'batch_size': 8,
        'lr': 2e-4,
        'd_shared': 512,
    }
}
```

### Old Format (Legacy)
Old checkpoints only contain model weights (state_dict). The script handles these gracefully.

## 🔄 What Happens When You Resume

1. **Model weights are loaded** - Restores learned parameters
2. **Optimizer state is restored** - Continues from same optimization state
3. **Training continues from next epoch** - If stopped at epoch 10, resumes at epoch 11
4. **Global step counter is restored** - TensorBoard logging continues seamlessly
5. **Best validation loss is tracked** - Won't save worse checkpoints

### Example Output When Resuming

```
============================================================
Resuming from checkpoint: checkpoints_brain_v2/best_brain_v2.pt
============================================================
✓ New format checkpoint with full training state
✓ Model state loaded (strict)
✓ Optimizer state loaded
✓ Starting from epoch 43
✓ Global step restored to 158250
✓ Best validation loss: 0.2345
✓ GradScaler state loaded
============================================================

Train samples: 30000
Val samples:   5000
Device: cuda
Precision: fp16
Batch size: 8
Learning rate: 0.0002
Training epochs: 43 -> 100
```

## 🎯 Common Use Cases

### 1. Training Was Interrupted

```bash
# Training stopped at epoch 25 due to crash/interrupt
# Resume to complete all 50 epochs
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 50 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

### 2. Extend Training Beyond Original Plan

```bash
# Originally trained for 50 epochs, want to train for 100 more
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 150 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

### 3. Fine-tune with Different Learning Rate

```bash
# Resume but with lower learning rate for fine-tuning
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --lr 5e-5 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

### 4. Continue with Larger Batch Size

```bash
# If you have more GPU memory available
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --batch_size 16 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

## ⚠️ Important Notes

### Compatibility

**✅ Fully Compatible Changes:**
- Different `--epochs` value
- Different `--batch_size`
- Different `--lr` (learning rate)
- Different `--precision` (fp32/fp16/bf16)

**⚠️ May Require Retraining:**
- Changes to model architecture (`d_shared`, layer counts, etc.)
- Different modalities enabled/disabled
- Major code refactoring

### Legacy Checkpoints

If resuming from old checkpoints (saved before this update):
- ✓ Model weights will load
- ⚠️ Optimizer state won't be available (will start fresh)
- ⚠️ Epoch number won't be known (starts from 0)
- ⚠️ May see warnings about "unexpected keys" - this is normal

You'll see output like:
```
✓ Old format checkpoint (model weights only)
⚠ Strict loading failed: ...unexpected keys...
  Attempting to load with strict=False...
✓ Model state loaded (partial - may need fine-tuning)
```

This is **safe** and **expected** for old checkpoints.

## 🔍 Troubleshooting

### "RuntimeError: Error(s) in loading state_dict"

**Cause:** Model architecture has changed significantly.

**Solutions:**
1. If it's a minor change, the script loads with `strict=False` automatically
2. If training fails, you may need to retrain from scratch
3. Check if you're using the same `d_shared` value

### "Checkpoint file not found"

**Cause:** Wrong path to checkpoint.

**Solution:** Check the path:
```bash
ls -lh checkpoints_brain_v2/best_brain_v2.pt
```

### Training Starts from Epoch 0

**Cause:** Old checkpoint format or checkpoint doesn't contain epoch info.

**Solution:** This is expected for old checkpoints. Model weights are still loaded correctly.

### TensorBoard Shows Discontinuity

**Cause:** If `global_step` isn't restored, TensorBoard may show a gap.

**Solution:** 
- New checkpoints preserve `global_step`
- For old checkpoints, the graphs will restart from 0
- Consider using different `--logdir` for continued runs

## 📊 TensorBoard with Resume

When resuming, TensorBoard logs continue seamlessly if using new checkpoint format:

**Option 1: Same logdir (continuous graphs)**
```bash
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --logdir runs_brain_v2 \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

**Option 2: New logdir (separate comparison)**
```bash
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 100 \
    --logdir runs_brain_v2_continued \
    --resume checkpoints_brain_v2/best_brain_v2.pt
```

Then compare both in TensorBoard:
```bash
tensorboard --logdir=. --port=6006
```

## 💡 Best Practices

1. **Always save checkpoints regularly** - The script saves best checkpoint automatically

2. **Keep old checkpoints** - Consider backing up before resuming:
   ```bash
   cp checkpoints_brain_v2/best_brain_v2.pt checkpoints_brain_v2/best_brain_v2_backup_epoch42.pt
   ```

3. **Monitor the first few batches** after resuming to ensure training is stable

4. **Use meaningful savedir names** for different experiments:
   ```bash
   --savedir checkpoints_experiment1
   --savedir checkpoints_experiment2_finetuned
   ```

5. **Document your resume commands** to track your training history

## 🎉 Summary

You can now:
- ✅ Resume interrupted training
- ✅ Extend training for more epochs
- ✅ Fine-tune with different hyperparameters
- ✅ Seamlessly continue TensorBoard logging
- ✅ Load both old and new checkpoint formats

Happy training! 🚀

