# Training Multimodal Model on Flickr8k

This guide shows you how to train the multimodal thinking model on the Flickr8k dataset.

## Quick Start

```bash
# Basic training with default settings
python train_multimodal_flickr8k.py

# Custom training parameters
python train_multimodal_flickr8k.py --epochs 20 --batch_size 16 --lr 1e-4

# Resume from checkpoint
python train_multimodal_flickr8k.py --checkpoint checkpoints/best.pt --epochs 10
```

## Dataset Structure

The script expects Flickr8k to be in the following location:
```
data/flickr8k/
├── Flicker8k_Dataset/          # Images
├── Flickr8k.token.txt          # Captions
├── Flickr_8k.trainImages.txt   # Train split
├── Flickr_8k.devImages.txt     # Dev split
└── Flickr_8k.testImages.txt    # Test split
```

If your Flickr8k is in a different location, use the `--root_dir` flag:
```bash
python train_multimodal_flickr8k.py --root_dir /path/to/flickr8k
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | `data/flickr8k` | Path to Flickr8k root directory |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `8` | Batch size (reduce if OOM) |
| `--lr` | `2e-4` | Learning rate |
| `--precision` | `fp16` | Training precision (fp32/fp16/bf16) |
| `--checkpoint` | `None` | Path to checkpoint to resume from |
| `--savedir` | `checkpoints` | Directory to save checkpoints |
| `--logdir` | `runs` | Directory for tensorboard logs |

## Memory-Efficient Training

If you run out of GPU memory, try these settings:

```bash
# Reduce batch size
python train_multimodal_flickr8k.py --batch_size 4

# Use mixed precision (default, but you can ensure it's on)
python train_multimodal_flickr8k.py --precision fp16 --batch_size 4

# Minimal memory usage
python train_multimodal_flickr8k.py --batch_size 2 --precision fp16
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

## Output Files

Training will create:
- `checkpoints/best.pt` - Best model (lowest validation loss)
- `checkpoints/last.pt` - Latest model checkpoint
- `runs/MMT_Flickr8k/` - TensorBoard logs

## Training Details

The model:
- Uses **E5-small-v2** for text encoding (384-dim)
- Uses **CLIP ViT-B/32** for image encoding (768-dim)
- Uses **Whisper-tiny** for audio encoding (384-dim)
- Projects all modalities to a shared 384-dim thinking space
- Uses 64 memory slots with 2-layer transformer walker
- Applies InfoNCE contrastive loss + reconstruction loss

Training features:
- **Gradient clipping** (norm=1.0) for stability
- **Learning rate warmup** (100 steps)
- **NaN detection** with automatic recovery
- **Mixed precision** (FP16) for faster training
- **8-bit AdamW** optimizer (if bitsandbytes available)

## Example Training Session

```bash
# Start training
python train_multimodal_flickr8k.py --epochs 10 --batch_size 8

# Expected output:
# Loading Flickr8k dataset from data/flickr8k...
# Train samples: 6000
# Val samples: 1000
# Initializing trainer...
# Starting training...
# Device: cuda
# Precision: fp16
# Batch size: 8
# Learning rate: 0.0002
# Epochs: 10
# --------------------------------------------------------------------------------
# [Epoch 0 Step 0] loss=1.6432 grad_norm=1.60 lr=2.00e-06
# [Epoch 0 Step 50] loss=1.4512 grad_norm=0.23 lr=1.02e-04
# [epoch 0] new best 1.3421 -> saved checkpoints/best.pt
# ...
```

## Troubleshooting

### "Flickr8k directory not found"
Make sure the dataset path is correct:
```bash
python train_multimodal_flickr8k.py --root_dir data/flickr8k
```

### Out of memory (OOM)
Reduce batch size:
```bash
python train_multimodal_flickr8k.py --batch_size 2
```

### NaN losses
The script has automatic NaN recovery. If it persists:
```bash
python train_multimodal_flickr8k.py --lr 1e-4  # Lower learning rate
```

### Slow training
Check GPU utilization. If low, increase batch size:
```bash
python train_multimodal_flickr8k.py --batch_size 16
```

## Next Steps

After training:

1. **Evaluate the model** - The trainer automatically saves the best model
2. **Try the demo mode** - Test on synthetic data first:
   ```bash
   python train_multimodal_sota8gb.py --demo --epochs 1
   ```
3. **Visualize in TensorBoard** - Monitor metrics:
   ```bash
   tensorboard --logdir runs
   ```

## Notes

- The Flickr8k dataset has ~6,000 training images and ~1,000 validation images
- Each image has 5 captions, so you'll see ~30,000 training samples total
- Training takes ~1-2 hours on a modern GPU for 10 epochs
- The model architecture is designed to work with limited memory (8GB GPU)

