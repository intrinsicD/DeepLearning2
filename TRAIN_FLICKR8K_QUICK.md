# 🎯 How to Train on Flickr8k - TL;DR

## One-Line Command

```bash
python train_multimodal_flickr8k.py
```

That's it! This will:
- Load Flickr8k from `data/flickr8k/`
- Train for 10 epochs
- Use batch size 8
- Save best model to `checkpoints/best.pt`
- Log to TensorBoard at `runs/`

## Common Use Cases

### Start Fresh Training
```bash
python train_multimodal_flickr8k.py --epochs 20 --batch_size 8
```

### Resume from Checkpoint
```bash
python train_multimodal_flickr8k.py --checkpoint checkpoints/best.pt --epochs 10
```

### Low Memory (4GB GPU)
```bash
python train_multimodal_flickr8k.py --batch_size 2 --precision fp16
```

### High Memory (24GB GPU)
```bash
python train_multimodal_flickr8k.py --batch_size 32 --precision fp16
```

### Monitor Progress
```bash
tensorboard --logdir runs
```

## What You Get

✅ **Tri-modal model** - Text, Image, Audio  
✅ **Automatic checkpointing** - Best model saved  
✅ **NaN recovery** - Stable training  
✅ **Mixed precision** - Fast & memory-efficient  
✅ **TensorBoard logs** - Track everything  

## Output Files

- `checkpoints/best.pt` ← Your trained model
- `checkpoints/last.pt` ← Latest checkpoint
- `runs/MMT_Flickr8k/` ← TensorBoard logs

## If Something Goes Wrong

| Problem | Solution |
|---------|----------|
| "Flickr8k directory not found" | Add `--root_dir /path/to/flickr8k` |
| Out of memory | Add `--batch_size 2` |
| NaN losses | Add `--lr 1e-4` |

## Full Documentation

See `FLICKR8K_TRAINING_GUIDE.md` for complete details.

