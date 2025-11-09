# NL-MM Training Quick Reference & Monitoring Guide

## Quick Start Commands

### 1. Train on Flickr8k (Ready-to-use wrapper)
```bash
# Standard training with MultiModalMemoryNetwork
python train_flickr8k.py \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --optimizer universal_anderson
```

### 2. Train Core NL-MM on Flickr8k
```bash
# Using the pure nl_mm architecture
python train_nlmm_flickr8k.py \
    --config nl_mm/configs/tiny_single_gpu.yaml \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 32 \
    --use_amp
```

### 3. Continue Training from Checkpoint
```bash
python continue_training_flickr8k.py \
    --checkpoint outputs/best_model.pt \
    --epochs 20 \
    --lr 5e-4
```

## Monitoring During Training

### Console Output
Watch for these key indicators:

```
Epoch 1/30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | loss: 2.3456 | i2t: 1.8234 | ETA: 0:03:42

📈 Training metrics:
   Loss: 2.3456
   Image↔Text: 1.8234
   Image↔Audio: 1.9123
   Text↔Audio: 1.7890

🔍 Evaluating...
   Image→Text R@1: 15.23%
   Text→Image R@1: 13.45%
   Image→Audio R@1: 14.67%
   
   💾 Saved best model (avg R@1: 14.34%)
```

### Real-time Monitoring in Terminal

#### Option 1: Follow training log
```bash
# In a separate terminal
tail -f outputs/nlmm_flickr8k/training.log
```

#### Option 2: Watch GPU usage
```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi

# Or with better formatting
pip install gpustat
watch -n 1 gpustat -cpu
```

#### Option 3: Monitor system resources
```bash
# CPU, RAM, disk I/O
htop

# Or
pip install glances
glances
```

### TensorBoard Visualization

Add to your training script:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/nlmm_flickr8k')

# During training
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/i2t', i2t_loss, epoch)
writer.add_scalar('Retrieval/i2t_r1', i2t_r1, epoch)
writer.add_scalar('Retrieval/t2i_r1', t2i_r1, epoch)
```

Launch TensorBoard:
```bash
tensorboard --logdir runs/nlmm_flickr8k --port 6006
# Open browser to http://localhost:6006
```

### Weights & Biases Integration

```python
import wandb

# Initialize at start of training
wandb.init(
    project="nlmm-flickr8k",
    config={
        "d_model": 512,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }
)

# Log metrics during training
wandb.log({
    "epoch": epoch,
    "train/loss": train_loss,
    "train/i2t_loss": i2t_loss,
    "eval/i2t_r1": i2t_r1,
    "eval/t2i_r1": t2i_r1,
})

# Log best model
wandb.save('best_model.pt')
```

## Key Metrics to Watch

### 1. Training Loss (should decrease)
- **Initial:** 5-8 (random initialization)
- **After 5 epochs:** 2-3
- **After 20 epochs:** 0.5-1.5
- **Converged:** 0.3-0.8

### 2. Retrieval Recall@1 (should increase)
- **Random baseline:** ~0.1% (1/8000 for Flickr8k)
- **After 5 epochs:** 5-15%
- **After 20 epochs:** 30-45%
- **Well-trained:** 50-65%

### 3. Retrieval Recall@5 (should increase faster)
- **Random baseline:** ~0.06%
- **After 5 epochs:** 15-30%
- **After 20 epochs:** 55-70%
- **Well-trained:** 75-85%

### 4. GPU Metrics
- **Memory usage:** Should be stable (65-85% of GPU memory)
- **Utilization:** Should be >90% during training
- **Temperature:** <85°C is safe

## Good vs Bad Training Patterns

### ✅ Good Signs

1. **Smooth loss decrease**
   ```
   Epoch 1: 5.234
   Epoch 5: 2.145
   Epoch 10: 1.234
   Epoch 20: 0.678
   ```

2. **Improving retrieval**
   ```
   Epoch 5:  i2t_r1=12%, t2i_r1=10%
   Epoch 10: i2t_r1=28%, t2i_r1=25%
   Epoch 20: i2t_r1=45%, t2i_r1=42%
   ```

3. **Stable gradient norms**
   ```
   Gradient norm: 0.8-1.5 (stable)
   ```

4. **Consistent training time per epoch**
   ```
   Epoch 1: 5m 23s
   Epoch 2: 5m 19s
   Epoch 3: 5m 25s
   ```

### ⚠️ Warning Signs

1. **Loss plateau too early**
   ```
   Epoch 5: 3.234
   Epoch 10: 3.221
   Epoch 15: 3.218
   ```
   **Fix:** Increase learning rate or check data augmentation

2. **Loss explosion**
   ```
   Epoch 1: 5.234
   Epoch 2: 8.456
   Epoch 3: NaN
   ```
   **Fix:** Decrease learning rate, check for bugs

3. **Oscillating metrics**
   ```
   Epoch 5: 1.234
   Epoch 6: 3.456
   Epoch 7: 1.567
   ```
   **Fix:** Reduce learning rate, increase batch size

4. **No improvement in retrieval**
   ```
   Epoch 5:  i2t_r1=2%
   Epoch 15: i2t_r1=3%
   Epoch 25: i2t_r1=2%
   ```
   **Fix:** Check data loading, verify contrastive loss, increase model capacity

## Troubleshooting Commands

### Check if data is loading correctly
```bash
python -c "
from src.utils.flickr8k_dataset import Flickr8kAudioDataset
ds = Flickr8kAudioDataset('flickr8k', 'train')
batch = ds[0]
print('Text shape:', batch['text'].shape)
print('Image shape:', batch['images'].shape)
print('Audio shape:', batch['audio'].shape)
"
```

### Test model forward pass
```bash
python -c "
import torch
from nl_mm.models.nl_mm_model import NLMM
from nl_mm.utils import load_config

cfg = load_config('nl_mm/configs/tiny_single_gpu.yaml')
model = NLMM(cfg)
batch = {
    'text': torch.randint(0, 1000, (2, 16)),
    'image': torch.randn(2, 3, 224, 224),
    'audio': torch.randn(2, 1, 80, 100),
}
outputs, state = model(batch)
print('Forward pass successful!')
print('Outputs:', outputs.keys())
"
```

### Check GPU memory usage
```bash
python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

### Validate checkpoint
```bash
python -c "
import torch
checkpoint = torch.load('outputs/best_model.pt', map_location='cpu')
print('Checkpoint keys:', checkpoint.keys())
print('Epoch:', checkpoint.get('epoch', 'N/A'))
print('Metrics:', checkpoint.get('metrics', 'N/A'))
"
```

## Performance Benchmarks

### Single GPU Training Times (NVIDIA RTX 3090)

| Configuration | Batch Size | Time/Epoch | Total (30 epochs) |
|--------------|-----------|------------|-------------------|
| Tiny (d=256) | 64 | 2 min | 1 hour |
| Small (d=512) | 32 | 5 min | 2.5 hours |
| Medium (d=768) | 16 | 12 min | 6 hours |
| Large (d=1024) | 8 | 25 min | 12.5 hours |

### Expected Memory Usage

| Configuration | Model Size | Peak Memory | Safe Batch Size |
|--------------|-----------|-------------|-----------------|
| Tiny (d=256) | 50M params | 4 GB | 128 |
| Small (d=512) | 150M params | 8 GB | 64 |
| Medium (d=768) | 350M params | 14 GB | 32 |
| Large (d=1024) | 600M params | 22 GB | 16 |

## Evaluation Commands

### During Training (automatic)
Metrics are computed every `--eval_every` epochs (default: 5)

### Manual Evaluation
```bash
python -c "
import torch
from nl_mm.models.nl_mm_model import NLMM
from src.utils.flickr8k_dataset import Flickr8kAudioDataset
from torch.utils.data import DataLoader

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pt')
cfg = checkpoint['config']
model = NLMM(cfg)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
test_ds = Flickr8kAudioDataset('flickr8k', 'test')
test_loader = DataLoader(test_ds, batch_size=32)

# Evaluate
# ... (add evaluation code)
"
```

### View Saved Metrics
```bash
# View JSON metrics
cat outputs/nlmm_flickr8k/metrics.json | python -m json.tool

# Or with jq
cat outputs/nlmm_flickr8k/metrics.json | jq .
```

### Generate Plots from Saved Metrics
```bash
python -c "
import json
import matplotlib.pyplot as plt

with open('outputs/nlmm_flickr8k/metrics.json') as f:
    metrics = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
print('Saved loss_curve.png')
"
```

## Export and Deployment

### Export to TorchScript
```bash
python nl_mm/export.py \
    --config nl_mm/configs/tiny_single_gpu.yaml \
    --checkpoint outputs/best_model.pt \
    --output nlmm_model.pt
```

### Load Exported Model
```python
import torch
model = torch.jit.load('nlmm_model.pt')
model.eval()

# Inference
batch = {
    'text': torch.randint(0, 1000, (1, 16)),
    'image': torch.randn(1, 3, 224, 224),
}
outputs, state = model(batch)
```

## Advanced Monitoring

### Custom Metrics Script
Create `monitor_training.py`:
```python
import json
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MetricsMonitor(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('metrics.json'):
            with open(event.src_path) as f:
                metrics = json.load(f)
            
            if metrics['epoch']:
                latest_epoch = metrics['epoch'][-1]
                latest_loss = metrics['train_loss'][-1]
                print(f"Epoch {latest_epoch}: Loss={latest_loss:.4f}")

observer = Observer()
observer.schedule(MetricsMonitor(), 'outputs/nlmm_flickr8k', recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

Run in separate terminal:
```bash
pip install watchdog
python monitor_training.py
```

## Tips for Faster Training

1. **Use larger batch sizes** (if GPU memory allows)
   ```bash
   --batch_size 64  # or 128
   ```

2. **Enable AMP** (automatic mixed precision)
   ```bash
   --use_amp
   ```

3. **Gradient accumulation** (simulate larger batch)
   ```bash
   --batch_size 16 --accumulation_steps 4  # effective batch=64
   ```

4. **More data workers**
   ```bash
   --num_workers 8
   ```

5. **Pin memory**
   ```python
   DataLoader(..., pin_memory=True)
   ```

6. **Reduce evaluation frequency**
   ```bash
   --eval_every 10  # instead of 5
   ```

## Summary Checklist

Before training:
- [ ] Dataset downloaded and extracted
- [ ] GPU available and recognized
- [ ] Config file reviewed
- [ ] Output directory created
- [ ] Monitoring tools ready

During training:
- [ ] Loss is decreasing
- [ ] GPU utilization >90%
- [ ] No OOM errors
- [ ] Retrieval metrics improving
- [ ] Checkpoints saving correctly

After training:
- [ ] Best model saved
- [ ] Metrics plotted
- [ ] Performance meets expectations
- [ ] Model exported if needed

## Getting Help

If you encounter issues:

1. Check the training log:
   ```bash
   tail -100 outputs/nlmm_flickr8k/training.log
   ```

2. Verify GPU status:
   ```bash
   nvidia-smi
   ```

3. Test with smaller config:
   ```bash
   python train_nlmm_flickr8k.py \
       --config nl_mm/configs/tiny_single_gpu.yaml \
       --epochs 5 \
       --batch_size 16
   ```

4. Enable debug mode:
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```

Good luck with your training! 🚀

