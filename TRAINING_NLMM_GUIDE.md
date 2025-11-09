# Training NL-MM on Real Data: Complete Guide

## Overview

The **NL-MM (Nested Learning Multimodal)** model is a cutting-edge multimodal architecture that can process text, images, and audio simultaneously. This guide explains how to train it on real datasets like Flickr8k.

## Quick Answer: Yes, You Can Use Flickr8k! ✅

You already have scripts set up for training on Flickr8k + Flickr Audio Caption Corpus (FACC). Here's how:

```bash
# 1. Download the Flickr8k dataset
bash download_flickr8k.sh

# 2. Train the model
python train_flickr8k.py --data_dir ./flickr8k --epochs 30 --batch_size 32
```

## What Kind of Data Do You Need?

### 1. **Multimodal Datasets** (Recommended)

The nl_mm model excels with multimodal data that includes 2-3 of these modalities:

#### **Tri-Modal: Text + Images + Audio**
- **Flickr8k + FACC** (You have this!)
  - 8,000 images
  - 40,000 text captions (5 per image)
  - 40,000 spoken audio captions
  - Perfect for cross-modal retrieval
  
- **AudioCaps**
  - YouTube audio clips with text descriptions
  
#### **Bi-Modal: Text + Images**
- **COCO Captions** (118K training images)
- **Conceptual Captions** (3M+ image-text pairs)
- **Visual Genome**

#### **Bi-Modal: Text + Audio**
- **Common Voice** (speech + transcriptions)
- **LibriSpeech** (audiobooks + text)

### 2. **Data Format Requirements**

The nl_mm model expects batches with these keys:

```python
batch = {
    "text": torch.Tensor,      # Shape: (batch_size, seq_len), token IDs
    "image": torch.Tensor,     # Shape: (batch_size, 3, H, W), RGB images
    "audio": torch.Tensor,     # Shape: (batch_size, 1, freq, time), spectrograms
    
    # Optional targets for supervised learning:
    "text_target": torch.Tensor,    # For next-token prediction
    "image_target": torch.Tensor,   # For reconstruction
    "audio_target": torch.Tensor,   # For reconstruction
}
```

## How to Train on Flickr8k (Step by Step)

### Step 1: Download the Dataset

```bash
# This script downloads Flickr8k images + text + FACC audio
bash download_flickr8k.sh
```

Expected structure:
```
flickr8k/
├── Flickr8k_Dataset/           # 8,000 images
├── Flickr8k_text/
│   ├── Flickr8k.token.txt      # Captions
│   ├── Flickr_8k.trainImages.txt
│   ├── Flickr_8k.devImages.txt
│   └── Flickr_8k.testImages.txt
└── flickr_audio/
    ├── wavs/                    # 40,000 .wav files
    └── wav2capt.txt             # Audio-to-caption mapping
```

### Step 2: Choose Your Training Script

You have **three options** depending on your goals:

#### Option A: Standard Training (Recommended for beginners)
```bash
python train_flickr8k.py \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --optimizer universal_anderson
```

#### Option B: Improved Training (Better hyperparameters)
```bash
python train_flickr8k_improved.py \
    --data_dir ./flickr8k \
    --epochs 50 \
    --batch_size 64 \
    --lr 5e-4
```

#### Option C: SGD Training (Stable baseline)
```bash
python train_flickr8k_sgd.py \
    --data_dir ./flickr8k \
    --epochs 40 \
    --batch_size 32
```

### Step 3: Monitor Training Progress

The training scripts automatically log:

#### **Console Output**
- Real-time progress bars with loss values
- Per-modality losses (image↔text, image↔audio, text↔audio)
- Evaluation metrics after each epoch

#### **Log Files**
Training logs are saved automatically:
```bash
# View training logs
tail -f flickr8k_training.log

# Or for improved version
tail -f flickr8k_improved_training.log
```

#### **Key Metrics to Watch**

1. **Training Losses** (should decrease):
   - `loss_i2t`: Image-to-text contrastive loss
   - `loss_i2a`: Image-to-audio contrastive loss
   - `loss_t2a`: Text-to-audio contrastive loss

2. **Retrieval Metrics** (should increase):
   - `i2t_r1`: Image→Text Recall@1 (% correct in top-1)
   - `t2i_r1`: Text→Image Recall@1
   - `i2a_r1`: Image→Audio Recall@1
   - Similar for R@5 and R@10

Example good progress:
```
Epoch 1:  i2t_r1=15%, t2i_r1=12%
Epoch 10: i2t_r1=35%, t2i_r1=32%
Epoch 30: i2t_r1=55%, t2i_r1=52%
```

### Step 4: Visualize Results

After training, visualize your results:

```bash
# Show training curves and metrics
python show_flickr8k_results.py
```

This generates plots showing:
- Loss curves over time
- Retrieval performance across epochs
- Cross-modal alignment quality

## Training the Core nl_mm Model

If you want to train the **core nl_mm architecture** (not the Flickr8k-specific wrapper), you'll need to adapt the data loading.

### Step 1: Create a Custom Data Loader

Create `train_nlmm_flickr8k.py`:

```python
"""Train core NL-MM model on Flickr8k."""
import torch
from torch.utils.data import DataLoader
from nl_mm.models.nl_mm_model import NLMM
from nl_mm.utils import load_config
from nl_mm.init import apply_nlmm_init
from src.utils.flickr8k_dataset import Flickr8kAudioDataset, collate_fn

# Load configuration
cfg = load_config("nl_mm/configs/tiny_single_gpu.yaml")

# Override vocab size for your tokenizer
cfg["vocab_size"] = 50000  # Adjust based on your tokenizer

# Create dataset
train_dataset = Flickr8kAudioDataset(
    root_dir="./flickr8k",
    split='train',
    image_size=224,
    audio_sample_rate=16000,
    n_mels=80,
    text_max_len=77,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NLMM(cfg).to(device)
apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))

# Configure optimizer
scheduler = model.configure_scheduler(cfg)

# Training loop
model.train()
for epoch in range(30):
    for batch_idx, batch in enumerate(train_loader):
        # Prepare batch for nl_mm format
        nl_batch = {
            "text": batch['text'].to(device),
            "image": batch['images'].to(device),
            "audio": batch['audio'].to(device),
            "text_target": batch['text'].to(device),  # Self-supervised
        }
        
        # Forward pass
        outputs, state = model(nl_batch)
        
        # Compute loss (you need to implement your loss function)
        loss = outputs["text"]  # Cross-entropy loss from text decoder
        
        # Backward pass
        loss.backward()
        
        # Update with NL scheduler
        scheduler.step_all(batch_idx)
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

### Step 2: Adapt Configuration

Edit `nl_mm/configs/flickr8k.yaml`:

```yaml
{
  "d_model": 512,
  "n_heads": 8,
  "ffn_mult": 4,
  "L_mem": 64,           # Increase memory for longer sequences
  "depth": {
    "text": 6,
    "image": 8,          # Vision needs more layers
    "audio": 6
  },
  "cms_levels": [
    {"name": "fast", "chunk_size": 1, "lr": 2.0e-4, "optimizer": "dmgd"},
    {"name": "mid", "chunk_size": 32, "lr": 1.0e-4, "optimizer": "dmgd"},
    {"name": "slow", "chunk_size": 512, "lr": 5.0e-5, "optimizer": "dmgd"}
  ],
  "ttt": {"enable": true, "eta": 1.0e-3, "max_steps": 3, "adapter_rank": 32},
  "precision": {"amp": "bf16", "grad_clip": 1.0},
  "max_position_embeddings": 2048,
  "vocab_size": 50000,
  "train_steps": 10000
}
```

## Monitoring Progress: Best Practices

### 1. **Use Weights & Biases (Recommended)**

Add W&B integration to track experiments:

```python
import wandb

# Initialize
wandb.init(project="nl-mm-flickr8k", config=cfg)

# Log during training
wandb.log({
    "train/loss": loss.item(),
    "train/i2t_loss": loss_i2t.item(),
    "eval/i2t_r1": i2t_r1,
    "epoch": epoch,
})
```

### 2. **TensorBoard (Built-in)**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/nlmm_flickr8k')

# Log metrics
writer.add_scalar('Loss/train', loss.item(), step)
writer.add_scalar('Retrieval/i2t_r1', i2t_r1, epoch)
```

View in browser:
```bash
tensorboard --logdir runs/nlmm_flickr8k
```

### 3. **Save Checkpoints Regularly**

```python
# Save checkpoint every N epochs
if epoch % 5 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
    }
    torch.save(checkpoint, f'checkpoints/nlmm_epoch_{epoch}.pt')
```

### 4. **Monitor GPU Usage**

```bash
# In a separate terminal
watch -n 1 nvidia-smi

# Or use
pip install gpustat
watch -n 1 gpustat
```

### 5. **Track Key Metrics**

Create a simple metrics tracker:

```python
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'i2t_r1': [],
            't2i_r1': [],
            'i2a_r1': [],
        }
    
    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot losses
        axes[0, 0].plot(self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['eval_loss'], label='Eval')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # Plot retrieval metrics
        axes[0, 1].plot(self.metrics['i2t_r1'], label='Image→Text')
        axes[0, 1].plot(self.metrics['t2i_r1'], label='Text→Image')
        axes[0, 1].set_title('Retrieval R@1')
        axes[0, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("Saved training_progress.png")

# Usage
tracker = MetricsTracker()
for epoch in range(num_epochs):
    # ... training ...
    tracker.add(
        train_loss=train_loss,
        i2t_r1=i2t_r1,
        t2i_r1=t2i_r1,
    )
    
    if epoch % 10 == 0:
        tracker.plot()
```

## Expected Training Times

On a single GPU (NVIDIA RTX 3090 / A100):

| Dataset | Batch Size | Epochs | Time per Epoch | Total Time |
|---------|-----------|--------|----------------|------------|
| Flickr8k (8K) | 32 | 30 | ~5 min | ~2.5 hours |
| Flickr8k (8K) | 64 | 50 | ~3 min | ~2.5 hours |
| COCO (118K) | 32 | 30 | ~45 min | ~22 hours |
| Conceptual Captions (3M) | 64 | 5 | ~8 hours | ~40 hours |

## What Success Looks Like

### Good Training Signs ✅
- Loss steadily decreases (not oscillating wildly)
- Retrieval R@1 improves from ~1% to 40-60% on Flickr8k
- R@5 and R@10 show even better improvement
- Model can retrieve correct captions for test images
- Cross-modal alignment is semantically meaningful

### Warning Signs ⚠️
- Loss plateaus early (learning rate too low)
- Loss explodes (learning rate too high)
- Metrics don't improve after 10+ epochs (architecture issue)
- GPU memory errors (batch size too large)

## Tips for Better Results

### 1. **Data Augmentation**
Already included in `Flickr8kAudioDataset`:
- Random crops for images
- Horizontal flips
- Color jittering (optional)

### 2. **Learning Rate Scheduling**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### 3. **Test-Time Training (TTT)**
Enable TTT during evaluation for better adaptation:
```python
# Enable TTT for test-time adaptation
metrics = evaluate(model, val_loader, device, enable_ttt=True)
```

### 4. **Gradient Accumulation** (for larger effective batch sizes)
```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = compute_loss(model, batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Next Steps After Training

1. **Evaluate on test set**
   ```bash
   python test_multimodal_trained.py --checkpoint best_model.pt
   ```

2. **Export for inference**
   ```bash
   python nl_mm/export.py --config nl_mm/configs/flickr8k.yaml --output model.pt
   ```

3. **Demo the model**
   ```bash
   python demo_multimodal_memory.py --model best_model.pt
   ```

4. **Continue training** (if needed)
   ```bash
   python continue_training_flickr8k.py --checkpoint best_model.pt --epochs 20
   ```

## Troubleshooting

### Out of Memory?
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--image_size 128`
- Enable gradient checkpointing
- Use mixed precision: `--use_amp`

### Training Too Slow?
- Increase `num_workers` in DataLoader
- Use larger batch size if memory allows
- Pre-compute image features with frozen CLIP

### Poor Performance?
- Train longer (50+ epochs)
- Increase model size (`d_model=768`)
- Use better data augmentation
- Try different optimizers

## Summary

**Yes, you can absolutely train nl_mm on Flickr8k!** You have everything you need:

1. ✅ Dataset loader: `src/utils/flickr8k_dataset.py`
2. ✅ Training scripts: `train_flickr8k.py`, `train_flickr8k_improved.py`
3. ✅ Model architecture: `nl_mm/models/nl_mm_model.py`
4. ✅ Evaluation code: Built-in retrieval metrics
5. ✅ Monitoring: Console logs, checkpoint saving

**Quick start command:**
```bash
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```

The model will learn to:
- Align images with text descriptions
- Align images with spoken audio
- Align text with audio
- Retrieve correct captions for images (and vice versa)
- Use test-time training for better adaptation

Good luck with your training! 🚀

