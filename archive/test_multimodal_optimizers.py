"""Test different optimizers on Multimodal Memory Network.

Tests various optimizers to find the best one for tri-modal learning:
- Adam (baseline)
- AdamW
- SGD with momentum
- UniversalMuon
- UniversalAndersonGDA
- Custom optimizers (CustomAdam, GDA2)

Evaluates on:
1. Training convergence speed
2. Cross-modal retrieval accuracy (I↔T, I↔A, T↔A)
3. Memory efficiency
4. Training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from pathlib import Path
import json

from src.architectures import MultiModalMemoryNetwork
from src.optimizers import UniversalMuon, UniversalAndersonGDA
from src.utils import get_device
from train_multimodal import SyntheticMultimodalDataset, contrastive_loss


device = get_device()
print(f"Using device: {device}\n")

# Create small synthetic dataset for quick testing
print("Creating synthetic multimodal dataset...")
train_dataset = SyntheticMultimodalDataset(
    num_samples=1000,
    num_classes=10,
    seq_len=16,
    vocab_size=1000,
    image_size=64,
    audio_size=64,
    seed=42,
)

val_dataset = SyntheticMultimodalDataset(
    num_samples=200,
    num_classes=10,
    seq_len=16,
    vocab_size=1000,
    image_size=64,
    audio_size=64,
    seed=123,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """CLIP-style InfoNCE loss (symmetric)."""
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    
    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(len(query), device=query.device)
    
    loss_q2k = F.cross_entropy(logits, labels)
    loss_k2q = F.cross_entropy(logits.T, labels)
    
    return (loss_q2k + loss_k2q) / 2


def train_epoch(model, dataloader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_i2t = 0
    total_i2a = 0
    total_t2a = 0
    
    for batch in dataloader:
        text = batch['text'].to(device)
        images = batch['image'].to(device)
        audio = batch['audio'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # Get modality embeddings
            text_outputs = model(text=text)
            image_outputs = model(images=images)
            audio_outputs = model(audio=audio)
            
            text_emb = text_outputs['central_latent']
            image_emb = image_outputs['central_latent']
            audio_emb = audio_outputs['central_latent']
            
            # InfoNCE losses
            loss_i2t = info_nce_loss(image_emb, text_emb)
            loss_i2a = info_nce_loss(image_emb, audio_emb)
            loss_t2a = info_nce_loss(text_emb, audio_emb)
            
            loss = loss_i2t + loss_i2a + loss_t2a
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_i2t += loss_i2t.item()
        total_i2a += loss_i2a.item()
        total_t2a += loss_t2a.item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'i2t_loss': total_i2t / n,
        'i2a_loss': total_i2a / n,
        't2a_loss': total_t2a / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate cross-modal retrieval."""
    model.eval()
    
    all_image_embs = []
    all_text_embs = []
    all_audio_embs = []
    
    for batch in dataloader:
        text = batch['text'].to(device)
        images = batch['image'].to(device)
        audio = batch['audio'].to(device)
        
        text_emb = model(text=text)['central_latent']
        image_emb = model(images=images)['central_latent']
        audio_emb = model(audio=audio)['central_latent']
        
        all_text_embs.append(text_emb)
        all_image_embs.append(image_emb)
        all_audio_embs.append(audio_emb)
    
    text_embs = F.normalize(torch.cat(all_text_embs), dim=-1)
    image_embs = F.normalize(torch.cat(all_image_embs), dim=-1)
    audio_embs = F.normalize(torch.cat(all_audio_embs), dim=-1)
    
    # Compute R@1
    def recall_at_1(query, keys):
        sim_matrix = torch.matmul(query, keys.T)
        preds = sim_matrix.argmax(dim=1)
        correct = (preds == torch.arange(len(query), device=query.device)).float().mean()
        return correct.item() * 100
    
    metrics = {
        'i2t_r1': recall_at_1(image_embs, text_embs),
        't2i_r1': recall_at_1(text_embs, image_embs),
        'i2a_r1': recall_at_1(image_embs, audio_embs),
        'a2i_r1': recall_at_1(audio_embs, image_embs),
        't2a_r1': recall_at_1(text_embs, audio_embs),
        'a2t_r1': recall_at_1(audio_embs, text_embs),
    }
    
    metrics['avg_r1'] = sum(metrics.values()) / 6
    
    return metrics


def test_optimizer(opt_name, opt_factory, num_epochs=20):
    """Test an optimizer on the multimodal network."""
    print(f"\n{'='*80}")
    print(f"Testing: {opt_name}")
    print(f"{'='*80}\n")
    
    # Create fresh model
    model = MultiModalMemoryNetwork(
        vocab_size=1000,
        text_embed_dim=256,
        text_seq_len=32,
        image_size=64,
        patch_size=8,
        image_channels=3,
        audio_channels=1,
        latent_dim=256,
        memory_size=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        enable_ttt_updates=False,  # Disabled during training
    ).to(device)
    
    # Create optimizer
    try:
        optimizer = opt_factory(model)
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        return None
    
    # Training setup
    scaler = GradScaler()
    
    start_time = time.time()
    history = []
    
    try:
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
            
            # Evaluate
            val_metrics = evaluate(model, val_loader, device)
            
            # Record
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_avg_r1': val_metrics['avg_r1'],
                'val_i2t_r1': val_metrics['i2t_r1'],
                'val_i2a_r1': val_metrics['i2a_r1'],
                'val_t2a_r1': val_metrics['t2a_r1'],
            }
            history.append(epoch_result)
            
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Loss={train_metrics['loss']:.4f}, "
                  f"Avg R@1={val_metrics['avg_r1']:.2f}%, "
                  f"I↔T={val_metrics['i2t_r1']:.1f}%, "
                  f"I↔A={val_metrics['i2a_r1']:.1f}%, "
                  f"T↔A={val_metrics['t2a_r1']:.1f}%")
        
        elapsed_time = time.time() - start_time
        
        # Final metrics
        final_val = history[-1]
        best_val = max(history, key=lambda x: x['val_avg_r1'])
        
        result = {
            'optimizer': opt_name,
            'final_avg_r1': final_val['val_avg_r1'],
            'best_avg_r1': best_val['val_avg_r1'],
            'best_epoch': best_val['epoch'],
            'final_loss': final_val['train_loss'],
            'i2t_r1': final_val['val_i2t_r1'],
            'i2a_r1': final_val['val_i2a_r1'],
            't2a_r1': final_val['val_t2a_r1'],
            'time_per_epoch': elapsed_time / num_epochs,
            'total_time': elapsed_time,
            'converged': final_val['val_avg_r1'] > 1.0,  # Basic convergence check
            'stable': max(h['train_loss'] for h in history) < 10.0,  # Didn't explode
            'history': history,
            'status': '✅',
        }
        
        print(f"\n✅ Completed: Avg R@1={result['final_avg_r1']:.2f}% "
              f"(best: {result['best_avg_r1']:.2f}% at epoch {result['best_epoch']})")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Failed: {str(e)[:100]}")
        return {
            'optimizer': opt_name,
            'final_avg_r1': 0.0,
            'best_avg_r1': 0.0,
            'best_epoch': 0,
            'final_loss': 999.0,
            'i2t_r1': 0.0,
            'i2a_r1': 0.0,
            't2a_r1': 0.0,
            'time_per_epoch': elapsed_time / max(1, len(history)) if history else 0,
            'total_time': elapsed_time,
            'converged': False,
            'stable': False,
            'history': history,
            'status': '❌',
            'error': str(e),
        }


# Define optimizer configurations
optimizer_configs = [
    {
        'name': 'Adam',
        'factory': lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    },
    {
        'name': 'AdamW',
        'factory': lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
    },
    {
        'name': 'SGD',
        'factory': lambda model: torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9),
    },
    {
        'name': 'UniversalMuon (auto)',
        'factory': lambda model: UniversalMuon(
            model.parameters(),
            lr=1e-3,
            ortho_mode='auto',
            ortho_threshold=128,
            scale_mode='adaptive',
        ),
    },
    {
        'name': 'UniversalMuon (preserve)',
        'factory': lambda model: UniversalMuon(
            model.parameters(),
            lr=1e-3,
            ortho_mode='preserve_magnitude',
            ortho_threshold=128,
            scale_mode='adaptive',
        ),
    },
    {
        'name': 'UniversalAndersonGDA',
        'factory': lambda model: UniversalAndersonGDA(
            model.parameters(),
            lr=1e-3,
            anderson_m=3,
            use_weighting=True,
            trust_region=1.5,
        ),
    },
]

# Try to import custom optimizers
try:
    from src.optimizers import CustomAdam, GDA2
    
    optimizer_configs.extend([
        {
            'name': 'CustomAdam',
            'factory': lambda model: CustomAdam(model.parameters(), lr=1e-3),
        },
        {
            'name': 'GDA2',
            'factory': lambda model: GDA2(model.parameters(), lr=1e-3),
        },
    ])
    print("✓ Loaded CustomAdam and GDA2")
except ImportError as e:
    print(f"Note: Could not import CustomAdam/GDA2: {e}")

print("\n" + "="*80)
print("MULTIMODAL NETWORK OPTIMIZER COMPARISON")
print("="*80)
print(f"\nTesting {len(optimizer_configs)} optimizers on multimodal network")
print(f"Architecture: Text (Transformer) + Image (ViT) + Audio (CNN)")
print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
print(f"Epochs per optimizer: 5")
print(f"Objective: CLIP-style InfoNCE (I↔T, I↔A, T↔A)")

# Run tests
results = []

for config in optimizer_configs:
    result = test_optimizer(config['name'], config['factory'], num_epochs=20)
    if result is not None:
        results.append(result)

# Sort by final average R@1
results.sort(key=lambda x: x['final_avg_r1'], reverse=True)

# Print summary
print("\n" + "="*80)
print("RESULTS SUMMARY (sorted by final Avg R@1)")
print("="*80)

print(f"\n{'Optimizer':<30} {'Avg R@1':<12} {'I↔T R@1':<10} {'I↔A R@1':<10} {'T↔A R@1':<10} {'Time/Epoch':<12} {'Status':<8}")
print("-" * 100)

for result in results:
    print(f"{result['optimizer']:<30} "
          f"{result['final_avg_r1']:<12.2f} "
          f"{result['i2t_r1']:<10.2f} "
          f"{result['i2a_r1']:<10.2f} "
          f"{result['t2a_r1']:<10.2f} "
          f"{result['time_per_epoch']:<12.1f}s "
          f"{result['status']:<8}")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Best overall
best = results[0]
print(f"\n🥇 Best Overall: {best['optimizer']}")
print(f"   Final Avg R@1: {best['final_avg_r1']:.2f}%")
print(f"   Best Avg R@1: {best['best_avg_r1']:.2f}% (epoch {best['best_epoch']})")
print(f"   Cross-modal: I↔T={best['i2t_r1']:.1f}%, I↔A={best['i2a_r1']:.1f}%, T↔A={best['t2a_r1']:.1f}%")

# Fastest
fastest = min(results, key=lambda x: x['time_per_epoch'])
print(f"\n⚡ Fastest: {fastest['optimizer']}")
print(f"   Time per epoch: {fastest['time_per_epoch']:.1f}s")
print(f"   Avg R@1: {fastest['final_avg_r1']:.2f}%")

# Most stable (lowest loss)
stable = min([r for r in results if r['converged']], key=lambda x: x['final_loss']) if any(r['converged'] for r in results) else None
if stable:
    print(f"\n🔒 Most Stable: {stable['optimizer']}")
    print(f"   Final loss: {stable['final_loss']:.4f}")
    print(f"   Avg R@1: {stable['final_avg_r1']:.2f}%")

# Best for each modality pair
i2t_best = max(results, key=lambda x: x['i2t_r1'])
i2a_best = max(results, key=lambda x: x['i2a_r1'])
t2a_best = max(results, key=lambda x: x['t2a_r1'])

print(f"\n📊 Best by Modality Pair:")
print(f"   I↔T: {i2t_best['optimizer']} ({i2t_best['i2t_r1']:.2f}%)")
print(f"   I↔A: {i2a_best['optimizer']} ({i2a_best['i2a_r1']:.2f}%)")
print(f"   T↔A: {t2a_best['optimizer']} ({t2a_best['t2a_r1']:.2f}%)")

# Universal optimizers performance
print(f"\n🌐 Universal Optimizers Performance:")
for result in results:
    if 'Universal' in result['optimizer']:
        print(f"   {result['optimizer']}: {result['final_avg_r1']:.2f}% Avg R@1, "
              f"{result['time_per_epoch']:.1f}s/epoch")

# Convergence rate
converged = [r for r in results if r['converged']]
print(f"\n✅ Converged: {len(converged)}/{len(results)} optimizers")

if not converged:
    print("   ⚠️ Note: Low convergence suggests need for more epochs or hyperparameter tuning")

# Save results
output_file = 'multimodal_optimizer_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'results': results,
        'summary': {
            'best_optimizer': best['optimizer'],
            'best_avg_r1': best['final_avg_r1'],
            'fastest_optimizer': fastest['optimizer'],
            'fastest_time': fastest['time_per_epoch'],
            'num_tested': len(results),
            'num_converged': len(converged),
        }
    }, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Recommendation logic
if best['final_avg_r1'] > 5.0:
    print(f"\n✅ Recommended: {best['optimizer']}")
    print(f"   Achieves best cross-modal alignment ({best['final_avg_r1']:.2f}% Avg R@1)")
    
    if fastest['optimizer'] != best['optimizer'] and fastest['time_per_epoch'] < best['time_per_epoch'] * 0.7:
        print(f"\n💡 Alternative (faster): {fastest['optimizer']}")
        print(f"   {fastest['time_per_epoch']/best['time_per_epoch']*100:.0f}% of best's time, "
              f"{fastest['final_avg_r1']/best['final_avg_r1']*100:.0f}% of best's performance")
else:
    print(f"\n⚠️ All optimizers showed weak convergence (<5% R@1)")
    print(f"   Consider:")
    print(f"   - Training for more epochs (tested only 5)")
    print(f"   - Adjusting learning rates")
    print(f"   - Using learning rate schedules")
    print(f"   - Checking data quality")

print("\n" + "="*80)

