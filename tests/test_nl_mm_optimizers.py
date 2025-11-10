"""
Test different optimizers for nl_mm model on Flickr8k.

This script trains the nl_mm model with different optimizers and compares:
- Training speed
- Final accuracy
- Memory usage
- Convergence behavior
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.nl_mm.models.nl_mm_model import NLMM
from modules.nl_mm.utils import load_config
from modules.nl_mm.init import apply_nlmm_init
from utils.flickr8k_dataset import Flickr8kAudioDataset, collate_fn
from utils import get_device


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07):
    """CLIP-style InfoNCE contrastive loss."""
    if query is None or key is None:
        return None
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(len(query), device=query.device)
    loss_q2k = F.cross_entropy(logits, labels)
    loss_k2q = F.cross_entropy(logits.T, labels)
    return (loss_q2k + loss_k2q) / 2


def create_optimizer(model, opt_name, lr=1e-3):
    """Create optimizer based on name."""
    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def train_with_optimizer(model, train_loader, val_loader, optimizer, opt_name, 
                         device, epochs=5, use_amp=True):
    """Train model with specific optimizer and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing Optimizer: {opt_name.upper()}")
    print(f"{'='*60}")
    
    scaler = GradScaler(device='cuda', enabled=use_amp)
    
    metrics = {
        'train_losses': [],
        'val_r1': [],
        'epoch_times': [],
        'total_time': 0,
        'final_loss': 0,
        'final_r1': 0,
        'best_r1': 0,
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in pbar:
            # Prepare batch
            audio = batch['audio'].to(device)
            B, C, n_mels, t = audio.shape
            audio_flat = audio.reshape(B, C, n_mels * t)
            
            nl_batch = {
                "text": batch['text'].to(device),
                "image": batch['images'].to(device),
                "audio": audio_flat,
                "text_target": batch['text'].to(device),
            }
            
            # Forward pass
            with autocast(device_type='cuda', enabled=use_amp):
                outputs, state = model(nl_batch, return_embeddings=True)
                
                # Compute loss
                loss = outputs.get("text", torch.tensor(0.0, device=device))
                
                if "embeddings" in outputs:
                    embs = outputs["embeddings"]
                    loss_i2t = info_nce_loss(embs.get("image"), embs.get("text"))
                    if loss_i2t is not None:
                        loss = loss + 0.5 * loss_i2t
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        metrics['train_losses'].append(avg_loss)
        
        # Evaluate
        val_r1 = evaluate_quick(model, val_loader, device)
        metrics['val_r1'].append(val_r1)
        
        if val_r1 > metrics['best_r1']:
            metrics['best_r1'] = val_r1
        
        epoch_time = time.time() - epoch_start
        metrics['epoch_times'].append(epoch_time)
        
        print(f"  Loss: {avg_loss:.4f} | R@1: {val_r1:.2f}% | Time: {epoch_time:.1f}s")
    
    metrics['total_time'] = time.time() - start_time
    metrics['final_loss'] = metrics['train_losses'][-1]
    metrics['final_r1'] = metrics['val_r1'][-1]
    
    return metrics


@torch.no_grad()
def evaluate_quick(model, dataloader, device, max_batches=50):
    """Quick evaluation on subset of validation data."""
    model.eval()
    
    all_text_embs = []
    all_image_embs = []
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        audio = batch['audio'].to(device)
        B, C, n_mels, t = audio.shape
        audio_flat = audio.reshape(B, C, n_mels * t)
        
        nl_batch = {
            "text": batch['text'].to(device),
            "image": batch['images'].to(device),
            "audio": audio_flat,
        }
        
        outputs, _ = model(nl_batch, return_embeddings=True)
        
        if "embeddings" in outputs:
            embs = outputs["embeddings"]
            if "text" in embs:
                all_text_embs.append(embs["text"])
            if "image" in embs:
                all_image_embs.append(embs["image"])
    
    if all_text_embs and all_image_embs:
        text_embs = F.normalize(torch.cat(all_text_embs), dim=-1)
        image_embs = F.normalize(torch.cat(all_image_embs), dim=-1)
        
        # Compute R@1
        sim = torch.matmul(image_embs, text_embs.T)
        ranks = torch.argsort(sim, dim=1, descending=True)
        gt = torch.arange(len(image_embs), device=image_embs.device)
        r1 = (ranks[:, 0] == gt).float().mean().item() * 100
        return r1
    
    return 0.0


def plot_comparison(all_metrics, output_dir):
    """Plot comparison of all optimizers."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training loss
    ax = axes[0, 0]
    for opt_name, metrics in all_metrics.items():
        ax.plot(metrics['train_losses'], label=opt_name, marker='o', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation R@1
    ax = axes[0, 1]
    for opt_name, metrics in all_metrics.items():
        ax.plot(metrics['val_r1'], label=opt_name, marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R@1 (%)')
    ax.set_title('Validation R@1 Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epoch time
    ax = axes[0, 2]
    opt_names = list(all_metrics.keys())
    avg_times = [sum(m['epoch_times']) / len(m['epoch_times']) for m in all_metrics.values()]
    ax.bar(opt_names, avg_times, color='skyblue', edgecolor='navy')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Epoch Time')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Final metrics comparison
    ax = axes[1, 0]
    final_losses = [m['final_loss'] for m in all_metrics.values()]
    ax.bar(opt_names, final_losses, color='lightcoral', edgecolor='darkred')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Training Loss')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Best R@1
    ax = axes[1, 1]
    best_r1 = [m['best_r1'] for m in all_metrics.values()]
    ax.bar(opt_names, best_r1, color='lightgreen', edgecolor='darkgreen')
    ax.set_ylabel('Best R@1 (%)')
    ax.set_title('Best Validation R@1')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Total training time
    ax = axes[1, 2]
    total_times = [m['total_time'] / 60 for m in all_metrics.values()]  # In minutes
    ax.bar(opt_names, total_times, color='plum', edgecolor='purple')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Total Training Time')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved plot to {output_dir / 'optimizer_comparison.png'}")
    plt.close()


def print_summary(all_metrics):
    """Print summary table of all optimizers."""
    print(f"\n{'='*80}")
    print("OPTIMIZER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Optimizer':<12} {'Final Loss':<12} {'Best R@1':<12} {'Avg Time/Epoch':<15} {'Total Time':<12}")
    print(f"{'-'*80}")
    
    for opt_name, metrics in all_metrics.items():
        avg_time = sum(metrics['epoch_times']) / len(metrics['epoch_times'])
        total_time = metrics['total_time'] / 60  # In minutes
        print(f"{opt_name:<12} {metrics['final_loss']:<12.4f} {metrics['best_r1']:<12.2f} "
              f"{avg_time:<15.1f} {total_time:<12.1f}")
    
    print(f"{'='*80}")
    
    # Find best optimizer for each metric
    best_loss = min(all_metrics.items(), key=lambda x: x[1]['final_loss'])
    best_r1 = max(all_metrics.items(), key=lambda x: x[1]['best_r1'])
    fastest = min(all_metrics.items(), key=lambda x: x[1]['total_time'])
    
    print(f"\n🏆 Best Results:")
    print(f"  Lowest Loss:  {best_loss[0]} ({best_loss[1]['final_loss']:.4f})")
    print(f"  Highest R@1:  {best_r1[0]} ({best_r1[1]['best_r1']:.2f}%)")
    print(f"  Fastest:      {fastest[0]} ({fastest[1]['total_time']/60:.1f} min)")
    print()


def main(args):
    device = get_device()
    print(f"🚀 NL-MM Optimizer Comparison Test")
    print(f"   Device: {device}")
    print(f"   Config: {args.config}")
    print(f"   Epochs per optimizer: {args.epochs}")
    
    # Load config
    cfg = load_config(args.config)
    cfg['batch_size'] = args.batch_size
    
    print(f"\n📊 Model Configuration:")
    print(f"   d_model: {cfg['d_model']}")
    print(f"   depth: {cfg['depth']}")
    print(f"   batch_size: {args.batch_size}")
    
    # Create datasets
    print(f"\n📂 Loading datasets...")
    train_dataset = Flickr8kAudioDataset(
        root_dir=args.data_dir,
        split='train',
        image_size=args.image_size,
        audio_sample_rate=16000,
        n_mels=80,
        text_max_len=77,
    )
    
    val_dataset = Flickr8kAudioDataset(
        root_dir=args.data_dir,
        split='val',
        image_size=args.image_size,
        audio_sample_rate=16000,
        n_mels=80,
        text_max_len=77,
    )
    
    # Use subset for faster testing
    if args.subset:
        from torch.utils.data import Subset
        train_indices = torch.randperm(len(train_dataset))[:args.subset].tolist()
        val_indices = torch.randperm(len(val_dataset))[:min(1000, len(val_dataset))].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Test each optimizer
    optimizers_to_test = args.optimizers.split(',')
    all_metrics = {}
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for opt_name in optimizers_to_test:
        # Create fresh model for each optimizer
        print(f"\n🏗️  Creating model for {opt_name}...")
        model = NLMM(cfg).to(device)
        apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))
        
        # Create optimizer
        optimizer = create_optimizer(model, opt_name, lr=args.lr)
        
        # Train and collect metrics
        metrics = train_with_optimizer(
            model, train_loader, val_loader, optimizer, opt_name,
            device, epochs=args.epochs, use_amp=args.use_amp
        )
        
        all_metrics[opt_name] = metrics
        
        # Clear GPU memory
        del model
        del optimizer
        torch.cuda.empty_cache()
    
    # Save results
    with open(output_dir / 'optimizer_comparison.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    print_summary(all_metrics)
    
    # Plot comparison
    plot_comparison(all_metrics, output_dir)
    
    print(f"\n✅ Comparison complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   - optimizer_comparison.json")
    print(f"   - optimizer_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare optimizers for nl_mm model")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="./flickr8k",
                        help="Path to Flickr8k dataset")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size")
    
    # Model
    parser.add_argument("--config", type=str, default="modules/nl_mm/configs/nano_8gb.yaml",
                        help="Path to config file")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs per optimizer")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    
    # Optimizers
    parser.add_argument("--optimizers", type=str, 
                        default="adam,adamw,sgd,rmsprop",
                        help="Comma-separated list of optimizers to test")
    
    # Testing
    parser.add_argument("--subset", type=int, default=5000,
                        help="Use subset of data (0 for full dataset)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results/folder_per_model/nl_mm/outputs/optimizer_comparison",
                        help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    
    args = parser.parse_args()
    main(args)

