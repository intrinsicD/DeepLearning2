"""Train improved Flickr8k model with BPE tokenizer and scaled-up architecture.

Improvements:
- BPE tokenizer (8K vocab instead of 33 chars)
- Scaled-up model (but still fits single GPU)
- Better training with data augmentation
- SGD optimizer (proven best)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import time

from architectures import MultiModalMemoryNetwork
from utils.flickr8k_improved import Flickr8kImprovedDataset, collate_fn
from utils import get_device


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """CLIP-style InfoNCE loss (symmetric)."""
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    
    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(len(query), device=query.device)
    
    loss_q2k = F.cross_entropy(logits, labels)
    loss_k2q = F.cross_entropy(logits.T, labels)
    
    return (loss_q2k + loss_k2q) / 2


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['images'].to(device)
        text = batch['text'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            text_outputs = model(text=text)
            image_outputs = model(images=images)
            
            text_emb = text_outputs['central_latent']
            image_emb = image_outputs['central_latent']
            
            loss = info_nce_loss(image_emb, text_emb)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'loss': total_loss / len(dataloader)}


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate image-text retrieval."""
    model.eval()
    
    all_image_embs = []
    all_text_embs = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        text = batch['text'].to(device)
        
        image_emb = model(images=images)['central_latent']
        text_emb = model(text=text)['central_latent']
        
        all_image_embs.append(image_emb)
        all_text_embs.append(text_emb)
    
    image_embs = F.normalize(torch.cat(all_image_embs), dim=-1)
    text_embs = F.normalize(torch.cat(all_text_embs), dim=-1)
    
    def compute_recall(query, keys, k_values=[1, 5, 10]):
        sim_matrix = torch.matmul(query, keys.T)
        ranks = torch.argsort(sim_matrix, dim=1, descending=True)
        gt_indices = torch.arange(len(query), device=query.device)
        
        recalls = {}
        for k in k_values:
            correct = (ranks[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
            recalls[f'R@{k}'] = correct.float().mean().item() * 100
        return recalls
    
    i2t = compute_recall(image_embs, text_embs)
    t2i = compute_recall(text_embs, image_embs)
    
    metrics = {
        'i2t_r1': i2t['R@1'], 'i2t_r5': i2t['R@5'], 'i2t_r10': i2t['R@10'],
        't2i_r1': t2i['R@1'], 't2i_r5': t2i['R@5'], 't2i_r10': t2i['R@10'],
    }
    metrics['avg_r1'] = (metrics['i2t_r1'] + metrics['t2i_r1']) / 2
    
    return metrics


def main(args):
    device = get_device()
    print(f"Using device: {device}\n")
    
    print("="*80)
    print("IMPROVED FLICKR8K TRAINING")
    print("="*80)
    print("\nImprovements:")
    print(f"  ✓ BPE tokenizer ({args.vocab_size} vocab vs 33 chars)")
    print(f"  ✓ Scaled-up model ({args.latent_dim}D, {args.num_layers} layers)")
    print(f"  ✓ Larger memory ({args.memory_size} slots)")
    print(f"  ✓ Data augmentation (ColorJitter, RandomCrop)")
    print(f"  ✓ SGD optimizer (proven best)")
    print()
    
    # Create datasets with BPE tokenizer
    print("Loading Flickr8k with BPE tokenizer...")
    train_dataset = Flickr8kImprovedDataset(
        root_dir=args.data_dir,
        split='train',
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_length=77,
    )
    
    val_dataset = Flickr8kImprovedDataset(
        root_dir=args.data_dir,
        split='val',
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_length=77,
    )
    
    # Create dataloaders
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create improved model
    print("\nCreating improved model...")
    print(f"  Latent dim: {args.latent_dim} (was 512)")
    print(f"  Num layers: {args.num_layers} (was 4)")
    print(f"  Memory size: {args.memory_size} (was 128)")
    print(f"  Vocab size: {args.vocab_size} (was 34)")
    
    model = MultiModalMemoryNetwork(
        vocab_size=args.vocab_size,
        text_embed_dim=args.latent_dim,
        text_seq_len=77,
        image_size=args.image_size,
        patch_size=16,
        image_channels=3,
        audio_channels=1,
        latent_dim=args.latent_dim,
        memory_size=args.memory_size,
        num_heads=8,
        num_layers=args.num_layers,
        dropout=0.1,
        enable_ttt_updates=False,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB (fp32)")
    
    # Check GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    # Create SGD optimizer
    print(f"\nUsing SGD optimizer (lr={args.lr}, momentum={args.momentum})")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=1e-4,
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_recall = 0.0
    history = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Learning rate step
        scheduler.step()
        
        avg_recall = val_metrics['avg_r1']
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  I→T: R@1={val_metrics['i2t_r1']:.2f}%, R@5={val_metrics['i2t_r5']:.2f}%, R@10={val_metrics['i2t_r10']:.2f}%")
        print(f"  T→I: R@1={val_metrics['t2i_r1']:.2f}%, R@5={val_metrics['t2i_r5']:.2f}%, R@10={val_metrics['t2i_r10']:.2f}%")
        print(f"  Avg R@1: {avg_recall:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': scheduler.get_last_lr()[0],
        })
        
        if avg_recall > best_recall:
            best_recall = avg_recall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args),
            }, args.output_dir / 'best_model_improved.pt')
            print(f"  ✓ Saved best model (avg R@1: {avg_recall:.2f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Save history
    with open(args.output_dir / 'history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best average R@1: {best_recall:.2f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Time per epoch: {elapsed_time/args.epochs:.1f} seconds")
    
    # Test set evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    test_dataset = Flickr8kImprovedDataset(
        root_dir=args.data_dir,
        split='test',
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_length=77,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    checkpoint = torch.load(args.output_dir / 'best_model_improved.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Set Results:")
    print(f"  I→T: R@1={test_metrics['i2t_r1']:.2f}%, R@5={test_metrics['i2t_r5']:.2f}%, R@10={test_metrics['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={test_metrics['t2i_r1']:.2f}%, R@5={test_metrics['t2i_r5']:.2f}%, R@10={test_metrics['t2i_r10']:.2f}%")
    print(f"  Avg R@1: {test_metrics['avg_r1']:.2f}%")
    
    # Compare to baseline
    print("\n" + "="*80)
    print("IMPROVEMENT OVER BASELINE")
    print("="*80)
    print(f"  Baseline (33 chars): 0.36% R@1")
    print(f"  Improved (8K BPE):   {best_recall:.2f}% R@1")
    print(f"  Gain: {best_recall/0.36:.1f}x improvement")
    
    with open(args.output_dir / 'test_results_improved.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nModel saved to: {args.output_dir / 'best_model_improved.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train improved Flickr8k model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/flickr8k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Reduced from 64 due to larger model')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model - scaled up but GPU-friendly
    parser.add_argument('--vocab_size', type=int, default=8192,
                       help='BPE vocab size (was 34 chars)')
    parser.add_argument('--latent_dim', type=int, default=768,
                       help='Increased from 512')
    parser.add_argument('--memory_size', type=int, default=256,
                       help='Increased from 128')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Increased from 4')
    
    # Training - SGD with warmup
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results/folder_per_model/multimodal_memory/outputs/flickr8k_improved')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)

