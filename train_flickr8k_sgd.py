"""Train Multimodal Memory Network on Flickr8k (Image-Text) using SGD.

Uses the best optimizer (SGD) from comprehensive testing.
CLIP-style InfoNCE for image-text alignment.
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

from src.architectures import MultiModalMemoryNetwork
from src.utils.flickr8k_simple import Flickr8kImageTextDataset, collate_fn
from src.utils import get_device


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
    total_i2t = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        text = batch['text'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # Get embeddings
            text_outputs = model(text=text)
            image_outputs = model(images=images)
            
            text_emb = text_outputs['central_latent']
            image_emb = image_outputs['central_latent']
            
            # InfoNCE loss for image-text alignment
            loss = info_nce_loss(image_emb, text_emb)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_i2t += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'i2t_loss': total_i2t / n,
    }


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
    
    # Compute R@K
    def compute_recall(query, keys, k_values=[1, 5, 10]):
        sim_matrix = torch.matmul(query, keys.T)
        ranks = torch.argsort(sim_matrix, dim=1, descending=True)
        
        gt_indices = torch.arange(len(query), device=query.device)
        
        recalls = {}
        for k in k_values:
            correct = (ranks[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
            recalls[f'R@{k}'] = correct.float().mean().item() * 100
        
        return recalls
    
    # Image -> Text
    i2t = compute_recall(image_embs, text_embs)
    # Text -> Image
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
    
    # Create datasets
    print("="*80)
    print("FLICKR8K IMAGE-TEXT TRAINING WITH SGD")
    print("="*80)
    print("\nLoading Flickr8k dataset...")
    
    train_dataset = Flickr8kImageTextDataset(
        root_dir=args.data_dir,
        split='train',
        image_size=args.image_size,
        text_max_len=77,
    )
    
    val_dataset = Flickr8kImageTextDataset(
        root_dir=args.data_dir,
        split='val',
        image_size=args.image_size,
        text_max_len=77,
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
    
    # Create model
    print("\nCreating model...")
    model = MultiModalMemoryNetwork(
        vocab_size=len(' abcdefghijklmnopqrstuvwxyz.,!?\'-') + 1,
        text_embed_dim=args.latent_dim,
        text_seq_len=77,
        image_size=args.image_size,
        patch_size=16,
        image_channels=3,
        audio_channels=1,
        latent_dim=args.latent_dim,
        memory_size=args.memory_size,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        enable_ttt_updates=False,  # Disabled during training
    ).to(device)
    
    model.print_model_info()
    
    # Create SGD optimizer (winner from testing!)
    print(f"\nUsing SGD optimizer (lr={args.lr}, momentum={args.momentum})")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Gradient scaler for mixed precision
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
        
        # Compute average recall
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
            }, args.output_dir / 'best_model_flickr8k.pt')
            print(f"  ✓ Saved best model (avg R@1: {avg_recall:.2f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Save history
    with open(args.output_dir / 'history_flickr8k.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best average R@1: {best_recall:.2f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Time per epoch: {elapsed_time/args.epochs:.1f} seconds")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    test_dataset = Flickr8kImageTextDataset(
        root_dir=args.data_dir,
        split='test',
        image_size=args.image_size,
        text_max_len=77,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    checkpoint = torch.load(args.output_dir / 'best_model_flickr8k.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Set Results:")
    print(f"  I→T: R@1={test_metrics['i2t_r1']:.2f}%, R@5={test_metrics['i2t_r5']:.2f}%, R@10={test_metrics['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={test_metrics['t2i_r1']:.2f}%, R@5={test_metrics['t2i_r5']:.2f}%, R@10={test_metrics['t2i_r10']:.2f}%")
    print(f"  Avg R@1: {test_metrics['avg_r1']:.2f}%")
    
    # Save test results
    with open(args.output_dir / 'test_results_flickr8k.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nModel saved to: {args.output_dir / 'best_model_flickr8k.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Flickr8k with SGD')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/flickr8k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--memory_size', type=int, default=128)
    
    # Training - SGD parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/flickr8k_sgd')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)

