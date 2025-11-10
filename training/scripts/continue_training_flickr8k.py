"""Continue training Flickr8k model from checkpoint with SGD.

Resumes from best saved checkpoint and trains for additional epochs.
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
from utils.flickr8k_simple import Flickr8kImageTextDataset, collate_fn
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
    print("CONTINUE TRAINING FLICKR8K WITH SGD")
    print("="*80)
    
    # Load datasets
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
        enable_ttt_updates=False,
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_metrics = checkpoint['metrics']
    best_recall = best_metrics['avg_r1']
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best Avg R@1: {best_recall:.2f}%")
    print(f"  I→T: {best_metrics['i2t_r1']:.2f}%, T→I: {best_metrics['t2i_r1']:.2f}%")
    
    # Create optimizer and load state if available
    print(f"\nUsing SGD optimizer (lr={args.lr}, momentum={args.momentum})")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
    )
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Learning rate scheduler (continue from where we left off)
    total_epochs = start_epoch + args.additional_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=args.lr * 0.01
    )
    
    # Fast-forward scheduler to current epoch
    for _ in range(start_epoch):
        scheduler.step()
    
    scaler = GradScaler()
    
    # Load previous history if exists
    history_file = Path(args.checkpoint).parent / 'history_flickr8k.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"Loaded previous history ({len(history)} epochs)")
    else:
        history = []
    
    # Training loop
    print("\n" + "="*80)
    print(f"CONTINUING TRAINING FOR {args.additional_epochs} MORE EPOCHS")
    print("="*80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch + 1, start_epoch + args.additional_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Learning rate step
        scheduler.step()
        
        avg_recall = val_metrics['avg_r1']
        
        # Print summary
        print(f"\nEpoch {epoch}/{total_epochs}:")
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
        
        # Save if improved
        if avg_recall > best_recall:
            best_recall = avg_recall
            save_path = Path(args.checkpoint).parent / 'best_model_flickr8k_continued.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args),
            }, save_path)
            print(f"  ✓ Saved improved model (avg R@1: {avg_recall:.2f}%)")
        
        # Always save latest
        latest_path = Path(args.checkpoint).parent / 'latest_model_flickr8k.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'args': vars(args),
        }, latest_path)
    
    elapsed_time = time.time() - start_time
    
    # Save updated history
    with open(Path(args.checkpoint).parent / 'history_flickr8k_continued.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("CONTINUED TRAINING COMPLETE")
    print("="*80)
    print(f"Best average R@1: {best_recall:.2f}%")
    print(f"Additional training time: {elapsed_time/60:.1f} minutes")
    print(f"Time per epoch: {elapsed_time/args.additional_epochs:.1f} seconds")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
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
    
    # Load best continued model
    best_path = Path(args.checkpoint).parent / 'best_model_flickr8k_continued.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best continued model from epoch {checkpoint['epoch']}")
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Set Results:")
    print(f"  I→T: R@1={test_metrics['i2t_r1']:.2f}%, R@5={test_metrics['i2t_r5']:.2f}%, R@10={test_metrics['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={test_metrics['t2i_r1']:.2f}%, R@5={test_metrics['t2i_r5']:.2f}%, R@10={test_metrics['t2i_r10']:.2f}%")
    print(f"  Avg R@1: {test_metrics['avg_r1']:.2f}%")
    
    # Save test results
    with open(Path(args.checkpoint).parent / 'test_results_continued.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nModels saved to: {Path(args.checkpoint).parent}")
    print("  - best_model_flickr8k_continued.pt")
    print("  - latest_model_flickr8k.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue training Flickr8k from checkpoint')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/flickr8k')
    parser.add_argument('--checkpoint', type=str, default='./results/folder_per_model/multimodal_memory/outputs/flickr8k_sgd/best_model_flickr8k.pt')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--memory_size', type=int, default=128)
    
    # Training
    parser.add_argument('--additional_epochs', type=int, default=50,
                       help='Number of additional epochs to train')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate (lower than initial for fine-tuning)')
    parser.add_argument('--momentum', type=float, default=0.9)
    
    args = parser.parse_args()
    
    main(args)

