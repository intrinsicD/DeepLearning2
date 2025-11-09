"""Training script for Multimodal Memory Network on Flickr8k + FACC.

Uses CLIP-style InfoNCE for cross-modal alignment:
- Image ↔ Text
- Image ↔ Audio  
- Text ↔ Audio

With proper test-time training (TTT) support.
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

from src.architectures import MultiModalMemoryNetwork
from src.optimizers import UniversalAndersonGDA
from src.utils.flickr8k_dataset import Flickr8kAudioDataset, collate_fn
from src.utils import get_device


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """CLIP-style InfoNCE loss (symmetric).
    
    Args:
        query: Query embeddings (B, D)
        key: Key embeddings (B, D)
        temperature: Temperature parameter
    
    Returns:
        Scalar loss
    """
    # Normalize
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(query, key.T) / temperature  # (B, B)
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(query), device=query.device)
    
    # Symmetric loss
    loss_q2k = F.cross_entropy(logits, labels)
    loss_k2q = F.cross_entropy(logits.T, labels)
    
    return (loss_q2k + loss_k2q) / 2


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_i2t = 0
    total_i2a = 0
    total_t2a = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        
        optimizer.zero_grad()
        
        # Forward with mixed precision
        with autocast(enabled=args.use_amp):
            # Get modality embeddings
            text_outputs = model(text=text)
            image_outputs = model(images=images)
            audio_outputs = model(audio=audio)
            
            text_emb = text_outputs['central_latent']
            image_emb = image_outputs['central_latent']
            audio_emb = audio_outputs['central_latent']
            
            # InfoNCE losses for each modality pair
            loss_i2t = info_nce_loss(image_emb, text_emb, args.temperature)
            loss_i2a = info_nce_loss(image_emb, audio_emb, args.temperature)
            loss_t2a = info_nce_loss(text_emb, audio_emb, args.temperature)
            
            # Total loss
            loss = loss_i2t + loss_i2a + loss_t2a
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate stats
        total_loss += loss.item()
        total_i2t += loss_i2t.item()
        total_i2a += loss_i2a.item()
        total_t2a += loss_t2a.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'i2t': f'{loss_i2t.item():.4f}',
            'i2a': f'{loss_i2a.item():.4f}',
            't2a': f'{loss_t2a.item():.4f}',
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'i2t_loss': total_i2t / n,
        'i2a_loss': total_i2a / n,
        't2a_loss': total_t2a / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, enable_ttt=False):
    """Evaluate cross-modal retrieval with R@1, R@5, R@10.
    
    Args:
        model: The model
        dataloader: Evaluation dataloader
        device: Device
        enable_ttt: Whether to enable test-time training during eval
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Temporarily enable TTT if requested
    original_ttt_states = {}
    if enable_ttt:
        for name, module in model.named_modules():
            if hasattr(module, 'enable_ttt_updates'):
                original_ttt_states[name] = module.enable_ttt_updates
                module.enable_ttt_updates = True
    
    all_image_embs = []
    all_text_embs = []
    all_audio_embs = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        
        # Get embeddings
        image_emb = model(images=images)['central_latent']
        text_emb = model(text=text)['central_latent']
        audio_emb = model(audio=audio)['central_latent']
        
        all_image_embs.append(image_emb)
        all_text_embs.append(text_emb)
        all_audio_embs.append(audio_emb)
    
    # Concatenate all embeddings
    image_embs = F.normalize(torch.cat(all_image_embs), dim=-1)
    text_embs = F.normalize(torch.cat(all_text_embs), dim=-1)
    audio_embs = F.normalize(torch.cat(all_audio_embs), dim=-1)
    
    # Compute retrieval metrics
    def compute_recall(query, keys, k_values=[1, 5, 10]):
        """Compute R@K for query->keys retrieval."""
        sim_matrix = torch.matmul(query, keys.T)  # (N, N)
        ranks = torch.argsort(sim_matrix, dim=1, descending=True)
        
        # Ground truth: diagonal (i-th query matches i-th key)
        gt_indices = torch.arange(len(query), device=query.device)
        
        recalls = {}
        for k in k_values:
            # Check if ground truth is in top-k
            correct = (ranks[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
            recalls[f'R@{k}'] = correct.float().mean().item() * 100
        
        return recalls
    
    # Image -> Text
    i2t = compute_recall(image_embs, text_embs)
    # Text -> Image
    t2i = compute_recall(text_embs, image_embs)
    # Image -> Audio
    i2a = compute_recall(image_embs, audio_embs)
    # Audio -> Image
    a2i = compute_recall(audio_embs, image_embs)
    # Text -> Audio
    t2a = compute_recall(text_embs, audio_embs)
    # Audio -> Text
    a2t = compute_recall(audio_embs, text_embs)
    
    # Restore original TTT states
    if enable_ttt:
        for name, module in model.named_modules():
            if name in original_ttt_states:
                module.enable_ttt_updates = original_ttt_states[name]
    
    metrics = {
        'i2t_r1': i2t['R@1'], 'i2t_r5': i2t['R@5'], 'i2t_r10': i2t['R@10'],
        't2i_r1': t2i['R@1'], 't2i_r5': t2i['R@5'], 't2i_r10': t2i['R@10'],
        'i2a_r1': i2a['R@1'], 'i2a_r5': i2a['R@5'], 'i2a_r10': i2a['R@10'],
        'a2i_r1': a2i['R@1'], 'a2i_r5': a2i['R@5'], 'a2i_r10': a2i['R@10'],
        't2a_r1': t2a['R@1'], 't2a_r5': t2a['R@5'], 't2a_r10': t2a['R@10'],
        'a2t_r1': a2t['R@1'], 'a2t_r5': a2t['R@5'], 'a2t_r10': a2t['R@10'],
    }
    
    return metrics


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
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
        vocab_size=len(' abcdefghijklmnopqrstuvwxyz.,!?\'-') + 1,  # Simple char vocab
        text_embed_dim=args.latent_dim,
        text_seq_len=77,
        image_size=args.image_size,
        patch_size=16,
        image_channels=3,
        audio_channels=1,
        latent_dim=args.latent_dim,
        memory_size=args.memory_size,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        enable_ttt_updates=False,  # Disabled during training
        ttt_topk=8,
        ttt_lr=0.1,
    ).to(device)
    
    model.print_model_info()
    
    # Create optimizer
    print(f"\nUsing optimizer: {args.optimizer}")
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'universal_anderson':
        optimizer = UniversalAndersonGDA(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            anderson_m=3,
            use_weighting=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_recall = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, enable_ttt=False)
        
        # Learning rate step
        scheduler.step()
        
        # Compute average recall
        avg_recall = (val_metrics['i2t_r1'] + val_metrics['t2i_r1'] + 
                     val_metrics['i2a_r1'] + val_metrics['a2i_r1'] +
                     val_metrics['t2a_r1'] + val_metrics['a2t_r1']) / 6
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  I↔T: R@1={val_metrics['i2t_r1']:.2f}%, R@5={val_metrics['i2t_r5']:.2f}%")
        print(f"  I↔A: R@1={val_metrics['i2a_r1']:.2f}%, R@5={val_metrics['i2a_r5']:.2f}%")
        print(f"  T↔A: R@1={val_metrics['t2a_r1']:.2f}%, R@5={val_metrics['t2a_r5']:.2f}%")
        print(f"  Avg R@1: {avg_recall:.2f}%")
        
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
            }, args.output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (avg R@1: {avg_recall:.2f}%)")
    
    # Save history
    with open(args.output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best average R@1: {best_recall:.2f}%")
    
    # Test with TTT enabled
    if args.test_ttt:
        print("\nTesting with TTT enabled...")
        checkpoint = torch.load(args.output_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        ttt_metrics = evaluate(model, val_loader, device, enable_ttt=True)
        
        print("\nTTT Results:")
        print(f"  I↔T: R@1={ttt_metrics['i2t_r1']:.2f}% (no TTT: {val_metrics['i2t_r1']:.2f}%)")
        print(f"  I↔A: R@1={ttt_metrics['i2a_r1']:.2f}% (no TTT: {val_metrics['i2a_r1']:.2f}%)")
        print(f"  T↔A: R@1={ttt_metrics['t2a_r1']:.2f}% (no TTT: {val_metrics['t2a_r1']:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal Memory Network on Flickr8k+FACC')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/flickr8k',
                       help='Path to Flickr8k+FACC dataset')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--memory_size', type=int, default=64)
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'universal_anderson'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # TTT
    parser.add_argument('--test_ttt', action='store_true',
                       help='Test with TTT enabled after training')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/flickr8k')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)

