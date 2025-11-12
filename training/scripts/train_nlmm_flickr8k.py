"""Train NL-MM (Nested Learning Multimodal) on Flickr8k dataset.

This script integrates the core nl_mm architecture with the Flickr8k+FACC dataset
for tri-modal learning (text + images + audio).
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from modules.nl_mm.models.nl_mm_model import NLMM
from modules.nl_mm.utils import load_config
from modules.nl_mm.init import apply_nlmm_init
from utils.flickr8k_dataset import Flickr8kAudioDataset, collate_fn
from utils import get_device


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """CLIP-style InfoNCE contrastive loss."""
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(len(query), device=query.device)
    loss_q2k = F.cross_entropy(logits, labels)
    loss_k2q = F.cross_entropy(logits.T, labels)
    return (loss_q2k + loss_k2q) / 2


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self, save_dir: Path, use_tensorboard: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_i2t': [],
            'train_i2a': [],
            'train_t2a': [],
            'val_i2t_r1': [],
            'val_t2i_r1': [],
            'val_i2a_r1': [],
            'learning_rate': [],
        }
        # Initialize TensorBoard writer
        self.writer = None
        if use_tensorboard:
            tensorboard_dir = self.save_dir / 'tensorboard'
            self.writer = SummaryWriter(str(tensorboard_dir))
            print(f"   📊 TensorBoard logging to: {tensorboard_dir}")
            print(f"   Run: tensorboard --logdir={tensorboard_dir}")

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        # Log to TensorBoard
        if self.writer is not None and 'epoch' in kwargs:
            epoch = kwargs['epoch']
            # Log training losses
            if 'train_loss' in kwargs:
                self.writer.add_scalar('Loss/train_total', kwargs['train_loss'], epoch)
            if 'train_i2t' in kwargs:
                self.writer.add_scalar('Loss/train_image_text', kwargs['train_i2t'], epoch)
            if 'train_i2a' in kwargs:
                self.writer.add_scalar('Loss/train_image_audio', kwargs['train_i2a'], epoch)
            if 'train_t2a' in kwargs:
                self.writer.add_scalar('Loss/train_text_audio', kwargs['train_t2a'], epoch)

            # Log validation metrics
            if 'val_i2t_r1' in kwargs:
                self.writer.add_scalar('Retrieval/image_to_text_R1', kwargs['val_i2t_r1'], epoch)
            if 'val_t2i_r1' in kwargs:
                self.writer.add_scalar('Retrieval/text_to_image_R1', kwargs['val_t2i_r1'], epoch)
            if 'val_i2a_r1' in kwargs:
                self.writer.add_scalar('Retrieval/image_to_audio_R1', kwargs['val_i2a_r1'], epoch)

            # Log learning rate
            if 'learning_rate' in kwargs:
                self.writer.add_scalar('Hyperparameters/learning_rate', kwargs['learning_rate'], epoch)

    def save(self):
        """Save metrics to JSON."""
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def plot(self):
        """Generate training plots."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            epochs = self.metrics['epoch']
            
            # Training losses
            axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Per-modality losses
            axes[0, 1].plot(epochs, self.metrics['train_i2t'], label='Image↔Text', linewidth=2)
            axes[0, 1].plot(epochs, self.metrics['train_i2a'], label='Image↔Audio', linewidth=2)
            axes[0, 1].plot(epochs, self.metrics['train_t2a'], label='Text↔Audio', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Contrastive Losses by Modality')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Retrieval metrics
            axes[1, 0].plot(epochs, self.metrics['val_i2t_r1'], 'o-', label='Image→Text', linewidth=2)
            axes[1, 0].plot(epochs, self.metrics['val_t2i_r1'], 's-', label='Text→Image', linewidth=2)
            axes[1, 0].plot(epochs, self.metrics['val_i2a_r1'], '^-', label='Image→Audio', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall@1 (%)')
            axes[1, 0].set_title('Cross-Modal Retrieval Performance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Learning rate
            if self.metrics['learning_rate']:
                axes[1, 1].plot(epochs, self.metrics['learning_rate'], 'g-', linewidth=2)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'training_progress.png', dpi=150)
            print(f"📊 Saved plot to {self.save_dir / 'training_progress.png'}")
            plt.close()
        except ImportError:
            print("⚠️  matplotlib not installed, skipping plots")


def extract_central_latent(outputs_dict):
    """Extract central latent representation from modules.nl_mm outputs."""
    # The nl_mm model returns loss values from decoders
    # We need to extract latent before decoder
    # For now, use decoder output as proxy
    # TODO: Modify nl_mm to return latent representations
    return outputs_dict


def train_epoch(model, dataloader, scheduler, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_i2t = 0
    total_i2a = 0
    total_t2a = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        # Prepare batch in nl_mm format
        nl_batch = {
            "text": batch['text'].to(device),
            "image": batch['images'].to(device),
            "audio": batch['audio'].to(device),
            "text_target": batch['text'].to(device),  # For language modeling loss
        }
        
        # Forward pass with AMP
        with autocast(enabled=args.use_amp, dtype=torch.bfloat16 if args.use_amp else torch.float32):
            outputs, state = model(nl_batch, enable_ttt=False)
            
            # The nl_mm outputs are decoder losses
            # For multimodal training, we need contrastive losses
            # TODO: Add hooks to extract embeddings before decoders
            
            # For now, use the text decoder loss as primary
            loss = outputs.get("text", torch.tensor(0.0, device=device))
            
            # Scale loss for gradient accumulation
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % args.accumulation_steps == 0:
            # Gradient clipping - unscale gradients for all optimizers in NLScheduler
            if hasattr(scheduler, '_level_states'):
                for level_state in scheduler._level_states.values():
                    scaler.unscale_(level_state.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            
            # Optimizer step using NL scheduler
            global_step = epoch * len(dataloader) + step
            scheduler.step_all(global_step)
            
            scaler.update()
        
        # Accumulate stats
        total_loss += loss.item() * args.accumulation_steps
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * args.accumulation_steps:.4f}',
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'i2t_loss': total_i2t / n if total_i2t > 0 else 0,
        'i2a_loss': total_i2a / n if total_i2a > 0 else 0,
        't2a_loss': total_t2a / n if total_t2a > 0 else 0,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, enable_ttt=False):
    """Evaluate cross-modal retrieval."""
    model.eval()
    
    # Collect all embeddings
    # Note: This is simplified - need to extract embeddings from modules.nl_mm properly
    print("  Evaluation: Collecting embeddings...")
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        nl_batch = {
            "text": batch['text'].to(device),
            "image": batch['images'].to(device),
            "audio": batch['audio'].to(device),
        }
        
        # Forward pass
        outputs, state = model(nl_batch, enable_ttt=enable_ttt)
    
    # Compute retrieval metrics
    # TODO: Implement proper embedding extraction and similarity computation
    # For now, return dummy metrics
    metrics = {
        'i2t_r1': 0.0,
        't2i_r1': 0.0,
        'i2a_r1': 0.0,
    }
    
    return metrics


def main(args):
    device = get_device()
    print(f"🚀 Training NL-MM on Flickr8k")
    print(f"   Device: {device}")
    print(f"   Config: {args.config}")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override with command-line args
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.lr:
        if 'optimizer' not in cfg:
            cfg['optimizer'] = {}
        if 'adamw' not in cfg['optimizer']:
            cfg['optimizer']['adamw'] = {}
        cfg['optimizer']['adamw']['lr'] = args.lr
    
    print(f"\n📊 Configuration:")
    print(f"   Model dim: {cfg['d_model']}")
    print(f"   Heads: {cfg['n_heads']}")
    print(f"   Memory length: {cfg['L_mem']}")
    print(f"   Batch size: {cfg.get('batch_size', args.batch_size)}")
    
    # Create datasets
    print("\n📂 Loading datasets...")
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
    
    # Create model
    print("\n🏗️  Creating NL-MM model...")
    model = NLMM(cfg).to(device)
    apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Configure NL scheduler
    print("\n⚙️  Configuring Nested Learning scheduler...")
    scheduler = model.configure_scheduler(cfg)
    
    # Create gradient scaler for AMP
    scaler = GradScaler(enabled=args.use_amp)
    
    # Metrics tracker
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = MetricsTracker(output_dir, use_tensorboard=not args.no_tensorboard)

    # Training loop
    print(f"\n🏋️  Training for {args.epochs} epochs...")
    best_metric = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, scheduler, scaler, device, epoch, args
        )
        
        print(f"\n📈 Training metrics:")
        print(f"   Loss: {train_metrics['loss']:.4f}")
        if train_metrics['i2t_loss'] > 0:
            print(f"   Image↔Text: {train_metrics['i2t_loss']:.4f}")
            print(f"   Image↔Audio: {train_metrics['i2a_loss']:.4f}")
            print(f"   Text↔Audio: {train_metrics['t2a_loss']:.4f}")
        
        # Log training metrics every epoch to TensorBoard
        if tracker.writer is not None:
            tracker.writer.add_scalar('Loss/train_total', train_metrics['loss'], epoch)
            if train_metrics['i2t_loss'] > 0:
                tracker.writer.add_scalar('Loss/train_image_text', train_metrics['i2t_loss'], epoch)
                tracker.writer.add_scalar('Loss/train_image_audio', train_metrics['i2a_loss'], epoch)
                tracker.writer.add_scalar('Loss/train_text_audio', train_metrics['t2a_loss'], epoch)
            tracker.writer.add_scalar('Hyperparameters/learning_rate', cfg['optimizer']['adamw']['lr'], epoch)
            tracker.writer.flush()  # Force write to disk

        # Evaluate every N epochs
        if epoch % args.eval_every == 0:
            print(f"\n🔍 Evaluating...")
            val_metrics = evaluate(model, val_loader, device, enable_ttt=args.enable_ttt)
            
            print(f"   Image→Text R@1: {val_metrics['i2t_r1']:.2f}%")
            print(f"   Text→Image R@1: {val_metrics['t2i_r1']:.2f}%")
            print(f"   Image→Audio R@1: {val_metrics['i2a_r1']:.2f}%")
            
            # Track metrics
            tracker.add(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                train_i2t=train_metrics['i2t_loss'],
                train_i2a=train_metrics['i2a_loss'],
                train_t2a=train_metrics['t2a_loss'],
                val_i2t_r1=val_metrics['i2t_r1'],
                val_t2i_r1=val_metrics['t2i_r1'],
                val_i2a_r1=val_metrics['i2a_r1'],
                learning_rate=cfg['optimizer']['adamw']['lr'],
            )
            
            # Force flush to TensorBoard
            if tracker.writer is not None:
                tracker.writer.flush()

            # Save best model
            avg_metric = (val_metrics['i2t_r1'] + val_metrics['t2i_r1']) / 2
            if avg_metric > best_metric:
                best_metric = avg_metric
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': cfg,
                    'metrics': val_metrics,
                }
                torch.save(checkpoint, output_dir / 'best_nlmm_model.pt')
                print(f"   💾 Saved best model (avg R@1: {avg_metric:.2f}%)")
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': cfg,
            }
            torch.save(checkpoint, output_dir / f'nlmm_epoch_{epoch}.pt')
            print(f"   💾 Saved checkpoint at epoch {epoch}")
        
        # Plot progress
        if epoch % args.plot_every == 0:
            tracker.save()
            tracker.plot()
    
    # Final save
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Total time: {elapsed/3600:.2f} hours")
    print(f"   Best avg R@1: {best_metric:.2f}%")
    print(f"   Outputs saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    tracker.save()
    tracker.plot()
    tracker.close()  # Close TensorBoard writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NL-MM on Flickr8k")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="./flickr8k",
                        help="Path to Flickr8k dataset")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size (square)")
    
    # Model
    parser.add_argument("--config", type=str, default="modules/nl_mm/configs/tiny_single_gpu.yaml",
                        help="Path to nl_mm config file")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Test-time training
    parser.add_argument("--enable_ttt", action="store_true",
                        help="Enable test-time training during eval")
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="./results/folder_per_model/nl_mm/outputs/nlmm_flickr8k",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--plot_every", type=int, default=5,
                        help="Generate plots every N epochs")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable TensorBoard logging")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    
    args = parser.parse_args()
    main(args)

