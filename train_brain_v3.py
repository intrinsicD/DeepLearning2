"""
Training script for Multimodal Brain v3.

Optimized for 8GB GPU training with options to scale up.

Features:
- Mixed precision training (fp16/bf16)
- Gradient accumulation for larger effective batch sizes
- MoE auxiliary loss support
- Comprehensive logging to TensorBoard
- Checkpoint resumption
- Memory-efficient training modes

Usage:
    # Quick training on 8GB GPU
    python train_brain_v3.py --root_dir data/flickr8k --size small --batch_size 4

    # Full training with more resources
    python train_brain_v3.py --root_dir data/flickr8k --size base --batch_size 8 --accum_steps 4

    # Resume from checkpoint
    python train_brain_v3.py --root_dir data/flickr8k --resume checkpoints_v3/best.pt
"""

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# Dataset
from datasets.flickr8k_dataset import Flickr8kData, collate_fn as flickr_collate

# Brain V3
from multimodal_brain_v3 import BrainConfig, ModelSize, MultimodalBrainV3
from brain_v3_components import build_brain_v3, Preproc, PreprocConfig


# ============================================================================
# Configuration
# ============================================================================

class TrainConfig:
    """Training configuration with sensible defaults for 8GB GPU."""

    def __init__(
        self,
        # Data
        root_dir: str = "data/flickr8k",
        num_workers: int = 0,
        pin_memory: bool = False,

        # Model
        model_size: str = "small",
        freeze_text: bool = True,
        freeze_image: bool = True,
        train_audio: bool = True,

        # Training
        epochs: int = 20,
        batch_size: int = 4,
        accum_steps: int = 2,
        lr: float = 3e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 0.01,

        # Scheduler
        scheduler: str = "cosine",
        warmup_ratio: float = 0.1,

        # Loss
        temperature: float = 0.07,
        moe_aux_weight: float = 0.01,
        latent_l2_weight: float = 0.0,

        # Precision
        precision: str = "fp16",
        grad_clip: float = 1.0,
        nan_guard: bool = True,

        # Checkpointing
        save_dir: str = "checkpoints_v3",
        log_dir: str = "runs_v3",
        log_interval: int = 10,
        eval_interval: int = 1,
        save_every: int = 5,

        # Resume
        resume_from: Optional[str] = None,

        # Advanced
        compile_model: bool = False,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.model_size = model_size
        self.freeze_text = freeze_text
        self.freeze_image = freeze_image
        self.train_audio = train_audio

        self.epochs = epochs
        self.batch_size = batch_size
        self.accum_steps = max(1, accum_steps)
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay

        self.scheduler = scheduler
        self.warmup_ratio = warmup_ratio

        self.temperature = temperature
        self.moe_aux_weight = moe_aux_weight

        self.precision = precision
        self.grad_clip = grad_clip

        self.save_dir = save_dir
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_every = save_every

        self.resume_from = resume_from

        self.compile_model = compile_model
        self.seed = seed


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Multimodal Brain v3")

    # Data
    parser.add_argument("--root_dir", type=str, default="data/flickr8k")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    # Model
    parser.add_argument("--size", type=str, default="small",
                       choices=["tiny", "small", "base", "large"])
    parser.add_argument("--train_text", action="store_true", help="Unfreeze text encoder")
    parser.add_argument("--train_image", action="store_true", help="Unfreeze image encoder")
    parser.add_argument("--freeze_audio", action="store_true", help="Freeze audio encoder")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Loss
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--moe_aux_weight", type=float, default=0.01)
    parser.add_argument("--latent_l2_weight", type=float, default=0.0,
                        help="L2 regularization on modality latents")

    # Precision
    parser.add_argument("--precision", type=str, default="fp16",
                       choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--nan_guard", action="store_true",
                        help="Skip optimizer steps when loss/gradients are non-finite")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints_v3")
    parser.add_argument("--log_dir", type=str, default="runs_v3")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    # Advanced
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return TrainConfig(
        root_dir=args.root_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        model_size=args.size,
        freeze_text=not args.train_text,
        freeze_image=not args.train_image,
        train_audio=not args.freeze_audio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        moe_aux_weight=args.moe_aux_weight,
        latent_l2_weight=args.latent_l2_weight,
        precision=args.precision,
        grad_clip=args.grad_clip,
        nan_guard=args.nan_guard,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_every=args.save_every,
        resume_from=args.resume,
        compile_model=args.compile,
        seed=args.seed,
    )


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def info_nce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss for contrastive learning.

    Args:
        a, b: (batch, dim) normalized embeddings
        temperature: Temperature for softmax

    Returns:
        Scalar loss
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(a, b.T) / temperature

    # Labels: diagonal should be positive pairs
    labels = torch.arange(a.size(0), device=a.device)

    # Symmetric cross entropy
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_a2b + loss_b2a)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def is_finite_tensor(x: torch.Tensor) -> bool:
    """Return True if all entries are finite."""
    return torch.isfinite(x).all().item()


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Main trainer class for Brain v3."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = get_device()

        print(f"\n{'='*60}")
        print(f"Multimodal Brain v3 Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model size: {cfg.model_size}")
        print(f"Precision: {cfg.precision}")
        print(f"Batch size: {cfg.batch_size} x {cfg.accum_steps} accum = {cfg.batch_size * cfg.accum_steps} effective")

        # Set seed
        set_seed(cfg.seed)

        # Create directories
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # TensorBoard
        run_name = f"brain_v3_{cfg.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, run_name))

        # Data
        self._setup_data()

        # Model
        self._setup_model()

        # Optimizer and scheduler
        self._setup_optimizer()

        # Mixed precision
        self._setup_precision()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Resume if needed
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    def _setup_data(self):
        """Initialize data loaders."""
        root_path = Path(self.cfg.root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.cfg.root_dir}")

        print(f"\nLoading data from {self.cfg.root_dir}...")

        flickr = Flickr8kData(root_dir=str(root_path))
        self.train_data = flickr.train_samples()
        self.val_data = flickr.eval_samples()

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=flickr_collate,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=flickr_collate,
        )

        print(f"Train samples: {len(self.train_data)}")
        print(f"Val samples: {len(self.val_data)}")
        print(f"Train batches per epoch: {len(self.train_loader)}")

        # Preprocessing
        self.preproc = Preproc()

    def _setup_model(self):
        """Build and configure model."""
        print(f"\nBuilding model...")

        # Determine config overrides based on available memory
        config_overrides = {}

        if self.device.type == "cuda":
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU memory: {mem_gb:.1f} GB")

            if mem_gb < 8:
                print("Low memory mode: disabling MoE, reducing layers")
                config_overrides["use_moe"] = False
                config_overrides["n_layers"] = 3
            elif mem_gb < 12:
                print("Medium memory mode: reducing experts")
                config_overrides["n_experts"] = 2

        self.model = build_brain_v3(
            size=self.cfg.model_size,
            device=self.device,
            freeze_text=self.cfg.freeze_text,
            freeze_image=self.cfg.freeze_image,
            train_audio=self.cfg.train_audio,
            config_overrides=config_overrides,
        )

        # Optional: torch.compile
        if self.cfg.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"\nOptimizer: AdamW, lr={self.cfg.lr}, wd={self.cfg.weight_decay}")
        print(f"Trainable parameters: {sum(p.numel() for p in params):,}")

        # Try 8-bit optimizer for memory efficiency
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            print("Using 8-bit AdamW optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            print("Using standard AdamW optimizer")

        # Scheduler
        total_steps = (len(self.train_loader) // self.cfg.accum_steps) * self.cfg.epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        self.scheduler = self._create_scheduler(total_steps, warmup_steps)

    def _create_scheduler(self, total_steps: int, warmup_steps: int):
        """Create learning rate scheduler."""
        min_factor = self.cfg.min_lr / self.cfg.lr

        def lr_lambda(step: int) -> float:
            # Warmup
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)

            # Decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

            if self.cfg.scheduler == "linear":
                return max(min_factor, 1.0 - (1.0 - min_factor) * progress)
            elif self.cfg.scheduler == "constant":
                return 1.0
            else:  # cosine
                return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_precision(self):
        """Setup mixed precision training."""
        use_amp = self.cfg.precision in ["fp16", "bf16"] and self.device.type == "cuda"

        if use_amp:
            dtype = torch.float16 if self.cfg.precision == "fp16" else torch.bfloat16
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            self.autocast_dtype = dtype
            print(f"Mixed precision: {self.cfg.precision}")
        else:
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
            self.autocast_dtype = torch.float32
            print("Mixed precision: disabled")

    def _load_checkpoint(self, path: str):
        """Load checkpoint and restore training state."""
        print(f"\nLoading checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Model state loaded")

        # Optimizer state
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Could not load optimizer state: {e}")

        # Scheduler state
        if "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Scheduler state loaded")
            except Exception as e:
                print(f"Could not load scheduler state: {e}")

        # Training state
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "model_size": self.cfg.model_size,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
            },
        }

        # Save latest
        path = os.path.join(self.cfg.save_dir, "latest.pt")
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = os.path.join(self.cfg.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")

        # Save periodic
        if (epoch + 1) % self.cfg.save_every == 0:
            epoch_path = os.path.join(self.cfg.save_dir, f"epoch_{epoch:03d}.pt")
            torch.save(checkpoint, epoch_path)

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert batch to model inputs."""
        inputs = self.preproc.convert_batch(batch, self.device)
        return inputs

    def _grad_norm(self, module: nn.Module) -> Optional[float]:
        """Compute total grad norm for a module if grads exist."""
        grads = [p.grad.detach() for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not grads:
            return None
        flat = torch.cat([g.reshape(-1) for g in grads])
        if not is_finite_tensor(flat):
            return float("nan")
        return flat.norm().item()

    def _compute_loss(
        self,
        z_by_mod: Dict[str, torch.Tensor],
        z_global: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        modalities = list(z_by_mod.keys())

        # Pairwise InfoNCE losses
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                loss_name = f"{mod1}_{mod2}"
                losses[loss_name] = info_nce(
                    z_by_mod[mod1], z_by_mod[mod2],
                    temperature=self.cfg.temperature,
                )

        # Global alignment loss
        global_losses = []
        for mod in modalities:
            global_losses.append(info_nce(
                z_global, z_by_mod[mod],
                temperature=self.cfg.temperature,
            ))
        losses["global"] = sum(global_losses) / len(global_losses)

        # MoE auxiliary loss
        if aux_loss is not None and self.model.config.use_moe:
            losses["moe_aux"] = aux_loss * self.cfg.moe_aux_weight

        # Latent L2 regularization to avoid collapse/exploding norms
        if self.cfg.latent_l2_weight > 0:
            l2_terms = [z.pow(2).mean() for z in z_by_mod.values()]
            l2_terms.append(z_global.pow(2).mean())
            losses["latent_l2"] = self.cfg.latent_l2_weight * sum(l2_terms) / len(l2_terms)

        # Total
        losses["total"] = sum(losses.values())

        return losses

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_losses = []
        accum_counter = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            inputs = self._prepare_batch(batch)

            # Forward pass with autocast
            with torch.amp.autocast(self.device.type, dtype=self.autocast_dtype):
                # Encode and think
                # Encode inputs and run the thinking core to get refined latents
                z_by_mod_in = self.model.encode_inputs(inputs)
                _, z_global, z_by_mod = self.model.think(z_by_mod_in)

                # Get auxiliary loss
                aux_loss = self.model.core.get_aux_loss()

                # Compute losses using the refined modality latents
                losses = self._compute_loss(z_by_mod, z_global, aux_loss)
                loss = losses["total"]

            if self.cfg.nan_guard and not is_finite_tensor(loss.detach()):
                print(f"  [warn] Non-finite loss at step {self.global_step}, skipping update")
                self.writer.add_scalar("train/nonfinite_loss", 1, self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            # Scale loss for accumulation
            accum_counter += 1
            is_last = batch_idx == len(self.train_loader) - 1
            effective_accum = self.cfg.accum_steps if not is_last else max(1, accum_counter)
            scaled_loss = loss / effective_accum

            # Backward
            self.scaler.scale(scaled_loss).backward()

            # Record loss
            epoch_losses.append(loss.detach().item())

            # Step optimizer
            should_step = (accum_counter >= self.cfg.accum_steps) or is_last

            if not should_step:
                continue

            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            if self.cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            else:
                grad_norm = 0.0

            if self.cfg.nan_guard and isinstance(grad_norm, torch.Tensor) and not is_finite_tensor(grad_norm):
                print(f"  [warn] Non-finite grad norm at step {self.global_step}, skipping step")
                self.writer.add_scalar("train/nonfinite_grad", 1, self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                accum_counter = 0
                continue

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            accum_counter = 0
            self.global_step += 1

            # Logging
            if self.global_step % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                steps_per_sec = (batch_idx + 1) / elapsed

                # Log to TensorBoard
                for name, val in losses.items():
                    self.writer.add_scalar(f"train/{name}", val.item(), self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                self.writer.add_scalar("train/grad_norm", float(grad_norm), self.global_step)

                core_grad = self._grad_norm(self.model.core)
                if core_grad is not None:
                    self.writer.add_scalar("train/grad_norm_core", core_grad, self.global_step)

                for mod_name, iface in self.model.modalities.items():
                    mod_grad = self._grad_norm(iface)
                    if mod_grad is not None:
                        self.writer.add_scalar(f"train/grad_norm_{mod_name}", mod_grad, self.global_step)

                # Log embedding stats
                for mod, z in z_by_mod.items():
                    self.writer.add_scalar(f"embeddings/{mod}/norm", z.norm(dim=-1).mean().item(), self.global_step)

                # Console output
                print(f"  [{batch_idx+1}/{len(self.train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"({steps_per_sec:.1f} it/s)")

        return np.mean(epoch_losses)

    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        all_losses = []
        all_z_by_mod = {mod: [] for mod in ["text", "image", "audio"]}

        for batch in self.val_loader:
            inputs = self._prepare_batch(batch)

            with torch.amp.autocast(self.device.type, dtype=self.autocast_dtype):
                z_by_mod_in = self.model.encode_inputs(inputs)
                _, z_global, z_by_mod = self.model.think(z_by_mod_in)
                aux_loss = self.model.core.get_aux_loss()
                losses = self._compute_loss(z_by_mod, z_global, aux_loss)

            all_losses.append(losses["total"].item())

            for mod, z in z_by_mod.items():
                all_z_by_mod[mod].append(z.cpu())

        # Log average losses
        avg_loss = np.mean(all_losses)
        self.writer.add_scalar("val/total", avg_loss, epoch)

        # Log similarity metrics
        for i, mod1 in enumerate(["text", "image", "audio"]):
            if not all_z_by_mod[mod1]:
                continue
            z1 = torch.cat(all_z_by_mod[mod1])[:100]
            z1 = F.normalize(z1, dim=-1)

            for mod2 in ["text", "image", "audio"][i+1:]:
                if not all_z_by_mod[mod2]:
                    continue
                z2 = torch.cat(all_z_by_mod[mod2])[:100]
                z2 = F.normalize(z2, dim=-1)

                # Diagonal similarity (alignment)
                n = min(len(z1), len(z2))
                diag_sim = (z1[:n] * z2[:n]).sum(dim=-1).mean().item()
                self.writer.add_scalar(f"val/alignment/{mod1}_{mod2}", diag_sim, epoch)

        return avg_loss

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch+1}/{self.cfg.epochs}")
            print("-" * 40)

            # Train
            train_loss = self.train_epoch(epoch)

            epoch_time = time.time() - epoch_start
            print(f"Epoch complete in {format_time(epoch_time)}, avg loss: {train_loss:.4f}")

            # Evaluate
            if (epoch + 1) % self.cfg.eval_interval == 0:
                val_loss = self.evaluate(epoch)
                print(f"Validation loss: {val_loss:.4f}")

                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss!")

                # Save checkpoint
                self._save_checkpoint(epoch, is_best=is_best)

        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {format_time(total_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.cfg.save_dir}")
        print(f"{'='*60}")

        self.writer.close()


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = parse_args()

    # Print config
    print("\nConfiguration:")
    for key, val in vars(cfg).items():
        print(f"  {key}: {val}")

    # Train
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
