"""
Train the multimodal brain (v2) on Flickr8k (image + text + audio).

Usage:
    python train_multimodal_brain_flickr8k.py --root_dir data/flickr8k --epochs 10 --batch_size 8

This script:
- Builds a MultimodalBrain with:
    - Text encoder: E5 small (intfloat/e5-small-v2) -> UpAdapter -> shared latent
    - Image encoder: CLIP ViT-B/32 vision tower -> UpAdapter -> shared latent
    - Audio encoder: small CNN over mel-spectrograms -> UpAdapter -> shared latent
- Trains ONLY adapters + thinking core on InfoNCE-style multimodal alignment losses.
- Logs all losses and some sample captions+images to TensorBoard so you can see if it works.

Requires:
    pip install torch torchvision torchaudio transformers tensorboard
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Dataset (your existing code)
from datasets.flickr8k_dataset import Flickr8kData, collate_fn as flickr_collate

# The new architecture
from brain_v2_components import Preproc, build_brain
from multimodal_brain_v2 import ThinkControl

# ------------------------------------------------------------
# Config / args
# ------------------------------------------------------------

class Config:
    def __init__(
        self,
        *,
        root_dir: str,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 2e-4,
        precision: str = "fp16",
        savedir: str = "checkpoints_brain_v2",
        logdir: str = "runs_brain_v2",
        d_shared: int = 512,
        use_8bit_optim: bool = False,
        log_interval: int = 10,  # Log scalars every N steps
        histogram_interval: int = 5,  # Log histograms every N epochs
        resume_from: str = None,  # Path to checkpoint to resume from
    ):
        self.root_dir = root_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.precision = precision
        self.savedir = savedir
        self.logdir = logdir
        self.d_shared = d_shared
        self.use_8bit_optim = use_8bit_optim
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.log_interval = log_interval
        self.histogram_interval = histogram_interval
        self.resume_from = resume_from


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train multimodal brain v2 on Flickr8k")
    p.add_argument("--root_dir", type=str, default="data/flickr8k", help="Path to Flickr8k root dir")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--savedir", type=str, default="checkpoints_brain_v2")
    p.add_argument("--logdir", type=str, default="runs_brain_v2")
    p.add_argument("--no_8bit", action="store_true", help="disable 8-bit optimizer even if available")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()
    return Config(
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        precision=args.precision,
        savedir=args.savedir,
        logdir=args.logdir,
        use_8bit_optim=not args.no_8bit,
        resume_from=args.resume,
    )


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE between two batches of embeddings.

    a, b: (B, D)
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device()
        set_seed(1234)

        os.makedirs(cfg.logdir, exist_ok=True)
        os.makedirs(cfg.savedir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, "brain_v2_flickr8k"))

        # Data
        root_path = Path(cfg.root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Flickr8k directory not found at {cfg.root_dir}")

        flickr_data = Flickr8kData(root_dir=str(root_path))
        self.train_data = flickr_data.train_samples()
        self.val_data = flickr_data.eval_samples()

        self.pre = Preproc()

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=flickr_collate,
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=flickr_collate,
        )

        # Model
        self.model = build_brain(
            d_shared=cfg.d_shared,
            device=self.device,
            freeze_text=True,
            freeze_image=True,
            train_audio_encoder=True,
        )

        # Optim
        params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            HAS_BNB = True
        except Exception:
            HAS_BNB = False

        if cfg.use_8bit_optim and HAS_BNB:
            self.optim = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Precision / scaler
        use_amp = (cfg.precision == "fp16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # LR warmup
        self.global_step = 0
        self.best_val = float("inf")
        self.current_epoch = 0
        self.start_epoch = 0

        # Resume from checkpoint if provided
        if cfg.resume_from and os.path.exists(cfg.resume_from):
            self._load_checkpoint(cfg.resume_from)

        # Setup custom TensorBoard layout
        self._setup_tensorboard_layout()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore training state."""
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✓ New format checkpoint with full training state")
        else:
            # Old format - just model weights
            state_dict = checkpoint
            print("✓ Old format checkpoint (model weights only)")

        # Try to load state dict, handling mismatches gracefully
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("✓ Model state loaded (strict)")
        except RuntimeError as e:
            print(f"⚠ Strict loading failed: {e}")
            print("  Attempting to load with strict=False...")

            # Load with strict=False to handle mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

            print("✓ Model state loaded (partial - may need fine-tuning)")

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state loaded")
            except Exception as e:
                print(f"⚠ Could not load optimizer state: {e}")
                print("  Continuing with fresh optimizer")

        # Load training progress
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            self.current_epoch = self.start_epoch
            print(f"✓ Starting from epoch {self.start_epoch}")

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            print(f"✓ Global step restored to {self.global_step}")

        if 'best_val' in checkpoint:
            self.best_val = checkpoint['best_val']
            print(f"✓ Best validation loss: {self.best_val:.4f}")
        elif 'val_loss' in checkpoint:
            self.best_val = checkpoint['val_loss']
            print(f"✓ Best validation loss: {self.best_val:.4f}")

        # Load scaler state if available
        if 'scaler_state_dict' in checkpoint and self.scaler.is_enabled():
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ GradScaler state loaded")
            except Exception as e:
                print(f"⚠ Could not load scaler state: {e}")

        print(f"{'='*60}\n")

    def _setup_tensorboard_layout(self):
        """Configure custom TensorBoard layout for better organization."""
        layout = {
            "Training Progress": {
                "Loss/Total": ["Multiline", ["loss/train/total", "loss/val/total"]],
                "Loss/Pairwise": ["Multiline", [
                    "loss/train/text_image",
                    "loss/train/text_audio",
                    "loss/train/image_audio"
                ]],
                "Loss/Global": ["Multiline", ["loss/train/global_modal", "loss/val/global_modal"]],
            },
            "Embeddings": {
                "Mean Values": ["Multiline", [
                    "embeddings/text/mean",
                    "embeddings/image/mean",
                    "embeddings/audio/mean"
                ]],
                "Std Dev": ["Multiline", [
                    "embeddings/text/std",
                    "embeddings/image/std",
                    "embeddings/audio/std"
                ]],
                "Norms": ["Multiline", [
                    "embeddings/text/norm",
                    "embeddings/image/norm",
                    "embeddings/audio/norm"
                ]],
            },
            "Optimization": {
                "Learning Rate": ["Multiline", ["optimization/lr"]],
                "Gradient Norm": ["Multiline", ["optimization/grad_norm"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def _to_device_nested(self, inputs: Dict[str, Dict]) -> Dict[str, Dict]:
        out = {}
        for k, d in inputs.items():
            if isinstance(d, dict):
                out[k] = {kk: (vv.to(self.device) if torch.is_tensor(vv) else vv) for kk, vv in d.items()}
            else:
                # Keep non-dict values as-is (e.g., _raw_texts list)
                out[k] = d
        return out

    def _prepare_inputs(self, flickr_batch: Dict[str, Any]) -> Dict[str, Dict]:
        converted = self.pre.convert_flickr_batch(flickr_batch)
        return self._to_device_nested(converted)

    def _compute_losses(self, z_by_mod: Dict[str, torch.Tensor], z_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        present = list(z_by_mod.keys())

        # Pairwise InfoNCE
        if "text" in present and "image" in present:
            losses["text_image"] = info_nce(z_by_mod["text"], z_by_mod["image"])
        if "text" in present and "audio" in present:
            losses["text_audio"] = info_nce(z_by_mod["text"], z_by_mod["audio"])
        if "image" in present and "audio" in present:
            losses["image_audio"] = info_nce(z_by_mod["image"], z_by_mod["audio"])

        # Global alignment with each modality
        if len(present) > 0:
            gl_losses = []
            for name in present:
                gl_losses.append(info_nce(z_global, z_by_mod[name]))
            losses["global_modal"] = sum(gl_losses) / len(gl_losses)

        return losses

    def train(self):
        print(f"Device: {self.device}")
        print(f"Precision: {self.cfg.precision}")
        print(f"Batch size: {self.cfg.batch_size}")
        print(f"Learning rate: {self.cfg.lr}")
        print(f"Training epochs: {self.start_epoch} -> {self.cfg.epochs}")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_losses = []

            for batch in self.train_loader:
                inputs = self._prepare_inputs(batch)

                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    # Encode & think
                    z_by_mod = self.model.encode_inputs(inputs)
                    ctrl = ThinkControl(steps=2, mode="default")
                    _tokens, z_global, _z_mod_out = self.model.think(z_by_mod, ctrl)

                    # Losses
                    losses = self._compute_losses(z_by_mod, z_global)
                    total_loss = sum(losses.values())

                self.optim.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(total_loss).backward()
                    # Unscale for gradient norm computation
                    self.scaler.unscale_(self.optim)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    self.optim.step()

                # Track epoch losses
                epoch_losses.append(total_loss.item())

                # Log every N steps
                if self.global_step % self.cfg.log_interval == 0:
                    # Hierarchical loss logging
                    for name, val in losses.items():
                        self.writer.add_scalar(f"loss/train/{name}", val.item(), self.global_step)
                    self.writer.add_scalar("loss/train/total", total_loss.item(), self.global_step)

                    # Optimization metrics
                    self.writer.add_scalar("optimization/lr", self.optim.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("optimization/grad_norm", grad_norm.item(), self.global_step)

                    # Embedding statistics
                    for mod_name, emb in z_by_mod.items():
                        self.writer.add_scalar(f"embeddings/{mod_name}/mean", emb.mean().item(), self.global_step)
                        self.writer.add_scalar(f"embeddings/{mod_name}/std", emb.std().item(), self.global_step)
                        self.writer.add_scalar(f"embeddings/{mod_name}/norm", emb.norm(dim=-1).mean().item(), self.global_step)

                    # Global token statistics
                    self.writer.add_scalar("embeddings/global/mean", z_global.mean().item(), self.global_step)
                    self.writer.add_scalar("embeddings/global/std", z_global.std().item(), self.global_step)
                    self.writer.add_scalar("embeddings/global/norm", z_global.norm(dim=-1).mean().item(), self.global_step)

                self.global_step += 1

            # End of epoch: log epoch-level metrics
            self.writer.add_scalar("loss/train/epoch_mean", np.mean(epoch_losses), epoch)
            self.writer.add_scalar("loss/train/epoch_std", np.std(epoch_losses), epoch)

            # Log histograms periodically
            if epoch % self.cfg.histogram_interval == 0:
                self._log_histograms(epoch)

            # Evaluation
            val_loss = self.evaluate(epoch)
            self.writer.add_scalar("loss/val/epoch_mean", val_loss, epoch)

            if val_loss < self.best_val:
                self.best_val = val_loss
                best_path = os.path.join(self.cfg.savedir, "best_brain_v2.pt")

                # Save comprehensive checkpoint for resuming
                checkpoint = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'best_val': self.best_val,
                    'val_loss': val_loss,
                    'config': {
                        'epochs': self.cfg.epochs,
                        'batch_size': self.cfg.batch_size,
                        'lr': self.cfg.lr,
                        'd_shared': self.cfg.d_shared,
                    }
                }

                # Add scaler state if using mixed precision
                if self.scaler.is_enabled():
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()

                torch.save(checkpoint, best_path)
                print(f"[epoch {epoch}] new best val {val_loss:.4f} -> saved {best_path}")

        print("Training complete.")

    @torch.no_grad()
    def _log_histograms(self, epoch: int):
        """Log weight and gradient histograms for model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Clean up parameter names for better readability
                clean_name = name.replace('.', '/')

                # Skip if parameter is empty or too small
                if param.numel() == 0:
                    continue

                # Log weights if they contain valid values
                param_data = param.data.cpu()
                if not torch.isnan(param_data).any() and not torch.isinf(param_data).any():
                    try:
                        self.writer.add_histogram(f"weights/{clean_name}", param_data, epoch)
                    except (ValueError, RuntimeError) as e:
                        # Skip this parameter if histogram creation fails
                        pass

                # Log gradients if available and valid
                if param.grad is not None and param.grad.numel() > 0:
                    grad_data = param.grad.cpu()
                    if not torch.isnan(grad_data).any() and not torch.isinf(grad_data).any():
                        try:
                            self.writer.add_histogram(f"gradients/{clean_name}", grad_data, epoch)
                        except (ValueError, RuntimeError) as e:
                            # Skip this gradient if histogram creation fails
                            pass

    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_losses = {key: [] for key in ["text_image", "text_audio", "image_audio", "global_modal"]}

        for i, batch in enumerate(self.val_loader):
            inputs = self._prepare_inputs(batch)
            with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                z_by_mod = self.model.encode_inputs(inputs)
                ctrl = ThinkControl(steps=2, mode="default")
                _tokens, z_global, _z_mod_out = self.model.think(z_by_mod, ctrl)
                losses = self._compute_losses(z_by_mod, z_global)
                total = sum(losses.values())

            total_loss += total.item()
            n_batches += 1

            # Accumulate losses for averaging
            for key in losses.keys():
                all_losses[key].append(losses[key].item())

            # Log a few qualitative samples from first batch of epoch
            if i == 0:
                self._log_eval_examples(batch, z_by_mod, z_global, epoch)

        # Log averaged validation losses
        avg_loss = total_loss / max(1, n_batches)
        self.writer.add_scalar("loss/val/total", avg_loss, epoch)

        for key, values in all_losses.items():
            if values:
                self.writer.add_scalar(f"loss/val/{key}", np.mean(values), epoch)

        print(f"[epoch {epoch}] val_loss = {avg_loss:.4f}")
        return avg_loss

    def _log_eval_examples(self, flickr_batch: Dict[str, Any], z_by_mod: Dict[str, torch.Tensor],
                          z_global: torch.Tensor, epoch: int):
        """
        Log comprehensive evaluation examples including:
        - Sample captions and images
        - Similarity matrices between modalities
        - Embedding projections
        """
        # Captions
        if "caption_strs" in flickr_batch:
            texts = flickr_batch["caption_strs"][:8]
            caption_text = "\n\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
            self.writer.add_text("eval/sample_captions", caption_text, epoch)

        # Images
        if "images" in flickr_batch:
            imgs = flickr_batch["images"][:8]  # (B,3,H,W), assumed normalized
            grid = make_grid(imgs.cpu(), nrow=4, normalize=True)
            self.writer.add_image("eval/sample_images", grid, epoch)

        # Similarity matrices between modalities
        if HAS_MATPLOTLIB and len(z_by_mod) >= 2:
            self._log_similarity_matrices(z_by_mod, epoch)

        # Embedding projections (for visualization in TensorBoard)
        if "text" in z_by_mod and "caption_strs" in flickr_batch:
            n_samples = min(100, z_by_mod["text"].size(0))
            self.writer.add_embedding(
                z_by_mod["text"][:n_samples].cpu(),
                metadata=flickr_batch["caption_strs"][:n_samples],
                tag="embeddings/text_projection",
                global_step=epoch
            )

        # Log embedding statistics per modality
        for mod_name, emb in z_by_mod.items():
            self.writer.add_scalar(f"eval/embeddings/{mod_name}/mean", emb.mean().item(), epoch)
            self.writer.add_scalar(f"eval/embeddings/{mod_name}/std", emb.std().item(), epoch)
            self.writer.add_scalar(f"eval/embeddings/{mod_name}/norm", emb.norm(dim=-1).mean().item(), epoch)

    def _log_similarity_matrices(self, z_by_mod: Dict[str, torch.Tensor], epoch: int):
        """Create and log similarity matrices between all pairs of modalities."""
        modalities = list(z_by_mod.keys())

        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                # Compute cosine similarity matrix
                z1 = F.normalize(z_by_mod[mod1], dim=-1).cpu()
                z2 = F.normalize(z_by_mod[mod2], dim=-1).cpu()

                # Take subset for visualization (max 32x32)
                n = min(32, z1.size(0))
                z1_subset = z1[:n]
                z2_subset = z2[:n]

                sim_matrix = (z1_subset @ z2_subset.t()).numpy()

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_title(f"{mod1.capitalize()} vs {mod2.capitalize()} Similarity Matrix")
                ax.set_xlabel(f"{mod2.capitalize()} samples")
                ax.set_ylabel(f"{mod1.capitalize()} samples")

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

                # Add diagonal line for reference (perfect alignment)
                if sim_matrix.shape[0] == sim_matrix.shape[1]:
                    ax.plot([0, n-1], [0, n-1], 'g--', alpha=0.5, linewidth=2, label='Diagonal')
                    ax.legend()

                plt.tight_layout()
                self.writer.add_figure(f"similarity/{mod1}_vs_{mod2}", fig, epoch)
                plt.close(fig)

        # Also log mean diagonal similarity (alignment metric)
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                z1 = F.normalize(z_by_mod[mod1], dim=-1)
                z2 = F.normalize(z_by_mod[mod2], dim=-1)
                n = min(z1.size(0), z2.size(0))

                # Diagonal similarity (how well matched pairs align)
                diag_sim = (z1[:n] * z2[:n]).sum(dim=-1).mean().item()
                self.writer.add_scalar(f"alignment/{mod1}_vs_{mod2}/diagonal_similarity", diag_sim, epoch)

                # Full matrix mean (average similarity)
                full_sim = (z1 @ z2.t()).mean().item()
                self.writer.add_scalar(f"alignment/{mod1}_vs_{mod2}/mean_similarity", full_sim, epoch)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    cfg = parse_args()
    os.makedirs(cfg.savedir, exist_ok=True)
    os.makedirs(cfg.logdir, exist_ok=True)
    trainer = Trainer(cfg)
    print(f"Train samples: {len(trainer.train_data)}")
    print(f"Val samples:   {len(trainer.val_data)}")
    trainer.train()
    print(f"Best val loss: {trainer.best_val:.4f}")
    print(f"Best model saved under: {cfg.savedir}")


if __name__ == "__main__":
    main()
