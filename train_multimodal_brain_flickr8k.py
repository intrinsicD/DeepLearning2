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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

import torchvision.transforms as T
from torchvision.utils import make_grid

import torchaudio

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPModel,
    CLIPImageProcessor,
)

# Dataset (your existing code)
from datasets.flickr8k_dataset import Flickr8kData, collate_fn as flickr_collate

# The new architecture
from multimodal_brain_v2 import (
    MultimodalBrain,
    ModalityInterface,
    UpAdapter,
    DownAdapter,     # not used in training, but available if you add decoders later
    ThinkingCore,
    ThinkControl,
)

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
# Encoders
# ------------------------------------------------------------

class E5TextEncoderWrapper(nn.Module):
    """
    Wraps intfloat/e5-small-v2 as a text encoder:
    input: dict with 'input_ids' and 'attention_mask'
    output: last_hidden_state (B, T, 384)
    """
    def __init__(self, model_name: str = "intfloat/e5-small-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**inputs, return_dict=True)
        return out.last_hidden_state  # (B,T,384)


class CLIPVisionEncoderWrapper(nn.Module):
    """
    Wraps CLIP ViT-B/32 vision tower:
    input: dict with 'pixel_values' (B,3,H,W)
    output: last_hidden_state (B, seq, 768)
    """
    def __init__(self, name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pv = inputs["pixel_values"]
        out = self.model.vision_model(pixel_values=pv)
        return out.last_hidden_state  # (B, seq, 768)


class AudioCNNEncoder(nn.Module):
    """
    Small CNN encoder over mel-spectrograms.
    Input: tensor (B, 1, n_mels, T) or (B, n_mels, T)
    Output: (B, d_audio) vector.
    """
    def __init__(self, n_mels: int = 80, d_out: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B,128,1,1)
        )
        self.proj = nn.Linear(128, d_out)

    def forward(self, inputs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle both direct tensor and dict input
        if isinstance(inputs, dict):
            x = inputs["mel"]
        else:
            x = inputs

        # x: (B,1,n_mels,T) or (B,n_mels,T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.net(x)  # (B,128,1,1)
        h = h.view(h.size(0), 128)
        return self.proj(h)  # (B,d_out)


# ------------------------------------------------------------
# Preprocessing (Flickr batch -> encoder inputs)
# ------------------------------------------------------------

class Preproc:
    def __init__(self):
        self.txt_tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        self.clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def convert_flickr_batch(self, batch: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Flickr8k batch format (from your collate_fn):
          {
            'images': (B,3,H,W) normalized,
            'text': token indices (ignored here),
            'audio': (B,1,n_mels,T) mel-spec,
            'caption_strs': List[str],
            'image_ids': List[str],
            ...
          }

        We build encoder inputs compatible with our wrappers.
        """
        out: Dict[str, Dict] = {}

        # Text
        if "caption_strs" in batch:
            texts = batch["caption_strs"]
            # E5 recommends prefix like "query: " or "passage: "; we keep it simple here.
            tk = self.txt_tok(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out["text"] = {"input_ids": tk.input_ids, "attention_mask": tk.attention_mask}
            # Store raw texts separately if needed for logging
            out["_raw_texts"] = texts

        # Image
        if "images" in batch:
            imgs = batch["images"]
            # imgs are already normalized by dataset, but CLIP processor expects unnormalized [0,1] range
            # Denormalize first using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
            imgs_denorm = imgs * std + mean
            imgs_denorm = torch.clamp(imgs_denorm, 0, 1)

            # Now CLIP processor can handle it
            px = self.clip_proc(images=imgs_denorm, return_tensors="pt")
            out["image"] = {"pixel_values": px.pixel_values}

        # Audio
        if "audio" in batch:
            # Use mel-spectrograms directly.
            # batch["audio"]: (B,1,n_mels,T)
            out["audio"] = {"mel": batch["audio"].float()}

        return out


# ------------------------------------------------------------
# Model builder
# ------------------------------------------------------------

def build_brain(cfg: Config, device: torch.device) -> MultimodalBrain:
    d_shared = cfg.d_shared

    # Encoders
    print("Loading E5 text encoder...")
    text_enc = E5TextEncoderWrapper()
    print("Loading CLIP vision encoder...")
    img_enc = CLIPVisionEncoderWrapper()
    print("Initializing audio CNN encoder...")
    aud_enc = AudioCNNEncoder(n_mels=80, d_out=384)

    # UpAdapters into shared latent
    print("Creating UpAdapters...")
    text_up = UpAdapter(d_in=384, d_shared=d_shared)
    img_up = UpAdapter(d_in=768, d_shared=d_shared)
    aud_up = UpAdapter(d_in=384, d_shared=d_shared)

    # No decoders by default (we only train thinking + alignment). You can add these later.
    print("Building modality interfaces...")
    text_iface = ModalityInterface(
        name="text",
        encoder=text_enc,
        up_adapter=text_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=True,
        freeze_decoder=True,
    )
    image_iface = ModalityInterface(
        name="image",
        encoder=img_enc,
        up_adapter=img_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=True,
        freeze_decoder=True,
    )
    audio_iface = ModalityInterface(
        name="audio",
        encoder=aud_enc,
        up_adapter=aud_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=False,  # small CNN; we train it
        freeze_decoder=True,
    )

    modalities = {
        "text": text_iface,
        "image": image_iface,
        "audio": audio_iface,
    }

    print("Creating ThinkingCore...")
    core = ThinkingCore(d_shared=d_shared, n_layers=3, n_heads=8, dropout=0.0, use_memory_token=True)

    print("Assembling MultimodalBrain...")
    brain = MultimodalBrain(
        d_shared=d_shared,
        modalities=modalities,
        thinking_core=core,
        use_memory=True,
    )

    print(f"Moving model to {device}...")
    return brain.to(device)


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
        self.model = build_brain(cfg, self.device)

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
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # LR warmup
        self.global_step = 0
        self.best_val = float("inf")

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

        for epoch in range(self.cfg.epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs = self._prepare_inputs(batch)

                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
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
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optim.step()

                # Log scalar losses
                for name, val in losses.items():
                    self.writer.add_scalar(f"train/loss_{name}", val.item(), self.global_step)
                self.writer.add_scalar("train/loss_total", total_loss.item(), self.global_step)
                self.writer.add_scalar("train/num_modalities", len(z_by_mod), self.global_step)

                self.global_step += 1

            # End of epoch: eval
            val_loss = self.evaluate(epoch)
            if val_loss < self.best_val:
                self.best_val = val_loss
                best_path = os.path.join(self.cfg.savedir, "best_brain_v2.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"[epoch {epoch}] new best val {val_loss:.4f} -> saved {best_path}")

        print("Training complete.")

    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(self.val_loader):
            inputs = self._prepare_inputs(batch)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                z_by_mod = self.model.encode_inputs(inputs)
                ctrl = ThinkControl(steps=2, mode="default")
                _tokens, z_global, _z_mod_out = self.model.think(z_by_mod, ctrl)
                losses = self._compute_losses(z_by_mod, z_global)
                total = sum(losses.values())

            total_loss += total.item()
            n_batches += 1

            # Log eval scalars per batch
            for name, val in losses.items():
                self.writer.add_scalar(f"val/loss_{name}", val.item(), self.global_step)
            self.writer.add_scalar("val/loss_total", total.item(), self.global_step)

            # Log a few qualitative samples from first batch of epoch
            if i == 0:
                self._log_eval_examples(batch, epoch)

        avg_loss = total_loss / max(1, n_batches)
        print(f"[epoch {epoch}] val_loss = {avg_loss:.4f}")
        return avg_loss

    def _log_eval_examples(self, flickr_batch: Dict[str, Any], epoch: int):
        """
        Log some raw captions and images so you can see training is sane.
        """
        # Captions
        if "caption_strs" in flickr_batch:
            texts = flickr_batch["caption_strs"][:4]
            self.writer.add_text("eval/sample_captions", "\n".join(texts), epoch)

        # Images
        if "images" in flickr_batch:
            imgs = flickr_batch["images"][:4]  # (B,3,H,W), assumed normalized
            grid = make_grid(imgs.cpu(), nrow=4, normalize=True)
            self.writer.add_image("eval/sample_images", grid, epoch)


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
