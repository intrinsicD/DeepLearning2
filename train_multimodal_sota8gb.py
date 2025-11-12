# train_multimodal_sota8gb.py
# Single-file trainer + evaluator for the modular multimodal "thinking" model
# Defaults target a single 8 GB GPU by freezing heavy I/O backbones and training
# only lightweight adapters + aggregator + walker. Includes TensorBoard logging,
# checkpointing, and an evaluation mode that loads the best snapshot and emits
# sample outputs for each modality combination.
#
# Requirements (minimal):
#   pip install torch torchvision torchaudio transformers diffusers tensorboard
#   pip install bitsandbytes  # optional (8-bit optimizers)
#
# Usage examples:
#   # demo run on synthetic toy data (sanity check; very light)
#   python train_multimodal_sota8gb.py --demo --epochs 1
#
#   # training/eval on a JSONL dataset with triplets (image,text,audio)
#   # each line: {"image": "/path/to/img.png", "text": "caption", "audio": "/path/to.wav"}
#   python train_multimodal_sota8gb.py --jsonl data/triples.jsonl --epochs 3
#   tensorboard --logdir runs
#
# Notes on the default backbones / dims (frozen):
# - Text encoder: intfloat/e5-small-v2 (embedding size 384)  [HF card]
# - Image encoder: CLIP ViT-B/32 via transformers (vision width 768; pooled proj 512)
# - Audio encoder: openai/whisper-tiny encoder (d_model 384)
# - Text decoder (eval): GPT-2 small (embedding size 768) with prefix-embeds
# - Image decoder (eval): diffusers AutoencoderKL 'sd-vae-ft-mse' (latents 4x32x32 for 256px)
# - Audio decoder (eval): Griffin-Lim inversion from mel (fallback if HiFi-GAN not present)
#
# This script expects the architecture file `multimodal_thinking_fixed.py` in the same folder.

from __future__ import annotations
import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# HF / Diffusers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    CLIPModel, CLIPImageProcessor,
    WhisperFeatureExtractor, WhisperModel,
)
from diffusers import AutoencoderKL

import torchvision.transforms as T
import torchaudio
from torchaudio import transforms as AT

# Try bitsandbytes (optional)
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except Exception:
    HAS_BNB = False

# Local architecture (from canvas file)
from multimodal_thinking_fixed import (
    MultiModalThinking, ThinkControl,
    TextDecoderAdapter, ImageDecoderAdapter, AudioDecoderAdapter,
)

# -----------------------------------
# Config
# -----------------------------------
@dataclass
class Config:
    # Latent workspace
    d_think: int = 384
    n_slots: int = 64
    walker_kind: str = "transformer"  # transformer|diffusion|retention|temporal|world
    walker_depth: int = 2
    walker_heads: int = 6
    think_steps: int = 2

    # Training
    batch_size: int = 4
    epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    use_8bit_optim: bool = True
    precision: str = "fp16"  # fp32|fp16|bf16
    logdir: str = "runs"
    savedir: str = "checkpoints"

    # Stability
    grad_clip: float = 1.0  # gradient clipping norm
    warmup_steps: int = 100  # learning rate warmup
    info_nce_temp: float = 0.07  # temperature for contrastive loss
    loss_scale_check: bool = True  # check for NaN/Inf
    log_every: int = 50  # log every N steps
    use_ema: bool = False  # exponential moving average (adds memory)

    # Data / demo
    demo: bool = True
    jsonl: Optional[str] = None
    num_workers: int = 2

    # Eval
    eval_only: bool = False
    checkpoint: Optional[str] = None
    samples_per_eval: int = 2

    # Image sizes
    image_size: int = 256  # px for VAE decode
    vae_latent_h: int = 32  # 256 / 8
    vae_latent_w: int = 32
    vae_latent_c: int = 4

# -----------------------------------
# Utility
# -----------------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# Encoder wrappers (so ModalEncoderPort can accept a single positional arg)
# Each wrapper takes a single positional arg (inputs), which may be a dict
# and returns (feats, mask) as expected by the architecture ports.
# -----------------------------------
class E5TextEncoderWrapper(nn.Module):
    def __init__(self, name: str = "intfloat/e5-small-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs):
        # inputs: {input_ids, attention_mask, raw (optional)}
        if isinstance(inputs, dict):
            # Filter out non-model keys like 'raw'
            model_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
            out = self.model(**model_inputs, return_dict=True)
            last_hidden = out.last_hidden_state  # (B,T,384)
            mask = ~model_inputs["attention_mask"].bool()  # True at pads
            return last_hidden, mask
        raise ValueError("E5TextEncoderWrapper expects a dict of tokenized inputs")


class CLIPVisionEncoderWrapper(nn.Module):
    def __init__(self, name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs):
        # inputs: {pixel_values}
        if isinstance(inputs, dict) and "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            out = self.model.vision_model(pv)
            hidden = out.last_hidden_state  # (B, seq, 768)
            mask = None
            return hidden, mask
        raise ValueError("CLIPVisionEncoderWrapper expects dict with 'pixel_values'")


class WhisperTinyEncoderWrapper(nn.Module):
    def __init__(self, name: str = "openai/whisper-tiny"):
        super().__init__()
        self.model = WhisperModel.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs):
        # inputs: {input_features}
        if isinstance(inputs, dict) and "input_features" in inputs:
            feats = inputs["input_features"]
            out = self.model.encoder(feats, return_dict=True)
            hidden = out.last_hidden_state  # (B, T', 384)
            mask = None
            return hidden, mask
        raise ValueError("WhisperTinyEncoderWrapper expects dict with 'input_features'")


# -----------------------------------
# Decoder wrappers used with DecoderAdapters (eval-time)
# -----------------------------------
class GPT2TextGenDecoder(nn.Module):
    def __init__(self, name: str = "gpt2", max_new_tokens: int = 32):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def forward(self, prefix_embeds: torch.Tensor, max_new_tokens: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Greedy generation from soft prefix embeddings.
        prefix_embeds: (B, P, d_model=768)
        Returns: {"text": List[str]}
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        device = prefix_embeds.device
        B = prefix_embeds.size(0)
        # start with EOS as first discrete token
        input_ids = torch.full((B, 1), self.tok.eos_token_id, device=device, dtype=torch.long)
        past = None
        # First call includes the prefix embeddings
        outputs = self.model(inputs_embeds=torch.cat([prefix_embeds, self.model.transformer.wte(input_ids)], dim=1),
                             use_cache=True, return_dict=True)
        past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = [next_ids]
        # Following steps use only the last token + past
        for _ in range(max_new_tokens - 1):
            embeds = self.model.transformer.wte(generated[-1])
            outputs = self.model(inputs_embeds=embeds, use_cache=True, past_key_values=past, return_dict=True)
            past = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_ids)
        ids = torch.cat(generated, dim=1)
        texts = self.tok.batch_decode(ids, skip_special_tokens=True)
        return {"text": texts}


class VAEImageDecoder(nn.Module):
    def __init__(self, name: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(name)
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        # Stable Diffusion convention: divide by scale factor before decode
        scale = getattr(self.vae.config, "scaling_factor", getattr(self.vae.config, "scale_factor", 0.18215))
        lat = latent_grid / scale
        imgs = self.vae.decode(lat).sample  # (B,3,H*8,W*8) in [-1,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # [0,1]
        return imgs


class GriffinLimVocoder(nn.Module):
    """Mel -> waveform via InverseMelScale + GriffinLim.
    This avoids deprecated/removed `create_fb_matrix` and works across torchaudio versions.
    """
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, n_mels: int = 80, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        # Inverse mel scale (mel -> linear magnitude)
        self.inv_mel = AT.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)
        # Griffin-Lim to reconstruct phase and waveform
        self.gl = AT.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_iter=32)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        # mel: (B, T, F)
        mel = mel.transpose(1, 2)  # (B, F, T) where F = n_mels
        # approximate inversion to linear magnitude spectrogram (B, n_stft, T)
        mag = self.inv_mel(mel)
        # Griffin-Lim expects (B, n_freq, T)
        wav = self.gl(mag)
        return {"audio": wav, "sample_rate": self.sr}


class DemoTriplesDataset(Dataset):
    """Tiny synthetic tri-modal dataset to sanity-check the pipeline."""
    def __init__(self, n: int = 256, image_size: int = 256, sr: int = 16000):
        super().__init__()
        self.n = n
        self.image_size = image_size
        self.sr = sr
        self.texts = [f"a synthetic sample #{i}" for i in range(n)]
        self.to_pil = T.ToPILImage()

    def __len__(self):
        return self.n

    def _rand_image(self):
        x = torch.rand(3, self.image_size, self.image_size)
        return x

    def _rand_audio(self, seconds: float = 1.0):
        t = torch.linspace(0, seconds, int(self.sr * seconds))
        freq = random.choice([220.0, 440.0, 660.0])
        wav = 0.2 * torch.sin(2 * math.pi * freq * t)
        return wav

    def __getitem__(self, idx):
        return {
            "image": self._rand_image(),
            "text": self.texts[idx],
            "audio": self._rand_audio(),
        }


class JSONLTriplesDataset(Dataset):
    """Reads a JSONL file with keys {image, text, audio}. Image can be any format path; audio WAV/FLAC.
    You can omit 'audio' for some rows (pairs still work)."""
    def __init__(self, path: str):
        super().__init__()
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.rows.append(j)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        out = {"text": row.get("text", None)}
        # load image if present
        if row.get("image") and os.path.exists(row["image"]):
            from PIL import Image
            img = Image.open(row["image"]).convert("RGB")
            out["image"] = T.ToTensor()(img)
        # load audio if present
        if row.get("audio") and os.path.exists(row["audio"]):
            wav, sr = torchaudio.load(row["audio"])  # (C,T)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            out["audio"] = wav.squeeze(0)
        return out


# -----------------------------------
# Collation + preprocessing for encoders
# -----------------------------------
class Preproc:
    def __init__(self):
        # Text encoder stuff (E5)
        self.txt_tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        # Image encoder stuff (CLIP)
        self.clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Audio encoder stuff (Whisper)
        self.whisper_feat = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def __call__(self, batch: List[Dict]) -> Dict[str, Dict]:
        texts = [b.get("text", "") for b in batch]
        images = [b["image"] for b in batch if "image" in b]
        audios = [b["audio"] for b in batch if "audio" in b]

        out: Dict[str, Dict] = {}
        # text
        if any(texts):
            # E5 suggests prefixes like "query: "/"passage: ", we use simple input here
            tk = self.txt_tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out["text"] = {"input_ids": tk.input_ids, "attention_mask": tk.attention_mask, "raw": texts}
        # image
        if len(images) > 0:
            # to PIL, then processor
            pil = [T.ToPILImage()(img) if torch.is_tensor(img) else img for img in images]
            px = self.clip_proc(images=pil, return_tensors="pt")
            out["image"] = {"pixel_values": px.pixel_values}
        # audio
        if len(audios) > 0:
            # WhisperFeatureExtractor expects a list of 1D numpy arrays
            wavs_list = [a.numpy() if a.dim()==1 else a.squeeze(0).numpy() for a in audios]
            feats = self.whisper_feat(wavs_list, sampling_rate=16000, return_tensors="pt")
            out["audio"] = {"input_features": feats.input_features}
        return out


# -----------------------------------
# Losses
# -----------------------------------
def info_nce(a: torch.Tensor, b: torch.Tensor, t: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss with symmetric cross-entropy."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / t
    labels = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) * 0.5


# -----------------------------------
# Training / Eval
# -----------------------------------

def build_model(cfg: Config, device: torch.device) -> MultiModalThinking:
    # Encoders (frozen wrappers)
    text_enc = E5TextEncoderWrapper()
    img_enc = CLIPVisionEncoderWrapper()
    aud_enc = WhisperTinyEncoderWrapper()

    # Decoders (eval only wrappers)
    txt_dec = GPT2TextGenDecoder()
    img_dec = VAEImageDecoder()
    aud_dec = GriffinLimVocoder()

    encoders = {
        "text":  (text_enc, 384, 384),   # d_feats, d_modal
        "image": (img_enc, 768, 768),
        "audio": (aud_enc, 384, 384),
    }
    decoders = {
        # d_modal_out is unused when adapters are present, but must be provided
        "text":  (txt_dec, 768),
        "image": (img_dec, cfg.vae_latent_c),
        "audio": (aud_dec, 80),
    }

    # Decoder adapters
    adapters = {
        "text":  TextDecoderAdapter(d_think=cfg.d_think, d_text=768, prefix_len=8, mode="prefix"),
        "image": ImageDecoderAdapter(d_think=cfg.d_think, C_lat=cfg.vae_latent_c, H=cfg.vae_latent_h, W=cfg.vae_latent_w),
        "audio": AudioDecoderAdapter(d_think=cfg.d_think, F=80, T=200),
    }

    model = MultiModalThinking(
        d_think=cfg.d_think,
        encoders=encoders,
        decoders=decoders,
        freeze_io=True,
        n_slots=cfg.n_slots,
        aggregator_depth=1,
        walker_kind=cfg.walker_kind,
        walker_kwargs={"depth": cfg.walker_depth, "heads": cfg.walker_heads},
        cond_dims={"text": 32, "image": 32, "audio": 32},
        decoder_adapters=nn.ModuleDict(adapters),
    )
    return model.to(device)


@torch.no_grad()
def sample_and_log(model: MultiModalThinking, batch_inputs: Dict[str, Dict], writer: SummaryWriter, step: int, device: torch.device):
    model.eval()
    ctrl = ThinkControl(steps=2)

    def to_device(d: Dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}

    # 1) image -> text
    if "image" in batch_inputs:
        outs = model({"image": to_device(batch_inputs["image"])}, request_outputs=["text"], control=ctrl)
        texts = outs["text"]["text"] if isinstance(outs["text"], dict) else outs["text"]
        writer.add_text("eval/image->text", "\n".join(texts), step)

    # 2) text -> image
    if "text" in batch_inputs:
        outs = model({"text": to_device(batch_inputs["text"])}, request_outputs=["image"], control=ctrl)
        imgs = outs["image"]
        grid = make_grid(imgs.cpu(), nrow=imgs.size(0), normalize=True)
        writer.add_image("eval/text->image", grid, step)

    # 3) audio -> text
    if "audio" in batch_inputs:
        outs = model({"audio": to_device(batch_inputs["audio"])}, request_outputs=["text"], control=ctrl)
        texts = outs["text"]["text"] if isinstance(outs["text"], dict) else outs["text"]
        writer.add_text("eval/audio->text", "\n".join(texts), step)

    # 4) text -> audio (mel + griffin-lim)
    if "text" in batch_inputs:
        outs = model({"text": to_device(batch_inputs["text"])}, request_outputs=["audio"], control=ctrl)
        audio = outs["audio"]["audio"] if isinstance(outs["audio"], dict) else outs["audio"]
        sr = outs["audio"].get("sample_rate", 22050) if isinstance(outs["audio"], dict) else 22050
        writer.add_audio("eval/text->audio", audio[0].cpu(), step, sample_rate=sr)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device()
        os.makedirs(cfg.logdir, exist_ok=True)
        os.makedirs(cfg.savedir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, "MMT"))
        set_seed(1234)
        self.precision = cfg.precision

        # Data
        if cfg.demo:
            dataset = DemoTriplesDataset(n=512, image_size=cfg.image_size)
            n_val = max(16, len(dataset)//16)
            idx = list(range(len(dataset)))
            random.shuffle(idx)
            train_idx, val_idx = idx[n_val:], idx[:n_val]
            self.train_data = torch.utils.data.Subset(dataset, train_idx)
            self.val_data = torch.utils.data.Subset(dataset, val_idx)
        else:
            assert cfg.jsonl and os.path.exists(cfg.jsonl), "--jsonl path required when not using --demo"
            full = JSONLTriplesDataset(cfg.jsonl)
            n_val = max(64, len(full)//20)
            self.train_data = torch.utils.data.Subset(full, list(range(n_val, len(full))))
            self.val_data = torch.utils.data.Subset(full, list(range(n_val)))
        self.pre = Preproc()

        self.train_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=self.pre)
        self.val_loader = DataLoader(self.val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=self.pre)

        # Model
        self.model = build_model(cfg, self.device)

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if cfg.use_8bit_optim and HAS_BNB:
            self.optim = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.precision=="fp16" and self.device.type=="cuda"))

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            return 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

        self.best_val = float("inf")
        self.global_step = 0
        self.nan_count = 0
        self.max_nan_tolerance = 5

    def _to_device_nested(self, inputs: Dict[str, Dict]) -> Dict[str, Dict]:
        out = {}
        for k, v in inputs.items():
            out[k] = {kk: (vv.to(self.device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        return out

    def _encode_modal(self, inputs: Dict[str, Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        """Use the model's encoder ports to obtain modal latents and masks (no graph)."""
        modal_lat, modal_mask = {}, {}
        for m, port in self.model.enc_ports.items():
            if m in inputs:
                feats, mask = port(inputs[m])
                modal_lat[m] = feats
                modal_mask[m] = mask
        return modal_lat, modal_mask

    def train(self):
        if self.cfg.eval_only:
            assert self.cfg.checkpoint and os.path.exists(self.cfg.checkpoint), "--checkpoint required for --eval_only"
            self.model.load_state_dict(torch.load(self.cfg.checkpoint, map_location=self.device))
            self.evaluate(save_samples=True)
            print(f"Eval-only done. Loaded {self.cfg.checkpoint}")
            return

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                inputs = self._to_device_nested(batch)
                with torch.amp.autocast('cuda', enabled=(self.precision!="fp32" and self.device.type=="cuda")):
                    # 1) Encode for losses
                    modal_lat, _ = self._encode_modal(inputs)
                    # 2) Forward to get Z
                    outs = self.model(inputs, request_outputs=[], control=ThinkControl(steps=self.cfg.think_steps))
                    Z = outs["_Z"]  # (B, N, D)
                    Zp = Z.mean(dim=1)
                    # 3) Alignment losses (Z vs each modal-to-think pooled)
                    align_loss = 0.0
                    cnt = 0
                    for m, z_m in modal_lat.items():
                        zt = self.model.modal2think[m](z_m).mean(dim=1)
                        align_loss = align_loss + info_nce(Zp, zt, t=self.cfg.info_nce_temp)
                        cnt += 1
                    if cnt > 0:
                        align_loss = align_loss / cnt
                    # 4) Reconstruction-in-latent (compare in thinking space)
                    recon_loss = 0.0
                    cnt2 = 0
                    for m, z_m in modal_lat.items():
                        # Map both to thinking space and compare there
                        z_from_think = self.model.modal2think[m](z_m).mean(dim=1)
                        # Use cosine similarity in thinking space
                        recon_loss = recon_loss + (1.0 - F.cosine_similarity(Zp, z_from_think, dim=-1).mean())
                        cnt2 += 1
                    if cnt2 > 0:
                        recon_loss = recon_loss / cnt2
                    total = 0.6 * align_loss + 0.4 * recon_loss

                # NaN/Inf check
                if self.cfg.loss_scale_check:
                    if torch.isnan(total) or torch.isinf(total):
                        self.nan_count += 1
                        print(f"⚠️  Warning: NaN/Inf loss at step {self.global_step} (count: {self.nan_count}/{self.max_nan_tolerance})")
                        if self.nan_count >= self.max_nan_tolerance:
                            print("❌ Too many NaN/Inf losses. Stopping training.")
                            return
                        continue  # Skip this batch
                    else:
                        self.nan_count = 0  # Reset on successful batch

                self.optim.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(total).backward()
                    # Gradient clipping (unscale first for accurate clipping)
                    self.scaler.unscale_(self.optim)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optim.step()

                # Learning rate warmup
                self.scheduler.step()

                # logs
                epoch_losses.append(total.item())
                self.writer.add_scalar("train/align_loss", align_loss.item(), self.global_step)
                self.writer.add_scalar("train/recon_loss", recon_loss.item(), self.global_step)
                self.writer.add_scalar("train/total", total.item(), self.global_step)
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

                # Periodic logging
                if self.global_step % self.cfg.log_every == 0:
                    avg_loss = sum(epoch_losses[-self.cfg.log_every:]) / len(epoch_losses[-self.cfg.log_every:])
                    print(f"[Epoch {epoch} Step {self.global_step}] loss={avg_loss:.4f} grad_norm={grad_norm:.2f} lr={self.scheduler.get_last_lr()[0]:.2e}")

                self.global_step += 1

            # end epoch: validate & save
            val_total = self.evaluate(save_samples=False)
            if val_total < self.best_val:
                self.best_val = val_total
                best_path = os.path.join(self.cfg.savedir, "best.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"[epoch {epoch}] new best {val_total:.4f} -> saved {best_path}")
            last_path = os.path.join(self.cfg.savedir, "last.pt")
            torch.save(self.model.state_dict(), last_path)

        # final samples
        self.evaluate(save_samples=True)
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self, save_samples: bool = True) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            inputs = self._to_device_nested(batch)
            with torch.amp.autocast('cuda', enabled=(self.precision!="fp32" and self.device.type=="cuda")):
                modal_lat, _ = self._encode_modal(inputs)
                outs = self.model(inputs, request_outputs=[], control=ThinkControl(steps=self.cfg.think_steps))
                Z = outs["_Z"]
                Zp = Z.mean(dim=1)
                align_loss = 0.0
                cnt = 0
                for m, z_m in modal_lat.items():
                    zt = self.model.modal2think[m](z_m).mean(dim=1)
                    align_loss = align_loss + info_nce(Zp, zt, t=self.cfg.info_nce_temp)
                    cnt += 1
                if cnt > 0:
                    align_loss = align_loss / cnt
                recon_loss = 0.0
                cnt2 = 0
                for m, z_m in modal_lat.items():
                    # Map both to thinking space and compare there
                    z_from_think = self.model.modal2think[m](z_m).mean(dim=1)
                    # Use cosine similarity in thinking space
                    recon_loss = recon_loss + (1.0 - F.cosine_similarity(Zp, z_from_think, dim=-1).mean())
                    cnt2 += 1
                if cnt2 > 0:
                    recon_loss = recon_loss / cnt2
                total = 0.6 * align_loss + 0.4 * recon_loss

            # Get batch size properly from Z tensor
            batch_size = Z.size(0)
            total_loss += total.item() * batch_size
            n += batch_size

        total_loss = total_loss / max(1, n)
        self.writer.add_scalar("val/total", total_loss, self.global_step)

        if save_samples:
            # grab first batch and log multimodal generations
            try:
                batch = next(iter(self.val_loader))
                sample_and_log(self.model, batch, self.writer, self.global_step, self.device)
            except Exception as e:
                print(f"Warning: Could not generate samples: {e}")
        return total_loss


# -----------------------------------
# Main
# -----------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="use synthetic demo data")
    p.add_argument("--jsonl", type=str, default=None, help="path to JSONL triples dataset")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--savedir", type=str, default="checkpoints")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32","fp16","bf16"])
    p.add_argument("--no_8bit", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()
    return Config(
        demo=args.demo or (args.jsonl is None),
        jsonl=args.jsonl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        logdir=args.logdir,
        savedir=args.savedir,
        precision=args.precision,
        use_8bit_optim=(not args.no_8bit),
        eval_only=args.eval_only,
        checkpoint=args.checkpoint,
    )


def main():
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
