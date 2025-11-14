"""Shared components and preprocessing helpers for Multimodal Brain v2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image

from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel

from multimodal_brain_v2 import (
    ModalityInterface,
    MultimodalBrain,
    ThinkingCore,
    UpAdapter,
)


# -----------------------------------------------------------------------------
# Encoder wrappers
# -----------------------------------------------------------------------------


class E5TextEncoderWrapper(nn.Module):
    """Wraps intfloat/e5-small-v2 as a text encoder."""

    def __init__(self, model_name: str = "intfloat/e5-small-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**inputs, return_dict=True)
        return out.last_hidden_state  # (B, T, 384)


class CLIPVisionEncoderWrapper(nn.Module):
    """Wraps CLIP ViT-B/32 vision tower."""

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
    """Small CNN encoder over mel-spectrograms."""

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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, d_out)

    def forward(self, inputs: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, dict):
            x = inputs["mel"]
        else:
            x = inputs

        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.net(x)
        h = h.view(h.size(0), 128)
        return self.proj(h)


# -----------------------------------------------------------------------------
# Preprocessing helpers
# -----------------------------------------------------------------------------


@dataclass
class PreprocConfig:
    audio_sample_rate: int = 16000
    n_mels: int = 80
    mel_win_size: int = 400
    mel_hop_size: int = 160
    clip_image_size: int = 224
    text_max_length: int = 128


class Preproc:
    """Utility to prepare inputs for training or inference."""

    def __init__(self, cfg: Optional[PreprocConfig] = None):
        self.cfg = cfg or PreprocConfig()
        self.txt_tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        self.clip_proc = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", do_rescale=False
        )
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.audio_sample_rate,
            n_fft=self.cfg.mel_win_size,
            hop_length=self.cfg.mel_hop_size,
            n_mels=self.cfg.n_mels,
            f_min=0,
            f_max=self.cfg.audio_sample_rate // 2,
        )

    # ------------------------------------------------------------------
    # Batch conversion (used during training)
    # ------------------------------------------------------------------
    def convert_flickr_batch(self, batch: Dict[str, Any]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}

        if "caption_strs" in batch:
            texts = batch["caption_strs"]
            tk = self.txt_tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.text_max_length,
                return_tensors="pt",
            )
            out["text"] = {"input_ids": tk.input_ids, "attention_mask": tk.attention_mask}
            out["_raw_texts"] = texts

        if "images" in batch:
            imgs = batch["images"]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
            imgs_denorm = torch.clamp(imgs * std + mean, 0, 1)
            px = self.clip_proc(images=imgs_denorm, return_tensors="pt")
            out["image"] = {"pixel_values": px.pixel_values}

        if "audio" in batch:
            out["audio"] = {"mel": batch["audio"].float()}

        return out

    # ------------------------------------------------------------------
    # Direct user input helpers (used for standalone inference)
    # ------------------------------------------------------------------
    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(image)).float() / 255.0
        arr = arr.permute(2, 0, 1)
        return arr

    def _load_audio_mel(self, path: Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(str(path))
        if sr != self.cfg.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.cfg.audio_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform).squeeze(0)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        return mel_spec

    def prepare_user_inputs(
        self,
        *,
        texts: Optional[Iterable[str]] = None,
        image_paths: Optional[Iterable[str]] = None,
        audio_paths: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        device = device or torch.device("cpu")
        inputs: Dict[str, Dict[str, torch.Tensor]] = {}

        counts = []
        if texts:
            text_list = list(texts)
            counts.append(len(text_list))
        else:
            text_list = []
        if image_paths:
            image_list = list(image_paths)
            counts.append(len(image_list))
        else:
            image_list = []
        if audio_paths:
            audio_list = list(audio_paths)
            counts.append(len(audio_list))
        else:
            audio_list = []

        if not counts:
            raise ValueError("At least one modality must be provided for inference.")

        batch_sizes = set(counts)
        if len(batch_sizes) > 1:
            raise ValueError(
                "All provided modalities must have the same number of samples."
            )

        if text_list:
            tk = self.txt_tok(
                text_list,
                padding=True,
                truncation=True,
                max_length=self.cfg.text_max_length,
                return_tensors="pt",
            )
            inputs["text"] = {
                "input_ids": tk.input_ids.to(device),
                "attention_mask": tk.attention_mask.to(device),
            }
            inputs["_raw_texts"] = text_list

        if image_list:
            images = torch.stack([self._load_image_tensor(Path(p)) for p in image_list])
            px = self.clip_proc(images=images, return_tensors="pt")
            inputs["image"] = {"pixel_values": px.pixel_values.to(device)}

        if audio_list:
            mels = [self._load_audio_mel(Path(p)) for p in audio_list]
            max_time = max(mel.shape[1] for mel in mels)
            padded = []
            for mel in mels:
                if mel.shape[1] < max_time:
                    pad = torch.zeros(mel.shape[0], max_time - mel.shape[1])
                    mel = torch.cat([mel, pad], dim=1)
                padded.append(mel)
            mel_batch = torch.stack(padded).unsqueeze(1)
            inputs["audio"] = {"mel": mel_batch.to(device)}

        return inputs


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------


def build_brain(
    *,
    d_shared: int,
    device: torch.device,
    freeze_text: bool = True,
    freeze_image: bool = True,
    train_audio_encoder: bool = True,
) -> MultimodalBrain:
    print("Loading E5 text encoder...")
    text_enc = E5TextEncoderWrapper()
    print("Loading CLIP vision encoder...")
    img_enc = CLIPVisionEncoderWrapper()
    print("Initializing audio CNN encoder...")
    aud_enc = AudioCNNEncoder(n_mels=80, d_out=384)

    text_up = UpAdapter(d_in=384, d_shared=d_shared)
    img_up = UpAdapter(d_in=768, d_shared=d_shared)
    aud_up = UpAdapter(d_in=384, d_shared=d_shared)

    text_iface = ModalityInterface(
        name="text",
        encoder=text_enc,
        up_adapter=text_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=freeze_text,
        freeze_decoder=True,
    )
    image_iface = ModalityInterface(
        name="image",
        encoder=img_enc,
        up_adapter=img_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=freeze_image,
        freeze_decoder=True,
    )
    audio_iface = ModalityInterface(
        name="audio",
        encoder=aud_enc,
        up_adapter=aud_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=not train_audio_encoder,
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


__all__ = [
    "AudioCNNEncoder",
    "CLIPVisionEncoderWrapper",
    "E5TextEncoderWrapper",
    "Preproc",
    "PreprocConfig",
    "build_brain",
]
