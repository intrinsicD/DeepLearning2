"""
Brain v3 Components - Encoder wrappers, preprocessing, and model factory.

This module provides:
- Encoder wrappers for text (E5), vision (CLIP), and audio (CNN/Whisper)
- Preprocessing utilities for converting raw inputs
- Model factory for building complete MultimodalBrainV3 models
- Decoder components for output generation

Optimized for 8GB GPU training with option to scale up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from PIL import Image

from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel


# ============================================================================
# Import Brain V3 components
# ============================================================================

from multimodal_brain_v3 import (
    BrainConfig,
    ModelSize,
    MultimodalBrainV3,
    ModalityInterface,
    UpAdapter,
    DownAdapter,
    RMSNorm,
    freeze_module,
)


# ============================================================================
# Encoder Wrappers
# ============================================================================

class E5TextEncoderWrapper(nn.Module):
    """
    Wraps E5 text encoder (intfloat/e5-small-v2 or intfloat/e5-base-v2).

    E5 is an efficient text encoder with strong performance on
    semantic similarity tasks.

    Output: (batch, seq, 384) for small, (batch, seq, 768) for base
    """

    MODELS = {
        "small": ("intfloat/e5-small-v2", 384),
        "base": ("intfloat/e5-base-v2", 768),
    }

    def __init__(
        self,
        size: str = "small",
        freeze: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        model_name, self.output_dim = self.MODELS[size]

        self.model = AutoModel.from_pretrained(model_name)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if freeze:
            freeze_module(self.model, trainable=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Dict with "input_ids" and "attention_mask"

        Returns:
            (batch, seq, output_dim) hidden states
        """
        out = self.model(**inputs, return_dict=True)
        return out.last_hidden_state


class CLIPVisionEncoderWrapper(nn.Module):
    """
    Wraps CLIP ViT vision encoder.

    Available models:
    - openai/clip-vit-base-patch32 (faster, 768 dim)
    - openai/clip-vit-base-patch16 (more accurate, 768 dim)
    - openai/clip-vit-large-patch14 (most accurate, 1024 dim)

    Output: (batch, num_patches+1, dim) where +1 is CLS token
    """

    MODELS = {
        "base32": ("openai/clip-vit-base-patch32", 768),
        "base16": ("openai/clip-vit-base-patch16", 768),
        "large14": ("openai/clip-vit-large-patch14", 1024),
    }

    def __init__(
        self,
        size: str = "base32",
        freeze: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        model_name, self.output_dim = self.MODELS[size]

        self.model = CLIPModel.from_pretrained(model_name)

        if gradient_checkpointing:
            self.model.vision_model.encoder.gradient_checkpointing = True

        if freeze:
            freeze_module(self.model, trainable=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Dict with "pixel_values" (batch, 3, 224, 224)

        Returns:
            (batch, num_patches+1, dim) vision features
        """
        pixel_values = inputs["pixel_values"]
        out = self.model.vision_model(pixel_values=pixel_values)
        return out.last_hidden_state


class AudioCNNEncoder(nn.Module):
    """
    CNN encoder for audio mel-spectrograms.

    A lightweight CNN that processes mel-spectrograms and outputs
    a fixed-size representation. Efficient and trainable.

    Input: (batch, 1, n_mels, time) or (batch, n_mels, time)
    Output: (batch, output_dim)
    """

    def __init__(
        self,
        n_mels: int = 80,
        output_dim: int = 384,
        hidden_channels: List[int] = [32, 64, 128],
    ):
        super().__init__()
        self.n_mels = n_mels
        self.output_dim = output_dim

        layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d((2, 2)),
            ])
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)

        self.proj = nn.Sequential(
            nn.Linear(hidden_channels[-1], output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            inputs: Tensor or Dict with "mel" key
                   Shapes: (batch, n_mels, time), (batch, 1, n_mels, time)

        Returns:
            (batch, output_dim)
        """
        if isinstance(inputs, dict):
            x = inputs["mel"]
        else:
            x = inputs

        # Handle different input shapes
        if x.dim() == 3:
            # (batch, n_mels, time) - add channel dim
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # (batch, 1, n_mels, time) - already correct
            pass
        else:
            raise ValueError(f"AudioCNNEncoder expects 3D or 4D input, got {x.dim()}D: {x.shape}")

        # CNN forward
        h = self.conv(x)
        h = h.view(h.size(0), -1)

        return self.proj(h)


class AudioTransformerEncoder(nn.Module):
    """
    Transformer encoder for audio (more powerful, more memory).

    Uses a small ViT-style architecture on mel-spectrograms.
    Better for longer audio and complex patterns.

    Input: (batch, 1, n_mels, time)
    Output: (batch, num_patches, dim)
    """

    def __init__(
        self,
        n_mels: int = 80,
        output_dim: int = 384,
        patch_size: int = 16,
        n_layers: int = 4,
        n_heads: int = 6,
        max_length: int = 512,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.output_dim = output_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, output_dim,
            kernel_size=(n_mels, patch_size),
            stride=(n_mels, patch_size),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, output_dim) * 0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            inputs: Tensor (batch, n_mels, time) or Dict with "mel" key

        Returns:
            (batch, num_patches+1, output_dim)
        """
        if isinstance(inputs, dict):
            x = inputs["mel"]
        else:
            x = inputs

        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # Patch embedding: (batch, dim, 1, n_patches) -> (batch, n_patches, dim)
        x = self.patch_embed(x)
        x = x.squeeze(2).transpose(1, 2)

        n_patches = x.size(1)

        # Add positional embedding
        x = x + self.pos_embed[:, :n_patches]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        return x


# ============================================================================
# Decoder Components
# ============================================================================

class TextDecoder(nn.Module):
    """
    Simple text decoder that maps latent to vocabulary logits.

    For inference, converts to text strings.
    """

    def __init__(
        self,
        d_in: int,
        tokenizer_name: str = "intfloat/e5-small-v2",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size

        self.head = nn.Sequential(
            RMSNorm(d_in),
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Linear(d_in * 2, self.vocab_size),
        )

    def forward(self, x: torch.Tensor) -> List[str]:
        """
        Args:
            x: (batch, d_in)

        Returns:
            List of decoded strings
        """
        logits = self.head(x)
        tokens = logits.argmax(dim=-1).cpu().tolist()
        texts = [
            self.tokenizer.decode([t], skip_special_tokens=True).strip()
            for t in tokens
        ]
        return texts


class VAEImageDecoder(nn.Module):
    """
    Image decoder using pretrained Stable Diffusion VAE.

    Maps latent vectors to VAE latent space, then decodes to images.
    """

    def __init__(
        self,
        d_in: int,
        vae_name: str = "stabilityai/sd-vae-ft-mse",
        latent_channels: int = 4,
        latent_size: int = 32,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_channels * latent_size * latent_size

        # Projection to VAE latent
        self.proj = nn.Sequential(
            RMSNorm(d_in),
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Linear(d_in * 2, self.latent_dim),
        )

        # Load VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(vae_name)
        freeze_module(self.vae, trainable=False)

        # Scaling factor
        self.scale = getattr(
            self.vae.config, "scaling_factor",
            getattr(self.vae.config, "scale_factor", 0.18215)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_in)

        Returns:
            (batch, 3, H, W) images in [0, 1]
        """
        batch_size = x.size(0)

        # Project to VAE latent space
        latent = self.proj(x)
        latent = latent.view(batch_size, self.latent_channels, self.latent_size, self.latent_size)

        # Decode through VAE
        latent_scaled = latent / self.scale
        with torch.no_grad():
            images = self.vae.decode(latent_scaled).sample

        # Clamp to [0, 1]
        images = (images.clamp(-1, 1) + 1) / 2

        return images


# ============================================================================
# Preprocessing
# ============================================================================

@dataclass
class PreprocConfig:
    """Configuration for preprocessing."""
    audio_sample_rate: int = 16000
    n_mels: int = 80
    mel_win_size: int = 400
    mel_hop_size: int = 160
    clip_image_size: int = 224
    text_max_length: int = 128
    text_encoder_size: str = "small"
    vision_encoder_size: str = "base32"


class Preproc:
    """
    Unified preprocessing for all modalities.

    Handles:
    - Text tokenization (E5)
    - Image preprocessing (CLIP)
    - Audio mel-spectrogram computation
    """

    def __init__(self, cfg: Optional[PreprocConfig] = None):
        self.cfg = cfg or PreprocConfig()

        # Text tokenizer
        text_model = E5TextEncoderWrapper.MODELS[self.cfg.text_encoder_size][0]
        self.txt_tokenizer = AutoTokenizer.from_pretrained(text_model)

        # Image processor
        vision_model = CLIPVisionEncoderWrapper.MODELS[self.cfg.vision_encoder_size][0]
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            vision_model, do_rescale=False
        )

        # Audio mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.audio_sample_rate,
            n_fft=self.cfg.mel_win_size,
            hop_length=self.cfg.mel_hop_size,
            n_mels=self.cfg.n_mels,
            f_min=0,
            f_max=self.cfg.audio_sample_rate // 2,
        )

    def tokenize_text(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text for E5 encoder."""
        tokens = self.txt_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.text_max_length,
            return_tensors="pt",
        )

        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}

        return tokens

    def process_images(
        self,
        images: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process images for CLIP encoder."""
        # Denormalize if needed (assuming ImageNet normalization)
        if images.min() < 0 or images.max() > 1:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images = torch.clamp(images * std + mean, 0, 1)

        # Convert to list of numpy arrays for CLIP processor
        # CLIP processor expects (H, W, C) format in range [0, 255] or [0, 1]
        images_np = images.cpu().numpy()
        # Convert from (B, C, H, W) to list of (H, W, C)
        images_list = [img.transpose(1, 2, 0) for img in images_np]

        # Process through CLIP processor
        px = self.clip_processor(images=images_list, return_tensors="pt")

        if device is not None:
            px = {k: v.to(device) for k, v in px.items()}

        return px

    def process_audio(
        self,
        audio: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio input to mel-spectrogram format.

        Handles multiple input formats:
        - Raw waveform: (batch, samples) or (batch, 1, samples)
        - Pre-computed mel: (batch, n_mels, time) or (batch, 1, n_mels, time)
        """
        # Ensure float
        audio = audio.float()

        # Detect if input is already a mel-spectrogram
        # Mel spectrograms have shape (batch, [1,] n_mels, time) where n_mels is typically 80
        # Raw audio has shape (batch, samples) or (batch, 1, samples) where samples >> n_mels

        is_mel_spectrogram = False

        if audio.dim() == 4:
            # (batch, 1, n_mels, time) - definitely a mel-spectrogram with channel dim
            is_mel_spectrogram = True
            mel = audio  # Keep as-is, already in correct format
        elif audio.dim() == 3:
            # Could be (batch, 1, samples) raw audio OR (batch, n_mels, time) mel
            # Heuristic: if dim 1 == n_mels (80), it's likely a mel
            if audio.shape[1] == self.cfg.n_mels:
                # (batch, n_mels, time) - mel without channel dim
                is_mel_spectrogram = True
                mel = audio.unsqueeze(1)  # Add channel dim -> (batch, 1, n_mels, time)
            elif audio.shape[1] == 1:
                # Check if it looks like (batch, 1, n_mels, time) squeezed wrong
                # or (batch, 1, samples) raw audio
                if audio.shape[2] == self.cfg.n_mels:
                    # Likely transposed mel
                    is_mel_spectrogram = True
                    mel = audio.unsqueeze(1)
                else:
                    # (batch, 1, samples) - raw audio with channel dim
                    is_mel_spectrogram = False
        elif audio.dim() == 2:
            # (batch, samples) - raw audio
            is_mel_spectrogram = False

        if is_mel_spectrogram:
            # Input is already a mel-spectrogram
            # Ensure shape is (batch, 1, n_mels, time)
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)

            # The dataset may have already normalized, but let's ensure consistency
            # Only normalize if values seem unnormalized (large range)
            if mel.abs().max() > 20:  # Likely unnormalized log-mel
                mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-9)
        else:
            # Need to compute mel-spectrogram from raw audio
            if audio.dim() == 3:
                # (batch, channels, samples) - use first channel
                audio = audio[:, 0]
            # audio is now (batch, samples)
            mel = self.mel_transform(audio)  # (batch, n_mels, time)

            # Log mel and normalize
            mel = torch.log(mel + 1e-9)
            mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-9)

            # Add channel dim
            mel = mel.unsqueeze(1)  # (batch, 1, n_mels, time)

        out = {"mel": mel}

        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}

        return out

    def convert_batch(
        self,
        batch: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert a batch dict to model inputs.

        Expected batch keys:
        - "caption_strs" or "texts": List of strings
        - "images": (batch, 3, H, W) tensor
        - "audio": (batch, samples) or (batch, 1, samples) tensor
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}

        # Text
        if "caption_strs" in batch:
            out["text"] = self.tokenize_text(batch["caption_strs"], device)
        elif "texts" in batch:
            out["text"] = self.tokenize_text(batch["texts"], device)

        # Images
        if "images" in batch:
            out["image"] = self.process_images(batch["images"], device)

        # Audio
        if "audio" in batch:
            out["audio"] = self.process_audio(batch["audio"], device)

        return out

    def prepare_inference_inputs(
        self,
        texts: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepare inputs for inference from file paths or raw text.
        """
        device = device or torch.device("cpu")
        inputs: Dict[str, Dict[str, torch.Tensor]] = {}

        # Validate batch sizes
        sizes = []
        if texts:
            sizes.append(len(texts))
        if image_paths:
            sizes.append(len(image_paths))
        if audio_paths:
            sizes.append(len(audio_paths))

        if len(set(sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes: {sizes}")

        if not sizes:
            raise ValueError("At least one modality must be provided")

        # Text
        if texts:
            inputs["text"] = self.tokenize_text(texts, device)

        # Images
        if image_paths:
            images = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                arr = torch.from_numpy(np.array(img)).float() / 255.0
                arr = arr.permute(2, 0, 1)
                images.append(arr)
            images = torch.stack(images)
            inputs["image"] = self.process_images(images, device)

        # Audio
        if audio_paths:
            mels = []
            for path in audio_paths:
                waveform, sr = torchaudio.load(str(path))
                if sr != self.cfg.audio_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.cfg.audio_sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                mel = self.mel_transform(waveform).squeeze(0)
                mel = torch.log(mel + 1e-9)
                mel = (mel - mel.mean()) / (mel.std() + 1e-9)
                mels.append(mel)

            # Pad to same length
            max_time = max(m.shape[-1] for m in mels)
            padded = []
            for mel in mels:
                if mel.shape[-1] < max_time:
                    pad = torch.zeros(mel.shape[0], max_time - mel.shape[-1])
                    mel = torch.cat([mel, pad], dim=-1)
                padded.append(mel)

            mel_batch = torch.stack(padded)
            inputs["audio"] = {"mel": mel_batch.to(device)}

        return inputs


# ============================================================================
# Model Factory
# ============================================================================

def build_brain_v3(
    size: Union[ModelSize, str] = "small",
    device: torch.device = None,
    freeze_text: bool = True,
    freeze_image: bool = True,
    train_audio: bool = True,
    use_image_decoder: bool = False,
    use_text_decoder: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> MultimodalBrainV3:
    """
    Factory function to build a complete MultimodalBrainV3.

    Args:
        size: Model size preset ("tiny", "small", "base", "large", "xlarge")
        device: Target device
        freeze_text: Freeze text encoder
        freeze_image: Freeze image encoder
        train_audio: Train audio encoder
        use_image_decoder: Include VAE image decoder
        use_text_decoder: Include text decoder
        config_overrides: Override config values

    Returns:
        Configured MultimodalBrainV3 model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create config
    overrides = config_overrides or {}
    config = BrainConfig.from_size(size, **overrides)

    print(f"Building MultimodalBrainV3 ({size})...")
    print(f"  d_shared: {config.d_shared}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_experts: {config.n_experts if config.use_moe else 'disabled'}")
    print(f"  gradient_checkpointing: {config.use_gradient_checkpointing}")

    # Select encoder dimensions based on config
    if config.d_shared <= 384:
        text_size, vision_size = "small", "base32"
    else:
        text_size, vision_size = "base", "base16"

    # Build encoders
    print(f"\nLoading E5 text encoder ({text_size})...")
    text_enc = E5TextEncoderWrapper(
        size=text_size,
        freeze=freeze_text,
        gradient_checkpointing=config.use_gradient_checkpointing and not freeze_text,
    )

    print(f"Loading CLIP vision encoder ({vision_size})...")
    image_enc = CLIPVisionEncoderWrapper(
        size=vision_size,
        freeze=freeze_image,
        gradient_checkpointing=config.use_gradient_checkpointing and not freeze_image,
    )

    print("Creating audio CNN encoder...")
    audio_enc = AudioCNNEncoder(n_mels=80, output_dim=384)
    if not train_audio:
        freeze_module(audio_enc, trainable=False)

    # Build adapters
    text_up = UpAdapter(
        d_in=text_enc.output_dim,
        d_shared=config.d_shared,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        pooling="attention",
    )

    image_up = UpAdapter(
        d_in=image_enc.output_dim,
        d_shared=config.d_shared,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        pooling="attention",
    )

    audio_up = UpAdapter(
        d_in=audio_enc.output_dim,
        d_shared=config.d_shared,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        pooling="mean",
    )

    # Optional decoders
    text_decoder = None
    text_down = None
    if use_text_decoder:
        text_down = DownAdapter(config.d_shared, config.d_shared)
        text_decoder = TextDecoder(config.d_shared)

    image_decoder = None
    image_down = None
    if use_image_decoder:
        image_down = DownAdapter(config.d_shared, config.d_shared)
        image_decoder = VAEImageDecoder(config.d_shared)

    # Build modality interfaces
    text_iface = ModalityInterface(
        name="text",
        encoder=text_enc,
        up_adapter=text_up,
        decoder=text_decoder,
        down_adapter=text_down,
        freeze_encoder=freeze_text,
        freeze_decoder=True if text_decoder else True,
    )

    image_iface = ModalityInterface(
        name="image",
        encoder=image_enc,
        up_adapter=image_up,
        decoder=image_decoder,
        down_adapter=image_down,
        freeze_encoder=freeze_image,
        freeze_decoder=True if image_decoder else True,
    )

    audio_iface = ModalityInterface(
        name="audio",
        encoder=audio_enc,
        up_adapter=audio_up,
        decoder=None,
        down_adapter=None,
        freeze_encoder=not train_audio,
        freeze_decoder=True,
    )

    modalities = {
        "text": text_iface,
        "image": image_iface,
        "audio": audio_iface,
    }

    # Build complete model
    print("\nAssembling MultimodalBrainV3...")
    brain = MultimodalBrainV3(config=config, modalities=modalities)

    # Move to device
    print(f"Moving model to {device}...")
    brain = brain.to(device)

    # Print parameter counts
    counts = brain.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {counts['total']:,}")
    print(f"  Trainable: {counts['trainable']:,}")
    print(f"  Frozen: {counts['frozen']:,}")

    return brain


# ============================================================================
# Quick sanity check
# ============================================================================

if __name__ == "__main__":
    print("Testing brain_v3_components...")

    # Test preprocessing
    preproc = Preproc()
    print("Preproc initialized")

    # Test tokenization
    tokens = preproc.tokenize_text(["Hello world", "Test sentence"])
    print(f"Text tokens shape: {tokens['input_ids'].shape}")

    # Test model factory (without loading pretrained models for speed)
    print("\nTesting model factory...")

    # Just test config
    config = BrainConfig.from_size("tiny")
    print(f"Tiny config: {config.d_shared}d, {config.n_layers}L")

    print("\nAll tests passed!")
