"""Multimodal Neural Architecture with Test-Time Training Memory.

A unified multimodal architecture that processes text, images, and audio through
modality-specific encoders, maintains a central latent memory with test-time
training capabilities, and can decode back to any modality.

Architecture:
    Input → Modality Encoder → Update Memory → Fuse with Memory → Central Latent
         ↓                                                            ↓
    Decoder ← Memory Readout ←────────── Feedback Loop ──────────────┘

Key Features:
- Modality-specific encoders (Transformer for text, ViT for images, CNN for audio)
- Central latent memory with attention-based test-time training
- Cross-modal fusion and modality-specific decoding
- Feedback loops for iterative refinement
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseArchitecture


class TestTimeMemory(nn.Module):
    """Test-Time Training Memory Block with proper multi-head attention.

    Maintains a learnable memory that can be updated during inference
    using attention-weighted updates with proper multi-head attention.

    Args:
        memory_size: Number of memory slots
        memory_dim: Dimensionality of each memory slot
        num_heads: Number of attention heads for memory access
        dropout: Dropout probability
        enable_ttt_updates: Enable test-time training updates (works in eval mode)
        ttt_topk: Number of top-k memory slots to update during TTT
        ttt_lr: Step size for TTT updates (or use inner optimizer)
    """
    
    def __init__(
        self,
        memory_size: int = 128,
        memory_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        enable_ttt_updates: bool = False,
        ttt_topk: int = 8,
        ttt_lr: float = 0.1,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.enable_ttt_updates = enable_ttt_updates
        self.ttt_topk = ttt_topk
        self.ttt_lr = ttt_lr

        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.02)
        
        # Use PyTorch's proper MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Memory update network (for TTT) - vectorized
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid(),
        )
        
        self.memory_update = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(memory_dim)
        
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory using proper multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, dim) or (batch, dim)
        
        Returns:
            Memory-augmented representation
        """
        if query.ndim == 2:
            query = query.unsqueeze(1)  # (batch, 1, dim)
        
        batch_size = query.shape[0]

        # Expand memory for batch: (memory_size, dim) -> (batch, memory_size, dim)
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        # Use PyTorch's MultiheadAttention: query attends to memory
        # MHA expects (L, N, E) when batch_first=False, or (N, L, E) when batch_first=True
        # We set batch_first=True, so: (batch, seq_len, dim)
        attended, _ = self.mha(query, memory, memory, need_weights=False)

        # Residual connection and norm
        output = self.norm(attended + query)

        if output.shape[1] == 1:
            output = output.squeeze(1)
        
        return output
    
    @torch.no_grad()
    def ttt_update(self, content: torch.Tensor):
        """Vectorized top-k test-time training update (works in eval mode).

        Args:
            content: Content to use for update (batch, ..., dim)
        """
        # Aggregate content across batch and spatial dimensions -> (dim,)
        content_summary = content.flatten(0, -2).mean(dim=0) if content.ndim > 2 else content.mean(dim=0)

        # Compute similarity: (memory_size,)
        similarity = self.memory @ content_summary

        # Select top-k slots to update
        topk_indices = similarity.topk(self.ttt_topk).indices

        # Prepare inputs for update networks: (topk, dim*2)
        selected_memory = self.memory[topk_indices]  # (topk, dim)
        content_expanded = content_summary.unsqueeze(0).expand(self.ttt_topk, -1)  # (topk, dim)
        combined = torch.cat([selected_memory, content_expanded], dim=-1)  # (topk, dim*2)

        # Compute gated update
        gate = self.update_gate(combined)  # (topk, dim)
        update_values = self.memory_update(combined)  # (topk, dim)

        # Apply update with step size
        new_values = selected_memory * (1 - self.ttt_lr * gate) + self.ttt_lr * gate * update_values

        # Write back (in-place but within no_grad)
        self.memory[topk_indices] = new_values

    def write(self, content: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        """Write to memory with optional test-time training.

        Args:
            content: Content to write (batch, dim) or (batch, seq_len, dim)
            update_memory: Whether to trigger TTT update

        Returns:
            Memory-augmented content
        """
        if content.ndim == 2:
            content = content.unsqueeze(1)
        
        # Read from memory first (uses MHA)
        augmented = self.read(content)
        
        # Test-time training: Update memory if enabled (works in eval mode)
        if update_memory and self.enable_ttt_updates:
            self.ttt_update(content)

        if augmented.ndim == 3 and augmented.shape[1] == 1:
            augmented = augmented.squeeze(1)
        
        return augmented
    
    def forward(self, x: torch.Tensor, mode: str = "read") -> torch.Tensor:
        """Forward pass through memory.
        
        Args:
            x: Input tensor
            mode: "read" or "write"
        
        Returns:
            Memory-augmented output
        """
        if mode == "read":
            return self.read(x)
        else:
            # For write mode, TTT updates are controlled by enable_ttt_updates flag
            return self.write(x, update_memory=True)


class TextEncoder(nn.Module):
    """Transformer-based text encoder.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text tokens.
        
        Args:
            x: Token IDs (batch, seq_len)
            mask: Attention mask
        
        Returns:
            Text embeddings (batch, embed_dim)
        """
        batch_size, seq_len = x.shape
        
        # Token and position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Pool to single vector (mean pooling)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        return self.norm(x)


class ImageEncoder(nn.Module):
    """Vision Transformer for image encoding.
    
    Args:
        image_size: Input image size
        patch_size: Patch size for ViT
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images.
        
        Args:
            x: Images (batch, channels, height, width)
        
        Returns:
            Image embeddings (batch, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Extract CLS token
        x = x[:, 0]
        
        return self.norm(x)


class AudioEncoder(nn.Module):
    """CNN-based audio encoder for spectrograms.
    
    Args:
        in_channels: Number of input channels (1 for mono spectrogram)
        embed_dim: Output embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # CNN for spectrogram processing
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio spectrograms.
        
        Args:
            x: Spectrograms (batch, channels, freq, time)
        
        Returns:
            Audio embeddings (batch, embed_dim)
        """
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return self.norm(x)


class ModalityDecoder(nn.Module):
    """Generic decoder from central latent to modality-specific output.
    
    Args:
        latent_dim: Central latent dimension
        output_dim: Output dimension for the modality
        num_layers: Number of decoder layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_dim = latent_dim
        
        for i in range(num_layers - 1):
            next_dim = (latent_dim + output_dim) // 2 if i == 0 else current_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode from central latent.
        
        Args:
            x: Central latent representation (batch, latent_dim)
        
        Returns:
            Modality-specific representation (batch, output_dim)
        """
        return self.decoder(x)


class MultiModalMemoryNetwork(BaseArchitecture):
    """Multimodal Neural Network with Test-Time Training Memory.
    
    Processes text, images, and audio through modality-specific encoders,
    maintains a central latent memory with TTT, and can decode to any modality.
    
    Args:
        # Text encoder params
        vocab_size: Vocabulary size for text
        text_embed_dim: Text embedding dimension
        text_seq_len: Maximum text sequence length
        
        # Image encoder params
        image_size: Input image size
        patch_size: Patch size for ViT
        image_channels: Number of image channels
        
        # Audio encoder params
        audio_channels: Number of audio channels
        
        # Central params
        latent_dim: Central latent dimension
        memory_size: Number of memory slots
        num_heads: Number of attention heads
        num_layers: Number of layers in encoders/decoders
        dropout: Dropout probability
        
        # TTT params
        ttt_mode: Test-time training mode ("attention" or "gradient")
        feedback_steps: Number of feedback loop iterations
    """
    
    def __init__(
        self,
        # Text params
        vocab_size: int = 10000,
        text_embed_dim: int = 512,
        text_seq_len: int = 512,
        
        # Image params
        image_size: int = 224,
        patch_size: int = 16,
        image_channels: int = 3,
        
        # Audio params
        audio_channels: int = 1,
        
        # Central params
        latent_dim: int = 512,
        memory_size: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        
        # TTT params
        ttt_mode: str = "attention",
        feedback_steps: int = 2,
        enable_ttt_updates: bool = False,
        ttt_topk: int = 8,
        ttt_lr: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.feedback_steps = feedback_steps
        
        # Modality encoders
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=text_seq_len,
            dropout=dropout,
        )
        
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=image_channels,
            embed_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.audio_encoder = AudioEncoder(
            in_channels=audio_channels,
            embed_dim=latent_dim,
            dropout=dropout,
        )
        
        # Projection to unified latent space
        self.text_proj = nn.Linear(text_embed_dim, latent_dim)
        
        # Learned modality type embeddings (helps fusion distinguish modalities)
        self.modality_type_embeddings = nn.Parameter(torch.randn(3, latent_dim) * 0.02)

        # Central memory with TTT
        self.central_memory = TestTimeMemory(
            memory_size=memory_size,
            memory_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_ttt_updates=enable_ttt_updates,
            ttt_topk=ttt_topk,
            ttt_lr=ttt_lr,
        )
        
        # Modality-specific memories with TTT
        self.text_memory = TestTimeMemory(
            memory_size=memory_size // 2,
            memory_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_ttt_updates=enable_ttt_updates,
            ttt_topk=ttt_topk // 2,
            ttt_lr=ttt_lr,
        )
        
        self.image_memory = TestTimeMemory(
            memory_size=memory_size // 2,
            memory_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_ttt_updates=enable_ttt_updates,
            ttt_topk=ttt_topk // 2,
            ttt_lr=ttt_lr,
        )
        
        self.audio_memory = TestTimeMemory(
            memory_size=memory_size // 2,
            memory_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_ttt_updates=enable_ttt_updates,
            ttt_topk=ttt_topk // 2,
            ttt_lr=ttt_lr,
        )
        
        # Cross-modal fusion with presence signals (latent_dim*3 + 3 presence bits)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3 + 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        # Feedback loop
        self.feedback = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Modality decoders
        self.text_decoder = ModalityDecoder(
            latent_dim=latent_dim,
            output_dim=text_embed_dim,
            num_layers=3,
            dropout=dropout,
        )
        
        self.image_decoder = ModalityDecoder(
            latent_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=3,
            dropout=dropout,
        )
        
        self.audio_decoder = ModalityDecoder(
            latent_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=3,
            dropout=dropout,
        )
        
    def encode_modality(
        self,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode modalities and update their respective memories.
        
        Args:
            text: Text token IDs (batch, seq_len)
            images: Images (batch, channels, height, width)
            audio: Audio spectrograms (batch, channels, freq, time)
            text_mask: Text attention mask
        
        Returns:
            Dictionary of encoded representations
        """
        encodings = {}
        
        if text is not None:
            text_enc = self.text_encoder(text, text_mask)
            text_enc = self.text_proj(text_enc)
            text_enc = self.text_memory.write(text_enc)
            encodings['text'] = text_enc
        
        if images is not None:
            image_enc = self.image_encoder(images)
            image_enc = self.image_memory.write(image_enc)
            encodings['image'] = image_enc
        
        if audio is not None:
            audio_enc = self.audio_encoder(audio)
            audio_enc = self.audio_memory.write(audio_enc)
            encodings['audio'] = audio_enc
        
        return encodings
    
    def fuse_modalities(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modalities into unified representation with presence signals.

        Args:
            encodings: Dictionary of modality encodings
        
        Returns:
            Fused representation (batch, latent_dim)
        """
        if not encodings:
            raise ValueError("Cannot fuse empty encodings - at least one modality required")

        batch_size = next(iter(encodings.values())).shape[0]
        device = next(iter(encodings.values())).device

        # Get modality encodings or zeros for missing modalities
        text_enc = encodings.get('text', torch.zeros(batch_size, self.latent_dim, device=device))
        image_enc = encodings.get('image', torch.zeros(batch_size, self.latent_dim, device=device))
        audio_enc = encodings.get('audio', torch.zeros(batch_size, self.latent_dim, device=device))

        # Create presence mask (1 if modality present, 0 otherwise)
        presence = torch.tensor([
            1.0 if 'text' in encodings else 0.0,
            1.0 if 'image' in encodings else 0.0,
            1.0 if 'audio' in encodings else 0.0,
        ], device=device).unsqueeze(0).expand(batch_size, -1)  # (batch, 3)

        # Add learned modality type embeddings (helps fusion know what's present)
        text_enc = text_enc + self.modality_type_embeddings[0]
        image_enc = image_enc + self.modality_type_embeddings[1]
        audio_enc = audio_enc + self.modality_type_embeddings[2]

        # Concatenate: (batch, latent_dim*3 + 3)
        combined = torch.cat([text_enc, image_enc, audio_enc, presence], dim=1)
        fused = self.fusion(combined)
        
        return fused
    
    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        decode_to: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multimodal network.
        
        Args:
            text: Text token IDs (batch, seq_len)
            images: Images (batch, channels, height, width)
            audio: Audio spectrograms (batch, channels, freq, time)
            text_mask: Text attention mask
            decode_to: List of modalities to decode to (e.g., ['text', 'image'])
        
        Returns:
            Dictionary containing:
                - central_latent: Unified representation
                - decoded_{modality}: Decoded outputs if requested
        """
        # Encode modalities
        encodings = self.encode_modality(text, images, audio, text_mask)
        
        # Fuse modalities
        fused = self.fuse_modalities(encodings)
        
        # Update central memory and apply feedback loop
        central_latent = self.central_memory.write(fused)
        
        for _ in range(self.feedback_steps):
            central_latent = central_latent + self.feedback(central_latent)
            central_latent = self.central_memory.read(central_latent)
        
        outputs = {'central_latent': central_latent}
        
        # Decode to requested modalities
        if decode_to is not None:
            if 'text' in decode_to:
                outputs['decoded_text'] = self.text_decoder(central_latent)
            if 'image' in decode_to:
                outputs['decoded_image'] = self.image_decoder(central_latent)
            if 'audio' in decode_to:
                outputs['decoded_audio'] = self.audio_decoder(central_latent)
        
        return outputs
    
    def cross_modal_retrieval(
        self,
        query_modality: str,
        query_data: torch.Tensor,
        target_modality: str,
    ) -> torch.Tensor:
        """Cross-modal retrieval: encode one modality, decode to another.
        
        Args:
            query_modality: Source modality ('text', 'image', or 'audio')
            query_data: Query data tensor
            target_modality: Target modality to decode to
        
        Returns:
            Decoded representation in target modality
        """
        # Map modality names to forward parameters
        modality_map = {
            'text': 'text',
            'image': 'images',
            'audio': 'audio',
        }

        # Encode query
        kwargs = {modality_map[query_modality]: query_data}
        outputs = self.forward(**kwargs, decode_to=[target_modality])
        
        return outputs[f'decoded_{target_modality}']

