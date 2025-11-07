"""Training script for Multimodal Memory Network on synthetic dataset.

Creates a small multimodal dataset with text, images, and audio pairs,
then trains the network with various loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.architectures import MultiModalMemoryNetwork
from src.optimizers import UniversalAndersonGDA
from src.utils import get_device


class SyntheticMultimodalDataset(Dataset):
    """Synthetic multimodal dataset with text, images, and audio.
    
    Generates paired data where all modalities share semantic information:
    - Text: Token sequences representing concepts (0-9 classes)
    - Images: 64x64 colored patterns corresponding to concepts
    - Audio: Simple waveform spectrograms corresponding to concepts
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of semantic classes
        seq_len: Text sequence length
        vocab_size: Vocabulary size for text
        image_size: Size of generated images
        audio_size: Size of audio spectrograms
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        seq_len: int = 16,
        vocab_size: int = 1000,
        image_size: int = 64,
        audio_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.audio_size = audio_size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self.labels = np.random.randint(0, num_classes, num_samples)
        
        # Pre-generate prototypes for each class
        self.text_prototypes = self._generate_text_prototypes()
        self.image_prototypes = self._generate_image_prototypes()
        self.audio_prototypes = self._generate_audio_prototypes()
        
    def _generate_text_prototypes(self):
        """Generate text prototypes for each class."""
        prototypes = []
        for c in range(self.num_classes):
            # Each class has a characteristic token pattern
            base_tokens = np.arange(c * 10, c * 10 + 10) % self.vocab_size
            prototype = np.random.choice(base_tokens, self.seq_len)
            prototypes.append(prototype)
        return np.array(prototypes)
    
    def _generate_image_prototypes(self):
        """Generate image prototypes for each class."""
        prototypes = []
        for c in range(self.num_classes):
            # Each class has a characteristic color and pattern
            img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            
            # Color channel based on class
            channel = c % 3
            img[channel] = 0.8
            
            # Add geometric pattern
            x = np.linspace(0, 2 * np.pi * (c + 1), self.image_size)
            y = np.linspace(0, 2 * np.pi * (c + 1), self.image_size)
            xx, yy = np.meshgrid(x, y)
            pattern = np.sin(xx) * np.cos(yy)
            
            for ch in range(3):
                img[ch] += pattern * 0.3
            
            img = np.clip(img, 0, 1)
            prototypes.append(img)
        
        return np.array(prototypes)
    
    def _generate_audio_prototypes(self):
        """Generate audio spectrogram prototypes for each class."""
        prototypes = []
        for c in range(self.num_classes):
            # Each class has a characteristic frequency pattern
            audio = np.zeros((1, self.audio_size, self.audio_size), dtype=np.float32)
            
            # Create frequency bands
            freq = (c + 1) * 3
            for i in range(self.audio_size):
                for j in range(self.audio_size):
                    audio[0, i, j] = 0.5 * (np.sin(i * freq / 10) + np.cos(j * freq / 15))
            
            audio = (audio - audio.min()) / (audio.max() - audio.min() + 1e-8)
            prototypes.append(audio)
        
        return np.array(prototypes)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Get prototypes and add noise
        text = self.text_prototypes[label].copy()
        # Add some random variations
        mask = np.random.rand(self.seq_len) < 0.2
        text[mask] = np.random.randint(0, self.vocab_size, mask.sum())
        
        image = self.image_prototypes[label].copy()
        image += np.random.randn(3, self.image_size, self.image_size) * 0.1
        image = np.clip(image, 0, 1)
        
        audio = self.audio_prototypes[label].copy()
        audio += np.random.randn(1, self.audio_size, self.audio_size) * 0.1
        audio = np.clip(audio, 0, 1)
        
        return {
            'text': torch.from_numpy(text).long(),
            'image': torch.from_numpy(image).float(),
            'audio': torch.from_numpy(audio).float(),
            'label': torch.tensor(label).long(),
        }


def contrastive_loss(emb1, emb2, temperature=0.07):
    """InfoNCE / CLIP-style contrastive loss."""
    # Normalize embeddings
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(emb1, emb2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(emb1), device=emb1.device)
    
    # Symmetric loss
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    
    return (loss1 + loss2) / 2


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_recon = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        text = batch['text'].to(device)
        images = batch['image'].to(device)
        audio = batch['audio'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with all modalities
        outputs_all = model(text=text, images=images, audio=audio)
        central_latent = outputs_all['central_latent']
        
        # Get individual modality embeddings for contrastive loss
        text_outputs = model(text=text)
        image_outputs = model(images=images)
        audio_outputs = model(audio=audio)
        
        text_emb = text_outputs['central_latent']
        image_emb = image_outputs['central_latent']
        audio_emb = audio_outputs['central_latent']
        
        # Contrastive losses (align modalities)
        loss_text_image = contrastive_loss(text_emb, image_emb)
        loss_text_audio = contrastive_loss(text_emb, audio_emb)
        loss_image_audio = contrastive_loss(image_emb, audio_emb)
        
        contrastive_total = loss_text_image + loss_text_audio + loss_image_audio
        
        # Reconstruction loss (cross-modal prediction)
        outputs_recon = model(text=text, images=images, decode_to=['audio'])
        
        # Target: actual audio embedding
        with torch.no_grad():
            target_audio_emb = model.audio_encoder(audio)
        
        recon_loss = F.mse_loss(outputs_recon['decoded_audio'], target_audio_emb)
        
        # Total loss
        loss = contrastive_total + 0.5 * recon_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_contrastive += contrastive_total.item()
        total_recon += recon_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'contrastive': f'{contrastive_total.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'contrastive': total_contrastive / len(dataloader),
        'recon': total_recon / len(dataloader),
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate cross-modal retrieval accuracy."""
    model.eval()
    
    all_text_emb = []
    all_image_emb = []
    all_audio_emb = []
    all_labels = []
    
    for batch in dataloader:
        text = batch['text'].to(device)
        images = batch['image'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Get embeddings
        text_emb = model(text=text)['central_latent']
        image_emb = model(images=images)['central_latent']
        audio_emb = model(audio=audio)['central_latent']
        
        all_text_emb.append(text_emb)
        all_image_emb.append(image_emb)
        all_audio_emb.append(audio_emb)
        all_labels.append(labels)
    
    all_text_emb = torch.cat(all_text_emb)
    all_image_emb = torch.cat(all_image_emb)
    all_audio_emb = torch.cat(all_audio_emb)
    all_labels = torch.cat(all_labels)
    
    # Normalize for similarity computation
    all_text_emb = F.normalize(all_text_emb, dim=-1)
    all_image_emb = F.normalize(all_image_emb, dim=-1)
    all_audio_emb = F.normalize(all_audio_emb, dim=-1)
    
    # Text -> Image retrieval accuracy (R@1)
    sim_text_image = torch.matmul(all_text_emb, all_image_emb.T)
    text_to_image_preds = sim_text_image.argmax(dim=1)
    text_to_image_acc = (text_to_image_preds == torch.arange(len(all_labels), device=device)).float().mean()
    
    # Image -> Text retrieval
    sim_image_text = torch.matmul(all_image_emb, all_text_emb.T)
    image_to_text_preds = sim_image_text.argmax(dim=1)
    image_to_text_acc = (image_to_text_preds == torch.arange(len(all_labels), device=device)).float().mean()
    
    # Text -> Audio retrieval
    sim_text_audio = torch.matmul(all_text_emb, all_audio_emb.T)
    text_to_audio_preds = sim_text_audio.argmax(dim=1)
    text_to_audio_acc = (text_to_audio_preds == torch.arange(len(all_labels), device=device)).float().mean()
    
    return {
        'text_to_image': text_to_image_acc.item() * 100,
        'image_to_text': image_to_text_acc.item() * 100,
        'text_to_audio': text_to_audio_acc.item() * 100,
    }


def plot_training_curves(history, save_path='multimodal_training.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Contrastive loss
    axes[0, 1].plot(history['train_contrastive'], label='Contrastive Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Contrastive Loss (Cross-Modal Alignment)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Retrieval accuracy
    axes[1, 0].plot(history['text_to_image_acc'], label='Text→Image', marker='o')
    axes[1, 0].plot(history['image_to_text_acc'], label='Image→Text', marker='s')
    axes[1, 0].plot(history['text_to_audio_acc'], label='Text→Audio', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Retrieval Accuracy (%)')
    axes[1, 0].set_title('Cross-Modal Retrieval Accuracy (R@1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Reconstruction loss
    axes[1, 1].plot(history['train_recon'], label='Reconstruction Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Cross-Modal Reconstruction Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def main():
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-2  # Higher LR for SGD

    print("=" * 80)
    print("MULTIMODAL MEMORY NETWORK TRAINING")
    print("=" * 80)
    
    # Create dataset
    print("\nCreating synthetic multimodal dataset...")
    train_dataset = SyntheticMultimodalDataset(
        num_samples=2000,
        num_classes=10,
        seq_len=16,
        vocab_size=1000,
        image_size=64,
        audio_size=64,
    )
    
    val_dataset = SyntheticMultimodalDataset(
        num_samples=500,
        num_classes=10,
        seq_len=16,
        vocab_size=1000,
        image_size=64,
        audio_size=64,
        seed=123,  # Different seed for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create model
    print("\nCreating model...")
    model = MultiModalMemoryNetwork(
        vocab_size=1000,
        text_embed_dim=256,
        text_seq_len=32,
        image_size=64,
        patch_size=8,
        image_channels=3,
        audio_channels=1,
        latent_dim=256,
        memory_size=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        ttt_mode="attention",
        feedback_steps=2,
    ).to(device)
    
    model.print_model_info()
    
    # Create optimizer - SGD works best for multimodal network based on testing
    print(f"\nUsing SGD optimizer (lr={learning_rate}, momentum=0.9)")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    history = {
        'train_loss': [],
        'train_contrastive': [],
        'train_recon': [],
        'text_to_image_acc': [],
        'image_to_text_acc': [],
        'text_to_audio_acc': [],
    }
    
    best_retrieval_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_contrastive'].append(train_metrics['contrastive'])
        history['train_recon'].append(train_metrics['recon'])
        history['text_to_image_acc'].append(val_metrics['text_to_image'])
        history['image_to_text_acc'].append(val_metrics['image_to_text'])
        history['text_to_audio_acc'].append(val_metrics['text_to_audio'])
        
        # Print summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Contrastive: {train_metrics['contrastive']:.4f}")
        print(f"  Reconstruction: {train_metrics['recon']:.4f}")
        print(f"  Text→Image R@1: {val_metrics['text_to_image']:.2f}%")
        print(f"  Image→Text R@1: {val_metrics['image_to_text']:.2f}%")
        print(f"  Text→Audio R@1: {val_metrics['text_to_audio']:.2f}%")
        
        # Save best model
        avg_retrieval = (val_metrics['text_to_image'] + val_metrics['image_to_text'] + val_metrics['text_to_audio']) / 3
        if avg_retrieval > best_retrieval_acc:
            best_retrieval_acc = avg_retrieval
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, 'best_multimodal_model.pt')
            print(f"  ✓ Saved best model (avg retrieval: {avg_retrieval:.2f}%)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest average retrieval accuracy: {best_retrieval_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(history)
    
    # Test memory adaptation
    print("\n" + "=" * 80)
    print("TEST-TIME MEMORY ADAPTATION")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(val_loader))
        text = batch['text'].to(device)
        
        # Check memory before
        initial_memory_norm = model.central_memory.memory.norm().item()
        print(f"\nInitial central memory norm: {initial_memory_norm:.4f}")
        
        # Process batch (memory updates via TTT)
        _ = model(text=text)
        
        # Check memory after
        updated_memory_norm = model.central_memory.memory.norm().item()
        print(f"Updated central memory norm: {updated_memory_norm:.4f}")
        print(f"Memory change: {abs(updated_memory_norm - initial_memory_norm):.4f}")
        print("✓ Test-time training is active")
    
    print("\n" + "=" * 80)
    print("All training complete! Model saved to: best_multimodal_model.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()

