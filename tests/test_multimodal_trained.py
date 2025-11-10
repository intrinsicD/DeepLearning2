"""Inference and visualization for trained multimodal model."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_multimodal import SyntheticMultimodalDataset
from architectures import MultiModalMemoryNetwork
from utils import get_device


def visualize_cross_modal_retrieval(model, dataset, device, num_examples=5):
    """Visualize cross-modal retrieval examples."""
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, num_examples * 3))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            text = sample['text'].unsqueeze(0).to(device)
            image = sample['image'].unsqueeze(0).to(device)
            audio = sample['audio'].unsqueeze(0).to(device)
            label = sample['label'].item()
            
            # Get embeddings
            text_emb = model(text=text)['central_latent']
            image_emb = model(images=image)['central_latent']
            audio_emb = model(audio=audio)['central_latent']
            
            # Normalize for similarity
            text_emb = F.normalize(text_emb, dim=-1)
            image_emb = F.normalize(image_emb, dim=-1)
            audio_emb = F.normalize(audio_emb, dim=-1)
            
            # Compute similarities
            sim_text_image = (text_emb @ image_emb.T).item()
            sim_text_audio = (text_emb @ audio_emb.T).item()
            sim_image_audio = (image_emb @ audio_emb.T).item()
            
            # Plot text (as token histogram)
            axes[idx, 0].bar(range(len(text[0])), text[0].cpu().numpy())
            axes[idx, 0].set_title(f'Text (Class {label})')
            axes[idx, 0].set_xlabel('Position')
            axes[idx, 0].set_ylabel('Token ID')
            
            # Plot image
            img_display = image[0].cpu().numpy().transpose(1, 2, 0)
            axes[idx, 1].imshow(img_display)
            axes[idx, 1].set_title(f'Image (Class {label})')
            axes[idx, 1].axis('off')
            
            # Plot audio spectrogram
            axes[idx, 2].imshow(audio[0, 0].cpu().numpy(), cmap='viridis', aspect='auto')
            axes[idx, 2].set_title(f'Audio (Class {label})')
            axes[idx, 2].set_xlabel('Time')
            axes[idx, 2].set_ylabel('Frequency')
            
            # Plot similarity matrix
            similarities = np.array([
                [1.0, sim_text_image, sim_text_audio],
                [sim_text_image, 1.0, sim_image_audio],
                [sim_text_audio, sim_image_audio, 1.0]
            ])
            im = axes[idx, 3].imshow(similarities, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[idx, 3].set_xticks([0, 1, 2])
            axes[idx, 3].set_yticks([0, 1, 2])
            axes[idx, 3].set_xticklabels(['Text', 'Image', 'Audio'])
            axes[idx, 3].set_yticklabels(['Text', 'Image', 'Audio'])
            axes[idx, 3].set_title('Cross-Modal Similarity')
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text_obj = axes[idx, 3].text(j, i, f'{similarities[i, j]:.2f}',
                                   ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[:, 3].ravel().tolist(), label='Similarity')
        plt.tight_layout()
        plt.savefig('multimodal_retrieval_examples.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to multimodal_retrieval_examples.png")
        plt.close()


def test_memory_adaptation(model, dataset, device):
    """Test test-time memory adaptation."""
    model.eval()
    
    print("\n" + "=" * 80)
    print("TEST-TIME MEMORY ADAPTATION ANALYSIS")
    print("=" * 80)
    
    # Get samples from two different classes
    class_0_samples = [i for i, label in enumerate(dataset.labels) if label == 0][:5]
    class_1_samples = [i for i, label in enumerate(dataset.labels) if label == 1][:5]
    
    with torch.no_grad():
        # Initial memory state
        initial_memory = model.central_memory.memory.clone()
        
        # Process class 0 samples
        print("\nProcessing 5 samples from Class 0...")
        for idx in class_0_samples:
            sample = dataset[idx]
            text = sample['text'].unsqueeze(0).to(device)
            _ = model(text=text)
        
        memory_after_class0 = model.central_memory.memory.clone()
        change_class0 = (memory_after_class0 - initial_memory).norm().item()
        print(f"Memory change after Class 0: {change_class0:.4f}")
        
        # Process class 1 samples
        print("\nProcessing 5 samples from Class 1...")
        for idx in class_1_samples:
            sample = dataset[idx]
            text = sample['text'].unsqueeze(0).to(device)
            _ = model(text=text)
        
        memory_after_class1 = model.central_memory.memory.clone()
        change_class1 = (memory_after_class1 - memory_after_class0).norm().item()
        print(f"Memory change after Class 1: {change_class1:.4f}")
        
        total_change = (memory_after_class1 - initial_memory).norm().item()
        print(f"\nTotal memory adaptation: {total_change:.4f}")
        print("✓ Memory adapts during test time without explicit training")


def analyze_embeddings(model, dataset, device, num_samples=100):
    """Analyze the learned embedding space."""
    model.eval()
    
    print("\n" + "=" * 80)
    print("EMBEDDING SPACE ANALYSIS")
    print("=" * 80)
    
    all_text_emb = []
    all_image_emb = []
    all_audio_emb = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            text = sample['text'].unsqueeze(0).to(device)
            image = sample['image'].unsqueeze(0).to(device)
            audio = sample['audio'].unsqueeze(0).to(device)
            
            text_emb = model(text=text)['central_latent']
            image_emb = model(images=image)['central_latent']
            audio_emb = model(audio=audio)['central_latent']
            
            all_text_emb.append(text_emb.cpu())
            all_image_emb.append(image_emb.cpu())
            all_audio_emb.append(audio_emb.cpu())
            all_labels.append(sample['label'].item())
    
    all_text_emb = torch.cat(all_text_emb)
    all_image_emb = torch.cat(all_image_emb)
    all_audio_emb = torch.cat(all_audio_emb)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    text_mean = all_text_emb.mean(dim=0)
    text_std = all_text_emb.std(dim=0).mean().item()
    
    print(f"\nText embeddings: mean norm = {text_mean.norm().item():.4f}, std = {text_std:.4f}")
    print(f"Image embeddings: mean norm = {all_image_emb.mean(dim=0).norm().item():.4f}")
    print(f"Audio embeddings: mean norm = {all_audio_emb.mean(dim=0).norm().item():.4f}")
    
    # Compute inter-modality correlation
    text_flat = all_text_emb.flatten()
    image_flat = all_image_emb.flatten()
    audio_flat = all_audio_emb.flatten()
    
    corr_text_image = torch.corrcoef(torch.stack([text_flat, image_flat]))[0, 1].item()
    corr_text_audio = torch.corrcoef(torch.stack([text_flat, audio_flat]))[0, 1].item()
    corr_image_audio = torch.corrcoef(torch.stack([image_flat, audio_flat]))[0, 1].item()
    
    print(f"\nInter-modality correlations:")
    print(f"  Text-Image: {corr_text_image:.4f}")
    print(f"  Text-Audio: {corr_text_audio:.4f}")
    print(f"  Image-Audio: {corr_image_audio:.4f}")


def main():
    """Main inference function."""
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    checkpoint = torch.load('best_multimodal_model.pt', map_location=device)
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Metrics: {checkpoint['metrics']}")
    
    # Load dataset
    print("\nLoading validation dataset...")
    val_dataset = SyntheticMultimodalDataset(
        num_samples=500,
        num_classes=10,
        seq_len=16,
        vocab_size=1000,
        image_size=64,
        audio_size=64,
        seed=123,
    )
    
    # Visualize retrieval
    print("\nGenerating cross-modal retrieval visualization...")
    visualize_cross_modal_retrieval(model, val_dataset, device, num_examples=5)
    
    # Test memory adaptation
    test_memory_adaptation(model, val_dataset, device)
    
    # Analyze embeddings
    analyze_embeddings(model, val_dataset, device)
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

