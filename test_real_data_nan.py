"""
Test with real Flickr8k data to see if NaN occurs.
"""

import torch
from torch.utils.data import DataLoader
from nl_mm.models.nl_mm_model import NLMM
from nl_mm.utils import load_config
from nl_mm.init import apply_nlmm_init
from src.utils.flickr8k_dataset import Flickr8kAudioDataset, collate_fn
from src.utils import get_device

def test_real_data():
    """Test with real data."""
    print("="*60)
    print("Testing NL-MM with Real Flickr8k Data")
    print("="*60)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Load config
    cfg = load_config('nl_mm/configs/nano_8gb.yaml')
    
    # Create model
    print("\nCreating model...")
    model = NLMM(cfg).to(device)
    apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))
    
    # Load data
    print("\nLoading Flickr8k dataset...")
    dataset = Flickr8kAudioDataset(
        root_dir='./flickr8k',
        split='train',
        image_size=224,
        audio_sample_rate=16000,
        n_mels=80,
        text_max_len=77,
    )
    
    # Use very small batch
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first batch
    print("\nTesting first batch...")
    batch = next(iter(loader))
    
    print(f"  Text shape: {batch['text'].shape}")
    print(f"  Image shape: {batch['images'].shape}")
    print(f"  Audio shape: {batch['audio'].shape}")
    print(f"  Text range: [{batch['text'].min()}, {batch['text'].max()}]")
    
    # Check for vocab overflow
    vocab_size = cfg.get('vocab_size', 32000)
    overflow = (batch['text'] >= vocab_size).sum().item()
    if overflow > 0:
        print(f"  ❌ WARNING: {overflow} tokens >= vocab_size ({vocab_size})")
        print(f"     Max token ID: {batch['text'].max().item()}")
    
    # Prepare batch
    audio = batch['audio'].to(device)
    B, C, n_mels, t = audio.shape
    audio_flat = audio.reshape(B, C, n_mels * t)
    
    nl_batch = {
        "text": batch['text'].to(device),
        "image": batch['images'].to(device),
        "audio": audio_flat,
        "text_target": batch['text'].to(device),
    }
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    
    with torch.no_grad():
        outputs, state = model(nl_batch, return_embeddings=True)
    
    print("\nOutput losses:")
    if "text" in outputs:
        text_loss = outputs["text"]
        if torch.isnan(text_loss):
            print(f"  ❌ Text loss is NaN!")
        else:
            print(f"  ✓ Text loss: {text_loss.item():.6f}")
    
    if "embeddings" in outputs:
        print("\nEmbeddings:")
        for key, val in outputs["embeddings"].items():
            has_nan = torch.isnan(val).any().item()
            has_inf = torch.isinf(val).any().item()
            print(f"  {key}: NaN={has_nan}, Inf={has_inf}, norm={val.norm().item():.4f}")
    
    # Test multiple batches
    print("\nTesting 10 batches...")
    model.train()
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        
        audio = batch['audio'].to(device)
        B, C, n_mels, t = audio.shape
        audio_flat = audio.reshape(B, C, n_mels * t)
        
        nl_batch = {
            "text": batch['text'].to(device),
            "image": batch['images'].to(device),
            "audio": audio_flat,
            "text_target": batch['text'].to(device),
        }
        
        outputs, state = model(nl_batch, return_embeddings=True)
        
        text_loss = outputs.get("text", torch.tensor(0.0, device=device))
        
        if torch.isnan(text_loss):
            print(f"  Batch {i}: ❌ NaN!")
            break
        else:
            print(f"  Batch {i}: loss={text_loss.item():.4f}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_real_data()

