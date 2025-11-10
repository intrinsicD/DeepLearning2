"""
Debug script to test nl_mm model forward pass and identify NaN sources.
"""

import torch
import torch.nn.functional as F
from modules.nl_mm.models.nl_mm_model import NLMM
from modules.nl_mm.utils import load_config
from modules.nl_mm.init import apply_nlmm_init

def check_for_nan(tensor, name):
    """Check tensor for NaN/Inf."""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
        print(f"   Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
        return True
    return False

def debug_model():
    """Debug model forward pass."""
    print("="*60)
    print("NL-MM Model Debug - Looking for NaN sources")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load config
    cfg = load_config('modules/nl_mm/configs/nano_8gb.yaml')
    print(f"\nConfig loaded:")
    print(f"  d_model: {cfg['d_model']}")
    print(f"  vocab_size: {cfg.get('vocab_size', 32000)}")
    print(f"  depth: {cfg['depth']}")
    
    # Create model
    print("\nCreating model...")
    model = NLMM(cfg).to(device)
    apply_nlmm_init(model, cfg.get("depth", {}), cfg.get("arch_kind", "encoder"))
    
    # Check model parameters
    print("\nChecking model parameters...")
    has_nan = False
    for name, param in model.named_parameters():
        if check_for_nan(param, f"Param: {name}"):
            has_nan = True
            break
    
    if has_nan:
        print("\n❌ Model has NaN in parameters after initialization!")
        return
    
    print("✓ Model parameters OK")
    
    # Create test batch
    batch_size = 2
    seq_len = 16
    vocab_size = cfg.get('vocab_size', 32000)
    
    print(f"\nCreating test batch (batch_size={batch_size}, seq_len={seq_len})...")
    
    # Make sure token IDs are within vocab range
    text_tokens = torch.randint(1, min(1000, vocab_size), (batch_size, seq_len), device=device)
    images = torch.randn(batch_size, 3, 64, 64, device=device) * 0.1  # Small values
    audio = torch.randn(batch_size, 1, 8000, device=device) * 0.1  # Small values
    
    nl_batch = {
        "text": text_tokens,
        "image": images,
        "audio": audio,
        "text_target": text_tokens,
    }
    
    print(f"  Text tokens range: [{text_tokens.min().item()}, {text_tokens.max().item()}]")
    print(f"  Image range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    
    try:
        with torch.no_grad():
            outputs, state = model(nl_batch, return_embeddings=True)
        
        print("\n✓ Forward pass completed")
        
        # Check outputs
        print("\nChecking outputs...")
        for key, value in outputs.items():
            if key == "embeddings":
                print(f"\nEmbeddings:")
                for emb_key, emb_val in value.items():
                    check_for_nan(emb_val, f"  {emb_key}")
            else:
                if isinstance(value, torch.Tensor):
                    if check_for_nan(value, f"Output: {key}"):
                        print(f"\n❌ NaN found in {key} output!")
                        
                        # Try to debug text decoder specifically
                        if key == "text":
                            print("\nDebugging text decoder...")
                            print(f"  Text target shape: {nl_batch['text_target'].shape}")
                            print(f"  Text target range: [{nl_batch['text_target'].min()}, {nl_batch['text_target'].max()}]")
                            print(f"  Vocab size: {vocab_size}")
                            
                            # Check if any target tokens are >= vocab_size
                            invalid_tokens = (nl_batch['text_target'] >= vocab_size).sum().item()
                            if invalid_tokens > 0:
                                print(f"  ❌ Found {invalid_tokens} tokens >= vocab_size!")
                    else:
                        print(f"  {key}: {value.item():.6f}")
        
        # Test contrastive loss
        if "embeddings" in outputs:
            print("\nTesting contrastive loss...")
            embs = outputs["embeddings"]
            
            if "image" in embs and "text" in embs:
                query = embs["image"]
                key = embs["text"]
                
                check_for_nan(query, "Image embeddings")
                check_for_nan(key, "Text embeddings")
                
                # Normalize
                query_norm = F.normalize(query, dim=-1, eps=1e-8)
                key_norm = F.normalize(key, dim=-1, eps=1e-8)
                
                check_for_nan(query_norm, "Image embeddings (normalized)")
                check_for_nan(key_norm, "Text embeddings (normalized)")
                
                # Compute logits
                logits = torch.matmul(query_norm, key_norm.T) / 0.07
                check_for_nan(logits, "Contrastive logits")
                
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
                # Compute loss
                labels = torch.arange(len(query), device=device)
                loss = F.cross_entropy(logits, labels)
                
                if check_for_nan(loss, "Contrastive loss"):
                    print("  ❌ NaN in contrastive loss!")
                else:
                    print(f"  ✓ Contrastive loss: {loss.item():.6f}")
        
        print("\n" + "="*60)
        print("Debug complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()

