"""Extract and display Flickr8k SGD training results."""

import json
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / 'folder_per_model' / 'multimodal_memory' / 'outputs'
output_dir = OUTPUT_ROOT / 'flickr8k_sgd'

print("=" * 80)
print("FLICKR8K TRAINING WITH SGD - RESULTS SUMMARY")
print("=" * 80)
print()

# Load history
history_file = output_dir / 'history_flickr8k.json'
if history_file.exists():
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    print(f"Training completed: {len(history)} epochs")
    print()

    # Best validation
    best = max(history, key=lambda x: x['val']['avg_r1'])
    print("Best Validation Performance:")
    print(f"  Epoch: {best['epoch']}")
    print(f"  Avg R@1: {best['val']['avg_r1']:.2f}%")
    print(f"  I→T: R@1={best['val']['i2t_r1']:.2f}%, R@5={best['val']['i2t_r5']:.2f}%, R@10={best['val']['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={best['val']['t2i_r1']:.2f}%, R@5={best['val']['t2i_r5']:.2f}%, R@10={best['val']['t2i_r10']:.2f}%")
    print()

    # Final performance
    final = history[-1]
    print(f"Final Performance (Epoch {final['epoch']}):")
    print(f"  Train Loss: {final['train']['loss']:.4f}")
    print(f"  Avg R@1: {final['val']['avg_r1']:.2f}%")
    print(f"  I→T: R@1={final['val']['i2t_r1']:.2f}%, R@5={final['val']['i2t_r5']:.2f}%, R@10={final['val']['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={final['val']['t2i_r1']:.2f}%, R@5={final['val']['t2i_r5']:.2f}%, R@10={final['val']['t2i_r10']:.2f}%")
    print()

    # Training progress
    print("Training Progress:")
    epochs_to_show = [1, 5, 10, 15, 20, 25, 30]
    for ep in epochs_to_show:
        if ep <= len(history):
            h = history[ep - 1]
            print(f"  Epoch {ep:2d}: Loss={h['train']['loss']:.4f}, Avg R@1={h['val']['avg_r1']:.2f}%")
    print()

    # Loss convergence
    first_loss = history[0]['train']['loss']
    last_loss = history[-1]['train']['loss']
    improvement = (first_loss - last_loss) / first_loss * 100
    print(f"Loss Improvement: {first_loss:.4f} → {last_loss:.4f} ({improvement:.1f}% reduction)")
    print()

    # R@1 improvement
    first_r1 = history[0]['val']['avg_r1']
    best_r1 = best['val']['avg_r1']
    r1_gain = best_r1 / first_r1 if first_r1 > 0 else 0
    print(f"R@1 Improvement: {first_r1:.2f}% → {best_r1:.2f}% ({r1_gain:.1f}x gain)")

else:
    print(f"History file not found at {history_file}!")

print()
print("=" * 80)
print("MODEL CHECKPOINT")
print("=" * 80)

checkpoint_file = output_dir / 'best_model_flickr8k.pt'
if checkpoint_file.exists():
    checkpoint = torch.load(checkpoint_file, weights_only=False)
    print(f"  Saved at epoch: {checkpoint['epoch']}")
    print(f"  Model size: {checkpoint_file.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Location: {checkpoint_file}")
else:
    print(f"  Checkpoint not found at {checkpoint_file}!")

print()
print("=" * 80)
print("DATASET INFO")
print("=" * 80)
print("  Dataset: Flickr8k (Image-Text only)")
print("  Train: 30,000 image-text pairs (6,000 images × 5 captions)")
print("  Val: 5,000 pairs (1,000 images × 5 captions)")
print("  Test: 5,000 pairs (1,000 images × 5 captions)")
print()
print("=" * 80)
print("MODEL CONFIG")
print("=" * 80)
print("  Architecture: MultiModalMemoryNetwork")
print("  Latent dim: 512")
print("  Memory size: 128 slots")
print("  Num heads: 8")
print("  Num layers: 4")
print("  Total params: ~45M")
print()
print("=" * 80)
print("OPTIMIZER: SGD (WINNER FROM TESTING)")
print("=" * 80)
print("  Learning rate: 0.01")
print("  Momentum: 0.9")
print("  Nesterov: True")
print("  Scheduler: CosineAnnealing")
print("  Batch size: 64")
print("  Epochs: 30")
print()
print("=" * 80)
