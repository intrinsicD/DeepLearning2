"""Show continued training results from Flickr8k."""

import json
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / 'folder_per_model' / 'multimodal_memory' / 'outputs'
output_dir = OUTPUT_ROOT / 'flickr8k_sgd'

print("=" * 80)
print("FLICKR8K CONTINUED TRAINING RESULTS")
print("=" * 80)
print()

# Load continued history
history_file = output_dir / 'history_flickr8k_continued.json'
if history_file.exists():
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    print(f"Total training: {len(history)} epochs")

    # Find where continued training started
    original_epochs = 30
    continued_epochs = len(history) - original_epochs
    print(f"Original training: Epochs 1-{original_epochs}")
    print(f"Continued training: Epochs {original_epochs + 1}-{len(history)} ({continued_epochs} additional)")
    print()

    # Best overall
    best = max(history, key=lambda x: x['val']['avg_r1'])
    print("Best Overall Performance:")
    print(f"  Epoch: {best['epoch']}")
    print(f"  Avg R@1: {best['val']['avg_r1']:.2f}%")
    print(f"  I→T: R@1={best['val']['i2t_r1']:.2f}%, R@5={best['val']['i2t_r5']:.2f}%, R@10={best['val']['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={best['val']['t2i_r1']:.2f}%, R@5={best['val']['t2i_r5']:.2f}%, R@10={best['val']['t2i_r10']:.2f}%")
    print()

    # Compare original vs continued
    original_end = history[original_epochs - 1]
    continued_end = history[-1]

    print("Original Training End (Epoch 30):")
    print(f"  Avg R@1: {original_end['val']['avg_r1']:.2f}%")
    print(f"  Loss: {original_end['train']['loss']:.4f}")
    print()

    print(f"Continued Training End (Epoch {continued_end['epoch']}):")
    print(f"  Avg R@1: {continued_end['val']['avg_r1']:.2f}%")
    print(f"  Loss: {continued_end['train']['loss']:.4f}")
    print()

    improvement = best['val']['avg_r1'] - original_end['val']['avg_r1']
    print(f"Improvement from continued training: {improvement:+.2f}% R@1")
    print()

    # Show training progression
    print("Training Progression:")
    milestones = [1, 10, 20, 30, 40, 50, 60, len(history)]
    for ep in milestones:
        if ep <= len(history):
            h = history[ep - 1]
            marker = " (original)" if ep <= original_epochs else " (continued)"
            print(f"  Epoch {ep:2d}: Loss={h['train']['loss']:.4f}, Avg R@1={h['val']['avg_r1']:.2f}%{marker}")
    print()

    # Loss improvement
    first_loss = history[0]['train']['loss']
    last_loss = history[-1]['train']['loss']
    total_reduction = (first_loss - last_loss) / first_loss * 100
    print(f"Total Loss Reduction: {first_loss:.4f} → {last_loss:.4f} ({total_reduction:.1f}%)")

    # Continued training specific reduction
    continued_start_loss = history[original_epochs]['train']['loss']
    continued_reduction = (continued_start_loss - last_loss) / continued_start_loss * 100
    print(f"Continued Training Loss Reduction: {continued_start_loss:.4f} → {last_loss:.4f} ({continued_reduction:.1f}%)")

print()
print("=" * 80)
print("TEST SET RESULTS")
print("=" * 80)

test_results_file = output_dir / 'test_results_continued.json'
if test_results_file.exists():
    with open(test_results_file, 'r', encoding='utf-8') as f:
        test_results = json.load(f)

    print(f"  I→T: R@1={test_results['i2t_r1']:.2f}%, R@5={test_results['i2t_r5']:.2f}%, R@10={test_results['i2t_r10']:.2f}%")
    print(f"  T→I: R@1={test_results['t2i_r1']:.2f}%, R@5={test_results['t2i_r5']:.2f}%, R@10={test_results['t2i_r10']:.2f}%")
    print(f"  Avg R@1: {test_results['avg_r1']:.2f}%")

print()
print("=" * 80)
print("MODEL CHECKPOINTS")
print("=" * 80)

checkpoints = [
    ('best_model_flickr8k.pt', 'Original best model'),
    ('best_model_flickr8k_continued.pt', 'Continued best model'),
    ('latest_model_flickr8k.pt', 'Latest model'),
]

for filename, description in checkpoints:
    path = output_dir / filename
    if path.exists():
        size = path.stat().st_size / (1024 * 1024)
        checkpoint = torch.load(path, weights_only=False)
        print(f"  {description}:")
        print(f"    File: {filename}")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Avg R@1: {checkpoint['metrics']['avg_r1']:.2f}%")
        print(f"    Size: {size:.1f} MB")
        print()

print("=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print("  Dataset: Flickr8k (8K images, 40K captions)")
print("  Optimizer: SGD (lr=0.01 → 0.005)")
print("  Architecture: MultiModalMemoryNetwork (45M params)")
print("  Original: 30 epochs in 12 minutes")
print("  Continued: 50 epochs in 21 minutes")
print("  Total: 80 epochs in 33 minutes")
print()
print("=" * 80)
