"""Test canonical Muon improvements: exclude final layer, update scaling, quintic NS."""

import torch
import torch.nn as nn
from architectures import SimpleCNN
from utils import get_mnist_loaders, get_device, Trainer
from optimizers import MuonFast

def exclude_final_layer(name: str, param: torch.Tensor) -> bool:
    """Exclude final output layer from Muon (Keras guidance).
    
    For SimpleCNN, fc2 is the final layer (128 -> 10).
    We can detect it by shape: output_dim == 10 (num_classes).
    """
    if param.ndim == 2:
        # Check if this looks like a final classification layer
        out_features, in_features = param.shape
        if out_features == 10:  # MNIST has 10 classes
            return True
    return False

def test_canonical_improvements():
    """Test all canonical Muon improvements."""
    
    device = get_device()
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    print("=" * 70)
    print("Testing Canonical Muon Improvements")
    print("=" * 70)
    
    configs = [
        {
            "name": "Baseline (no exclusions, no scaling)",
            "kwargs": {
                "lr": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "scale_mode": None,
                "exclude_fn": None,
            }
        },
        {
            "name": "With spectral scaling (NeMo default)",
            "kwargs": {
                "lr": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "scale_mode": "spectral",
                "scale_extra": 1.0,
                "exclude_fn": None,
            }
        },
        {
            "name": "Exclude final layer (Keras guidance)",
            "kwargs": {
                "lr": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "scale_mode": None,
                "exclude_fn": exclude_final_layer,
            }
        },
        {
            "name": "Full canonical (exclude + spectral)",
            "kwargs": {
                "lr": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "scale_mode": "spectral",
                "scale_extra": 1.0,
                "exclude_fn": exclude_final_layer,
                "ns_tol": 1e-3,  # Size-normalized default
                "verbose": True,
            }
        },
    ]
    
    results = {}
    
    for config in configs:
        print("\n" + "=" * 70)
        print(f"Testing: {config['name']}")
        print("=" * 70)
        
        model = SimpleCNN(input_channels=1, num_classes=10).to(device)
        optimizer = MuonFast(model.parameters(), **config['kwargs'])
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
        history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=5)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100.0 * correct / total
        test_loss = test_loss / len(test_loader)
        
        results[config['name']] = {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_acc': history['train_acc'][-1],
            'train_loss': history['train_loss'][-1],
        }
        
        print(f"\nFinal Results:")
        print(f"  Train: {history['train_acc'][-1]:.2f}% acc, {history['train_loss'][-1]:.4f} loss")
        print(f"  Test:  {test_acc:.2f}% acc, {test_loss:.4f} loss")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Canonical Muon Improvements")
    print("=" * 70)
    print(f"{'Configuration':<45} {'Test Acc':<12} {'Test Loss'}")
    print("-" * 70)
    
    for name, res in results.items():
        print(f"{name:<45} {res['test_acc']:>6.2f}%      {res['test_loss']:>6.4f}")
    
    print("=" * 70)
    
    # Check if improvements help
    baseline_acc = results['Baseline (no exclusions, no scaling)']['test_acc']
    full_canonical_acc = results['Full canonical (exclude + spectral)']['test_acc']

    print("\nImpact Analysis:")
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    print(f"  Full canonical: {full_canonical_acc:.2f}%")
    print(f"  Improvement: {full_canonical_acc - baseline_acc:+.2f}%")
    
    if full_canonical_acc >= 99.0:
        print("\n✓ Full canonical Muon achieves competitive accuracy (≥99%)!")
    else:
        print(f"\n⚠️  Still below 99% (at {full_canonical_acc:.2f}%)")

if __name__ == "__main__":
    test_canonical_improvements()

