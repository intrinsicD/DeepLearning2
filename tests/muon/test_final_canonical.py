"""Test final canonical Muon with all fixes: correct spectral scaling, quintic, etc."""

import torch
import torch.nn as nn
from architectures import SimpleCNN
from utils import get_mnist_loaders, get_device, Trainer
from optimizers import MuonFast

def exclude_head_and_small(name: str, param: torch.Tensor) -> bool:
    """Exclude classifier head and biases (Keras guidance).
    
    Detects head by: output_dim <= 10 (num_classes)
    Also excludes {0,1}-D parameters (biases, norms)
    """
    if param.ndim < 2:
        return True  # Biases, layer norms, etc.
    
    if param.ndim == 2:
        # Detect classifier head by small output dimension
        if param.shape[0] <= 10:  # Likely a classifier
            return True
    
    return False

def test_final_canonical():
    """Test Muon with all canonical features enabled."""
    
    device = get_device()
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    print("=" * 70)
    print("Final Canonical Muon Test")
    print("=" * 70)
    
    configs = [
        {
            "name": "MuonFast (Full Canonical)",
            "kwargs": {
                "lr": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "exclude_fn": exclude_head_and_small,
                "scale_mode": "spectral",
                "scale_extra": 1.0,
                "ns_coefficients": "simple",  # Quintic is too aggressive
                "ns_tol": 1e-3,
                "verbose": True,
            }
        },
        {
            "name": "Adam (Baseline)",
            "optimizer_class": torch.optim.Adam,
            "kwargs": {
                "lr": 0.001,
            }
        },
    ]
    
    results = {}
    
    for config in configs:
        print("\n" + "=" * 70)
        print(f"Testing: {config['name']}")
        print("=" * 70)
        
        model = SimpleCNN(input_channels=1, num_classes=10).to(device)
        
        if 'optimizer_class' in config:
            optimizer = config['optimizer_class'](model.parameters(), **config['kwargs'])
        else:
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
            'final_train_acc': history['train_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
        }
        
        print(f"\n{config['name']} Results:")
        print(f"  Train: {history['train_acc'][-1]:.2f}% acc, {history['train_loss'][-1]:.4f} loss")
        print(f"  Test:  {test_acc:.2f}% acc, {test_loss:.4f} loss")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    for name, res in results.items():
        print(f"{name:40s}: {res['test_acc']:6.2f}% acc, {res['test_loss']:.4f} loss")
    
    muon_acc = results['MuonFast (Full Canonical)']['test_acc']
    adam_acc = results['Adam (Baseline)']['test_acc']
    
    print("\n" + "=" * 70)
    print(f"MuonFast: {muon_acc:.2f}%")
    print(f"Adam:     {adam_acc:.2f}%")
    print(f"Delta:    {muon_acc - adam_acc:+.2f}%")
    
    if muon_acc >= 99.0:
        print("\n✓ SUCCESS: MuonFast achieves ≥99% accuracy!")
    elif muon_acc >= adam_acc - 0.3:
        print("\n✓ COMPETITIVE: Within 0.3% of Adam (expected for conv-heavy CNN)")
    else:
        print(f"\n⚠️  Below target ({muon_acc:.2f}% < {adam_acc - 0.3:.2f}%)")
    
    print("=" * 70)

if __name__ == "__main__":
    test_final_canonical()

