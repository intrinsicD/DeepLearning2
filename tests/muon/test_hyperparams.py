"""Test with lower learning rate and momentum to see if it stabilizes."""

import torch
import torch.nn as nn
from architectures import SimpleCNN
from utils import get_mnist_loaders, get_device, Trainer
from optimizers import MuonFast

def test_different_hyperparams():
    """Test MuonFast with different learning rates and momentum."""
    
    device = get_device()
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    configs = [
        {"lr": 0.001, "momentum": 0.95, "name": "Default (lr=0.001, mom=0.95)"},
        {"lr": 0.0001, "momentum": 0.95, "name": "Lower LR (lr=0.0001, mom=0.95)"},
        {"lr": 0.001, "momentum": 0.0, "name": "No momentum (lr=0.001, mom=0.0)"},
        {"lr": 0.001, "momentum": 0.5, "name": "Lower momentum (lr=0.001, mom=0.5)"},
    ]
    
    for config in configs:
        print("=" * 60)
        print(f"Testing: {config['name']}")
        print("=" * 60)
        
        model = SimpleCNN(input_channels=1, num_classes=10)
        optimizer = MuonFast(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=5)
        
        # Check if it diverged
        final_loss = history['train_loss'][-1]
        if final_loss > 1.0:
            print(f"❌ DIVERGED - Final loss: {final_loss:.4f}")
        else:
            print(f"✓ STABLE - Final loss: {final_loss:.4f}, Final acc: {history['train_acc'][-1]:.2f}%")
        print()

if __name__ == "__main__":
    test_different_hyperparams()

