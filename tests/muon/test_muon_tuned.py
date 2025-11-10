"""Test MuonFast with properly tuned learning rate."""

import torch
import torch.nn as nn
from architectures import SimpleCNN
from utils import get_mnist_loaders, get_device, Trainer
from optimizers import MuonFast

def test_muon_tuned():
    """Test MuonFast with smaller learning rate suitable for orthogonalized updates."""
    
    device = get_device()
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # Muon typically needs much smaller learning rates (like 0.00001-0.0001)
    # because the orthogonalized updates have controlled magnitude
    for lr in [0.0001, 0.00005, 0.00003, 0.00001]:
        print("=" * 60)
        print(f"Testing MuonFast with LR={lr}")
        print("=" * 60)
        
        model = SimpleCNN(input_channels=1, num_classes=10)
        optimizer = MuonFast(model.parameters(), lr=lr, momentum=0.95)
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
        
        history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=5)
        
        # Check results
        final_loss = history['train_loss'][-1]
        final_acc = history['train_acc'][-1]
        max_loss = max(history['train_loss'])
        
        if max_loss > 2.0:
            print(f"❌ DIVERGED at some point (max loss: {max_loss:.4f})")
        elif final_acc < 80:
            print(f"⚠️  TOO SLOW - Final acc: {final_acc:.2f}%, loss: {final_loss:.4f}")
        else:
            print(f"✓ GOOD - Final acc: {final_acc:.2f}%, loss: {final_loss:.4f}")
        print()

if __name__ == "__main__":
    test_muon_tuned()

