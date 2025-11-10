"""Full 5-epoch test of MuonFast to match the benchmark."""

import torch
import torch.nn as nn
from architectures import SimpleCNN
from utils import get_mnist_loaders, get_device, Trainer
from optimizers import MuonFast

def full_muon_test():
    """Run full 5-epoch training like in the benchmark."""
    
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load data
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    print("=" * 60)
    print("Training MuonFast for 5 epochs")
    print("=" * 60)
    
    # Create model
    model = SimpleCNN(input_channels=1, num_classes=10)
    optimizer = MuonFast(model.parameters(), lr=0.001, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
    )
    
    # Evaluate
    test_metrics = trainer.validate(test_loader)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print()
    
    # Print training history
    print("Training history:")
    for epoch in range(5):
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.2f}%")
        print(f"  Val Loss: {history['val_loss'][epoch]:.4f}, Val Acc: {history['val_acc'][epoch]:.2f}%")
    
    # Check if model learned anything
    if test_metrics['accuracy'] > 50:
        print("\n✓ MuonFast trained successfully!")
    else:
        print(f"\n✗ MuonFast failed to learn (only {test_metrics['accuracy']:.2f}% accuracy)")

if __name__ == "__main__":
    full_muon_test()

