"""
Quick Start Guide - Simple demonstration of the DeepLearning2 framework.
"""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN, ResNet, FullyConnectedNet
from src.optimizers import CustomSGD, CustomAdam
from src.utils import get_device, print_gpu_info, Trainer


def demo_architectures():
    """Demonstrate different neural network architectures."""
    print("\n" + "=" * 60)
    print("DEMO 1: Neural Network Architectures")
    print("=" * 60)
    
    # Create different architectures
    architectures = {
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'ResNet': ResNet(input_channels=1, num_classes=10, num_blocks=[2, 2, 2]),
        'FullyConnected': FullyConnectedNet(input_size=784, hidden_sizes=[256, 128], num_classes=10)
    }
    
    for name, model in architectures.items():
        print(f"\n{name}:")
        print(f"  Parameters: {model.get_num_parameters():,}")
        
        # Test forward pass
        if name == 'FullyConnected':
            test_input = torch.randn(1, 784)
        else:
            test_input = torch.randn(1, 1, 28, 28)
        
        with torch.no_grad():
            output = model(test_input)
        print(f"  Output shape: {output.shape}")


def demo_optimizers():
    """Demonstrate different optimizers."""
    print("\n" + "=" * 60)
    print("DEMO 2: Custom Optimizers")
    print("=" * 60)
    
    model = SimpleCNN(input_channels=1, num_classes=10)
    
    # Create different optimizers
    optimizers = {
        'PyTorch SGD': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'PyTorch Adam': torch.optim.Adam(model.parameters(), lr=0.001),
        'Custom SGD': CustomSGD(model.parameters(), lr=0.01, momentum=0.9),
        'Custom Adam': CustomAdam(model.parameters(), lr=0.001),
    }
    
    for name, optimizer in optimizers.items():
        print(f"\n{name}:")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Number of parameter groups: {len(optimizer.param_groups)}")


def demo_training():
    """Demonstrate training with dummy data."""
    print("\n" + "=" * 60)
    print("DEMO 3: Training with Custom Components")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create model
    model = SimpleCNN(input_channels=1, num_classes=10)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create optimizer
    optimizer = CustomAdam(model.parameters(), lr=0.001)
    
    # Create trainer
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Create dummy data
    print("\nGenerating dummy data for demonstration...")
    dummy_train_data = []
    for _ in range(10):  # 10 batches
        data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (32,))
        dummy_train_data.append((data, target))
    
    # Simulate a few training steps
    model.train()
    print("\nPerforming 5 training steps...")
    for i, (data, target) in enumerate(dummy_train_data[:5]):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"  Step {i+1}/5 - Loss: {loss.item():.4f}")
    
    print("\nTraining demonstration completed!")


def demo_gpu_info():
    """Demonstrate GPU detection and information."""
    print("\n" + "=" * 60)
    print("DEMO 4: GPU Detection and Information")
    print("=" * 60)
    
    print_gpu_info()
    
    device = get_device()
    print(f"\nSelected device for computation: {device}")
    
    if device.type == 'cuda':
        print("\nYou have CUDA available and can run experiments on GPU!")
        print("This will significantly speed up training.")
    else:
        print("\nCUDA not available. Running on CPU.")
        print("For GPU acceleration, install PyTorch with CUDA support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("DeepLearning2 Framework Quick Start")
    print("=" * 60)
    
    demo_architectures()
    demo_optimizers()
    demo_training()
    demo_gpu_info()
    
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python -m src.experiments.mnist_example --arch cnn --optimizer adam --epochs 5")
    print("2. Run: python -m src.experiments.compare_architectures --epochs 3")
    print("3. Run: python -m src.experiments.compare_optimizers --epochs 5")
    print("4. Create your own custom architectures in src/architectures/")
    print("5. Create your own custom optimizers in src/optimizers/")
    print("=" * 60)


if __name__ == '__main__':
    main()
