"""
Example experiment: MNIST classification with different architectures and optimizers.
"""

import torch
import torch.nn as nn
from src.architectures import SimpleCNN, ResNet, FullyConnectedNet
from src.optimizers import CustomSGD, CustomAdam
from src.utils import get_device, print_gpu_info, Trainer, get_mnist_loaders


def run_experiment(architecture='cnn', optimizer_type='adam', epochs=5):
    """
    Run MNIST classification experiment.
    
    Args:
        architecture (str): Architecture type ('cnn', 'resnet', 'fc')
        optimizer_type (str): Optimizer type ('sgd', 'adam', 'custom_sgd', 'custom_adam')
        epochs (int): Number of training epochs
    """
    # Get device
    device = get_device()
    print_gpu_info()
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {architecture} architecture...")
    if architecture == 'cnn':
        model = SimpleCNN(input_channels=1, num_classes=10)
    elif architecture == 'resnet':
        model = ResNet(input_channels=1, num_classes=10, num_blocks=[2, 2, 2])
    elif architecture == 'fc':
        model = FullyConnectedNet(input_size=28*28, hidden_sizes=[256, 128], num_classes=10)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model.print_model_info()
    
    # Create optimizer
    print(f"\nCreating {optimizer_type} optimizer...")
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'custom_sgd':
        optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == 'custom_adam':
        optimizer = CustomAdam(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Create trainer
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        epochs=epochs,
        val_loader=val_loader
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    
    return model, trainer, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MNIST Classification Experiment')
    parser.add_argument('--arch', type=str, default='cnn', 
                       choices=['cnn', 'resnet', 'fc'],
                       help='Neural network architecture')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'custom_sgd', 'custom_adam'],
                       help='Optimizer type')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MNIST Classification Experiment")
    print("=" * 60)
    print(f"Architecture: {args.arch}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    model, trainer, history = run_experiment(
        architecture=args.arch,
        optimizer_type=args.optimizer,
        epochs=args.epochs
    )
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)
