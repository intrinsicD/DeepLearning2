"""
Example experiment: Compare different architectures on MNIST.
"""

from pathlib import Path

import torch.nn as nn
from src.architectures import SimpleCNN, ResNet, FullyConnectedNet
from src.optimizers import CustomAdam
from src.utils import (
    Trainer,
    get_device,
    get_mnist_loaders,
    plot_bar_chart,
    plot_metric_curves,
    print_gpu_info,
)


def compare_architectures(epochs=3):
    """
    Compare different neural network architectures on MNIST.
    
    Args:
        epochs (int): Number of training epochs for each architecture
    """
    # Get device
    device = get_device()
    print_gpu_info()
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # Define architectures to compare
    architectures = {
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'ResNet': ResNet(input_channels=1, num_classes=10, num_blocks=[2, 2, 2]),
        'FullyConnected': FullyConnectedNet(input_size=28*28, hidden_sizes=[256, 128], num_classes=10)
    }
    
    results = {}
    
    # Train and evaluate each architecture
    for name, model in architectures.items():
        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)
        
        model.print_model_info()
        
        # Create optimizer and trainer
        optimizer = CustomAdam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            val_loader=val_loader
        )
        
        # Evaluate
        test_loss, test_acc = trainer.validate(test_loader)
        
        results[name] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'num_params': model.get_num_parameters(),
            'history': history
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Architecture':<20} {'Parameters':<15} {'Test Acc':<15} {'Test Loss':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<20} {result['num_params']:<15,} {result['test_acc']:<15.2f} {result['test_loss']:<15.4f}")
    
    print("=" * 60)
    
    # Generate visual summaries
    output_dir = Path("figures") / "architecture_comparison"
    histories = {name: result['history'] for name, result in results.items()}

    plot_metric_curves(
        histories,
        metric='val_acc',
        title='Validation Accuracy by Architecture',
        ylabel='Accuracy (%)',
        save_path=output_dir / 'validation_accuracy.png',
    )

    plot_metric_curves(
        histories,
        metric='val_loss',
        title='Validation Loss by Architecture',
        ylabel='Loss',
        save_path=output_dir / 'validation_loss.png',
    )

    plot_bar_chart(
        {name: result['test_acc'] for name, result in results.items()},
        title='Test Accuracy by Architecture',
        ylabel='Accuracy (%)',
        save_path=output_dir / 'test_accuracy.png',
    )

    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Different Architectures')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs per architecture')
    
    args = parser.parse_args()
    
    results = compare_architectures(epochs=args.epochs)
