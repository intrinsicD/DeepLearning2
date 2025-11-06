"""
Example experiment: Compare different optimizers on MNIST.
"""

from pathlib import Path

import torch
import torch.nn as nn
from src.architectures import SimpleCNN
from src.optimizers import CustomAdam, CustomSGD, MuonFast
from src.utils import (
    Trainer,
    get_device,
    get_mnist_loaders,
    plot_bar_chart,
    plot_metric_curves,
    print_gpu_info,
)


def compare_optimizers(epochs=5):
    """
    Compare different optimizers on the same architecture.
    
    Args:
        epochs (int): Number of training epochs for each optimizer
    """
    # Get device
    device = get_device()
    print_gpu_info()
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # Define optimizers to compare
    def get_optimizers(model):
        return {
            'SGD': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
            'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
            'CustomSGD': CustomSGD(model.parameters(), lr=0.01, momentum=0.9),
            'CustomAdam': CustomAdam(model.parameters(), lr=0.001),
            'MuonFast': MuonFast(model.parameters(), lr=0.001, momentum=0.95),
        }

    results = {}

    # Train and evaluate with each optimizer
    for name in ['SGD', 'Adam', 'CustomSGD', 'CustomAdam', 'MuonFast']:
        print("\n" + "=" * 60)
        print(f"Training with {name} optimizer")
        print("=" * 60)
        
        # Create fresh model for each optimizer
        model = SimpleCNN(input_channels=1, num_classes=10)
        model.print_model_info()
        
        # Create optimizer and trainer
        optimizers = get_optimizers(model)
        optimizer = optimizers[name]
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
            'history': history
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("OPTIMIZER COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Optimizer':<20} {'Test Accuracy':<20} {'Test Loss':<20}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<20} {result['test_acc']:<20.2f} {result['test_loss']:<20.4f}")
    
    print("=" * 60)
    
    # Generate visual summaries
    output_dir = Path("figures") / "optimizer_comparison"
    histories = {name: result['history'] for name, result in results.items()}

    plot_metric_curves(
        histories,
        metric='val_acc',
        title='Validation Accuracy by Optimizer',
        ylabel='Accuracy (%)',
        save_path=output_dir / 'validation_accuracy.png',
    )

    plot_metric_curves(
        histories,
        metric='val_loss',
        title='Validation Loss by Optimizer',
        ylabel='Loss',
        save_path=output_dir / 'validation_loss.png',
    )

    plot_bar_chart(
        {name: result['test_acc'] for name, result in results.items()},
        title='Test Accuracy by Optimizer',
        ylabel='Accuracy (%)',
        save_path=output_dir / 'test_accuracy.png',
    )

    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Different Optimizers')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs per optimizer')
    
    args = parser.parse_args()
    
    results = compare_optimizers(epochs=args.epochs)
