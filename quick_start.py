"""Quick Start Guide for the DeepLearning2 framework."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from src.architectures import FullyConnectedNet, ResNet, SimpleCNN
from src.experiments.compare_architectures import compare_architectures
from src.experiments.compare_optimizers import compare_optimizers
from src.optimizers import CustomAdam, CustomSGD
from src.utils import Trainer, get_device, print_gpu_info


def demo_architectures() -> None:
    """Demonstrate different neural network architectures."""
    print("\n" + "=" * 60)
    print("DEMO 1: Neural Network Architectures")
    print("=" * 60)

    architectures = {
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'ResNet': ResNet(input_channels=1, num_classes=10, num_blocks=[2, 2, 2]),
        'FullyConnected': FullyConnectedNet(
            input_size=784, hidden_sizes=[256, 128], num_classes=10
        ),
    }

    for name, model in architectures.items():
        print(f"\n{name}:")
        print(f"  Parameters: {model.get_num_parameters():,}")

        if name == 'FullyConnected':
            test_input = torch.randn(1, 784)
        else:
            test_input = torch.randn(1, 1, 28, 28)

        with torch.no_grad():
            output = model(test_input)
        print(f"  Output shape: {output.shape}")


def demo_optimizers() -> None:
    """Demonstrate different optimizers."""
    print("\n" + "=" * 60)
    print("DEMO 2: Custom Optimizers")
    print("=" * 60)

    model = SimpleCNN(input_channels=1, num_classes=10)

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


def demo_training() -> None:
    """Demonstrate training with dummy data."""
    print("\n" + "=" * 60)
    print("DEMO 3: Training with Custom Components")
    print("=" * 60)

    device = get_device()
    print(f"\nUsing device: {device}")

    model = SimpleCNN(input_channels=1, num_classes=10)
    print(f"Model parameters: {model.get_num_parameters():,}")

    optimizer = CustomAdam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    print("\nGenerating dummy data for demonstration...")
    dummy_train_data = []
    for _ in range(10):
        data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (32,))
        dummy_train_data.append((data, target))

    model.train()
    print("\nPerforming 5 training steps...")
    for i, (data, target) in enumerate(dummy_train_data[:5]):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"  Step {i + 1}/5 - Loss: {loss.item():.4f}")

    print("\nTraining demonstration completed!")


def demo_gpu_info() -> None:
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


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Utility to print aligned tables for experiment summaries."""

    if not rows:
        print("(no data)")
        return

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(value)))

    header_line = " ".join(
        f"{header:<{col_widths[idx]}}" for idx, header in enumerate(headers)
    )
    separator = "-" * len(header_line)
    print(header_line)
    print(separator)
    for row in rows:
        print(" ".join(f"{str(value):<{col_widths[idx]}}" for idx, value in enumerate(row)))


def _print_figure_locations(title: str, figure_paths: Dict[str, Path]) -> None:
    """Display where generated figures have been saved."""

    print(title)
    if not figure_paths:
        print("  (no figures generated)")
        return

    for name, path in figure_paths.items():
        label = name.replace('_', ' ').title()
        print(f"  - {label}: {path}")


def _report_best_result(results: Dict[str, Dict[str, Any]], metric: str) -> None:
    """Highlight the configuration with the highest metric value."""

    if not results:
        return

    best_name, best_metrics = max(
        results.items(), key=lambda item: item[1].get(metric, float('-inf'))
    )
    value = best_metrics.get(metric)
    if value is not None:
        print(f"Best {metric.replace('_', ' ')}: {best_name} ({value:.2f})")


def run_architecture_benchmark(
    epochs: int, output_dir: Path
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Path]]:
    """Train all reference architectures and collect comparison visuals."""

    print("\n" + "=" * 60)
    print("EXTENSIVE TEST: Architecture Comparison")
    print("=" * 60)

    results, figures = compare_architectures(
        epochs=epochs,
        output_dir=output_dir,
        return_figures=True,
    )

    rows = []
    for name, metrics in results.items():
        rows.append(
            [
                name,
                f"{metrics.get('num_params', 0):,}",
                f"{metrics.get('test_acc', 0.0):.2f}",
                f"{metrics.get('test_loss', 0.0):.4f}",
            ]
        )

    _print_table(
        headers=("Architecture", "Parameters", "Test Acc", "Test Loss"),
        rows=rows,
    )
    _report_best_result(results, metric='test_acc')
    _print_figure_locations(
        "Generated architecture comparison figures:", figures
    )

    return results, figures


def run_optimizer_benchmark(
    epochs: int, output_dir: Path
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Path]]:
    """Train the reference model with multiple optimizers and create visuals."""

    print("\n" + "=" * 60)
    print("EXTENSIVE TEST: Optimizer Comparison")
    print("=" * 60)

    results, figures = compare_optimizers(
        epochs=epochs,
        output_dir=output_dir,
        return_figures=True,
    )

    rows = []
    for name, metrics in results.items():
        rows.append(
            [
                name,
                f"{metrics.get('test_acc', 0.0):.2f}",
                f"{metrics.get('test_loss', 0.0):.4f}",
            ]
        )

    _print_table(
        headers=("Optimizer", "Test Acc", "Test Loss"),
        rows=rows,
    )
    _report_best_result(results, metric='test_acc')
    _print_figure_locations(
        "Generated optimizer comparison figures:", figures
    )

    return results, figures


def run_extensive_evaluations(
    arch_epochs: int,
    optimizer_epochs: int,
    output_root: Path,
) -> None:
    """Execute comprehensive training runs and save the resulting assets."""

    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"run-{timestamp}"
    architecture_dir = run_dir / "architectures"
    optimizer_dir = run_dir / "optimizers"

    print("\n" + "=" * 60)
    print("EXTENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Results will be stored under: {run_dir}")

    architecture_dir.mkdir(parents=True, exist_ok=True)
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    run_architecture_benchmark(epochs=arch_epochs, output_dir=architecture_dir)
    run_optimizer_benchmark(epochs=optimizer_epochs, output_dir=optimizer_dir)

    print("\n" + "=" * 60)
    print("Extensive evaluations completed!")
    print("=" * 60)
    print(f"All generated assets are available in: {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the quick start script."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the DeepLearning2 quick start demonstrations and the full "
            "benchmark suite that compares reference architectures and optimizers."
        )
    )
    parser.add_argument(
        "--arch-epochs",
        type=int,
        default=3,
        help="Number of epochs for each architecture benchmark run.",
    )
    parser.add_argument(
        "--opt-epochs",
        type=int,
        default=5,
        help="Number of epochs for each optimizer benchmark run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("quick_start_reports"),
        help="Root directory used to store benchmark reports and figures.",
    )
    parser.add_argument(
        "--skip-extensive",
        action="store_true",
        help="Skip the comprehensive benchmark suite and only run lightweight demos.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Run all demonstrations and optional benchmark suites."""
    if args is None:
        args = parse_args()

    print("=" * 60)
    print("DeepLearning2 Framework Quick Start")
    print("=" * 60)

    demo_architectures()
    demo_optimizers()
    demo_training()
    demo_gpu_info()

    if args.skip_extensive:
        print("\nSkipping extensive benchmarks (command-line flag provided).")
    else:
        run_extensive_evaluations(
            arch_epochs=args.arch_epochs,
            optimizer_epochs=args.opt_epochs,
            output_root=args.output_root,
        )

    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    if args.skip_extensive:
        print("\nNext steps:")
        print("1. Re-run this script without --skip-extensive to execute the full benchmark suite.")
        print("2. Explore python -m src.experiments.compare_architectures for custom comparisons.")
        print("3. Explore python -m src.experiments.compare_optimizers for optimizer studies.")
    else:
        print(
            "\nBenchmark assets saved to the configured output directory. Review the figures and training"
        )
        print(
            "histories to compare model behavior, then customize architectures and optimizers under"
        )
        print("src/architectures/ and src/optimizers/ for your own experiments.")
    print("=" * 60)


if __name__ == '__main__':
    main()
