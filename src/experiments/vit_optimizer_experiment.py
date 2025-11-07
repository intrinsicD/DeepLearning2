"""Experiment: Evaluate optimizers on a Vision Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.architectures import VisionTransformer
from src.optimizers import AndersonGDA, CustomAdam, CustomSGD, GDA2, MuonFast
from src.utils import (
    Trainer,
    get_device,
    get_mnist_loaders,
    plot_bar_chart,
    plot_metric_curves,
    print_gpu_info,
)


def run_vit_optimizer_experiment(
    epochs: int = 5,
    output_dir: Path | None = None,
    return_figures: bool = False,
) -> Dict[str, Dict[str, Any]] | Tuple[
    Dict[str, Dict[str, Any]], Dict[str, Path]
]:
    """Compare optimizers using a compact Vision Transformer on MNIST."""
    device = get_device()
    print_gpu_info()

    print("\nLoading MNIST dataset for ViT experiment...")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=128)

    def get_optimizers(model: nn.Module) -> Dict[str, torch.optim.Optimizer]:
        return {
            "SGD": torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
            "Adam": torch.optim.Adam(model.parameters(), lr=0.001),
            "CustomSGD": CustomSGD(model.parameters(), lr=0.05, momentum=0.9),
            "CustomAdam": CustomAdam(model.parameters(), lr=0.001),
            "AndersonGDA": AndersonGDA(model.parameters(), lr=0.01, beta=0.0, m=2),
            "GDA2": GDA2(model.parameters(), lr=0.001),
            "MuonFast": MuonFast(
                model.parameters(),
                lr=0.0003,  # Much lower LR for ViT with dimension-aware scaling
                momentum=0.95,
                nesterov=True,
                model=model,
                scale_mode="spectral",
                scale_extra=1.0,
                ns_coefficients="simple",
                ns_tol=1e-3,
            ),
        }

    results: Dict[str, Dict[str, Any]] = {}

    optimizer_names = [
        "SGD",
        "Adam",
        "CustomSGD",
        "CustomAdam",
        "AndersonGDA",
        "GDA2",
        "MuonFast",
    ]

    for name in optimizer_names:
        print("\n" + "=" * 60)
        print(f"Training VisionTransformer with {name}")
        print("=" * 60)

        model = VisionTransformer(
            image_size=28,
            patch_size=4,
            in_channels=1,
            num_classes=10,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_dim=128,
            dropout=0.1,
        )
        model.print_model_info()
        model.to(device)

        optimizers = get_optimizers(model)
        optimizer = optimizers[name]
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        history = trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            val_loader=val_loader,
        )

        test_loss, test_acc = trainer.validate(test_loader)

        results[name] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
        }

    print("\n" + "=" * 60)
    print("VISION TRANSFORMER OPTIMIZER RESULTS")
    print("=" * 60)
    print(f"{'Optimizer':<20} {'Test Accuracy':<20} {'Test Loss':<20}")
    print("-" * 60)
    for name, result in results.items():
        print(
            f"{name:<20} {result['test_acc']:<20.2f} "
            f"{result['test_loss']:<20.4f}"
        )
    print("=" * 60)

    if output_dir is None:
        output_dir = Path("figures") / "vit_optimizer_experiment"
    else:
        output_dir = Path(output_dir)

    histories = {name: result["history"] for name, result in results.items()}
    figure_paths = {
        "validation_accuracy": plot_metric_curves(
            histories,
            metric="val_acc",
            title="ViT Validation Accuracy by Optimizer",
            ylabel="Accuracy (%)",
            save_path=output_dir / "validation_accuracy.png",
        ),
        "validation_loss": plot_metric_curves(
            histories,
            metric="val_loss",
            title="ViT Validation Loss by Optimizer",
            ylabel="Loss",
            save_path=output_dir / "validation_loss.png",
        ),
        "test_accuracy": plot_bar_chart(
            {name: result["test_acc"] for name, result in results.items()},
            title="ViT Test Accuracy by Optimizer",
            ylabel="Accuracy (%)",
            save_path=output_dir / "test_accuracy.png",
        ),
    }

    if return_figures:
        return results, figure_paths

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare optimizers using a Vision Transformer on MNIST",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per optimizer",
    )

    args = parser.parse_args()

    run_vit_optimizer_experiment(epochs=args.epochs)
