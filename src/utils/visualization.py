"""Visualization utilities for training experiments."""

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt


def _prepare_output_path(save_path: Path) -> Path:
    """Ensure the output directory exists and return the path."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_metric_curves(
    histories: Mapping[str, Mapping[str, Iterable[float]]],
    metric: str,
    title: str,
    ylabel: str,
    save_path: Path,
) -> Path:
    """Plot the evolution of a metric across training histories.

    Args:
        histories: Mapping from label to history dictionaries containing metric lists.
        metric: Key to extract from each history.
        title: Plot title.
        ylabel: Axis label for the metric.
        save_path: Location to store the generated figure.

    Returns:
        Path pointing to the saved figure.
    """
    save_path = _prepare_output_path(save_path)

    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        if metric not in history or not history[metric]:
            continue
        values = list(history[metric])
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker='o', label=label)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_bar_chart(
    values: Mapping[str, float],
    title: str,
    ylabel: str,
    save_path: Path,
) -> Path:
    """Generate a bar chart comparing scalar values.

    Args:
        values: Mapping from label to scalar metric.
        title: Chart title.
        ylabel: Axis label for the metric.
        save_path: Location to store the generated figure.

    Returns:
        Path to the saved chart.
    """
    save_path = _prepare_output_path(save_path)

    labels = list(values.keys())
    metrics = [values[label] for label in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, metrics)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, metric in zip(bars, metrics):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{metric:.2f}",
            ha='center',
            va='bottom',
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
