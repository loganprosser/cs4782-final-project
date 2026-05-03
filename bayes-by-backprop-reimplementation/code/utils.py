from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="Train Accuracy")
    if history.get("val_accuracy"):
        axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regression_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    mean: np.ndarray,
    output_path: Path,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    title: str = "Regression Predictions",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, color="black", s=16, alpha=0.7, label="Training data")
    ax.plot(x_grid, mean, color="tab:blue", linewidth=2, label="Predictive mean")

    if lower is not None and upper is not None:
        ax.fill_between(
            x_grid.squeeze(),
            lower.squeeze(),
            upper.squeeze(),
            color="tab:blue",
            alpha=0.2,
            label="Interquartile range",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def flatten_history(history: list[dict[str, float]], key: str) -> list[float]:
    return [item[key] for item in history]
