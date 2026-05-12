from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_bayesian: bool = False,
    mc_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_bayesian:
            logits = torch.stack([model(inputs) for _ in range(mc_samples)], dim=0).mean(dim=0)
        else:
            logits = model(inputs)

        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    accuracy = total_correct / total_examples
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
    }


@torch.no_grad()
def predict_regression_mc(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    mc_samples: int = 100,
) -> dict[str, np.ndarray]:
    model.eval()
    inputs = inputs.to(device)
    preds = torch.stack([model(inputs) for _ in range(mc_samples)], dim=0).squeeze(-1)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    lower = torch.quantile(preds, q=0.25, dim=0)
    upper = torch.quantile(preds, q=0.75, dim=0)
    return {
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
        "lower": lower.cpu().numpy(),
        "upper": upper.cpu().numpy(),
        "samples": preds.cpu().numpy(),
    }


def save_accuracy_table(rows: list[dict[str, float | str]], output_path: Path) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)


def save_json(payload: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
