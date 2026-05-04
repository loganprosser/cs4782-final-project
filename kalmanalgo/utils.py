from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def choose_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_subset(dataset: torch.utils.data.Dataset, size: int, seed: int) -> torch.utils.data.Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    train_subset: int = 0,
    test_subset: int = 0,
    num_workers: int = 0,
    download: bool = True,
    fake_data: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if fake_data:
        train_dataset = datasets.FakeData(
            size=train_subset if train_subset > 0 else 256,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed,
        )
        test_dataset = datasets.FakeData(
            size=test_subset if test_subset > 0 else 128,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed + 1000,
        )
    else:
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        train_dataset = make_subset(train_dataset, train_subset, seed)
        test_dataset = make_subset(test_dataset, test_subset, seed + 1)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(targets).sum().item())
