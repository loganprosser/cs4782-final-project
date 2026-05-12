from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models import MODEL_NAMES, build_model
from sample_trust import SampleTrustState, compute_sample_trust, needs_features


TRUST_DIAG_KEYS = [
    "trust_mean",
    "trust_std",
    "trust_min",
    "trust_max",
    "frac_at_min_trust",
    "frac_at_max_trust",
    "mean_confidence",
    "mean_entropy",
    "mean_loss",
    "mean_alignment",
    "mean_grad_norm",
]


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


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    existing_fields: list[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
    fieldnames = list(dict.fromkeys([*existing_fields, *row.keys()]))
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_diagnostics(diags: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in TRUST_DIAG_KEYS:
        values = [float(diag[key]) for diag in diags if key in diag and diag[key] is not None]
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def make_subset(dataset: torch.utils.data.Dataset, size: int, seed: int) -> torch.utils.data.Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    transform = transforms.ToTensor()
    data_dir = Path(args.data_dir)
    full_train = datasets.MNIST(root=data_dir, train=True, download=not args.no_download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=not args.no_download, transform=transform)

    if args.train_subset > 0:
        full_train = make_subset(full_train, args.train_subset, args.seed)
    if args.test_subset > 0:
        test_dataset = make_subset(test_dataset, args.test_subset, args.seed + 1)

    train_size = int(args.train_split * len(full_train))
    val_size = len(full_train) - train_size
    if val_size <= 0:
        train_dataset = full_train
    else:
        train_dataset, _ = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader, len(train_dataset)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    trust_state: SampleTrustState,
) -> dict[str, float]:
    model.train()
    total_weighted_loss = 0.0
    total_ce_loss = 0.0
    total_correct = 0
    total_examples = 0
    trust_diags: list[dict[str, Any]] = []
    want_features = needs_features(args.sample_trust_mode)

    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if want_features:
            logits, features = model(inputs, return_features=True)
        else:
            logits = model(inputs)
            features = None

        per_example_loss = F.cross_entropy(logits, targets, reduction="none")
        trust, trust_diag = compute_sample_trust(
            mode=args.sample_trust_mode,
            logits=logits,
            targets=targets,
            per_example_loss=per_example_loss,
            features=features,
            state=trust_state,
            min_trust=args.min_trust,
            max_trust=args.max_trust,
            lambda_loss=args.lambda_loss,
            lambda_entropy=args.lambda_entropy,
            use_leave_one_out=args.use_leave_one_out,
        )
        trust_weight = trust.detach()
        loss = (trust_weight * per_example_loss).sum() / (trust_weight.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_weighted_loss += float(loss.item()) * batch_size
        total_ce_loss += float(per_example_loss.detach().mean().item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_examples += batch_size
        trust_diags.append(trust_diag)

        if args.log_trust_every > 0 and step % args.log_trust_every == 0:
            print(
                f"  step={step:04d} trust_mean={trust_diag.get('trust_mean', 0.0):.4f} "
                f"trust_min={trust_diag.get('trust_min', 0.0):.4f} "
                f"trust_max={trust_diag.get('trust_max', 0.0):.4f}"
            )

    metrics = {
        "train_loss": total_weighted_loss / max(total_examples, 1),
        "train_unweighted_loss": total_ce_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
    }
    metrics.update(average_diagnostics(trust_diags))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_examples += targets.size(0)
    return {
        "test_loss": total_loss / max(total_examples, 1),
        "test_accuracy": total_correct / max(total_examples, 1),
    }


def default_run_name(args: argparse.Namespace) -> str:
    loo = "_loo" if args.use_leave_one_out else ""
    lr = str(args.lr).replace(".", "p")
    return f"{args.model}_{args.sample_trust_mode}_bs{args.batch_size}_lr{lr}_seed{args.seed}{loo}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ordinary MNIST models with samplewise gradient trust.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/sample_trust/runs")
    parser.add_argument("--summary-csv", type=str, default="results/sample_trust/reports/summary.csv")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")

    parser.add_argument("--sample-trust-mode", default="none")
    parser.add_argument("--sample-trust-beta", type=float, default=0.95)
    parser.add_argument("--min-trust", type=float, default=0.05)
    parser.add_argument("--max-trust", type=float, default=1.0)
    parser.add_argument("--lambda-loss", type=float, default=1.0)
    parser.add_argument("--lambda-entropy", type=float, default=1.0)
    parser.add_argument("--use-leave-one-out", action="store_true")
    parser.add_argument("--log-trust-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = args.run_name or default_run_name(args)
    output_root = ensure_dir(Path(args.output_dir))
    run_dir = ensure_dir(output_root / run_name)
    summary_csv = Path(args.summary_csv)
    print(f"Using device: {device}")
    print(f"Run: {run_name}")

    train_loader, test_loader, train_size = build_loaders(args)
    model = build_model(args.model, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trust_state = SampleTrustState(beta=args.sample_trust_beta)

    rows: list[dict[str, Any]] = []
    best_test_acc = float("-inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, trust_state)
        test_metrics = evaluate(model, test_loader, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            **train_metrics,
            **test_metrics,
            "sample_trust_mode": args.sample_trust_mode,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "model": args.model,
            "run_name": run_name,
        }
        rows.append(row)
        write_csv_rows(run_dir / "metrics.csv", rows)

        if test_metrics["test_accuracy"] > best_test_acc:
            best_test_acc = test_metrics["test_accuracy"]
            best_epoch = epoch

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['train_loss']:.4f} "
            f"train_acc={train_metrics['train_accuracy']:.4f} "
            f"test_loss={test_metrics['test_loss']:.4f} "
            f"test_acc={test_metrics['test_accuracy']:.4f} "
            f"trust_mean={train_metrics.get('trust_mean', 1.0):.4f}"
        )

    final = rows[-1]
    summary_row = {
        "run_name": run_name,
        "sample_trust_mode": args.sample_trust_mode,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "min_trust": args.min_trust,
        "max_trust": args.max_trust,
        "lambda_loss": args.lambda_loss,
        "lambda_entropy": args.lambda_entropy,
        "use_leave_one_out": args.use_leave_one_out,
        "train_size": train_size,
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_accuracy"],
        "final_test_loss": final["test_loss"],
        "final_test_acc": final["test_accuracy"],
        "best_test_acc": best_test_acc,
        "epoch_best_test_acc": best_epoch,
        "final_trust_mean": final.get("trust_mean", ""),
        "final_trust_std": final.get("trust_std", ""),
        "final_trust_min": final.get("trust_min", ""),
        "final_trust_max": final.get("trust_max", ""),
        "notes": "ordinary_nn_samplewise_trust",
    }
    append_csv_row(summary_csv, summary_row)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    try:
        from plot_sample_trust_results import plot_run

        plot_run(run_dir / "metrics.csv", run_dir / "plots")
    except Exception as exc:
        print(f"Warning: failed to generate run plots: {exc}")

    print(f"Metrics saved to {run_dir / 'metrics.csv'}")
    print(f"Summary appended to {summary_csv}")


if __name__ == "__main__":
    main()
