from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
REPRO_DIR = REPO_ROOT / "results" / "reproduction"
sys.path.insert(0, str(CODE_ROOT))

from bayes_by_backprop.losses import bayesian_classification_loss  # noqa: E402
from bayes_by_backprop.models import BayesianMLP, DropoutMLP, StandardMLP  # noqa: E402


METHODS = {
    "standard_mlp": {
        "display": "Standard MLP",
        "architecture": "2 hidden ReLU layers, 400 units each; deterministic weights; no dropout",
    },
    "dropout_mlp": {
        "display": "Dropout MLP",
        "architecture": "2 hidden ReLU layers, 400 units each; deterministic weights; dropout p=0.5",
    },
    "bayes_by_backprop": {
        "display": "Bayes by Backprop",
        "architecture": "2 hidden ReLU layers, 400 units each; BayesianLinear layers; scale-mixture prior",
    },
    "bayes_by_backprop_dropout": {
        "display": "Bayes by Backprop + Dropout",
        "architecture": "2 hidden ReLU layers, 400 units each; BayesianLinear layers; scale-mixture prior; dropout p=0.1",
    },
}

TRAINING_COLUMNS = [
    "method",
    "seed",
    "epoch",
    "train_loss",
    "val_loss",
    "test_loss",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "test_error",
    "epoch_runtime_seconds",
    "cumulative_runtime_seconds",
    "cpu_memory_mb",
    "gpu_memory_allocated_mb",
    "gpu_peak_memory_allocated_mb",
]

FINAL_COLUMNS = [
    "method",
    "seed",
    "architecture",
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "best_epoch",
    "final_test_accuracy",
    "final_test_error",
    "best_test_accuracy",
    "best_test_error",
    "runtime_seconds",
    "average_epoch_time_seconds",
    "checkpoint_path",
]

RESOURCE_COLUMNS = [
    "method",
    "seed",
    "epoch",
    "epoch_runtime_seconds",
    "cumulative_runtime_seconds",
    "cpu_memory_mb",
    "gpu_memory_allocated_mb",
    "gpu_peak_memory_allocated_mb",
]

RUNTIME_COLUMNS = [
    "method",
    "seed",
    "total_runtime_seconds",
    "average_epoch_time_seconds",
    "median_epoch_time_seconds",
    "min_epoch_time_seconds",
    "max_epoch_time_seconds",
]

MEMORY_COLUMNS = [
    "method",
    "seed",
    "peak_cpu_memory_mb",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
]


def ensure_dirs() -> None:
    for path in [
        REPRO_DIR,
        REPRO_DIR / "logs",
        REPRO_DIR / "checkpoints",
        REPRO_DIR / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)


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


def parse_device(value: str) -> torch.device:
    value = value.lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def build_model(method: str) -> nn.Module:
    if method == "standard_mlp":
        return StandardMLP()
    if method == "dropout_mlp":
        return DropoutMLP(dropout=0.5)
    if method == "bayes_by_backprop":
        return BayesianMLP(dropout=0.0)
    if method == "bayes_by_backprop_dropout":
        return BayesianMLP(dropout=0.1)
    raise ValueError(f"Unsupported method: {method}")


def is_bayesian(method: str) -> bool:
    return method.startswith("bayes_by_backprop")


def get_cpu_memory_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        try:
            import resource

            usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                return usage / (1024**2)
            return usage / 1024.0
        except Exception:
            return math.nan


def write_warning(message: str) -> None:
    log_path = REPRO_DIR / "logs" / "run_warnings.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
    print(f"Warning: {message}")


def load_existing_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


def remove_run_rows(df: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    if df.empty or "method" not in df or "seed" not in df:
        return df
    return df[~((df["method"] == method) & (df["seed"].astype(int) == int(seed)))]


def completed(method: str, seed: int, epochs: int) -> bool:
    final_path = REPRO_DIR / "final_metrics.csv"
    if not final_path.exists():
        return False
    df = pd.read_csv(final_path)
    if df.empty:
        return False
    matches = df[(df["method"] == method) & (df["seed"].astype(int) == int(seed))]
    if matches.empty:
        return False
    latest = matches.iloc[-1]
    checkpoint = REPO_ROOT / str(latest["checkpoint_path"])
    return int(latest["epochs"]) >= epochs and checkpoint.exists()


def make_loaders(batch_size: int, seed: int, quick: bool) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_dir = REPO_ROOT / "data"
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    if quick:
        train_dataset = Subset(train_dataset, list(range(min(1024, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(512, len(val_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(512, len(test_dataset)))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, len(train_dataset)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    dataset_size: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if is_bayesian(method):
            loss, _ = bayesian_classification_loss(
                logits,
                targets,
                log_q=model.log_variational_posterior(),
                log_p=model.log_prior(),
                dataset_size=dataset_size,
                kl_weight=1.0,
            )
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    mc_samples: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if is_bayesian(method):
            logits = torch.stack([model(inputs) for _ in range(mc_samples)], dim=0).mean(dim=0)
        else:
            logits = model(inputs)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    accuracy = total_correct / total_examples
    return {"loss": total_loss / total_examples, "accuracy": accuracy, "error": 1.0 - accuracy}


def append_log(method: str, seed: int, message: str) -> None:
    path = REPRO_DIR / "logs" / f"{method}_seed{seed}.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def save_all_tables(
    training_rows: list[dict],
    final_rows: list[dict],
    resource_rows: list[dict],
    runtime_rows: list[dict],
    memory_rows: list[dict],
) -> None:
    base = REPRO_DIR
    save_csv(base / "training_curves.csv", training_rows, TRAINING_COLUMNS)
    save_csv(base / "final_metrics.csv", final_rows, FINAL_COLUMNS)
    save_csv(base / "resource_metrics.csv", resource_rows, RESOURCE_COLUMNS)
    save_csv(base / "runtime_summary.csv", runtime_rows, RUNTIME_COLUMNS)
    save_csv(base / "memory_summary.csv", memory_rows, MEMORY_COLUMNS)


def run_method(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> None:
    set_seed(seed)
    train_loader, val_loader, test_loader, train_size = make_loaders(args.batch_size, seed, args.quick)
    model = build_model(method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    base = REPRO_DIR
    training_rows = load_existing_csv(base / "training_curves.csv", TRAINING_COLUMNS).to_dict("records")
    final_rows = load_existing_csv(base / "final_metrics.csv", FINAL_COLUMNS).to_dict("records")
    resource_rows = load_existing_csv(base / "resource_metrics.csv", RESOURCE_COLUMNS).to_dict("records")
    runtime_rows = load_existing_csv(base / "runtime_summary.csv", RUNTIME_COLUMNS).to_dict("records")
    memory_rows = load_existing_csv(base / "memory_summary.csv", MEMORY_COLUMNS).to_dict("records")

    if args.force:
        for rows in [training_rows, final_rows, resource_rows, runtime_rows, memory_rows]:
            rows[:] = [
                row
                for row in rows
                if not (row.get("method") == method and int(row.get("seed", -1)) == int(seed))
            ]

    start_total = time.perf_counter()
    cumulative = 0.0
    epoch_times: list[float] = []
    best_test_accuracy = float("-inf")
    best_test_error = float("inf")
    best_epoch = 0
    best_state = None
    peak_cpu_memory = get_cpu_memory_mb()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, method, train_size)
        val_metrics = evaluate(model, val_loader, device, method, args.mc_samples)
        test_metrics = evaluate(model, test_loader, device, method, args.mc_samples)
        epoch_runtime = time.perf_counter() - epoch_start
        cumulative = time.perf_counter() - start_total
        epoch_times.append(epoch_runtime)

        cpu_memory = get_cpu_memory_mb()
        if math.isnan(peak_cpu_memory) or (not math.isnan(cpu_memory) and cpu_memory > peak_cpu_memory):
            peak_cpu_memory = cpu_memory

        gpu_alloc = math.nan
        gpu_peak = math.nan
        gpu_reserved = math.nan
        if torch.cuda.is_available() and device.type == "cuda":
            gpu_alloc = torch.cuda.memory_allocated(device) / (1024**2)
            gpu_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            gpu_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

        row = {
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_error": test_metrics["error"],
            "epoch_runtime_seconds": epoch_runtime,
            "cumulative_runtime_seconds": cumulative,
            "cpu_memory_mb": cpu_memory,
            "gpu_memory_allocated_mb": gpu_alloc,
            "gpu_peak_memory_allocated_mb": gpu_peak,
        }
        training_rows.append(row)
        resource_rows.append({key: row[key] for key in RESOURCE_COLUMNS})

        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_test_error = test_metrics["error"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        message = (
            f"epoch={epoch:03d} train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"test_error={100.0 * test_metrics['error']:.2f}% runtime={epoch_runtime:.2f}s"
        )
        print(f"{method} seed={seed} {message}")
        append_log(method, seed, message)

        save_all_tables(training_rows, final_rows, resource_rows, runtime_rows, memory_rows)

    total_runtime = time.perf_counter() - start_total
    final_test = training_rows[-1]["test_accuracy"]
    final_error = training_rows[-1]["test_error"]
    checkpoint_rel = Path("results") / "reproduction" / "checkpoints" / f"{method}_seed{seed}.pt"
    checkpoint_path = REPO_ROOT / checkpoint_rel
    torch.save(
        {
            "method": method,
            "seed": seed,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_model_state_dict": best_state,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "best_epoch": best_epoch,
            "best_test_accuracy": best_test_accuracy,
            "best_test_error": best_test_error,
        },
        checkpoint_path,
    )

    final_rows.append(
        {
            "method": method,
            "seed": seed,
            "architecture": METHODS[method]["architecture"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "best_epoch": best_epoch,
            "final_test_accuracy": final_test,
            "final_test_error": final_error,
            "best_test_accuracy": best_test_accuracy,
            "best_test_error": best_test_error,
            "runtime_seconds": total_runtime,
            "average_epoch_time_seconds": float(np.mean(epoch_times)),
            "checkpoint_path": checkpoint_rel.as_posix(),
        }
    )
    runtime_rows.append(
        {
            "method": method,
            "seed": seed,
            "total_runtime_seconds": total_runtime,
            "average_epoch_time_seconds": float(np.mean(epoch_times)),
            "median_epoch_time_seconds": float(np.median(epoch_times)),
            "min_epoch_time_seconds": float(np.min(epoch_times)),
            "max_epoch_time_seconds": float(np.max(epoch_times)),
        }
    )
    memory_rows.append(
        {
            "method": method,
            "seed": seed,
            "peak_cpu_memory_mb": peak_cpu_memory,
            "peak_gpu_memory_allocated_mb": gpu_peak if "gpu_peak" in locals() else math.nan,
            "peak_gpu_memory_reserved_mb": gpu_reserved if "gpu_reserved" in locals() else math.nan,
        }
    )
    save_all_tables(training_rows, final_rows, resource_rows, runtime_rows, memory_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing MNIST reproduction experiments.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        print("Quick mode: using small data subsets and at most 2 epochs.")
    device = parse_device(args.device)
    print(f"Using device: {device}")

    try:
        import psutil  # noqa: F401
    except Exception:
        write_warning("psutil is not installed; CPU memory uses resource.getrusage fallback when available.")

    for seed in args.seeds:
        for method in METHODS:
            if not args.force and completed(method, seed, args.epochs):
                print(f"Skipping completed run: {method} seed={seed}")
                continue
            run_method(args, method, seed, device)

    print("Reproduction experiment runner finished.")


if __name__ == "__main__":
    main()
