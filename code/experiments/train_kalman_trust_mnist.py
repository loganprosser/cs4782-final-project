from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from bayes_by_backprop.evaluate import save_json
from bayes_by_backprop.models import BayesianMLP, DropoutMLP, StandardMLP
from bayes_by_backprop.update_trust import apply_update_trust_scaling
from bayes_by_backprop.utils import ensure_dir, get_device, set_seed
from kalman_trust.kalman_trust_optim import KalmanTrustAdamW, OptimizerVariant

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


VARIANTS = [
    "adamw",
    "adamw_clip",
    "kalman_lr_controller",
    "direction_aware_trust",
    "per_unit_filter_trust",
    "combined_kalman_trust",
    "old_kalman_grad_scaling",
]

DATASETS = ["mnist", "fashion_mnist"]


def build_model(model_name: str, hidden_dim: int, hidden_layers: int) -> torch.nn.Module:
    if model_name == "standard":
        return StandardMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if model_name == "dropout":
        return DropoutMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if model_name == "bayesian":
        return BayesianMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    raise ValueError(f"Unsupported model: {model_name}")


class LabelNoiseDataset(Dataset):
    def __init__(self, dataset: Dataset, noise_rate: float, seed: int, num_classes: int = 10) -> None:
        if noise_rate < 0.0 or noise_rate >= 1.0:
            raise ValueError("--label-noise must be in [0, 1).")
        self.dataset = dataset
        self.noisy_targets: list[int] = []
        generator = torch.Generator().manual_seed(seed)
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            target_int = int(target)
            if torch.rand((), generator=generator).item() < noise_rate:
                replacement = int(torch.randint(num_classes - 1, (), generator=generator).item())
                if replacement >= target_int:
                    replacement += 1
                self.noisy_targets.append(replacement)
            else:
                self.noisy_targets.append(target_int)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        inputs, _ = self.dataset[idx]
        return inputs, self.noisy_targets[idx]


def load_image_dataset(dataset_name: str, root: Path, data_dir: str):
    transform = transforms.ToTensor()
    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "fashion_mnist":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    train_full = dataset_cls(root=root / data_dir, train=True, download=True, transform=transform)
    test_data = dataset_cls(root=root / data_dir, train=False, download=True, transform=transform)
    return train_full, test_data


def format_noise_tag(label_noise: float) -> str:
    return f"noise{label_noise:g}".replace(".", "p")


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    device = get_device()
    if device.type != "cuda":
        return device
    try:
        # Some PyTorch builds report CUDA as available but cannot run kernels on
        # older GPUs such as V100/sm_70. Probe before moving the model.
        torch.empty(1, device=device) + 1
        torch.cuda.synchronize()
        return device
    except Exception as exc:
        print(f"Warning: CUDA probe failed ({exc}); falling back to CPU.")
        return torch.device("cpu")


def ece_score(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    ece = torch.zeros((), device=logits.device)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_ece = 0.0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_ece += ece_score(logits, targets) * targets.size(0)
        total += targets.size(0)
    return {
        "loss": total_loss / total,
        "nll": total_loss / total,
        "accuracy": total_correct / total,
        "error": 1.0 - (total_correct / total),
        "ece": total_ece / total,
    }


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_curves(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path: Path, row: dict[str, float | int | str]) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing = []
            if path.exists():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    existing = list(csv.DictReader(handle))
                existing = [item for item in existing if not (item["run_name"] == row["run_name"])]
            existing.append({key: str(value) for key, value in row.items()})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(existing)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def plot_curves(curve_files: list[Path], output_path: Path, metric: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for path in curve_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        label = path.stem.replace("_curves", "")
        ax.plot([int(row["epoch"]) for row in rows], [float(row[metric]) for row in rows], label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trust_diagnostics(trust_history: list[dict[str, float]], output_dir: Path, run_name: str) -> None:
    if not trust_history:
        return
    keys = sorted({key for row in trust_history for key in row if key != "epoch"})
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in keys[:12]:
        ax.plot([row["epoch"] for row in trust_history], [row.get(key, float("nan")) for row in trust_history], label=key)
    ax.set_title(f"Mean Trust Over Time: {run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean trust")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_trust_over_time.png", dpi=180)
    plt.close(fig)

    final_values = [value for key, value in trust_history[-1].items() if key != "epoch"]
    if final_values:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(final_values, bins=15, range=(0.0, 1.0), color="#2a6f97", alpha=0.85)
        ax.set_title(f"Final Trust Distribution: {run_name}")
        ax.set_xlabel("mean trust")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / f"{run_name}_trust_hist.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST with AdamW Kalman-trust optimizer variants.")
    parser.add_argument("--optimizer_variant", choices=VARIANTS, default="adamw")
    parser.add_argument("--model", choices=["standard", "dropout", "bayesian"], default="standard")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of train labels to replace; val/test stay clean.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="kalman_trust_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--kalman_beta", type=float, default=0.95)
    parser.add_argument("--kalman_Q", type=float, default=1e-4)
    parser.add_argument("--kalman_P0", type=float, default=1.0)
    parser.add_argument("--kalman_R0", type=float, default=1.0)
    parser.add_argument("--trust_min", type=float, default=0.05)
    parser.add_argument("--trust_max", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root_dir / args.output_dir)
    diagnostics_dir = ensure_dir(output_dir / "trust_diagnostics")
    curves_dir = ensure_dir(output_dir / "curves")
    plots_dir = ensure_dir(output_dir / "plots")
    device = choose_device(args.device)
    dataset_tag = ""
    if args.dataset != "mnist" or args.label_noise > 0.0:
        noise_tag = f"_{format_noise_tag(args.label_noise)}" if args.label_noise > 0.0 else ""
        dataset_tag = f"{args.dataset}{noise_tag}_"
    run_name = f"{dataset_tag}{args.model}_{args.optimizer_variant}_s{args.seed}"

    train_full, test_data = load_image_dataset(args.dataset, root_dir, args.data_dir)
    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_data, val_data = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    if args.label_noise > 0.0:
        train_data = LabelNoiseDataset(train_data, args.label_noise, seed=args.seed + 10_000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, args.hidden_dim, args.hidden_layers).to(device)
    if args.optimizer_variant == "old_kalman_grad_scaling":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trust_state = {}
        trust_optimizer = None
    else:
        optimizer = None
        trust_state = None
        trust_optimizer = KalmanTrustAdamW(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_variant=args.optimizer_variant,  # type: ignore[arg-type]
            clip_grad_norm=args.clip_grad_norm,
            kalman_beta=args.kalman_beta,
            kalman_Q=args.kalman_Q,
            kalman_P0=args.kalman_P0,
            kalman_R0=args.kalman_R0,
            trust_min=args.trust_min,
            trust_max=args.trust_max,
        )

    start_time = time.perf_counter()
    peak_start = rss_mb()
    curves = []
    trust_history = []
    best_val_acc = -1.0
    best_test = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_trust: dict[str, list[float]] = {}
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                assert trust_optimizer is not None
                trust_optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            if args.optimizer_variant == "old_kalman_grad_scaling":
                assert optimizer is not None and trust_state is not None
                stats = apply_update_trust_scaling(
                    model,
                    trust_state,
                    mode="kalman_layer",
                    kalman_beta=args.kalman_beta,
                    kalman_process_noise=args.kalman_Q,
                    kalman_initial_P=args.kalman_P0,
                    kalman_initial_R=args.kalman_R0,
                    kalman_clip_min=args.trust_min,
                    kalman_clip_max=args.trust_max,
                )
                optimizer.step()
                for key, value in stats.items():
                    if key.endswith("/K"):
                        epoch_trust.setdefault(key.removeprefix("kalman/").removesuffix("/K"), []).append(value)
            else:
                assert trust_optimizer is not None
                stats = trust_optimizer.step()
                for key, value in stats.items():
                    epoch_trust.setdefault(key, []).append(value)
            train_loss += loss.item() * targets.size(0)
            train_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            train_total += targets.size(0)

        val = evaluate(model, val_loader, device)
        test = evaluate(model, test_loader, device)
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_test = test
        curve_row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_nll": val["nll"],
            "val_ece": val["ece"],
            "test_accuracy": test["accuracy"],
            "test_nll": test["nll"],
            "test_ece": test["ece"],
        }
        curves.append(curve_row)
        if epoch_trust:
            trust_history.append({"epoch": epoch, **{key: sum(values) / len(values) for key, values in epoch_trust.items()}})
        if not args.quiet:
            print(f"{run_name} epoch={epoch} val_acc={val['accuracy']:.4f} test_acc={test['accuracy']:.4f}")

    runtime = time.perf_counter() - start_time
    peak_memory_mb = max(rss_mb() - peak_start, 0.0)
    curve_path = curves_dir / f"{run_name}_curves.csv"
    write_curves(curve_path, curves)
    plot_trust_diagnostics(trust_history, diagnostics_dir, run_name)

    final = curves[-1]
    summary_row = {
        "run_name": run_name,
        "seed": args.seed,
        "dataset": args.dataset,
        "label_noise": args.label_noise,
        "model": args.model,
        "optimizer_variant": args.optimizer_variant,
        "final_train_loss": final["train_loss"],
        "final_val_accuracy": final["val_accuracy"],
        "final_test_accuracy": final["test_accuracy"],
        "best_val_test_accuracy": best_test.get("accuracy", final["test_accuracy"]),
        "final_test_nll": final["test_nll"],
        "final_test_ece": final["test_ece"],
        "runtime_sec": runtime,
        "peak_memory_mb": peak_memory_mb,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    append_summary(output_dir / "summary.csv", summary_row)
    save_json({"summary": summary_row, "curves": curves, "trust_history": trust_history}, output_dir / f"{run_name}.json")

    curve_files = sorted(curves_dir.glob("*_curves.csv"))
    plot_curves(curve_files, plots_dir / "val_accuracy_by_variant.png", "val_accuracy", "Validation Accuracy")
    plot_curves(curve_files, plots_dir / "val_loss_by_variant.png", "val_loss", "Validation Loss")
    plot_curves(curve_files, plots_dir / "test_accuracy_by_variant.png", "test_accuracy", "Test Accuracy")
    if args.quiet:
        print(
            f"{run_name}: best_val_test_acc={summary_row['best_val_test_accuracy']:.4f} "
            f"final_test_acc={summary_row['final_test_accuracy']:.4f} "
            f"ece={summary_row['final_test_ece']:.4f}"
        )


if __name__ == "__main__":
    main()
