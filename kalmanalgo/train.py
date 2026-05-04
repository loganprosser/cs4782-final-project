from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .kalman_optimizer import KalmanGradientTrust
    from .models import build_model, uses_kalman
    from .utils import accuracy, build_loaders, choose_device, ensure_dir, set_seed, timestamp, write_csv, write_json
except ImportError:
    from kalman_optimizer import KalmanGradientTrust
    from models import build_model, uses_kalman
    from utils import accuracy, build_loaders, choose_device, ensure_dir, set_seed, timestamp, write_csv, write_json


METHODS = ["baseline", "dropout", "kalman", "dropout_kalman"]
MODELS = ["mlp", "cnn"]
OPTIMIZERS = ["sgd", "adam"]
DIAG_MODES = ["tensor", "scalar"]


# Default experiment options. Change these constants to steer local runs without
# needing a long command line; CLI arguments still override them.
DEFAULT_MODEL = "mlp"
DEFAULT_METHOD = "baseline"
DEFAULT_RUN_ALL = False
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SEED = 0
DEFAULT_HIDDEN_SIZE = 400
DEFAULT_DROPOUT_P: float | None = None
DEFAULT_P0 = 1.0
DEFAULT_Q = 1e-5
DEFAULT_R_FLOOR = 1e-8
DEFAULT_BETA_R = 0.95
DEFAULT_EPS = 1e-8
DEFAULT_DIAG_MODE = "tensor"
DEFAULT_DEVICE = "auto"
DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_RUN_NAME = "first_run"
DEFAULT_USE_TIMESTAMPED_RESULT_DIR = True
DEFAULT_TRAIN_SUBSET = 0
DEFAULT_TEST_SUBSET = 0
DEFAULT_NUM_WORKERS = 0
DEFAULT_NO_DOWNLOAD = False
DEFAULT_FAKE_DATA = False
DEFAULT_NO_PLOTS = False


def build_optimizer(
    model: nn.Module,
    method: str,
    args: argparse.Namespace,
) -> torch.optim.Optimizer | KalmanGradientTrust:
    if uses_kalman(method):
        return KalmanGradientTrust(
            model.named_parameters(),
            base_optimizer=args.optimizer,
            lr=args.lr,
            P0=args.P0,
            Q=args.Q,
            R_floor=args.R_floor,
            beta_R=args.beta_R,
            eps=args.eps,
            diag_mode=args.diag_mode,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def flatten_layer_metrics(layer_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for layer, metrics in layer_metrics.items():
        safe_layer = layer.replace(".", "/")
        for key, value in metrics.items():
            row[f"layer:{safe_layer}:{key}"] = value
    return row


def average_layer_metrics(batch_metrics: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for layer_metrics in batch_metrics:
        for layer, metrics in layer_metrics.items():
            totals.setdefault(layer, {})
            counts.setdefault(layer, {})
            for key, value in metrics.items():
                totals[layer][key] = totals[layer].get(key, 0.0) + float(value)
                counts[layer][key] = counts[layer].get(key, 0) + 1
    averaged: dict[str, dict[str, float]] = {}
    for layer, metrics in totals.items():
        averaged[layer] = {key: value / max(counts[layer][key], 1) for key, value in metrics.items()}
    return averaged


def collect_raw_grad_metrics(model: nn.Module) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        metrics[name] = {"raw_grad_norm": float(param.grad.detach().norm().item())}
    return metrics


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | KalmanGradientTrust,
    device: torch.device,
    use_kalman: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batch_layer_metrics: list[dict[str, dict[str, float]]] = []

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        layer_metrics = collect_raw_grad_metrics(model)
        if use_kalman:
            assert isinstance(optimizer, KalmanGradientTrust)
            kalman_metrics = optimizer.filter_gradients(model.named_parameters())
            for layer, metrics in kalman_metrics.items():
                layer_metrics.setdefault(layer, {}).update(metrics)
            optimizer.step(filter_gradients=False)
        else:
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy(logits.detach(), targets)
        total_examples += batch_size
        batch_layer_metrics.append(layer_metrics)

    metrics = {
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
    }
    return metrics, average_layer_metrics(batch_layer_metrics)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy(logits, targets)
        total_examples += batch_size
    return {
        "test_loss": total_loss / max(total_examples, 1),
        "test_accuracy": total_correct / max(total_examples, 1),
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | KalmanGradientTrust,
    args: argparse.Namespace,
    epoch: int,
    test_accuracy: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "test_accuracy": test_accuracy,
    }


def run_one(
    model_name: str,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    result_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(args.seed)
    model = build_model(model_name, method, hidden_size=args.hidden_size, dropout_p=args.dropout_p).to(device)
    optimizer = build_optimizer(model, method, args)
    run_name = f"{model_name}_{method}_seed{args.seed}"
    use_filter = uses_kalman(method)
    metrics_rows: list[dict[str, Any]] = []
    best_test_acc = float("-inf")
    best_test_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, layer_metrics = train_epoch(model, train_loader, optimizer, device, use_filter)
        test_metrics = evaluate(model, test_loader, device)
        row: dict[str, Any] = {
            "run_name": run_name,
            "model": model_name,
            "method": method,
            "epoch": epoch,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "generalization_gap": train_metrics["train_accuracy"] - test_metrics["test_accuracy"],
            **train_metrics,
            **test_metrics,
            **flatten_layer_metrics(layer_metrics),
        }
        metrics_rows.append(row)

        if test_metrics["test_accuracy"] > best_test_acc:
            best_test_acc = test_metrics["test_accuracy"]
            best_test_loss = test_metrics["test_loss"]
            best_epoch = epoch
            checkpoint_path = result_dir / "checkpoints" / f"{run_name}_best.pt"
            ensure_dir(checkpoint_path.parent)
            torch.save(checkpoint_payload(model, optimizer, args, epoch, best_test_acc), checkpoint_path)

        print(
            f"{run_name} epoch {epoch:03d} | "
            f"train_loss={train_metrics['train_loss']:.4f} train_acc={train_metrics['train_accuracy']:.4f} "
            f"test_loss={test_metrics['test_loss']:.4f} test_acc={test_metrics['test_accuracy']:.4f}"
        )

    final = metrics_rows[-1]
    summary = {
        "run_name": run_name,
        "method": method,
        "model": model_name,
        "final_test_acc": final["test_accuracy"],
        "best_test_acc": best_test_acc,
        "final_test_loss": final["test_loss"],
        "best_test_loss": best_test_loss,
        "best_epoch": best_epoch,
        "final_train_acc": final["train_accuracy"],
        "final_train_loss": final["train_loss"],
        "final_generalization_gap": final["generalization_gap"],
    }
    return metrics_rows, summary


def write_report(result_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    by_key = {(row["model"], row["method"]): row for row in summary_rows}
    lines = [
        "# Kalman Gradient Trust MNIST Report",
        "",
        "This experiment compares deterministic MLP and CNN classifiers on MNIST with baseline training, dropout, Kalman-style gradient trust, and dropout plus Kalman-style gradient trust.",
        "",
        "The Kalman method keeps the forward pass deterministic. It filters each minibatch gradient with an EMA-like Kalman update, using observed gradient variance as measurement uncertainty and a diagonal covariance estimate to compute a layer/tensor trust gain.",
        "",
        "## Final Results",
        "",
        "| model | method | final test acc | best test acc | final test loss | best epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['final_test_acc']:.4f} | "
            f"{row['best_test_acc']:.4f} | {row['final_test_loss']:.4f} | {row['best_epoch']} |"
        )

    lines.extend(["", "## Kalman Comparisons", ""])
    for model in MODELS:
        baseline = by_key.get((model, "baseline"))
        dropout = by_key.get((model, "dropout"))
        kalman = by_key.get((model, "kalman"))
        dropout_kalman = by_key.get((model, "dropout_kalman"))
        if baseline and kalman:
            delta = kalman["best_test_acc"] - baseline["best_test_acc"]
            lines.append(f"- {model}: Kalman vs baseline best accuracy delta: {delta:+.4f}.")
        if dropout and dropout_kalman:
            delta = dropout_kalman["best_test_acc"] - dropout["best_test_acc"]
            lines.append(f"- {model}: dropout plus Kalman vs dropout best accuracy delta: {delta:+.4f}.")
        if baseline and dropout and kalman and dropout_kalman:
            complement = dropout_kalman["best_test_acc"] - max(dropout["best_test_acc"], kalman["best_test_acc"])
            lines.append(f"- {model}: complementarity signal over the better single method: {complement:+.4f}.")

    lines.extend(
        [
            "",
            "## Over-Damping Check",
            "",
            "Inspect `plots/kalman_gain_by_layer.png` and `plots/raw_vs_filtered_grad_norm.png`. Persistently low gains paired with slow accuracy improvement are evidence that the filter is over-damping useful gradients.",
            "",
        ]
    )
    report_path = result_dir / "summaries" / "report.md"
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kalman-style gradient trust on MNIST.")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--run-all", action="store_true", default=DEFAULT_RUN_ALL, help="Run all 8 model/method comparisons.")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_DROPOUT_P)
    parser.add_argument("--P0", type=float, default=DEFAULT_P0)
    parser.add_argument("--Q", type=float, default=DEFAULT_Q)
    parser.add_argument("--R-floor", dest="R_floor", type=float, default=DEFAULT_R_FLOOR)
    parser.add_argument("--beta-R", dest="beta_R", type=float, default=DEFAULT_BETA_R)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--diag-mode", choices=DIAG_MODES, default=DEFAULT_DIAG_MODE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--no-timestamped-result-dir",
        action="store_false",
        dest="use_timestamped_result_dir",
        default=DEFAULT_USE_TIMESTAMPED_RESULT_DIR,
        help="Write directly into --results-dir unless --run-name is set.",
    )
    parser.add_argument("--train-subset", type=int, default=DEFAULT_TRAIN_SUBSET)
    parser.add_argument("--test-subset", type=int, default=DEFAULT_TEST_SUBSET)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-download", action="store_true", default=DEFAULT_NO_DOWNLOAD)
    parser.add_argument("--fake-data", action="store_true", default=DEFAULT_FAKE_DATA, help="Use torchvision FakeData for quick smoke tests.")
    parser.add_argument("--no-plots", action="store_true", default=DEFAULT_NO_PLOTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    module_dir = Path(__file__).resolve().parent
    data_dir_arg = Path(args.data_dir)
    data_dir = data_dir_arg if data_dir_arg.is_absolute() else module_dir / data_dir_arg
    results_dir_arg = Path(args.results_dir)
    results_root = results_dir_arg if results_dir_arg.is_absolute() else module_dir / results_dir_arg
    if args.run_name:
        result_dir = ensure_dir(results_root / args.run_name)
    elif args.use_timestamped_result_dir:
        result_dir = ensure_dir(results_root / timestamp())
    else:
        result_dir = ensure_dir(results_root)
    print(f"Using device: {device}")
    print(f"Writing results to: {result_dir}")

    combos = [(model, method) for model in MODELS for method in METHODS] if args.run_all else [(args.model, args.method)]
    all_metrics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    write_json(
        result_dir / "hyperparameters.json",
        vars(args) | {"device": str(device), "combos": combos, "resolved_data_dir": str(data_dir)},
    )

    for model_name, method in combos:
        train_loader, test_loader = build_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            download=not args.no_download,
            fake_data=args.fake_data,
        )
        rows, summary = run_one(model_name, method, args, device, train_loader, test_loader, result_dir)
        all_metrics.extend(rows)
        summaries.append(summary)
        write_csv(result_dir / "metrics" / "metrics.csv", all_metrics)
        write_csv(result_dir / "summaries" / "final_summary.csv", summaries)

    write_report(result_dir, summaries)
    if not args.no_plots:
        try:
            try:
                from .plot_results import plot_all
            except ImportError:
                from plot_results import plot_all

            plot_all(result_dir / "metrics" / "metrics.csv", result_dir / "summaries" / "final_summary.csv", result_dir / "plots")
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}")

    print("")
    print("method, model, final_test_acc, best_test_acc, final_test_loss, best_epoch")
    for row in summaries:
        print(
            f"{row['method']}, {row['model']}, {row['final_test_acc']:.4f}, "
            f"{row['best_test_acc']:.4f}, {row['final_test_loss']:.4f}, {row['best_epoch']}"
        )


if __name__ == "__main__":
    main()
