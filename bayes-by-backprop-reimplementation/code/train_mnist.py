from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from evaluate import evaluate_classifier, save_accuracy_table, save_json
from losses import bayesian_classification_loss
from models import BayesianMLP, DropoutMLP, StandardMLP
from update_trust import (
    apply_update_trust_scaling,
    build_optimizer,
    get_update_trust_suffix,
    summarize_trust_logs,
)
from utils import ensure_dir, get_device, plot_training_curves, set_seed

DEFAULT_CONFIG = {
    "model": "bayesian",
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "kl_weight": 1.0,
    "mc_samples": 10,
    "bayesian_dropout": 0.1,
    "hidden_dim": 400,
    "hidden_layers": 2,
    "update_trust_mode": "none",
    "depth_decay_lambda": 0.15,
    "grad_trust_eps": 1e-8,
    "grad_trust_clip_min": 0.1,
    "grad_trust_clip_max": 10.0,
    "running_grad_beta": 0.95,
    "log_update_trust": True,
    "kalman_beta": 0.95,
    "kalman_process_noise": 1e-4,
    "kalman_initial_P": 1.0,
    "kalman_initial_R": 1.0,
    "kalman_eps": 1e-8,
    "kalman_clip_min": 0.05,
    "kalman_clip_max": 1.0,
    "log_kalman_trust": True,
    "seed": 0,
    "data_dir": "data",
    "output_dir": "results/mnist",
    "resume_checkpoint": "",
    "quiet": False,
    "run_tag": "",
}


def build_model(model_name: str, bayesian_dropout: float, hidden_dim: int, hidden_layers: int) -> nn.Module:
    if model_name == "standard":
        return StandardMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if model_name == "dropout":
        return DropoutMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if model_name == "bayesian":
        return BayesianMLP(dropout=bayesian_dropout, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    raise ValueError(f"Unsupported model: {model_name}")


def get_run_name(
    model_name: str,
    bayesian_dropout: float,
    update_trust_mode: str,
    depth_decay_lambda: float,
    running_grad_beta: float,
    kalman_beta: float,
    kalman_process_noise: float,
    kalman_initial_P: float,
    kalman_initial_R: float,
    hidden_dim: int = 400,
    hidden_layers: int = 2,
    run_tag: str = "",
) -> str:
    if model_name == "bayesian" and bayesian_dropout > 0.0:
        base_name = f"bayesian_dropout_{str(bayesian_dropout).replace('.', 'p')}"
    else:
        base_name = model_name
    trust_suffix = get_update_trust_suffix(
        update_trust_mode,
        depth_decay_lambda,
        running_grad_beta,
        kalman_beta=kalman_beta,
        kalman_process_noise=kalman_process_noise,
        kalman_initial_P=kalman_initial_P,
        kalman_initial_R=kalman_initial_R,
    )
    arch_suffix = "" if hidden_dim == 400 and hidden_layers == 2 else f"_arch_h{hidden_dim}_l{hidden_layers}"
    tag_suffix = f"_{run_tag}" if run_tag else ""
    return f"{base_name}_{trust_suffix}{arch_suffix}{tag_suffix}"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_size: int,
    model_name: str,
    kl_weight: float,
    trust_state: dict[str, float],
    update_trust_mode: str,
    grad_trust_eps: float,
    grad_trust_clip_min: float,
    grad_trust_clip_max: float,
    running_grad_beta: float,
    log_update_trust: bool,
    kalman_beta: float,
    kalman_process_noise: float,
    kalman_initial_P: float,
    kalman_initial_R: float,
    kalman_eps: float,
    kalman_clip_min: float,
    kalman_clip_max: float,
    log_kalman_trust: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    epoch_trust_logs: list[dict[str, float]] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        logits = model(inputs)
        if model_name == "bayesian":
            loss, _ = bayesian_classification_loss(
                logits,
                targets,
                log_q=model.log_variational_posterior(),
                log_p=model.log_prior(),
                dataset_size=dataset_size,
                kl_weight=kl_weight,
            )
        else:
            loss = F.cross_entropy(logits, targets)

        loss.backward()
        trust_logs = apply_update_trust_scaling(
            model=model,
            trust_state=trust_state,
            mode=update_trust_mode,
            beta=running_grad_beta,
            eps=grad_trust_eps,
            clip_min=grad_trust_clip_min,
            clip_max=grad_trust_clip_max,
            kalman_beta=kalman_beta,
            kalman_process_noise=kalman_process_noise,
            kalman_initial_P=kalman_initial_P,
            kalman_initial_R=kalman_initial_R,
            kalman_eps=kalman_eps,
            kalman_clip_min=kalman_clip_min,
            kalman_clip_max=kalman_clip_max,
        )
        if trust_logs:
            epoch_trust_logs.append(trust_logs)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
    if (log_update_trust or log_kalman_trust) and epoch_trust_logs:
        trust_summary = summarize_trust_logs(epoch_trust_logs)
        metrics.update({f"trust_{key}": value for key, value in trust_summary.items()})
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    mc_samples: int,
) -> dict[str, float]:
    return evaluate_classifier(
        model,
        loader,
        device=device,
        is_bayesian=model_name == "bayesian",
        mc_samples=mc_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST models for Bayes by Backprop reproduction.")
    parser.add_argument("--model", choices=["standard", "dropout", "bayesian"], default=DEFAULT_CONFIG["model"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--kl-weight", type=float, default=DEFAULT_CONFIG["kl_weight"])
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_CONFIG["mc_samples"])
    parser.add_argument("--bayesian-dropout", type=float, default=DEFAULT_CONFIG["bayesian_dropout"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--hidden-layers", type=int, default=DEFAULT_CONFIG["hidden_layers"])
    parser.add_argument("--update-trust-mode", choices=["none", "depth_decay", "grad_norm", "running_grad_var", "kalman_layer", "propagated_uncertainty"], default=DEFAULT_CONFIG["update_trust_mode"])
    parser.add_argument("--depth-decay-lambda", type=float, default=DEFAULT_CONFIG["depth_decay_lambda"])
    parser.add_argument("--grad-trust-eps", type=float, default=DEFAULT_CONFIG["grad_trust_eps"])
    parser.add_argument("--grad-trust-clip-min", type=float, default=DEFAULT_CONFIG["grad_trust_clip_min"])
    parser.add_argument("--grad-trust-clip-max", type=float, default=DEFAULT_CONFIG["grad_trust_clip_max"])
    parser.add_argument("--running-grad-beta", type=float, default=DEFAULT_CONFIG["running_grad_beta"])
    parser.add_argument("--log-update-trust", action="store_true", default=DEFAULT_CONFIG["log_update_trust"])
    parser.add_argument("--no-log-update-trust", action="store_false", dest="log_update_trust")
    parser.add_argument("--kalman-beta", type=float, default=DEFAULT_CONFIG["kalman_beta"])
    parser.add_argument("--kalman-process-noise", type=float, default=DEFAULT_CONFIG["kalman_process_noise"])
    parser.add_argument("--kalman-initial-P", type=float, default=DEFAULT_CONFIG["kalman_initial_P"])
    parser.add_argument("--kalman-initial-R", type=float, default=DEFAULT_CONFIG["kalman_initial_R"])
    parser.add_argument("--kalman-eps", type=float, default=DEFAULT_CONFIG["kalman_eps"])
    parser.add_argument("--kalman-clip-min", type=float, default=DEFAULT_CONFIG["kalman_clip_min"])
    parser.add_argument("--kalman-clip-max", type=float, default=DEFAULT_CONFIG["kalman_clip_max"])
    parser.add_argument("--log-kalman-trust", action="store_true", default=DEFAULT_CONFIG["log_kalman_trust"])
    parser.add_argument("--no-log-kalman-trust", action="store_false", dest="log_kalman_trust")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--data-dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--resume-checkpoint", type=str, default=DEFAULT_CONFIG["resume_checkpoint"])
    parser.add_argument("--quiet", action="store_true", default=DEFAULT_CONFIG["quiet"])
    parser.add_argument("--run-tag", type=str, default=DEFAULT_CONFIG["run_tag"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    run_name = get_run_name(
        args.model,
        args.bayesian_dropout,
        args.update_trust_mode,
        args.depth_decay_lambda,
        args.running_grad_beta,
        args.kalman_beta,
        args.kalman_process_noise,
        args.kalman_initial_P,
        args.kalman_initial_R,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        run_tag=args.run_tag,
    )

    root_dir = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root_dir / args.output_dir)
    data_dir = ensure_dir(root_dir / args.data_dir)

    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trust_state: dict[str, object] = {}
    model = build_model(args.model, args.bayesian_dropout, args.hidden_dim, args.hidden_layers).to(device)
    optimizer = build_optimizer(
        model=model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        update_trust_mode=args.update_trust_mode,
        depth_decay_lambda=args.depth_decay_lambda,
        optimizer_cls=torch.optim.Adam,
    )
    start_epoch = 1

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trust_state = checkpoint.get("trust_state", {})
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = checkpoint.get("history", history)
        best_val_accuracy = float(checkpoint.get("best_val_accuracy", best_val_accuracy))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        best_model_state = checkpoint.get("best_model_state_dict", best_model_state)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            dataset_size=len(train_dataset),
            model_name=args.model,
            kl_weight=args.kl_weight,
            trust_state=trust_state,
            update_trust_mode=args.update_trust_mode,
            grad_trust_eps=args.grad_trust_eps,
            grad_trust_clip_min=args.grad_trust_clip_min,
            grad_trust_clip_max=args.grad_trust_clip_max,
            running_grad_beta=args.running_grad_beta,
            log_update_trust=args.log_update_trust,
            kalman_beta=args.kalman_beta,
            kalman_process_noise=args.kalman_process_noise,
            kalman_initial_P=args.kalman_initial_P,
            kalman_initial_R=args.kalman_initial_R,
            kalman_eps=args.kalman_eps,
            kalman_clip_min=args.kalman_clip_min,
            kalman_clip_max=args.kalman_clip_max,
            log_kalman_trust=args.log_kalman_trust,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            device=device,
            model_name=args.model,
            mc_samples=args.mc_samples,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        is_better = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["loss"] < best_val_loss
            )
        )
        if is_better:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        if not args.quiet:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
        if not args.quiet and args.log_update_trust and "trust_avg_alpha" in train_metrics:
            print(
                f"  trust avg_alpha={train_metrics['trust_avg_alpha']:.4f} "
                f"min_alpha={train_metrics['trust_min_alpha']:.4f} "
                f"max_alpha={train_metrics['trust_max_alpha']:.4f}"
            )
        if not args.quiet and args.log_kalman_trust and "trust_kalman_mean_K" in train_metrics:
            print(
                f"  kalman mean_K={train_metrics['trust_kalman_mean_K']:.4f} "
                f"min_K={train_metrics['trust_kalman_min_K']:.4f} "
                f"max_K={train_metrics['trust_kalman_max_K']:.4f} "
                f"mean_P={train_metrics.get('trust_kalman_mean_P', 0.0):.4f} "
                f"mean_R={train_metrics.get('trust_kalman_mean_R', 0.0):.4f}"
            )
        if not args.quiet and args.log_kalman_trust and "trust_propagated_mean_K" in train_metrics:
            print(
                f"  propagated mean_K={train_metrics['trust_propagated_mean_K']:.4f} "
                f"min_K={train_metrics['trust_propagated_min_K']:.4f} "
                f"max_K={train_metrics['trust_propagated_max_K']:.4f} "
                f"mean_P={train_metrics.get('trust_propagated_mean_P', 0.0):.4f} "
                f"mean_R_eff={train_metrics.get('trust_propagated_mean_R_eff', 0.0):.4f}"
            )

    last_model_state = copy.deepcopy(model.state_dict())
    last_test_metrics = evaluate_classifier(
        model,
        test_loader,
        device=device,
        is_bayesian=args.model == "bayesian",
        mc_samples=args.mc_samples,
    )

    model.load_state_dict(best_model_state)
    best_test_metrics = evaluate_classifier(
        model,
        test_loader,
        device=device,
        is_bayesian=args.model == "bayesian",
        mc_samples=args.mc_samples,
    )

    metrics_payload = {
        "model": args.model,
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "mc_samples": args.mc_samples,
        "bayesian_dropout": args.bayesian_dropout,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "update_trust_mode": args.update_trust_mode,
        "depth_decay_lambda": args.depth_decay_lambda,
        "grad_trust_eps": args.grad_trust_eps,
        "grad_trust_clip_min": args.grad_trust_clip_min,
        "grad_trust_clip_max": args.grad_trust_clip_max,
        "running_grad_beta": args.running_grad_beta,
        "kalman_beta": args.kalman_beta,
        "kalman_process_noise": args.kalman_process_noise,
        "kalman_initial_P": args.kalman_initial_P,
        "kalman_initial_R": args.kalman_initial_R,
        "kalman_eps": args.kalman_eps,
        "kalman_clip_min": args.kalman_clip_min,
        "kalman_clip_max": args.kalman_clip_max,
        "seed": args.seed,
        "run_tag": args.run_tag,
        "selection_strategy": "best_validation_accuracy_then_loss",
        "best_epoch": best_epoch,
        "best_val_metrics": {
            "accuracy": best_val_accuracy,
            "loss": best_val_loss,
        },
        "test_metrics": best_test_metrics,
        "last_epoch_test_metrics": last_test_metrics,
        "history": history,
    }
    save_json(metrics_payload, output_dir / f"{run_name}_test_accuracy.json")

    aggregate_metrics_path = output_dir / "test_accuracy.json"
    if aggregate_metrics_path.exists():
        with aggregate_metrics_path.open("r", encoding="utf-8") as handle:
            aggregate_metrics = json.load(handle)
    else:
        aggregate_metrics = {}
    aggregate_metrics[run_name] = metrics_payload
    save_json(aggregate_metrics, aggregate_metrics_path)

    table_path = output_dir / "accuracy_table.csv"
    table_rows = []
    if table_path.exists():
        import pandas as pd

        table_rows = pd.read_csv(table_path).to_dict(orient="records")
        table_rows = [row for row in table_rows if row["Model"] != run_name]

    table_rows.append(
        {
            "Model": run_name,
            "Test Accuracy": best_test_metrics["accuracy"],
            "Test Error": best_test_metrics["error_rate"],
        }
    )
    save_accuracy_table(table_rows, table_path)

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": last_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "trust_state": trust_state,
            "history": history,
            "metrics_payload": metrics_payload,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state,
        },
        checkpoint_dir / f"{run_name}.pt",
    )
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "trust_state": trust_state,
            "history": history,
            "metrics_payload": metrics_payload,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state,
        },
        checkpoint_dir / f"{run_name}_best.pt",
    )

    try:
        plot_training_curves(
            history,
            output_dir / f"{run_name}_training_curves.png",
            title=f"MNIST {run_name.replace('_', ' ').title()} Training Curves",
        )
    except Exception as exc:
        print(f"Warning: failed to save training curves for {run_name}: {exc}")

    if args.quiet:
        print(
            f"{run_name}: best_epoch={best_epoch} "
            f"best_val_acc={best_val_accuracy:.4f} "
            f"test_acc={best_test_metrics['accuracy']:.4f} "
            f"last_test_acc={last_test_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
