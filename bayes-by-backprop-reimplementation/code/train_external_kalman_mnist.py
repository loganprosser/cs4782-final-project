from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kalmanalgo.kalman_optimizer import KalmanGradientTrust

from evaluate import evaluate_classifier, save_json
from losses import bayesian_classification_loss
from train_mnist import LabelNoiseDataset, build_model, load_image_dataset
from utils import ensure_dir, get_device, set_seed


def run_name(args: argparse.Namespace) -> str:
    if args.model == "bayesian" and args.bayesian_dropout > 0.0:
        base = f"bayesian_dropout_{str(args.bayesian_dropout).replace('.', 'p')}"
    else:
        base = args.model
    arch = "" if args.hidden_dim == 400 and args.hidden_layers == 2 else f"_arch_h{args.hidden_dim}_l{args.hidden_layers}"
    data = "clean" if args.label_noise == 0.0 else f"noise{args.label_noise:g}".replace(".", "p")
    return f"{args.dataset}_{data}_{base}_externalKalman_{args.diag_mode}{arch}_s{args.seed}"


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: KalmanGradientTrust,
    device: torch.device,
    *,
    model_name: str,
    dataset_size: int,
    kl_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    gain_total = 0.0
    gain_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
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
        diagnostics = optimizer.filter_gradients(model.named_parameters())
        optimizer.step(filter_gradients=False)

        gains = [item["kalman_gain"] for item in diagnostics.values()]
        if gains:
            gain_total += sum(gains) / len(gains)
            gain_count += 1
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_examples += targets.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "mean_kalman_gain": gain_total / max(gain_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BBB models with kalmanalgo.KalmanGradientTrust.")
    parser.add_argument("--model", choices=["standard", "dropout", "bayesian"], default="standard")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--bayesian-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/external_kalman_algo")
    parser.add_argument("--P0", type=float, default=1.0)
    parser.add_argument("--Q", type=float, default=1e-5)
    parser.add_argument("--R-floor", type=float, default=1e-8)
    parser.add_argument("--beta-R", type=float, default=0.95)
    parser.add_argument("--diag-mode", choices=["tensor", "scalar"], default="tensor")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root_dir / args.output_dir)
    data_dir = ensure_dir(root_dir / args.data_dir)
    device = get_device()

    full_train, test_dataset = load_image_dataset(args.dataset, data_dir)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.label_noise > 0.0:
        train_dataset = LabelNoiseDataset(train_dataset, args.label_noise, seed=args.seed + 10_000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, args.bayesian_dropout, args.hidden_dim, args.hidden_layers).to(device)
    optimizer = KalmanGradientTrust(
        model.named_parameters(),
        base_optimizer=args.base_optimizer,
        lr=args.lr,
        P0=args.P0,
        Q=args.Q,
        R_floor=args.R_floor,
        beta_R=args.beta_R,
        diag_mode=args.diag_mode,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "mean_kalman_gain": []}
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            model_name=args.model,
            dataset_size=len(train_dataset),
            kl_weight=args.kl_weight,
        )
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            device=device,
            is_bayesian=args.model == "bayesian",
            mc_samples=args.mc_samples,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["mean_kalman_gain"].append(train_metrics["mean_kalman_gain"])

        if val_metrics["accuracy"] > best_val_accuracy or (
            val_metrics["accuracy"] == best_val_accuracy and val_metrics["loss"] < best_val_loss
        ):
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        if not args.quiet:
            print(
                f"{run_name(args)} epoch={epoch} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"mean_K={train_metrics['mean_kalman_gain']:.4f}"
            )

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

    name = run_name(args)
    payload = {
        "model": args.model,
        "run_name": name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "base_optimizer": args.base_optimizer,
        "kl_weight": args.kl_weight,
        "mc_samples": args.mc_samples,
        "bayesian_dropout": args.bayesian_dropout,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "optimizer_family": "kalmanalgo.KalmanGradientTrust",
        "diag_mode": args.diag_mode,
        "P0": args.P0,
        "Q": args.Q,
        "R_floor": args.R_floor,
        "beta_R": args.beta_R,
        "seed": args.seed,
        "dataset": args.dataset,
        "label_noise": args.label_noise,
        "selection_strategy": "best_validation_accuracy_then_loss",
        "best_epoch": best_epoch,
        "best_val_metrics": {"accuracy": best_val_accuracy, "loss": best_val_loss},
        "test_metrics": best_test_metrics,
        "last_epoch_test_metrics": last_test_metrics,
        "history": history,
    }
    save_json(payload, output_dir / f"{name}_test_accuracy.json")

    aggregate_path = output_dir / "test_accuracy.json"
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)
    else:
        aggregate = {}
    aggregate[name] = payload
    save_json(aggregate, aggregate_path)

    if args.quiet:
        print(
            f"{name}: best_epoch={best_epoch} "
            f"best_val_acc={best_val_accuracy:.4f} "
            f"test_acc={best_test_metrics['accuracy']:.4f} "
            f"last_test_acc={last_test_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
