from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate import predict_regression_mc, save_json
from losses import bayesian_regression_loss
from models import BayesianRegressionMLP, RegressionMLP
from update_trust import (
    apply_update_trust_scaling,
    build_optimizer,
    get_update_trust_suffix,
    summarize_trust_logs,
)
from utils import ensure_dir, get_device, plot_regression_predictions, set_seed

DEFAULT_CONFIG = {
    "model": "bayesian",
    "epochs": 2000,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "kl_weight": 1e-4,
    "mc_samples": 100,
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
    "output_dir": "results/regression",
    "resume_checkpoint": "",
}


def generate_regression_data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x_train = torch.cat(
        [
            torch.linspace(0.0, 0.4, 100),
            torch.linspace(0.6, 1.0, 100),
        ]
    ).unsqueeze(1)
    epsilon = torch.randn(x_train.size(), generator=generator) * 0.02
    x_noisy = x_train + epsilon
    y_train = x_train + 0.3 * torch.sin(2 * math.pi * x_noisy) + 0.3 * torch.sin(4 * math.pi * x_noisy) + epsilon
    x_grid = torch.linspace(-0.2, 1.2, 500).unsqueeze(1)
    return x_train, y_train, x_grid


def target_function(x: torch.Tensor) -> torch.Tensor:
    return x + 0.3 * torch.sin(2 * math.pi * x) + 0.3 * torch.sin(4 * math.pi * x)


def build_model(model_name: str) -> nn.Module:
    if model_name == "standard":
        return RegressionMLP()
    if model_name == "bayesian":
        return BayesianRegressionMLP()
    raise ValueError(f"Unsupported model: {model_name}")


def get_run_name(
    model_name: str,
    update_trust_mode: str,
    depth_decay_lambda: float,
    running_grad_beta: float,
    kalman_beta: float,
    kalman_process_noise: float,
    kalman_initial_P: float,
    kalman_initial_R: float,
) -> str:
    return f"{model_name}_{get_update_trust_suffix(update_trust_mode, depth_decay_lambda, running_grad_beta, kalman_beta, kalman_process_noise, kalman_initial_P, kalman_initial_R)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression models for Bayes by Backprop uncertainty plots.")
    parser.add_argument("--model", choices=["standard", "bayesian"], default=DEFAULT_CONFIG["model"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--kl-weight", type=float, default=DEFAULT_CONFIG["kl_weight"])
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_CONFIG["mc_samples"])
    parser.add_argument("--update-trust-mode", choices=["none", "depth_decay", "grad_norm", "running_grad_var", "kalman_layer"], default=DEFAULT_CONFIG["update_trust_mode"])
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
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--resume-checkpoint", type=str, default=DEFAULT_CONFIG["resume_checkpoint"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    run_name = get_run_name(
        args.model,
        args.update_trust_mode,
        args.depth_decay_lambda,
        args.running_grad_beta,
        args.kalman_beta,
        args.kalman_process_noise,
        args.kalman_initial_P,
        args.kalman_initial_R,
    )

    root_dir = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root_dir / args.output_dir)

    x_train, y_train, x_grid = generate_regression_data(seed=args.seed)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model).to(device)
    optimizer = build_optimizer(
        model=model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        update_trust_mode=args.update_trust_mode,
        depth_decay_lambda=args.depth_decay_lambda,
        optimizer_cls=torch.optim.Adam,
    )
    trust_state: dict[str, object] = {}
    start_epoch = 1

    history = {"train_loss": [], "train_mse": []}

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trust_state = checkpoint.get("trust_state", {})
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = checkpoint.get("history", history)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_examples = 0
        epoch_trust_logs: list[dict[str, float]] = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            preds = model(inputs)
            if args.model == "bayesian":
                loss, parts = bayesian_regression_loss(
                    preds,
                    targets,
                    log_q=model.log_variational_posterior(),
                    log_p=model.log_prior(),
                    dataset_size=len(train_dataset),
                    kl_weight=args.kl_weight,
                )
                mse_value = parts["mse"]
            else:
                loss = F.mse_loss(preds, targets)
                mse_value = loss.item()

            loss.backward()
            trust_logs = apply_update_trust_scaling(
                model=model,
                trust_state=trust_state,
                mode=args.update_trust_mode,
                beta=args.running_grad_beta,
                eps=args.grad_trust_eps,
                clip_min=args.grad_trust_clip_min,
                clip_max=args.grad_trust_clip_max,
                kalman_beta=args.kalman_beta,
                kalman_process_noise=args.kalman_process_noise,
                kalman_initial_P=args.kalman_initial_P,
                kalman_initial_R=args.kalman_initial_R,
                kalman_eps=args.kalman_eps,
                kalman_clip_min=args.kalman_clip_min,
                kalman_clip_max=args.kalman_clip_max,
            )
            if trust_logs:
                epoch_trust_logs.append(trust_logs)
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_mse += mse_value * targets.size(0)
            total_examples += targets.size(0)

        history["train_loss"].append(total_loss / total_examples)
        history["train_mse"].append(total_mse / total_examples)

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={history['train_loss'][-1]:.6f} "
                f"train_mse={history['train_mse'][-1]:.6f}"
            )
            if args.log_update_trust and epoch_trust_logs:
                trust_summary = summarize_trust_logs(epoch_trust_logs)
                print(
                    f"  trust avg_alpha={trust_summary.get('avg_alpha', 0.0):.4f} "
                    f"min_alpha={trust_summary.get('min_alpha', 0.0):.4f} "
                    f"max_alpha={trust_summary.get('max_alpha', 0.0):.4f}"
                )
                if args.log_kalman_trust and "kalman_mean_K" in trust_summary:
                    print(
                        f"  kalman mean_K={trust_summary.get('kalman_mean_K', 0.0):.4f} "
                        f"min_K={trust_summary.get('kalman_min_K', 0.0):.4f} "
                        f"max_K={trust_summary.get('kalman_max_K', 0.0):.4f} "
                        f"mean_P={trust_summary.get('kalman_mean_P', 0.0):.4f} "
                        f"mean_R={trust_summary.get('kalman_mean_R', 0.0):.4f}"
                    )

    with torch.no_grad():
        train_preds = model(x_train.to(device))
        train_mse = F.mse_loss(train_preds, y_train.to(device)).item()

    if args.model == "bayesian":
        prediction_stats = predict_regression_mc(model, x_grid, device=device, mc_samples=args.mc_samples)
        mean_predictions = torch.from_numpy(prediction_stats["mean"]).unsqueeze(1)
        plot_regression_predictions(
            x_train.numpy(),
            y_train.numpy(),
            x_grid.numpy(),
            prediction_stats["mean"],
            output_dir / f"{run_name}_uncertainty.png",
            lower=prediction_stats["lower"],
            upper=prediction_stats["upper"],
            title="Bayesian Regression Predictive Uncertainty",
        )

        in_support_mask = ((x_grid.squeeze() >= 0.0) & (x_grid.squeeze() <= 0.4)) | (
            (x_grid.squeeze() >= 0.6) & (x_grid.squeeze() <= 1.0)
        )
        support_std = prediction_stats["std"][in_support_mask.numpy()].mean().item()
        gap_std = prediction_stats["std"][~in_support_mask.numpy()].mean().item()
        qualitative = "uncertainty increases outside observed regions" if gap_std > support_std else "uncertainty did not increase as expected"
        full_grid_mse = F.mse_loss(mean_predictions, target_function(x_grid)).item()
    else:
        model.eval()
        grid_preds_tensor = model(x_grid.to(device)).cpu()
        grid_preds = grid_preds_tensor.numpy()
        plot_regression_predictions(
            x_train.numpy(),
            y_train.numpy(),
            x_grid.numpy(),
            grid_preds,
            output_dir / f"{run_name}.png",
            title="Standard Regression Predictions",
        )
        support_std = None
        gap_std = None
        qualitative = "deterministic model"
        full_grid_mse = F.mse_loss(grid_preds_tensor, target_function(x_grid)).item()

    metrics = {
        "model": args.model,
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "mc_samples": args.mc_samples,
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
        "train_mse": train_mse,
        "full_grid_mse": full_grid_mse,
        "qualitative_uncertainty_behavior": qualitative,
        "support_std_mean": support_std,
        "gap_std_mean": gap_std,
        "history": history,
    }
    metrics_path = output_dir / "regression_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            all_metrics = json.load(handle)
    else:
        all_metrics = {}
    all_metrics[run_name] = metrics
    save_json(all_metrics, metrics_path)

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "trust_state": trust_state,
            "history": history,
            "metrics": metrics,
        },
        checkpoint_dir / f"{run_name}.pt",
    )


if __name__ == "__main__":
    main()
