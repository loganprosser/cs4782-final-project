from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import BayesianMLP, DropoutMLP, StandardMLP
from utils import ensure_dir, get_device, set_seed


DISPLAY_NAMES = {
    "standard": "Standard MLP",
    "dropout": "Dropout MLP",
    "bayesian": "Bayes by Backprop",
    "bayesian_dropout_0p1": "Bayesian + Dropout",
}

COLORS = {
    "standard": "#2a6f97",
    "dropout": "#4d908e",
    "bayesian": "#bc4749",
    "bayesian_dropout_0p1": "#7a3e9d",
}


def display_name(run_name: str) -> str:
    if run_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[run_name]
    if "_trust_" in run_name:
        base_name, trust_suffix = run_name.split("_trust_", maxsplit=1)
        trust_display = trust_suffix.replace("_", " ")
        trust_display = trust_display.replace("kalmanLayer", "kalman-layer")
        trust_display = trust_display.replace("propagatedUncertainty", "propagated uncertainty")
        trust_display = trust_display.replace("gradvar", "grad-var")
        trust_display = trust_display.replace("gradnorm", "grad-norm")
        return f"{display_name(base_name)} ({trust_display})"
    return run_name.replace("_", " ").title()


def checkpoint_base_name(run_name: str) -> str:
    if "_trust_" in run_name:
        return run_name.split("_trust_", maxsplit=1)[0]
    return run_name


def build_model(run_name: str) -> torch.nn.Module:
    base_name = checkpoint_base_name(run_name)
    if base_name == "standard":
        return StandardMLP()
    if base_name == "dropout":
        return DropoutMLP()
    if base_name == "bayesian":
        return BayesianMLP(dropout=0.0)
    if base_name == "bayesian_dropout_0p1":
        return BayesianMLP(dropout=0.1)
    raise ValueError(f"Unsupported run name: {run_name}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(run_name: str, checkpoint_dir: Path, selection_mode: str, device: torch.device) -> torch.nn.Module:
    suffix = "_best.pt" if selection_mode == "best_val" else ".pt"
    checkpoint_path = checkpoint_dir / f"{run_name}{suffix}"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {run_name}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(run_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def checkpoint_available(run_name: str, checkpoint_dir: Path, selection_mode: str) -> bool:
    suffix = "_best.pt" if selection_mode == "best_val" else ".pt"
    return (checkpoint_dir / f"{run_name}{suffix}").exists()


def enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def corrupt_batch(inputs: torch.Tensor, corruption: str, severity: float, generator: torch.Generator) -> torch.Tensor:
    if corruption == "clean":
        return inputs
    if corruption == "gaussian":
        noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype)
        return torch.clamp(inputs + severity * noise, 0.0, 1.0)
    if corruption == "occlusion":
        occluded = inputs.clone()
        size = int(severity)
        start = (28 - size) // 2
        occluded[:, :, start : start + size, start : start + size] = 0.0
        return occluded
    raise ValueError(f"Unsupported corruption: {corruption}")


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    run_name: str,
    mc_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_name = checkpoint_base_name(run_name)
    is_stochastic = base_name.startswith("bayesian") or base_name == "dropout"
    samples = mc_samples if is_stochastic else 1

    model.eval()
    if "dropout" in base_name:
        enable_dropout(model)

    probs = []
    for _ in range(samples):
        logits = model(inputs)
        probs.append(F.softmax(logits, dim=1))
    sample_probs = torch.stack(probs, dim=0)
    mean_probs = sample_probs.mean(dim=0)
    return mean_probs, sample_probs


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)


def mutual_information(mean_probs: torch.Tensor, sample_probs: torch.Tensor) -> torch.Tensor:
    if sample_probs.size(0) <= 1:
        return torch.zeros(mean_probs.size(0), device=mean_probs.device)
    expected_entropy = predictive_entropy(sample_probs.reshape(-1, sample_probs.size(-1))).reshape(
        sample_probs.size(0), sample_probs.size(1)
    ).mean(dim=0)
    return predictive_entropy(mean_probs) - expected_entropy


def roc_auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@torch.no_grad()
def evaluate_corruption(
    model: torch.nn.Module,
    loader: DataLoader,
    run_name: str,
    corruption: str,
    severity: float,
    device: torch.device,
    mc_samples: int,
    seed: int,
) -> dict[str, float | str]:
    total = 0
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    entropy_values = []
    mi_values = []
    error_labels = []
    generator = torch.Generator().manual_seed(seed)

    for inputs, targets in loader:
        inputs = corrupt_batch(inputs, corruption, severity, generator).to(device)
        targets = targets.to(device)
        mean_probs, sample_probs = predict_probabilities(model, inputs, run_name, mc_samples)
        predictions = mean_probs.argmax(dim=1)
        batch_correct = predictions.eq(targets)
        target_probs = mean_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        one_hot = F.one_hot(targets, num_classes=10).float()

        entropy = predictive_entropy(mean_probs)
        mi = mutual_information(mean_probs, sample_probs)

        total += targets.size(0)
        correct += int(batch_correct.sum().item())
        nll_sum += float((-torch.log(target_probs.clamp_min(1e-8))).sum().item())
        brier_sum += float((mean_probs - one_hot).pow(2).sum(dim=1).sum().item())
        entropy_values.extend(float(value) for value in entropy.cpu())
        mi_values.extend(float(value) for value in mi.cpu())
        error_labels.extend(0 if bool(value) else 1 for value in batch_correct.cpu())

    entropy_correct = [score for score, is_error in zip(entropy_values, error_labels) if is_error == 0]
    entropy_error = [score for score, is_error in zip(entropy_values, error_labels) if is_error == 1]
    mi_correct = [score for score, is_error in zip(mi_values, error_labels) if is_error == 0]
    mi_error = [score for score, is_error in zip(mi_values, error_labels) if is_error == 1]
    label = "clean" if corruption == "clean" else f"{corruption}_{severity:g}"

    return {
        "run_name": run_name,
        "display_name": display_name(run_name),
        "corruption": corruption,
        "severity": severity,
        "condition": label,
        "accuracy": correct / total,
        "error_rate": 1.0 - (correct / total),
        "nll": nll_sum / total,
        "brier": brier_sum / total,
        "entropy_mean": sum(entropy_values) / len(entropy_values),
        "entropy_correct_mean": sum(entropy_correct) / len(entropy_correct) if entropy_correct else float("nan"),
        "entropy_error_mean": sum(entropy_error) / len(entropy_error) if entropy_error else float("nan"),
        "entropy_error_gap": (
            (sum(entropy_error) / len(entropy_error)) - (sum(entropy_correct) / len(entropy_correct))
            if entropy_correct and entropy_error
            else float("nan")
        ),
        "entropy_error_auc": roc_auc(entropy_values, error_labels),
        "mutual_information_mean": sum(mi_values) / len(mi_values),
        "mutual_information_error_gap": (
            (sum(mi_error) / len(mi_error)) - (sum(mi_correct) / len(mi_correct))
            if mi_correct and mi_error
            else float("nan")
        ),
        "mutual_information_error_auc": roc_auc(mi_values, error_labels),
        "examples": total,
    }


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows: list[dict[str, float | str]], output_path: Path) -> None:
    condition_order = ["clean", "gaussian_0.15", "gaussian_0.3", "gaussian_0.45", "occlusion_8", "occlusion_12", "occlusion_16"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name in sorted({str(row["run_name"]) for row in rows}):
        model_rows = {str(row["condition"]): row for row in rows if row["run_name"] == run_name}
        xs = [condition for condition in condition_order if condition in model_rows]
        ys = [float(model_rows[condition]["accuracy"]) for condition in xs]
        base = checkpoint_base_name(run_name)
        ax.plot(xs, ys, marker="o", linewidth=2, label=display_name(run_name), color=COLORS.get(base))
    ax.set_title("Accuracy Under MNIST Distribution Shift")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_auc(rows: list[dict[str, float | str]], output_path: Path) -> None:
    shifted_rows = [row for row in rows if row["condition"] != "clean"]
    by_run = {}
    for row in shifted_rows:
        by_run.setdefault(str(row["run_name"]), []).append(float(row["entropy_error_auc"]))
    labels = list(by_run)
    values = [sum(scores) / len(scores) for scores in by_run.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS.get(checkpoint_base_name(label), "#555555") for label in labels]
    ax.bar([display_name(label) for label in labels], values, color=colors)
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1, label="Random ranking")
    ax.set_title("Uncertainty as an Error Detector Under Shift")
    ax.set_ylabel("Mean entropy AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary(rows: list[dict[str, float | str]], config: dict[str, object]) -> str:
    shifted_rows = [row for row in rows if row["condition"] != "clean"]
    clean_rows = [row for row in rows if row["condition"] == "clean"]

    def mean_for(run_name: str, key: str) -> float:
        values = [
            float(row[key])
            for row in shifted_rows
            if row["run_name"] == run_name and not math.isnan(float(row[key]))
        ]
        return sum(values) / len(values) if values else float("nan")

    def clean_accuracy(run_name: str) -> float:
        row = next((item for item in clean_rows if item["run_name"] == run_name), None)
        return float(row["accuracy"]) if row else float("nan")

    best_clean = max(clean_rows, key=lambda row: float(row["accuracy"]))
    best_shift_acc = max(
        {str(row["run_name"]) for row in rows},
        key=lambda name: mean_for(name, "accuracy"),
    )
    best_auc = max(
        {str(row["run_name"]) for row in rows},
        key=lambda name: mean_for(name, "entropy_error_auc"),
    )
    ranked_shift = sorted({str(row["run_name"]) for row in rows}, key=lambda name: mean_for(name, "accuracy"), reverse=True)
    ranked_auc = sorted({str(row["run_name"]) for row in rows}, key=lambda name: mean_for(name, "entropy_error_auc"), reverse=True)

    top_shift_text = ", ".join(
        f"`{display_name(name)}` (`{mean_for(name, 'accuracy'):.4f}`)" for name in ranked_shift[:3]
    )
    top_auc_text = ", ".join(
        f"`{display_name(name)}` (`{mean_for(name, 'entropy_error_auc'):.4f}`)" for name in ranked_auc[:3]
    )

    lines = [
        "# Uncertainty Shift Benchmark Summary",
        "",
        "## Test",
        "This benchmark evaluates trained MNIST checkpoints on clean inputs plus Gaussian noise and center occlusion corruptions.",
        "It is designed to expose the regime where Bayesian, dropout, and Kalman-trust variants should help most: not just average accuracy, but knowing when predictions have become unreliable.",
        "",
        "## Configuration",
        f"- Selection mode: `{config['selection_mode']}`.",
        f"- Examples per condition: `{config['max_examples']}`.",
        f"- MC samples for stochastic models: `{config['mc_samples']}`.",
        f"- Device: `{config['device']}`.",
        "",
        "## Key Findings",
        f"- Best clean accuracy in this run: `{display_name(str(best_clean['run_name']))}` at `{float(best_clean['accuracy']):.4f}`.",
        f"- Best average shifted accuracy: `{display_name(best_shift_acc)}` at `{mean_for(best_shift_acc, 'accuracy'):.4f}`.",
        f"- Best uncertainty/error ranking under shift: `{display_name(best_auc)}` at `{mean_for(best_auc, 'entropy_error_auc'):.4f}` mean entropy AUROC.",
        f"- Top shifted-accuracy runners: {top_shift_text}.",
        f"- Top uncertainty-ranking runners: {top_auc_text}.",
        "",
        "## Kalman Readout",
    ]

    kalman_pairs = [
        ("bayesian_trust_none", "bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0"),
        ("bayesian_dropout_0p1_trust_none", "bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0"),
    ]
    for baseline_name, kalman_name in kalman_pairs:
        if any(row["run_name"] == baseline_name for row in rows) and any(row["run_name"] == kalman_name for row in rows):
            shift_delta = mean_for(kalman_name, "accuracy") - mean_for(baseline_name, "accuracy")
            auc_delta = mean_for(kalman_name, "entropy_error_auc") - mean_for(baseline_name, "entropy_error_auc")
            clean_delta = clean_accuracy(kalman_name) - clean_accuracy(baseline_name)
            lines.append(
                f"- `{display_name(kalman_name)}` vs `{display_name(baseline_name)}`: clean accuracy delta `{clean_delta:+.4f}`, shifted accuracy delta `{shift_delta:+.4f}`, entropy AUROC delta `{auc_delta:+.4f}`."
            )
    if lines[-1] == "## Kalman Readout":
        lines.append("- No Kalman-layer checkpoints were included in this benchmark run.")

    lines.extend(
        [
            "",
            "## Reading The Metrics",
            "- `entropy_error_auc` measures whether higher predictive entropy ranks wrong predictions above correct predictions; `0.5` is random and `1.0` is ideal.",
            "- `entropy_error_gap` is mean entropy on wrong predictions minus mean entropy on correct predictions; positive values mean uncertainty rises on mistakes.",
            "- `mutual_information_*` uses MC disagreement, so it is most meaningful for Bayesian and dropout-style stochastic models.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark uncertainty quality under MNIST distribution shift.")
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--summary-dir", type=str, default="results/uncertainty_shift")
    parser.add_argument("--output-dir", type=str, default="results/uncertainty_shift")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-trust-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = root_dir / "results"
    output_dir = ensure_dir(root_dir / args.output_dir)
    summary_dir = ensure_dir(root_dir / args.summary_dir)
    device = get_device()

    metrics = load_json(results_dir / "mnist" / "test_accuracy.json")
    checkpoint_dir = results_dir / "mnist" / "checkpoints"
    if args.include_trust_runs:
        run_names = [name for name in metrics if checkpoint_available(name, checkpoint_dir, args.selection_mode)]
    else:
        preferred = [
            "standard_trust_none",
            "dropout_trust_none",
            "bayesian_trust_none",
            "bayesian_dropout_0p1_trust_none",
        ]
        run_names = [name for name in preferred if name in metrics]
        run_names = [name for name in run_names if checkpoint_available(name, checkpoint_dir, args.selection_mode)]
    if not run_names:
        raise RuntimeError("No MNIST runs found. Train checkpoints before running the uncertainty benchmark.")

    dataset = datasets.MNIST(root=root_dir / args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.max_examples > 0:
        dataset = Subset(dataset, list(range(min(args.max_examples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    conditions = [
        ("clean", 0.0),
        ("gaussian", 0.15),
        ("gaussian", 0.30),
        ("gaussian", 0.45),
        ("occlusion", 8.0),
        ("occlusion", 12.0),
        ("occlusion", 16.0),
    ]
    rows: list[dict[str, float | str]] = []
    for run_name in run_names:
        model = load_model(run_name, checkpoint_dir, args.selection_mode, device)
        for condition_index, (corruption, severity) in enumerate(conditions):
            rows.append(
                evaluate_corruption(
                    model=model,
                    loader=loader,
                    run_name=run_name,
                    corruption=corruption,
                    severity=severity,
                    device=device,
                    mc_samples=args.mc_samples,
                    seed=args.seed + condition_index,
                )
            )

    config = {
        "selection_mode": args.selection_mode,
        "max_examples": len(dataset),
        "batch_size": args.batch_size,
        "mc_samples": args.mc_samples,
        "seed": args.seed,
        "device": str(device),
        "include_trust_runs": args.include_trust_runs,
        "conditions": [{"corruption": name, "severity": severity} for name, severity in conditions],
    }
    payload = {"config": config, "results": rows}
    with (output_dir / "uncertainty_shift_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(rows, output_dir / "uncertainty_shift_metrics.csv")

    plot_accuracy(rows, summary_dir / "uncertainty_shift_accuracy.png")
    plot_uncertainty_auc(rows, summary_dir / "uncertainty_shift_entropy_auc.png")
    (summary_dir / "uncertainty_shift_summary.md").write_text(build_summary(rows, config), encoding="utf-8")
    with (summary_dir / "uncertainty_shift_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics_json": str((output_dir / "uncertainty_shift_metrics.json").relative_to(root_dir)),
                "metrics_csv": str((output_dir / "uncertainty_shift_metrics.csv").relative_to(root_dir)),
                "summary": str((summary_dir / "uncertainty_shift_summary.md").relative_to(root_dir)),
                "plots": [
                    str((summary_dir / "uncertainty_shift_accuracy.png").relative_to(root_dir)),
                    str((summary_dir / "uncertainty_shift_entropy_auc.png").relative_to(root_dir)),
                ],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
