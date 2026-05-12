from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        values.append(float(value) if value not in ("", None) else float("nan"))
    return values


def has_numeric(rows: list[dict[str, str]], key: str) -> bool:
    for row in rows:
        value = row.get(key, "")
        if value not in ("", None):
            try:
                float(value)
                return True
            except ValueError:
                pass
    return False


def line_plot(
    x: list[float],
    series: list[tuple[str, list[float]]],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in series:
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run(metrics_csv: Path, output_dir: Path | None = None) -> None:
    rows = read_rows(metrics_csv)
    if not rows:
        return
    output_dir = ensure_dir(output_dir or metrics_csv.parent / "plots")
    epochs = floats(rows, "epoch")

    line_plot(
        epochs,
        [("Train Accuracy", floats(rows, "train_accuracy")), ("Test Accuracy", floats(rows, "test_accuracy"))],
        output_dir / "accuracy_vs_epoch.png",
        "Accuracy vs Epoch",
        "Accuracy",
    )
    line_plot(
        epochs,
        [("Train Loss", floats(rows, "train_loss")), ("Test Loss", floats(rows, "test_loss"))],
        output_dir / "loss_vs_epoch.png",
        "Loss vs Epoch",
        "Loss",
    )

    if has_numeric(rows, "trust_mean"):
        trust_mean = floats(rows, "trust_mean")
        trust_std = floats(rows, "trust_std")
        lower = [m - s for m, s in zip(trust_mean, trust_std)]
        upper = [m + s for m, s in zip(trust_mean, trust_std)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, trust_mean, marker="o", linewidth=1.8, label="Trust Mean")
        ax.fill_between(epochs, lower, upper, alpha=0.2, label="Mean +/- Std")
        ax.set_title("Trust Mean and Std vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Trust")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "trust_mean_std_vs_epoch.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if has_numeric(rows, "trust_min") and has_numeric(rows, "trust_max"):
        line_plot(
            epochs,
            [("Trust Min", floats(rows, "trust_min")), ("Trust Max", floats(rows, "trust_max"))],
            output_dir / "trust_min_max_vs_epoch.png",
            "Trust Min and Max vs Epoch",
            "Trust",
        )

    optional = [
        ("mean_alignment", "Mean Alignment"),
        ("mean_grad_norm", "Mean Grad Norm"),
        ("mean_entropy", "Mean Entropy"),
    ]
    for key, label in optional:
        if has_numeric(rows, key):
            line_plot(
                epochs,
                [(label, floats(rows, key))],
                output_dir / f"{key}_vs_epoch.png",
                f"{label} vs Epoch",
                label,
            )


def grouped_mean(rows: list[dict[str, str]], value_key: str, group_keys: tuple[str, ...]) -> dict[tuple[str, ...], float]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key, "")
        if value in ("", None):
            continue
        buckets[tuple(row.get(key, "") for key in group_keys)].append(float(value))
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def bar_plot(labels: list[str], values: list[float], output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(labels)), 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(summary_csv: Path, output_dir: Path | None = None) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        return
    output_dir = ensure_dir(output_dir or summary_csv.parent / "plots")

    for value_key, filename, title, ylabel in [
        ("best_test_acc", "best_test_acc_by_trust_mode.png", "Best Test Accuracy by Trust Mode", "Best Test Accuracy"),
        ("final_test_acc", "final_test_acc_by_trust_mode.png", "Final Test Accuracy by Trust Mode", "Final Test Accuracy"),
        ("final_trust_mean", "final_trust_mean_by_trust_mode.png", "Final Trust Mean by Trust Mode", "Final Trust Mean"),
    ]:
        means = grouped_mean(rows, value_key, ("sample_trust_mode",))
        labels = sorted(key[0] for key in means)
        bar_plot(labels, [means[(label,)] for label in labels], output_dir / filename, title, ylabel)

    means = grouped_mean(rows, "best_test_acc", ("batch_size", "sample_trust_mode"))
    labels = sorted(means.keys(), key=lambda item: (int(item[0]) if item[0].isdigit() else item[0], item[1]))
    bar_labels = [f"bs{batch}\n{mode}" for batch, mode in labels]
    bar_plot(
        bar_labels,
        [means[label] for label in labels],
        output_dir / "best_test_acc_by_batch_size_and_trust_mode.png",
        "Best Test Accuracy by Batch Size and Trust Mode",
        "Best Test Accuracy",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sample trust run and aggregate metrics.")
    parser.add_argument("--metrics-csv", type=str, default="")
    parser.add_argument("--summary-csv", type=str, default="results/sample_trust/reports/summary.csv")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_csv and not args.aggregate_only:
        metrics_csv = Path(args.metrics_csv)
        output_dir = Path(args.output_dir) if args.output_dir else metrics_csv.parent / "plots"
        plot_run(metrics_csv, output_dir)
    summary_csv = Path(args.summary_csv)
    if summary_csv.exists():
        aggregate_dir = Path(args.output_dir) if args.output_dir and args.aggregate_only else summary_csv.parent / "plots"
        plot_aggregate(summary_csv, aggregate_dir)


if __name__ == "__main__":
    main()
