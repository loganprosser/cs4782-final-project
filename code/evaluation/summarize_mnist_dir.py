from __future__ import annotations

import argparse
from pathlib import Path

from summarize_results import (
    display_name,
    ensure_dir,
    load_json,
    load_mnist_rows_from_metrics,
    write_svg_bar_chart,
    write_svg_line_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one standalone MNIST results directory.")
    parser.add_argument("--mnist-dir", type=str, required=True)
    parser.add_argument("--summary-dir", type=str, required=True)
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    mnist_dir = root_dir / args.mnist_dir
    summary_dir = ensure_dir(root_dir / args.summary_dir)
    metrics_path = mnist_dir / "test_accuracy.json"

    mnist_rows = load_mnist_rows_from_metrics(metrics_path, args.selection_mode)
    if not mnist_rows:
        raise RuntimeError(f"No MNIST metrics found at {metrics_path}")

    write_svg_bar_chart(
        labels=[display_name(str(row["Model"])) for row in mnist_rows],
        values=[float(row["Test Accuracy"]) for row in mnist_rows],
        output_path=summary_dir / "mnist_accuracy_summary.svg",
        title="MNIST Test Accuracy",
        value_format="{:.4f}",
        colors=["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.9,
        y_max=1.0,
    )
    write_svg_bar_chart(
        labels=[display_name(str(row["Model"])) for row in mnist_rows],
        values=[float(row["Test Error"]) for row in mnist_rows],
        output_path=summary_dir / "mnist_error_summary.svg",
        title="MNIST Test Error",
        value_format="{:.4f}",
        colors=["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )

    metrics = load_json(metrics_path)
    colors = ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d", "#6f4e37", "#3d405b"]
    train_accuracy_series = []
    val_accuracy_series = []
    val_loss_series = []
    for idx, run_name in enumerate(metrics):
        history = metrics[run_name].get("history", {})
        color = colors[idx % len(colors)]
        if history.get("train_accuracy"):
            train_accuracy_series.append(
                {"label": display_name(run_name), "values": history["train_accuracy"], "color": color}
            )
        if history.get("val_accuracy"):
            val_accuracy_series.append(
                {"label": display_name(run_name), "values": history["val_accuracy"], "color": color}
            )
        if history.get("val_loss"):
            val_loss_series.append(
                {"label": display_name(run_name), "values": history["val_loss"], "color": color}
            )

    if train_accuracy_series:
        write_svg_line_chart(
            train_accuracy_series,
            summary_dir / "mnist_train_accuracy_comparison.svg",
            "MNIST Training Accuracy",
            "Accuracy",
            y_min=0.85,
            y_max=1.0,
        )
    if val_accuracy_series:
        write_svg_line_chart(
            val_accuracy_series,
            summary_dir / "mnist_val_accuracy_comparison.svg",
            "MNIST Validation Accuracy",
            "Accuracy",
            y_min=0.85,
            y_max=1.0,
        )
    if val_loss_series:
        write_svg_line_chart(
            val_loss_series,
            summary_dir / "mnist_val_loss_comparison.svg",
            "MNIST Validation Loss",
            "Loss",
        )

    best_row = max(mnist_rows, key=lambda row: float(row["Test Accuracy"]))
    lines = [
        "# Dropout Propagated-Uncertainty Summary",
        "",
        "## Runs",
    ]
    for row in mnist_rows:
        lines.append(
            f"- `{display_name(str(row['Model']))}`: accuracy `{float(row['Test Accuracy']):.4f}`, error `{float(row['Test Error']):.4f}`."
        )
    lines.extend(
        [
            "",
            "## Best Result",
            f"- Best saved run: `{display_name(str(best_row['Model']))}` at `{float(best_row['Test Accuracy']):.4f}` accuracy.",
            "",
            "## Interpretation",
            "- These runs do not use Bayes by Backprop. They use deterministic/dropout MLP weights, then apply propagated-uncertainty gradient trust before the optimizer step.",
        ]
    )
    (summary_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
