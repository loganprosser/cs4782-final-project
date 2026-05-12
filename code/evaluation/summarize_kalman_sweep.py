from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from summarize_results import ensure_dir, write_svg_bar_chart


SEED_SUFFIX_RE = re.compile(r"_s\d+$")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_name(run_name: str) -> str:
    return SEED_SUFFIX_RE.sub("", run_name)


def display_group(name: str, item: dict) -> str:
    model = item.get("model")
    dropout = float(item.get("bayesian_dropout") or 0.0)
    hidden_dim = int(item.get("hidden_dim") or 400)
    hidden_layers = int(item.get("hidden_layers") or 2)
    trust = item.get("update_trust_mode")

    if model == "standard":
        label = "Standard MLP"
    elif model == "dropout":
        label = "Dropout MLP"
    elif model == "bayesian" and dropout > 0.0:
        label = "BBB + Dropout"
    elif model == "bayesian":
        label = "BBB"
    else:
        label = name

    if trust == "kalman_layer":
        label += " + Kalman"
    elif trust == "none":
        label += " no trust"

    if hidden_dim != 400 or hidden_layers != 2:
        label += f" h{hidden_dim}/L{hidden_layers}"
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Kalman finalist seed/architecture sweep.")
    parser.add_argument("--mnist-dir", type=str, default="results/kalman_sweep/mnist")
    parser.add_argument("--summary-dir", type=str, default="results/kalman_sweep/summary")
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    mnist_dir = root_dir / args.mnist_dir
    summary_dir = ensure_dir(root_dir / args.summary_dir)
    metrics = load_json(mnist_dir / "test_accuracy.json")
    metric_key = "test_metrics" if args.selection_mode == "best_val" else "last_epoch_test_metrics"

    grouped: dict[str, list[dict]] = {}
    for run_name, payload in metrics.items():
        selected = payload.get(metric_key) or payload.get("test_metrics")
        grouped.setdefault(group_name(run_name), []).append(
            {
                "run_name": run_name,
                "seed": payload.get("seed"),
                "accuracy": float(selected["accuracy"]),
                "error": float(selected["error_rate"]),
                "best_epoch": payload.get("best_epoch"),
                "hidden_dim": payload.get("hidden_dim"),
                "hidden_layers": payload.get("hidden_layers"),
                "update_trust_mode": payload.get("update_trust_mode"),
                "model": payload.get("model"),
                "bayesian_dropout": payload.get("bayesian_dropout"),
            }
        )

    rows = []
    for name, items in grouped.items():
        accuracies = [item["accuracy"] for item in items]
        errors = [item["error"] for item in items]
        rows.append(
            {
                "group": name,
                "display": display_group(name, items[0]),
                "runs": len(items),
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "mean_error": statistics.mean(errors),
                "std_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "best_accuracy": max(accuracies),
                "worst_accuracy": min(accuracies),
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda item: str(item["seed"]))),
            }
        )

    rows.sort(key=lambda row: row["mean_accuracy"], reverse=True)
    with (summary_dir / "kalman_sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_rows = rows[:18]
    accuracy_values = [float(row["mean_accuracy"]) for row in plot_rows]
    accuracy_y_min = max(0.0, min(accuracy_values) - 0.02)
    accuracy_y_max = min(1.0, max(accuracy_values) + 0.01)
    write_svg_bar_chart(
        [row["display"] for row in plot_rows],
        accuracy_values,
        summary_dir / "kalman_sweep_mean_accuracy.svg",
        "Kalman Sweep Mean Test Accuracy",
        "{:.4f}",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=accuracy_y_min,
        y_max=accuracy_y_max,
    )
    write_svg_bar_chart(
        [row["display"] for row in plot_rows],
        [float(row["mean_error"]) for row in plot_rows],
        summary_dir / "kalman_sweep_mean_error.svg",
        "Kalman Sweep Mean Test Error",
        "{:.4f}",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )

    lines = ["# Kalman Sweep Summary", "", "## Ranked Mean Accuracy"]
    for row in rows:
        lines.append(
            f"- `{row['display']}`: mean accuracy `{row['mean_accuracy']:.4f}` +/- `{row['std_accuracy']:.4f}` over `{row['runs']}` run(s); best `{row['best_accuracy']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Reading",
            "- Multi-seed rows are the strongest evidence; single-seed deep/wide rows are architecture probes.",
            "- This sweep intentionally drops the underperforming grad-norm, grad-var, depth-decay, and propagated-uncertainty modes.",
        ]
    )
    (summary_dir / "kalman_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote sweep summary to {summary_dir.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
