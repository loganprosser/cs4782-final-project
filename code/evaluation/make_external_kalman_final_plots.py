from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

from make_final_poster_plots import FINAL_COMPARISON_COLORS, parse_final_comparison_rows, write_final_multiseed_plot
from summarize_results import ensure_dir


def load_external_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.values())


def model_label(payload: dict) -> str:
    model = payload["model"]
    dropout = float(payload.get("bayesian_dropout") or 0.0)
    hidden_dim = int(payload.get("hidden_dim") or 400)
    hidden_layers = int(payload.get("hidden_layers") or 2)
    diag = payload.get("diag_mode", "tensor")

    if model == "standard":
        label = "Standard MLP"
    elif model == "dropout":
        label = "Dropout MLP"
    elif model == "bayesian" and dropout > 0.0:
        label = "BBB + Dropout"
    elif model == "bayesian":
        label = "BBB"
    else:
        label = str(model)
    return f"{label} + kalmanalgo {diag} h{hidden_dim}/L{hidden_layers}"


def summarize_external_rows(runs: list[dict], *, dataset: str, noisy: bool) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict]] = {}
    for payload in runs:
        if payload.get("dataset") != dataset:
            continue
        label_noise = float(payload.get("label_noise") or 0.0)
        if noisy and label_noise <= 0.0:
            continue
        if not noisy and label_noise != 0.0:
            continue
        grouped.setdefault(model_label(payload), []).append(payload)

    rows: list[dict[str, float | int | str]] = []
    for label, items in grouped.items():
        accuracies = [float(item["test_metrics"]["accuracy"]) for item in items]
        errors = [float(item["test_metrics"]["error_rate"]) for item in items]
        rows.append(
            {
                "model": label,
                "runs": len(items),
                "mean_error": statistics.mean(errors),
                "std_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "best_accuracy": max(accuracies),
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_accuracy"]), reverse=True)


def write_markdown(rows: list[dict[str, float | int | str]], path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Model | Runs | Mean Error | Std Error | Mean Accuracy | Std Accuracy | Best Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['runs']} | {float(row['mean_error']):.4f} | "
            f"{float(row['std_error']):.4f} | {float(row['mean_accuracy']):.4f} | "
            f"{float(row['std_accuracy']):.4f} | {float(row['best_accuracy']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def padded_range(values: list[float], spreads: list[float], pad: float, lo_floor: float = 0.0, hi_ceil: float = 1.0) -> tuple[float, float]:
    lo = min(value - spread for value, spread in zip(values, spreads, strict=True))
    hi = max(value + spread for value, spread in zip(values, spreads, strict=True))
    lo = max(lo_floor, lo - pad)
    hi = min(hi_ceil, hi + pad)
    if hi - lo < pad * 4:
        mid = (hi + lo) / 2
        lo = max(lo_floor, mid - pad * 2)
        hi = min(hi_ceil, mid + pad * 2)
    return lo, hi


def register_external_colors(rows: list[dict[str, float | int | str]]) -> None:
    palette = ["#5f6c7b", "#7f5539", "#2f7d57", "#8b5cf6"]
    for idx, row in enumerate(rows):
        FINAL_COMPARISON_COLORS.setdefault(str(row["model"]), palette[idx % len(palette)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy final plots and append external kalmanalgo experiment rows.")
    parser.add_argument("--external-json", type=str, default="results/bayes_by_backprop/external_kalman_algo/test_accuracy.json")
    parser.add_argument("--output-dir", type=str, default="final_plots_with_external_kalman")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    final_dir = root / "final_plots"
    output_dir = ensure_dir(root / args.output_dir)
    if final_dir.exists():
        shutil.copytree(final_dir, output_dir, dirs_exist_ok=True)

    runs = load_external_runs(root / args.external_json)
    clean_base = parse_final_comparison_rows(root / "classe_plots" / "final_comparison" / "final_error_comparison.md")
    noisy_base = parse_final_comparison_rows(root / "classe_plots" / "final_noisy_comparison" / "final_noisy_comparison.md")
    clean_rows = clean_base + summarize_external_rows(runs, dataset="mnist", noisy=False)
    noisy_rows = noisy_base + summarize_external_rows(runs, dataset="fashion_mnist", noisy=True)
    register_external_colors(clean_rows + noisy_rows)

    write_markdown(clean_rows, output_dir / "final_error_comparison_with_external_kalman.md", "Final Multi-Seed Comparison With External Kalman")
    write_markdown(noisy_rows, output_dir / "final_noisy_comparison_with_external_kalman.md", "Final Noisy Multi-Seed Comparison With External Kalman")

    clean_acc_min, clean_acc_max = padded_range(
        [float(row["mean_accuracy"]) for row in clean_rows],
        [float(row["std_accuracy"]) for row in clean_rows],
        0.002,
    )
    clean_err_min, clean_err_max = padded_range(
        [float(row["mean_error"]) for row in clean_rows],
        [float(row["std_error"]) for row in clean_rows],
        0.002,
    )
    noisy_acc_min, noisy_acc_max = padded_range(
        [float(row["mean_accuracy"]) for row in noisy_rows],
        [float(row["std_accuracy"]) for row in noisy_rows],
        0.006,
    )
    noisy_err_min, noisy_err_max = padded_range(
        [float(row["mean_error"]) for row in noisy_rows],
        [float(row["std_error"]) for row in noisy_rows],
        0.006,
    )

    write_final_multiseed_plot(
        clean_rows,
        output_dir / "final_multiseed_accuracy_comparison.svg",
        metric_key="mean_accuracy",
        spread_key="std_accuracy",
        title="Final Multi-Seed Accuracy + kalmanalgo",
        x_label="Mean test accuracy",
        value_format="{:.4f}",
        x_min=clean_acc_min,
        x_max=clean_acc_max,
        subtitle="Existing final rows plus external kalmanalgo runs",
    )
    write_final_multiseed_plot(
        clean_rows,
        output_dir / "final_multiseed_error_comparison.svg",
        metric_key="mean_error",
        spread_key="std_error",
        title="Final Multi-Seed Error + kalmanalgo",
        x_label="Mean test error",
        value_format="{:.4f}",
        x_min=clean_err_min,
        x_max=clean_err_max,
        lower_is_better=True,
        subtitle="Existing final rows plus external kalmanalgo runs",
    )
    write_final_multiseed_plot(
        noisy_rows,
        output_dir / "final_noisy_multiseed_accuracy_comparison.svg",
        metric_key="mean_accuracy",
        spread_key="std_accuracy",
        title="Final Noisy Multi-Seed Accuracy + kalmanalgo",
        x_label="Mean noisy FashionMNIST test accuracy",
        value_format="{:.4f}",
        x_min=noisy_acc_min,
        x_max=noisy_acc_max,
        subtitle="Existing final rows plus external kalmanalgo runs",
    )
    write_final_multiseed_plot(
        noisy_rows,
        output_dir / "final_noisy_multiseed_error_comparison.svg",
        metric_key="mean_error",
        spread_key="std_error",
        title="Final Noisy Multi-Seed Error + kalmanalgo",
        x_label="Mean noisy FashionMNIST test error",
        value_format="{:.4f}",
        x_min=noisy_err_min,
        x_max=noisy_err_max,
        lower_is_better=True,
        subtitle="Existing final rows plus external kalmanalgo runs",
    )
    print(f"Wrote copied final plots with external Kalman rows under {output_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
