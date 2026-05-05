from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import ensure_dir


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    by_variant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_variant.setdefault(row["optimizer_variant"], []).append(row)
    out = []
    for variant, items in by_variant.items():
        accs = [float(item["best_val_test_accuracy"]) for item in items]
        errors = [1.0 - acc for acc in accs]
        nlls = [float(item["final_test_nll"]) for item in items]
        eces = [float(item["final_test_ece"]) for item in items]
        out.append(
            {
                "optimizer_variant": variant,
                "runs": len(items),
                "mean_best_val_test_accuracy": statistics.mean(accs),
                "std_best_val_test_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "mean_best_val_test_error": statistics.mean(errors),
                "std_best_val_test_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "mean_test_nll": statistics.mean(nlls),
                "std_test_nll": statistics.stdev(nlls) if len(nlls) > 1 else 0.0,
                "mean_test_ece": statistics.mean(eces),
                "std_test_ece": statistics.stdev(eces) if len(eces) > 1 else 0.0,
            }
        )
    return sorted(out, key=lambda row: float(row["mean_best_val_test_accuracy"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bar_plot(rows: list[dict[str, float | int | str]], output_path: Path, key: str, ylabel: str, title: str) -> None:
    labels = [str(row["optimizer_variant"]).replace("_", "\n") for row in rows]
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2a6f97")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    if "accuracy" in key:
        ax.set_ylim(max(0.0, min(values) - 0.01), min(1.0, max(values) + 0.005))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def errorbar_plot(
    summary_rows: list[dict[str, float | int | str]],
    raw_rows: list[dict[str, str]],
    output_path: Path,
    value_key: str,
    std_key: str,
    raw_key: str,
    ylabel: str,
    title: str,
    invert_raw: bool = False,
) -> None:
    labels = [str(row["optimizer_variant"]).replace("_", "\n") for row in summary_rows]
    values = [float(row[value_key]) for row in summary_rows]
    errors = [float(row[std_key]) for row in summary_rows]
    order = [str(row["optimizer_variant"]) for row in summary_rows]
    by_variant: dict[str, list[float]] = {variant: [] for variant in order}
    for row in raw_rows:
        value = float(row[raw_key])
        by_variant[row["optimizer_variant"]].append(1.0 - value if invert_raw else value)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_positions = list(range(len(summary_rows)))
    ax.bar(x_positions, values, yerr=errors, capsize=4, color="#2a6f97", alpha=0.86)
    for x_pos, variant in zip(x_positions, order, strict=True):
        seed_values = by_variant.get(variant, [])
        if not seed_values:
            continue
        if len(seed_values) == 1:
            offsets = [0.0]
        else:
            step = 0.12 if len(seed_values) <= 3 else 0.08
            center = (len(seed_values) - 1) / 2.0
            offsets = [(idx - center) * step for idx in range(len(seed_values))]
        ax.scatter(
            [x_pos + offset for offset in offsets],
            seed_values,
            color="#bc4749",
            edgecolor="white",
            linewidth=0.5,
            s=34,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ymin = max(0.0, min(min(values), *(value - err for value, err in zip(values, errors, strict=True))) - 0.01)
    ymax = min(1.0, max(max(values), *(value + err for value, err in zip(values, errors, strict=True))) + 0.01)
    if ymax > ymin:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AdamW Kalman trust optimizer results.")
    parser.add_argument("--results-dir", type=str, default="kalman_trust_results")
    parser.add_argument("--epochs", type=int, default=0, help="Only summarize rows with this epoch count; 0 uses all rows.")
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = ensure_dir(root_dir / args.results_dir)
    rows = read_rows(results_dir / "summary.csv")
    if args.epochs:
        rows = [row for row in rows if int(float(row.get("epochs", 0))) == args.epochs]
        if not rows:
            raise RuntimeError(f"No rows in {results_dir / 'summary.csv'} with epochs={args.epochs}")
    summary = grouped(rows)
    suffix = f"_epochs{args.epochs}" if args.epochs else ""
    plots_dir = ensure_dir(results_dir / "plots")
    write_csv(results_dir / f"summary_by_variant{suffix}.csv", summary)
    bar_plot(summary, plots_dir / f"mean_accuracy_by_optimizer{suffix}.png", "mean_best_val_test_accuracy", "Accuracy", "Mean Test Accuracy by Optimizer Variant")
    bar_plot(summary, plots_dir / f"mean_nll_by_optimizer{suffix}.png", "mean_test_nll", "NLL", "Mean Test NLL by Optimizer Variant")
    bar_plot(summary, plots_dir / f"mean_ece_by_optimizer{suffix}.png", "mean_test_ece", "ECE", "Mean Test ECE by Optimizer Variant")
    errorbar_plot(
        summary,
        rows,
        plots_dir / f"mean_accuracy_by_optimizer_errorbars{suffix}.png",
        "mean_best_val_test_accuracy",
        "std_best_val_test_accuracy",
        "best_val_test_accuracy",
        "Accuracy",
        "Mean Test Accuracy by Optimizer Variant",
    )
    errorbar_plot(
        summary,
        rows,
        plots_dir / f"mean_error_by_optimizer_errorbars{suffix}.png",
        "mean_best_val_test_error",
        "std_best_val_test_error",
        "best_val_test_accuracy",
        "Error",
        "Mean Test Error by Optimizer Variant",
        invert_raw=True,
    )
    lines = ["# Kalman Trust Optimizer Summary", "", "## Ranked Variants"]
    if args.epochs:
        lines.insert(1, f"\nFiltered to rows with `epochs={args.epochs}`.")
    for row in summary:
        lines.append(
            f"- `{row['optimizer_variant']}`: accuracy `{float(row['mean_best_val_test_accuracy']):.4f}` +/- `{float(row['std_best_val_test_accuracy']):.4f}` over `{row['runs']}` runs; NLL `{float(row['mean_test_nll']):.4f}`, ECE `{float(row['mean_test_ece']):.4f}`."
        )
    (results_dir / f"summary_by_variant{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summaries under {results_dir.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
