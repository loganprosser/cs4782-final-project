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
        nlls = [float(item["final_test_nll"]) for item in items]
        eces = [float(item["final_test_ece"]) for item in items]
        out.append(
            {
                "optimizer_variant": variant,
                "runs": len(items),
                "mean_best_val_test_accuracy": statistics.mean(accs),
                "std_best_val_test_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "mean_test_nll": statistics.mean(nlls),
                "mean_test_ece": statistics.mean(eces),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AdamW Kalman trust optimizer results.")
    parser.add_argument("--results-dir", type=str, default="kalman_trust_results")
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = ensure_dir(root_dir / args.results_dir)
    rows = read_rows(results_dir / "summary.csv")
    summary = grouped(rows)
    write_csv(results_dir / "summary_by_variant.csv", summary)
    bar_plot(summary, results_dir / "plots" / "mean_accuracy_by_optimizer.png", "mean_best_val_test_accuracy", "Accuracy", "Mean Test Accuracy by Optimizer Variant")
    bar_plot(summary, results_dir / "plots" / "mean_nll_by_optimizer.png", "mean_test_nll", "NLL", "Mean Test NLL by Optimizer Variant")
    bar_plot(summary, results_dir / "plots" / "mean_ece_by_optimizer.png", "mean_test_ece", "ECE", "Mean Test ECE by Optimizer Variant")
    lines = ["# Kalman Trust Optimizer Summary", "", "## Ranked Variants"]
    for row in summary:
        lines.append(
            f"- `{row['optimizer_variant']}`: accuracy `{float(row['mean_best_val_test_accuracy']):.4f}` +/- `{float(row['std_best_val_test_accuracy']):.4f}` over `{row['runs']}` runs; NLL `{float(row['mean_test_nll']):.4f}`, ECE `{float(row['mean_test_ece']):.4f}`."
        )
    (results_dir / "summary_by_variant.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summaries under {results_dir.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
