from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np


ORDERED_MODES = [
    "confidence",
    "entropy",
    "inverse_loss",
    "final_layer_align",
    "final_layer_align_mag",
    "ema_final_layer_align",
    "combined_simple",
]
ORDERED_BATCHES = [16, 32, 64, 128, 256]
MODE_LABELS = {
    "confidence": "confidence",
    "entropy": "entropy",
    "inverse_loss": "inverse loss",
    "final_layer_align": "align",
    "final_layer_align_mag": "align + mag",
    "ema_final_layer_align": "ema align",
    "combined_simple": "combined",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def latest_sample_trust_prefix(rows: list[dict[str, str]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        match = re.match(r"(sample_trust_\d{8}_\d{6})_", row.get("run_name", ""))
        if match:
            counts[match.group(1)] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def filter_rows(rows: list[dict[str, str]], run_prefix: str) -> list[dict[str, str]]:
    if run_prefix:
        return [row for row in rows if row.get("run_name", "").startswith(f"{run_prefix}_")]
    return rows


def matched_deltas(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    baselines = {
        (row["model"], row["batch_size"], row["seed"]): row
        for row in rows
        if row.get("sample_trust_mode") == "none"
    }
    deltas: list[dict[str, str | float]] = []
    for row in rows:
        mode = row.get("sample_trust_mode", "")
        if mode == "none":
            continue
        key = (row.get("model", ""), row.get("batch_size", ""), row.get("seed", ""))
        baseline = baselines.get(key)
        if baseline is None:
            continue
        deltas.append(
            {
                "model": row.get("model", ""),
                "batch_size": row.get("batch_size", ""),
                "seed": row.get("seed", ""),
                "mode": mode,
                "best_test_acc": as_float(row, "best_test_acc"),
                "final_test_acc": as_float(row, "final_test_acc"),
                "baseline_best_test_acc": as_float(baseline, "best_test_acc"),
                "baseline_final_test_acc": as_float(baseline, "final_test_acc"),
                "best_delta": as_float(row, "best_test_acc") - as_float(baseline, "best_test_acc"),
                "final_delta": as_float(row, "final_test_acc") - as_float(baseline, "final_test_acc"),
                "final_trust_mean": as_float(row, "final_trust_mean"),
            }
        )
    return deltas


def write_delta_csv(path: Path, deltas: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "batch_size",
        "seed",
        "mode",
        "best_test_acc",
        "baseline_best_test_acc",
        "best_delta",
        "final_test_acc",
        "baseline_final_test_acc",
        "final_delta",
        "final_trust_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(deltas)


def batch_sizes_from_rows(rows: list[dict[str, str]]) -> list[int]:
    found = sorted({int(row["batch_size"]) for row in rows if row.get("batch_size", "").isdigit()})
    ordered = [batch for batch in ORDERED_BATCHES if batch in found]
    extras = [batch for batch in found if batch not in ordered]
    return ordered + extras


def delta_matrix(deltas: list[dict[str, str | float]], model: str, key: str, batch_sizes: list[int]) -> np.ndarray:
    values = np.full((len(ORDERED_MODES), len(batch_sizes)), np.nan)
    for item in deltas:
        if item["model"] != model or item["mode"] not in ORDERED_MODES:
            continue
        row = ORDERED_MODES.index(str(item["mode"]))
        col = batch_sizes.index(int(str(item["batch_size"])))
        values[row, col] = float(item[key])
    return values


def draw_heatmap(ax: plt.Axes, values: np.ndarray, title: str, batch_sizes: list[int], show_y: bool = True, vmax: float | None = None) -> plt.AxesImage:
    if not np.isfinite(values).any():
        values = np.zeros_like(values)
    if vmax is None:
        vmax = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.001)
    else:
        vmax = max(vmax, 0.001)
    image = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(batch_sizes)), [str(batch) for batch in batch_sizes])
    ax.set_yticks(range(len(ORDERED_MODES)), [MODE_LABELS[mode] for mode in ORDERED_MODES] if show_y else [])
    ax.set_xlabel("Batch size")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    return image


def best_mode_by_batch(deltas: list[dict[str, str | float]]) -> dict[str, dict[int, dict[str, str | float]]]:
    grouped: dict[str, dict[int, list[dict[str, str | float]]]] = defaultdict(lambda: defaultdict(list))
    for item in deltas:
        grouped[str(item["model"])][int(str(item["batch_size"]))].append(item)
    best: dict[str, dict[int, dict[str, str | float]]] = {}
    for model, by_batch in grouped.items():
        best[model] = {}
        for batch_size, items in by_batch.items():
            best[model][batch_size] = max(items, key=lambda item: float(item["best_delta"]))
    return best


def mode_means(deltas: list[dict[str, str | float]], model: str) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in deltas:
        if item["model"] == model:
            grouped[str(item["mode"])].append(float(item["best_delta"]))
    labels = [mode for mode in ORDERED_MODES if mode in grouped]
    values = [sum(grouped[mode]) / len(grouped[mode]) for mode in labels]
    return labels, values


def make_dashboard(rows: list[dict[str, str]], deltas: list[dict[str, str | float]], output_path: Path, run_prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model"] for row in rows})
    batch_sizes = batch_sizes_from_rows(rows)
    heatmap_cols = min(3, max(1, len(models)))
    heatmap_rows = int(np.ceil(len(models) / heatmap_cols))
    fig = plt.figure(figsize=(6 * heatmap_cols + 1.0, 4.2 * heatmap_rows + 8), constrained_layout=True)
    grid = fig.add_gridspec(
        heatmap_rows + 2,
        heatmap_cols + 1,
        width_ratios=[*[1.0] * heatmap_cols, 0.04],
        height_ratios=[*[1.0] * heatmap_rows, 0.9, 0.9],
    )
    fig.suptitle(f"Sample Trust Findings: {run_prefix}", fontsize=16)

    matrices = [delta_matrix(deltas, model, "best_delta", batch_sizes) for model in models]
    finite_parts = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    finite_values = np.concatenate(finite_parts) if finite_parts else np.array([])
    vmax = max(abs(float(finite_values.min())), abs(float(finite_values.max())), 0.001) if finite_values.size else 0.001
    image = None
    for index, model in enumerate(models):
        row = index // heatmap_cols
        col = index % heatmap_cols
        ax = fig.add_subplot(grid[row, col])
        image = draw_heatmap(
            ax,
            matrices[index],
            f"{model}: Best Accuracy Delta vs `none`",
            batch_sizes,
            show_y=col == 0,
            vmax=vmax,
        )
    for index in range(len(models), heatmap_rows * heatmap_cols):
        fig.add_subplot(grid[index // heatmap_cols, index % heatmap_cols]).axis("off")
    if image is not None:
        colorbar_axis = fig.add_subplot(grid[:heatmap_rows, heatmap_cols])
        cbar = fig.colorbar(image, cax=colorbar_axis)
        cbar.set_label("Accuracy delta")

    best_axis = fig.add_subplot(grid[heatmap_rows, :heatmap_cols])
    best = best_mode_by_batch(deltas)
    for model in models:
        batches = [batch for batch in batch_sizes if batch in best.get(model, {})]
        values = [float(best[model][batch]["best_delta"]) for batch in batches]
        labels = [str(best[model][batch]["mode"]) for batch in batches]
        best_axis.plot(batches, values, marker="o", linewidth=2, label=model)
        for batch, value, label in zip(batches, values, labels):
            best_axis.annotate(MODE_LABELS[label], (batch, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    best_axis.axhline(0.0, color="black", linewidth=1)
    best_axis.set_title("Best Trust Improvement at Each Batch Size")
    best_axis.set_xlabel("Batch size")
    best_axis.set_ylabel("Best accuracy delta")
    best_axis.set_xticks(batch_sizes)
    best_axis.grid(True, alpha=0.25)
    best_axis.legend(ncol=min(4, len(models)))

    mean_axis = fig.add_subplot(grid[heatmap_rows + 1, :heatmap_cols])
    mean_values = np.full((len(models), len(ORDERED_MODES)), np.nan)
    for model_index, model in enumerate(models):
        labels, values = mode_means(deltas, model)
        lookup = dict(zip(labels, values))
        mean_values[model_index, :] = [lookup.get(mode, np.nan) for mode in ORDERED_MODES]
    finite_means = mean_values[np.isfinite(mean_values)]
    mean_vmax = max(abs(float(finite_means.min())), abs(float(finite_means.max())), 0.001) if finite_means.size else 0.001
    mean_image = mean_axis.imshow(mean_values, cmap="RdYlGn", vmin=-mean_vmax, vmax=mean_vmax, aspect="auto")
    mean_axis.set_title("Average Best-Accuracy Delta Across Batch Sizes")
    mean_axis.set_xticks(range(len(ORDERED_MODES)), [MODE_LABELS[mode] for mode in ORDERED_MODES], rotation=30, ha="right")
    mean_axis.set_yticks(range(len(models)), models)
    for row in range(mean_values.shape[0]):
        for col in range(mean_values.shape[1]):
            value = mean_values[row, col]
            if np.isfinite(value):
                mean_axis.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=7)
    mean_cbar_axis = fig.add_subplot(grid[heatmap_rows + 1, heatmap_cols])
    mean_cbar = fig.colorbar(mean_image, cax=mean_cbar_axis)
    mean_cbar.set_label("Mean accuracy delta")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create combined sample trust comparison plots.")
    parser.add_argument("--summary-csv", type=str, default="results/sample_trust/reports/summary.csv")
    parser.add_argument("--run-prefix", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/sample_trust/reports/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    rows = read_rows(summary_csv)
    run_prefix = args.run_prefix or latest_sample_trust_prefix(rows)
    rows = filter_rows(rows, run_prefix)
    deltas = matched_deltas(rows)
    output_dir = Path(args.output_dir)
    write_delta_csv(output_dir / f"{run_prefix}_matched_deltas.csv", deltas)
    make_dashboard(rows, deltas, output_dir / f"{run_prefix}_combined_findings.png", run_prefix)
    print(f"Combined findings plot saved to {output_dir / f'{run_prefix}_combined_findings.png'}")
    print(f"Matched delta CSV saved to {output_dir / f'{run_prefix}_matched_deltas.csv'}")


if __name__ == "__main__":
    main()
