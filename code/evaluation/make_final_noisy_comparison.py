from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_results import ensure_dir, svg_multiline_text, wrap_label


TARGET_ROWS = [
    ("standard_trust_none", "Standard MLP h400/L2"),
    ("standard_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "Standard MLP + Kalman h400/L2"),
    ("standard_trust_none_arch_h800_l2_wide", "Standard MLP h800/L2"),
    ("standard_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0_arch_h800_l2_wide", "Standard MLP + Kalman h800/L2"),
    ("dropout_trust_none", "Dropout MLP h400/L2"),
    ("dropout_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "Dropout MLP + Kalman h400/L2"),
    ("bayesian_trust_none", "BBB h400/L2"),
    ("bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "BBB + Kalman h400/L2"),
    ("bayesian_dropout_0p1_trust_none", "BBB + Dropout h400/L2"),
    ("bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "BBB + Dropout + Kalman h400/L2"),
]


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["group"]: row for row in csv.DictReader(handle)}


def value_range(values: list[float], stds: list[float], pad: float) -> tuple[float, float]:
    low = min(value - std for value, std in zip(values, stds, strict=True))
    high = max(value + std for value, std in zip(values, stds, strict=True))
    return max(0.0, low - pad), min(1.0, high + pad)


def write_horizontal_error_bar_svg(
    rows: list[dict[str, float | str]],
    output_path: Path,
    metric: str,
    title: str,
    x_min: float,
    x_max: float,
) -> None:
    width = 1360
    row_height = 58
    height = 130 + row_height * len(rows)
    left = 440
    right = 150
    top = 70
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]
    colors = ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"]
    bar_height = 24
    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    for idx, row in enumerate(rows):
        y = top + (idx + 0.5) * (plot_height / len(rows))
        mean = float(row[mean_key])
        std = float(row[std_key])
        x_mean = x_coord(mean)
        x_low = x_coord(max(x_min, mean - std))
        x_high = x_coord(min(x_max, mean + std))
        color = colors[idx % len(colors)]
        bar_x = min(left, x_mean)
        bar_width = abs(x_mean - left)
        svg.extend(svg_multiline_text(left - 18, y - 8, wrap_label(str(row["label"]), 34), anchor="end", font_size=13))
        svg.append(f'<rect x="{bar_x:.1f}" y="{y - bar_height / 2:.1f}" width="{bar_width:.1f}" height="{bar_height}" fill="{color}" rx="5"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y - 8:.1f}" x2="{x_low:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y - 8:.1f}" x2="{x_high:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(
            f'<text x="{min(width - 110, x_high + 10):.1f}" y="{y + 5:.1f}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#222222">{mean:.4f} +/- {std:.4f}</text>'
        )

    for tick_idx in range(7):
        value = x_min + (x_max - x_min) * tick_idx / 6
        x = x_coord(value)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{value:.3f}</text>')
        if tick_idx < 6:
            svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e7e1d7" stroke-width="1"/>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def collect_rows(summary: dict[str, dict[str, str]], min_runs: int) -> list[dict[str, float | str]]:
    rows = []
    for key, label in TARGET_ROWS:
        if key not in summary:
            continue
        row = summary[key]
        if int(row["runs"]) < min_runs:
            continue
        rows.append(
            {
                "label": label,
                "runs": int(row["runs"]),
                "mean_error": float(row["mean_error"]),
                "std_error": float(row["std_error"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "std_accuracy": float(row["std_accuracy"]),
                "best_accuracy": float(row["best_accuracy"]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make final noisy FashionMNIST multi-seed comparison plots.")
    parser.add_argument("--summary-csv", type=str, default="results/final_noisy_bbb_kalman/summary/kalman_sweep_summary.csv")
    parser.add_argument("--output-dir", type=str, default="results/final_noisy_bbb_kalman/final_comparison")
    parser.add_argument("--min-runs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    summary_path = root_dir / args.summary_csv
    output_dir = ensure_dir(root_dir / args.output_dir)
    summary = read_summary(summary_path)
    rows = collect_rows(summary, args.min_runs)
    if not rows:
        raise RuntimeError(f"No matching rows with runs >= {args.min_runs} in {summary_path}")

    accuracy_values = [float(row["mean_accuracy"]) for row in rows]
    accuracy_stds = [float(row["std_accuracy"]) for row in rows]
    error_values = [float(row["mean_error"]) for row in rows]
    error_stds = [float(row["std_error"]) for row in rows]
    acc_min, acc_max = value_range(accuracy_values, accuracy_stds, pad=0.01)
    err_min, err_max = value_range(error_values, error_stds, pad=0.01)

    write_horizontal_error_bar_svg(
        rows,
        output_dir / "final_noisy_multiseed_accuracy_comparison.svg",
        "accuracy",
        "Mean Noisy FashionMNIST Test Accuracy Across Seeds",
        acc_min,
        acc_max,
    )
    write_horizontal_error_bar_svg(
        rows,
        output_dir / "final_noisy_multiseed_error_comparison.svg",
        "error",
        "Mean Noisy FashionMNIST Test Error Across Seeds",
        err_min,
        err_max,
    )

    lines = [
        "# Final Noisy Multi-Seed Comparison",
        "",
        "| Model | Runs | Mean Error | Std Error | Mean Accuracy | Std Accuracy | Best Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['runs']} | {float(row['mean_error']):.4f} | {float(row['std_error']):.4f} | {float(row['mean_accuracy']):.4f} | {float(row['std_accuracy']):.4f} | {float(row['best_accuracy']):.4f} |"
        )
    (output_dir / "final_noisy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote noisy final comparison plots under {output_dir.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
