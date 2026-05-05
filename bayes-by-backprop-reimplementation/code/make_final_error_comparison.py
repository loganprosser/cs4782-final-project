from __future__ import annotations

import csv
from pathlib import Path

from summarize_results import ensure_dir, svg_multiline_text, wrap_label


TARGET_ROWS = [
    ("standard_trust_none", "Standard MLP h400/L2"),
    ("standard_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "Standard MLP + Kalman h400/L2"),
    ("dropout_trust_none", "Dropout MLP h400/L2"),
    ("dropout_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "Dropout MLP + Kalman h400/L2"),
    ("bayesian_trust_none", "BBB h400/L2"),
    ("bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "BBB + Kalman h400/L2"),
    ("bayesian_dropout_0p1_trust_none", "BBB + Dropout h400/L2"),
    ("bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0", "BBB + Dropout + Kalman h400/L2"),
    ("standard_trust_none_arch_h800_l2", "Standard MLP h800/L2"),
    ("standard_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0_arch_h800_l2", "Standard MLP + Kalman h800/L2"),
]


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["group"]: row for row in csv.DictReader(handle)}


def write_error_bar_svg(rows: list[dict[str, float | str]], output_path: Path) -> None:
    width = 1320
    row_height = 58
    height = 130 + row_height * len(rows)
    left = 430
    right = 130
    top = 70
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_error = max(float(row["mean_error"]) + float(row["std_error"]) for row in rows) * 1.2
    max_error = max(max_error, 0.025)

    def x_coord(value: float) -> float:
        return left + (value / max_error) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">Mean MNIST Test Error Across Seeds</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]
    colors = ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"]
    bar_height = 24
    for idx, row in enumerate(rows):
        y = top + (idx + 0.5) * (plot_height / len(rows))
        mean_error = float(row["mean_error"])
        std_error = float(row["std_error"])
        x_mean = x_coord(mean_error)
        x_low = x_coord(max(0.0, mean_error - std_error))
        x_high = x_coord(mean_error + std_error)
        color = colors[idx % len(colors)]
        svg.extend(svg_multiline_text(left - 18, y - 8, wrap_label(str(row["label"]), 34), anchor="end", font_size=13))
        svg.append(f'<rect x="{left:.1f}" y="{y - bar_height / 2:.1f}" width="{x_mean - left:.1f}" height="{bar_height}" fill="{color}" rx="5"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y - 8:.1f}" x2="{x_low:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y - 8:.1f}" x2="{x_high:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(
            f'<text x="{x_high + 10:.1f}" y="{y + 5:.1f}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#222222">{mean_error:.4f} +/- {std_error:.4f}</text>'
        )

    for tick_idx in range(6):
        value = max_error * tick_idx / 5
        x = x_coord(value)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{value:.3f}</text>')
        if tick_idx < 5:
            svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e7e1d7" stroke-width="1"/>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_accuracy_bar_svg(rows: list[dict[str, float | str]], output_path: Path) -> None:
    width = 1320
    row_height = 58
    height = 130 + row_height * len(rows)
    left = 430
    right = 130
    top = 70
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min = 0.977
    x_max = 0.984

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">Mean MNIST Test Accuracy Across Seeds</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]
    colors = ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"]
    bar_height = 24
    for idx, row in enumerate(rows):
        y = top + (idx + 0.5) * (plot_height / len(rows))
        mean_accuracy = float(row["mean_accuracy"])
        std_accuracy = float(row["std_accuracy"])
        x_mean = x_coord(mean_accuracy)
        x_low = x_coord(max(x_min, mean_accuracy - std_accuracy))
        x_high = x_coord(min(x_max, mean_accuracy + std_accuracy))
        color = colors[idx % len(colors)]
        svg.extend(svg_multiline_text(left - 18, y - 8, wrap_label(str(row["label"]), 34), anchor="end", font_size=13))
        svg.append(f'<rect x="{left:.1f}" y="{y - bar_height / 2:.1f}" width="{x_mean - left:.1f}" height="{bar_height}" fill="{color}" rx="5"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y - 8:.1f}" x2="{x_low:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y - 8:.1f}" x2="{x_high:.1f}" y2="{y + 8:.1f}" stroke="#111111" stroke-width="2"/>')
        svg.append(
            f'<text x="{x_high + 10:.1f}" y="{y + 5:.1f}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#222222">{mean_accuracy:.4f} +/- {std_accuracy:.4f}</text>'
        )

    for tick_idx in range(8):
        value = x_min + (x_max - x_min) * tick_idx / 7
        x = x_coord(value)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{value:.3f}</text>')
        if tick_idx < 7:
            svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e7e1d7" stroke-width="1"/>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root_dir / "classe_plots" / "final_comparison")
    sweep_rows = read_summary(root_dir / "classe_plots" / "kalman_sweep_summary" / "kalman_sweep_summary.csv")
    wide_rows = read_summary(root_dir / "classe_plots" / "wide_standard_confirm" / "summary" / "kalman_sweep_summary.csv")
    merged = {**sweep_rows, **wide_rows}

    rows = []
    lines = [
        "# Final Multi-Seed Error Comparison",
        "",
        "| Model | Runs | Mean Error | Std Error | Mean Accuracy | Best Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in TARGET_ROWS:
        if key not in merged:
            continue
        row = merged[key]
        if int(row["runs"]) < 2:
            continue
        rows.append(
            {
                "label": label,
                "runs": int(row["runs"]),
                "mean_error": float(row["mean_error"]),
                "std_error": float(row["std_error"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "std_accuracy": float(row["std_accuracy"]),
            }
        )
        lines.append(
            f"| {label} | {row['runs']} | {float(row['mean_error']):.4f} | {float(row['std_error']):.4f} | {float(row['mean_accuracy']):.4f} | {float(row['best_accuracy']):.4f} |"
        )

    write_error_bar_svg(rows, output_dir / "final_multiseed_error_comparison.svg")
    write_accuracy_bar_svg(rows, output_dir / "final_multiseed_accuracy_comparison.svg")
    (output_dir / "final_error_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'final_multiseed_error_comparison.svg'}")
    print(f"Wrote {output_dir / 'final_multiseed_accuracy_comparison.svg'}")


if __name__ == "__main__":
    main()
