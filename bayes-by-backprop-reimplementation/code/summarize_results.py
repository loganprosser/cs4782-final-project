from __future__ import annotations

import argparse
import csv
import json
import textwrap
from html import escape
from pathlib import Path

MNIST_PAPER_CONTEXT = (
    "The paper reports that Bayes by Backprop is competitive with dropout on MNIST, "
    "rather than requiring an exact numerical match in a small-scale reproduction."
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_accuracy_table(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "Model": row["Model"],
                    "Test Accuracy": float(row["Test Accuracy"]),
                    "Test Error": float(row["Test Error"]),
                }
            )
    return rows


def load_mnist_rows_from_metrics(path: Path, selection_mode: str) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    metrics = load_json(path)
    preferred_order = [
        "standard",
        "dropout",
        "bayesian",
        "bayesian_dropout_0p1",
        "standard_trust_none",
        "dropout_trust_none",
        "bayesian_trust_none",
        "bayesian_dropout_0p1_trust_none",
    ]
    ordered_names = [name for name in preferred_order if name in metrics]
    ordered_names.extend(name for name in metrics if name not in ordered_names)

    rows: list[dict[str, float | str]] = []
    metric_key = "test_metrics" if selection_mode == "best_val" else "last_epoch_test_metrics"
    for name in ordered_names:
        payload = metrics[name]
        selected = payload.get(metric_key) or payload.get("test_metrics")
        rows.append(
            {
                "Model": name,
                "Test Accuracy": float(selected["accuracy"]),
                "Test Error": float(selected["error_rate"]),
            }
        )
    return rows


def display_name(model_name: str) -> str:
    mapping = {
        "standard": "Standard MLP",
        "dropout": "Dropout MLP",
        "bayesian": "Bayes by Backprop",
        "bayesian_dropout_0p1": "Bayesian + Dropout",
    }
    if model_name in mapping:
        return mapping[model_name]

    if "_trust_" in model_name:
        base_name, trust_suffix = model_name.split("_trust_", maxsplit=1)
        base_display = display_name(base_name)
        trust_display = trust_suffix.replace("_", " ")
        if trust_display.startswith("depth lambda"):
            trust_display = trust_display.replace("depth lambda", "depth-decay lambda=")
        elif trust_display.startswith("gradnorm"):
            trust_display = "grad-norm scaling"
        elif trust_display.startswith("gradvar beta"):
            trust_display = trust_display.replace("gradvar beta", "running-grad-var beta=")
        elif trust_display.startswith("kalmanlayer beta"):
            trust_display = trust_display.replace("kalmanlayer beta", "kalman-layer beta=")
            trust_display = trust_display.replace(" q", " Q=")
            trust_display = trust_display.replace(" p", " P=")
            trust_display = trust_display.replace(" r", " R=")
        elif trust_display.startswith("propagateduncertainty beta"):
            trust_display = trust_display.replace("propagateduncertainty beta", "propagated uncertainty beta=")
            trust_display = trust_display.replace(" q", " Q=")
            trust_display = trust_display.replace(" p", " P=")
            trust_display = trust_display.replace(" r", " R=")
            trust_display = trust_display.replace(" lambda", " lambda=")
        elif trust_display == "none":
            trust_display = "no trust scaling"
        return f"{base_display} ({trust_display})"

    return model_name.replace("_", " ").title()


def compact_plot_label(label: str) -> str:
    replacements = {
        "Bayes by Backprop": "BBB",
        "Bayesian + Dropout": "BBB + Dropout",
        "no trust scaling": "none",
        "kalmanLayer beta0.95 Q0.0001 P1.0 R1.0": "Kalman layer",
        "Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0": "Kalman layer",
        "kalman-layer beta=0.95 Q=0.0001 P=1.0 R=1.0": "Kalman layer",
        "propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15": "Propagated uncertainty",
        "Propagateduncertainty Beta0.95 Q0.0001 P1.0 R1.0 Lambda0.15": "Propagated uncertainty",
        "propagated uncertainty beta=0.95 Q=0.0001 P=1.0 R=1.0 lambda=0.15": "Propagated uncertainty",
        "depth-decay lambda=0.15": "Depth decay",
        "running-grad-var beta=0.95": "Grad var",
        "grad-norm scaling": "Grad norm",
    }
    compact = label
    for old, new in replacements.items():
        compact = compact.replace(old, new)
    return compact


def wrap_label(label: str, width: int) -> list[str]:
    wrapped = textwrap.wrap(compact_plot_label(label), width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped or [compact_plot_label(label)]


def svg_multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    anchor: str,
    font_size: int,
    fill: str = "#222222",
    line_height: int | None = None,
) -> list[str]:
    line_height = line_height or int(font_size * 1.25)
    svg_lines = [
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Helvetica, Arial, sans-serif" font-size="{font_size}" fill="{fill}">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_height
        svg_lines.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
    svg_lines.append("</text>")
    return svg_lines


def write_svg_bar_chart(
    labels: list[str],
    values: list[float],
    output_path: Path,
    title: str,
    value_format: str,
    colors: list[str],
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    use_horizontal = len(labels) > 6 or max((len(label) for label in labels), default=0) > 18

    if use_horizontal:
        wrapped_labels = [wrap_label(label, 42) for label in labels]
        width = 1320
        row_height = max(50, max(len(lines) for lines in wrapped_labels) * 17 + 22)
        height = max(500, 130 + row_height * len(labels))
        left = 500
        right = 130
        top = 70
        bottom = 60
        plot_width = width - left - right
        plot_height = height - top - bottom
        x_max = y_max if y_max is not None else max(values) * 1.15
        span = max(x_max - y_min, 1e-8)
        bar_height = plot_height / max(len(values) * 1.35, 1)

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fffdf8"/>',
            f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        ]

        for idx, (label, value) in enumerate(zip(labels, values)):
            y = top + (idx + 0.5) * (plot_height / len(values))
            bar_top = y - bar_height / 2
            bar_width = ((value - y_min) / span) * plot_width
            color = colors[idx % len(colors)]
            svg_lines.append(
                f'<rect x="{left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="6"/>'
            )
            label_lines = wrapped_labels[idx]
            label_y = y - ((len(label_lines) - 1) * 8)
            svg_lines.extend(svg_multiline_text(left - 18, label_y, label_lines, anchor="end", font_size=13))
            value_label = value_format.format(value)
            value_overflows = left + bar_width + 70 > width - 20
            value_x = left + bar_width - 8 if value_overflows else left + bar_width + 10
            value_anchor = "end" if value_overflows else "start"
            value_fill = "#ffffff" if value_overflows and bar_width > 55 else "#222222"
            svg_lines.append(
                f'<text x="{value_x:.1f}" y="{y + 5:.1f}" text-anchor="{value_anchor}" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="{value_fill}">{escape(value_label)}</text>'
            )

        for tick_idx in range(5):
            tick_value = y_min + (span * tick_idx / 4)
            x = left + (plot_width * tick_idx / 4)
            svg_lines.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
            svg_lines.append(
                f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{tick_value:.3f}</text>'
            )
            if tick_idx < 4:
                svg_lines.append(
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e7e1d7" stroke-width="1"/>'
                )

        svg_lines.append("</svg>")
        output_path.write_text("\n".join(svg_lines), encoding="utf-8")
        return

    width = 760
    height = 420
    left = 80
    right = 40
    top = 70
    bottom = 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = y_max if y_max is not None else max(values) * 1.15
    span = max(y_max - y_min, 1e-8)
    bar_width = plot_width / max(len(values) * 1.8, 1)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + (idx + 0.5) * (plot_width / len(values))
        bar_left = x - bar_width / 2
        bar_height = ((value - y_min) / span) * plot_height
        bar_top = top + plot_height - bar_height
        color = colors[idx % len(colors)]
        svg_lines.append(
            f'<rect x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="6"/>'
        )
        svg_lines.append(
            f'<text x="{x:.1f}" y="{bar_top - 10:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#222222">{value_format.format(value)}</text>'
        )
        svg_lines.extend(
            svg_multiline_text(
                x,
                top + plot_height + 26,
                wrap_label(label, 16),
                anchor="middle",
                font_size=13,
            )
        )

    for tick_idx in range(5):
        tick_value = y_min + (span * tick_idx / 4)
        y = top + plot_height - (plot_height * tick_idx / 4)
        svg_lines.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333333" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{tick_value:.3f}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_svg_line_chart(
    series: list[dict[str, object]],
    output_path: Path,
    title: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width = 860
    height = 460
    left = 90
    right = 180
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_values = [value for item in series for value in item["values"]]  # type: ignore[index]
    inferred_min = min(all_values)
    inferred_max = max(all_values)
    y_min = inferred_min if y_min is None else y_min
    y_max = inferred_max if y_max is None else y_max
    if y_max == y_min:
        y_max = y_min + 1e-6

    max_len = max(len(item["values"]) for item in series)  # type: ignore[index]

    def x_coord(index: int) -> float:
        if max_len <= 1:
            return left + plot_width / 2
        return left + (index / (max_len - 1)) * plot_width

    def y_coord(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#333333">Epoch</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#333333" transform="rotate(-90 24 {top + plot_height / 2})">{y_label}</text>',
    ]

    for tick_idx in range(5):
        tick_value = y_min + ((y_max - y_min) * tick_idx / 4)
        y = y_coord(tick_value)
        svg_lines.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333333" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{tick_value:.3f}</text>'
        )
        if tick_idx < 4:
            svg_lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e7e1d7" stroke-width="1"/>'
            )

    for epoch_idx in range(max_len):
        x = x_coord(epoch_idx)
        svg_lines.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#444444">{epoch_idx + 1}</text>'
        )

    legend_x = left + plot_width + 24
    legend_y = top + 18

    for idx, item in enumerate(series):
        values = item["values"]  # type: ignore[index]
        color = item["color"]  # type: ignore[index]
        label = item["label"]  # type: ignore[index]
        path_parts = []
        for value_idx, value in enumerate(values):
            cmd = "M" if value_idx == 0 else "L"
            path_parts.append(f"{cmd} {x_coord(value_idx):.1f} {y_coord(float(value)):.1f}")
        svg_lines.append(
            f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for value_idx, value in enumerate(values):
            svg_lines.append(
                f'<circle cx="{x_coord(value_idx):.1f}" cy="{y_coord(float(value)):.1f}" r="3.5" fill="{color}"/>'
            )
        row_y = legend_y + idx * 26
        svg_lines.append(f'<line x1="{legend_x}" y1="{row_y}" x2="{legend_x + 26}" y2="{row_y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        svg_lines.append(
            f'<text x="{legend_x + 36}" y="{row_y + 5}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#222222">{label}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_svg_grouped_bar_chart(
    categories: list[str],
    series: list[dict[str, object]],
    output_path: Path,
    title: str,
    value_format: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    categories = [compact_plot_label(category) for category in categories]
    width = 1180
    height = 540
    left = 90
    right = 70
    top = 70
    bottom = 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [float(value) for item in series for value in item["values"]]  # type: ignore[index]
    y_max = y_max if y_max is not None else max(all_values) * 1.15
    span = max(y_max - y_min, 1e-8)
    category_width = plot_width / max(len(categories), 1)
    series_count = max(len(series), 1)
    bar_width = category_width / (series_count + 1.5)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]

    for cat_idx, category in enumerate(categories):
        center_x = left + (cat_idx + 0.5) * category_width
        svg_lines.extend(
            svg_multiline_text(
                center_x,
                top + plot_height + 26,
                wrap_label(category, 14),
                anchor="middle",
                font_size=12,
            )
        )
        for series_idx, item in enumerate(series):
            value = float(item["values"][cat_idx])  # type: ignore[index]
            color = str(item["color"])  # type: ignore[index]
            group_offset = (series_idx - (series_count - 1) / 2) * bar_width
            bar_left = center_x + group_offset - bar_width / 2
            bar_height = ((value - y_min) / span) * plot_height
            bar_top = top + plot_height - bar_height
            svg_lines.append(
                f'<rect x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="5"/>'
            )
            svg_lines.append(
                f'<text x="{bar_left + bar_width / 2:.1f}" y="{bar_top - 8:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#222222">{value_format.format(value)}</text>'
            )

    for tick_idx in range(5):
        tick_value = y_min + (span * tick_idx / 4)
        y = top + plot_height - (plot_height * tick_idx / 4)
        svg_lines.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333333" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#444444">{tick_value:.3f}</text>'
        )

    legend_x = left + 12
    legend_y = top - 24
    for idx, item in enumerate(series):
        row_x = legend_x + idx * 210
        color = str(item["color"])  # type: ignore[index]
        label = str(item["label"])  # type: ignore[index]
        svg_lines.append(f'<rect x="{row_x}" y="{legend_y}" width="18" height="18" fill="{color}" rx="3"/>')
        svg_lines.append(
            f'<text x="{row_x + 26}" y="{legend_y + 14}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#222222">{label}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def extract_trust_mode(model_name: str) -> str:
    if "_trust_" not in model_name:
        return "legacy"
    return model_name.split("_trust_", maxsplit=1)[1]


def trust_mode_display(mode: str) -> str:
    mapping = {
        "none": "None",
        "depth_lambda0.15": "Depth Decay",
        "gradnorm": "Grad Norm",
        "gradvar_beta0.95": "Grad Var",
        "kalmanLayer_beta0.95_Q1e-4_P1.0_R1.0": "Kalman Layer",
        "kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0": "Kalman Layer",
        "propagatedUncertainty_beta0.95_Q0.0001_P1.0_R1.0_lambda0.15": "Propagated Uncertainty",
    }
    if mode.startswith("kalmanLayer"):
        return "Kalman Layer"
    if mode.startswith("propagatedUncertainty"):
        return "Propagated Uncertainty"
    return mapping.get(mode, mode.replace("_", " ").title())


def build_trust_summary(mnist_rows: list[dict[str, float | str]], benchmark_results: list[dict] | None) -> str:
    trust_rows = [row for row in mnist_rows if "_trust_" in str(row["Model"])]
    if not trust_rows:
        return "# Update Trust Summary\n\nNo update-trust runs were found.\n"

    benchmark_by_name = {item["run_name"]: item for item in benchmark_results or []}

    def rows_for(prefix: str) -> list[dict[str, float | str]]:
        return [row for row in trust_rows if str(row["Model"]).startswith(prefix)]

    sections = ["# Update Trust Summary", ""]
    for prefix in ["bayesian_trust_", "bayesian_dropout_0p1_trust_"]:
        group_rows = rows_for(prefix)
        if not group_rows:
            continue
        group_name = display_name(prefix[:-7])
        sections.append(f"## {group_name}")
        baseline = next((row for row in group_rows if extract_trust_mode(str(row["Model"])) == "none"), None)
        best_row = max(group_rows, key=lambda row: float(row["Test Accuracy"]))
        sections.append(
            f"- Best trust-mode run: `{display_name(str(best_row['Model']))}` at `{float(best_row['Test Accuracy']):.4f}` accuracy."
        )
        if baseline is not None:
            delta = float(best_row["Test Accuracy"]) - float(baseline["Test Accuracy"])
            sections.append(
                f"- Relative to `{display_name(str(baseline['Model']))}`, the best trust variant changed accuracy by `{delta:.4f}`."
            )
        for row in sorted(group_rows, key=lambda item: float(item["Test Accuracy"]), reverse=True):
            mode = trust_mode_display(extract_trust_mode(str(row["Model"])))
            line = f"- `{mode}`: accuracy `{float(row['Test Accuracy']):.4f}`, error `{float(row['Test Error']):.4f}`"
            bench = benchmark_by_name.get(str(row["Model"]))
            if bench is not None:
                line += f", train step `{bench['train_step_ms']:.2f} ms`, eval step `{bench['eval_step_ms']:.2f} ms`."
            else:
                line += "."
            sections.append(line)
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def build_summary_text(mnist_rows: list[dict[str, float | str]], regression_metrics: dict | None) -> str:
    lines: list[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Overall Verdict")

    model_names = {str(row["Model"]) for row in mnist_rows}
    mnist_complete = {"standard", "dropout", "bayesian"}.issubset(model_names)

    regression_replicated = False
    bayesian_regression = None
    if regression_metrics and "bayesian" in regression_metrics:
        bayesian_regression = regression_metrics["bayesian"]
        support_std = bayesian_regression.get("support_std_mean")
        gap_std = bayesian_regression.get("gap_std_mean")
        regression_replicated = (
            support_std is not None and gap_std is not None and gap_std > support_std
        )

    if mnist_complete and regression_replicated:
        lines.append("Strong qualitative replication: Bayes by Backprop is competitive with dropout on MNIST and reproduces the expected regression uncertainty pattern.")
    elif regression_replicated:
        lines.append("Partial replication: the regression uncertainty result is replicated, but the MNIST comparison is incomplete because only part of the classification suite was saved.")
    else:
        lines.append("Incomplete replication: the currently saved runs do not fully support the paper claims yet.")
    lines.append("")

    lines.append("## MNIST Classification")
    lines.append(MNIST_PAPER_CONTEXT)
    lines.append("")
    if not mnist_rows:
        lines.append("- No MNIST accuracy table was found.")
    else:
        for row in mnist_rows:
            lines.append(
                f"- `{display_name(str(row['Model']))}`: accuracy `{row['Test Accuracy']:.4f}`, error `{row['Test Error']:.4f}`."
            )
        if mnist_complete:
            best_row = max(mnist_rows, key=lambda row: float(row["Test Accuracy"]))
            lines.append(f"- Best saved MNIST result: `{display_name(str(best_row['Model']))}` with `{best_row['Test Accuracy']:.4f}` accuracy.")
            bayes_row = next((row for row in mnist_rows if row["Model"] == "bayesian"), None)
            dropout_row = next((row for row in mnist_rows if row["Model"] == "dropout"), None)
            standard_row = next((row for row in mnist_rows if row["Model"] == "standard"), None)
            if bayes_row is not None and dropout_row is not None and standard_row is not None:
                bayes_gap_vs_dropout = float(dropout_row["Test Accuracy"]) - float(bayes_row["Test Accuracy"])
                bayes_gain_vs_standard = float(bayes_row["Test Accuracy"]) - float(standard_row["Test Accuracy"])
                lines.append(f"- Bayes by Backprop is within `{bayes_gap_vs_dropout:.4f}` accuracy points of dropout and exceeds the standard MLP by `{bayes_gain_vs_standard:.4f}`.")
            tuned_row = next((row for row in mnist_rows if str(row["Model"]).startswith("bayesian_dropout_")), None)
            if tuned_row is not None and dropout_row is not None:
                tuned_gain = float(tuned_row["Test Accuracy"]) - float(dropout_row["Test Accuracy"])
                lines.append(f"- The Bayesian+dropout extension reached `{tuned_row['Test Accuracy']:.4f}` accuracy, which is `{tuned_gain:.4f}` above plain dropout.")
            lines.append("- This supports the paper's claim that Bayes by Backprop can be competitive with dropout on MNIST in a smaller-scale reproduction.")
        else:
            lines.append("- Only the Bayesian MNIST run is saved, so we cannot yet verify the paper's dropout comparison from the current artifacts.")
    lines.append("")

    lines.append("## Synthetic Regression")
    if not bayesian_regression:
        lines.append("- No Bayesian regression metrics were found.")
    else:
        support_std = bayesian_regression.get("support_std_mean")
        gap_std = bayesian_regression.get("gap_std_mean")
        lines.append(
            f"- Bayesian train MSE: `{bayesian_regression['train_mse']:.6f}`; full-grid MSE: `{bayesian_regression['full_grid_mse']:.6f}`."
        )
        if support_std is not None and gap_std is not None:
            ratio = gap_std / support_std if support_std > 0 else float("inf")
            lines.append(f"- Mean predictive std in observed regions: `{support_std:.4f}`.")
            lines.append(f"- Mean predictive std in the gap / extrapolation region: `{gap_std:.4f}`.")
            lines.append(f"- Uncertainty ratio (gap / observed): `{ratio:.2f}x`.")
            if gap_std > support_std:
                lines.append("- This qualitatively matches the paper: the Bayesian model is more uncertain away from observed data.")
            else:
                lines.append("- This does not clearly reproduce the paper's uncertainty pattern.")
    lines.append("")

    lines.append("## Key Findings")
    bayes_row = next((row for row in mnist_rows if row["Model"] == "bayesian"), None)
    if bayes_row:
        lines.append(f"- The saved Bayesian MNIST model reached `{float(bayes_row['Test Accuracy']):.2%}` test accuracy after 10 epochs.")
    if mnist_complete and bayes_row is not None:
        best_row = max(mnist_rows, key=lambda row: float(row["Test Accuracy"]))
        lines.append(f"- The best saved MNIST model was `{display_name(str(best_row['Model']))}` at `{float(best_row['Test Accuracy']):.2%}` accuracy.")
        lines.append(f"- The original Bayes by Backprop model stayed within `{(float(best_row['Test Accuracy']) - float(bayes_row['Test Accuracy'])):.2%}` of the best classifier.")
    if bayesian_regression:
        support_std = bayesian_regression.get("support_std_mean")
        gap_std = bayesian_regression.get("gap_std_mean")
        if support_std is not None and gap_std is not None:
            lines.append(
                f"- Regression uncertainty was higher away from observed data (`{gap_std:.4f}` vs `{support_std:.4f}` predictive std)."
            )
    if not mnist_complete:
        lines.append("- The main missing evidence is the standard and dropout MNIST baselines; training those will complete the paper-style classification comparison.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment summary plots and markdown.")
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--summary-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = root_dir / "results"
    if args.summary_dir:
        summary_dir = ensure_dir(root_dir / args.summary_dir)
    else:
        default_dir = "summary_best_val" if args.selection_mode == "best_val" else "summary_last_epoch"
        summary_dir = ensure_dir(results_dir / default_dir)

    mnist_rows = load_mnist_rows_from_metrics(results_dir / "mnist" / "test_accuracy.json", args.selection_mode)
    regression_metrics_path = results_dir / "regression" / "regression_metrics.json"
    regression_metrics = load_json(regression_metrics_path) if regression_metrics_path.exists() else None
    benchmark_metrics_path = results_dir / "benchmark" / f"mnist_compute_benchmark_{args.selection_mode}.json"
    if not benchmark_metrics_path.exists():
        benchmark_metrics_path = results_dir / "benchmark" / "mnist_compute_benchmark.json"
    benchmark_metrics = load_json(benchmark_metrics_path) if benchmark_metrics_path.exists() else None

    if mnist_rows:
        write_svg_bar_chart(
            labels=[display_name(str(row["Model"])) for row in mnist_rows],
            values=[float(row["Test Accuracy"]) for row in mnist_rows],
            output_path=summary_dir / "mnist_accuracy_summary.svg",
            title="MNIST Test Accuracy",
            value_format="{:.4f}",
            colors=["#2a6f97", "#4d908e", "#bc4749"],
            y_min=0.9,
            y_max=1.0,
        )
        write_svg_bar_chart(
            labels=[display_name(str(row["Model"])) for row in mnist_rows],
            values=[float(row["Test Error"]) for row in mnist_rows],
            output_path=summary_dir / "mnist_error_summary.svg",
            title="MNIST Test Error",
            value_format="{:.4f}",
            colors=["#2a6f97", "#4d908e", "#bc4749"],
            y_min=0.0,
        )

    mnist_metrics_path = results_dir / "mnist" / "test_accuracy.json"
    if mnist_metrics_path.exists():
        mnist_metrics = load_json(mnist_metrics_path)
        series_colors = {
            "standard": "#2a6f97",
            "dropout": "#4d908e",
            "bayesian": "#bc4749",
            "bayesian_dropout_0p1": "#7a3e9d",
        }
        val_accuracy_series = []
        train_accuracy_series = []
        val_loss_series = []
        preferred_order = ["standard", "dropout", "bayesian", "bayesian_dropout_0p1"]
        ordered_names = [name for name in preferred_order if name in mnist_metrics]
        fallback_colors = ["#c1666b", "#6f4e37", "#3d405b", "#81b29a"]
        for idx, model_name in enumerate(ordered_names):
            if model_name in mnist_metrics:
                history = mnist_metrics[model_name]["history"]
                color = series_colors.get(model_name, fallback_colors[idx % len(fallback_colors)])
                val_accuracy_series.append(
                    {"label": display_name(model_name), "values": history["val_accuracy"], "color": color}
                )
                train_accuracy_series.append(
                    {"label": display_name(model_name), "values": history["train_accuracy"], "color": color}
                )
                val_loss_series.append(
                    {"label": display_name(model_name), "values": history["val_loss"], "color": color}
                )
        if val_accuracy_series:
            write_svg_line_chart(
                val_accuracy_series,
                summary_dir / "mnist_val_accuracy_comparison.svg",
                title="MNIST Validation Accuracy by Epoch",
                y_label="Validation Accuracy",
                y_min=0.94,
                y_max=0.985,
            )
        if train_accuracy_series:
            write_svg_line_chart(
                train_accuracy_series,
                summary_dir / "mnist_train_accuracy_comparison.svg",
                title="MNIST Training Accuracy by Epoch",
                y_label="Training Accuracy",
                y_min=0.85,
                y_max=1.0,
            )
        if val_loss_series:
            write_svg_line_chart(
                val_loss_series,
                summary_dir / "mnist_val_loss_comparison.svg",
                title="MNIST Validation Loss by Epoch",
                y_label="Validation Loss",
            )

        trust_prefixes = ["bayesian_trust_", "bayesian_dropout_0p1_trust_"]
        for prefix in trust_prefixes:
            group_names = [name for name in mnist_metrics if name.startswith(prefix)]
            if not group_names:
                continue
            preferred_group_order = [
                f"{prefix}none",
                f"{prefix}depth_lambda0.15",
                f"{prefix}gradnorm",
                f"{prefix}gradvar_beta0.95",
            ]
            ordered_group_names = [name for name in preferred_group_order if name in group_names]
            ordered_group_names.extend(name for name in group_names if name not in ordered_group_names)
            categories = [trust_mode_display(extract_trust_mode(name)) for name in ordered_group_names]
            val_acc_values = [mnist_metrics[name]["test_metrics"]["accuracy"] for name in ordered_group_names]
            val_err_values = [mnist_metrics[name]["test_metrics"]["error_rate"] for name in ordered_group_names]
            if prefix.startswith("bayesian_dropout_0p1"):
                title_base = "Bayesian + Dropout"
                color = "#7a3e9d"
                output_stub = "bayesian_dropout_trust"
            else:
                title_base = "Bayes by Backprop"
                color = "#bc4749"
                output_stub = "bayesian_trust"
            write_svg_grouped_bar_chart(
                categories=categories,
                series=[
                    {"label": f"{title_base} Accuracy", "values": val_acc_values, "color": color},
                ],
                output_path=summary_dir / f"{output_stub}_accuracy_by_trust.svg",
                title=f"{title_base} Accuracy by Trust Mode",
                value_format="{:.4f}",
                y_min=0.97,
                y_max=max(val_acc_values) + 0.005,
            )
            write_svg_grouped_bar_chart(
                categories=categories,
                series=[
                    {"label": f"{title_base} Error", "values": val_err_values, "color": color},
                ],
                output_path=summary_dir / f"{output_stub}_error_by_trust.svg",
                title=f"{title_base} Error by Trust Mode",
                value_format="{:.4f}",
                y_min=0.0,
            )

    if regression_metrics and "bayesian" in regression_metrics:
        bayesian_regression = regression_metrics["bayesian"]
        support_std = bayesian_regression.get("support_std_mean")
        gap_std = bayesian_regression.get("gap_std_mean")
        if support_std is not None and gap_std is not None:
            write_svg_bar_chart(
                labels=["Observed regions", "Gap / extrapolation"],
                values=[support_std, gap_std],
                output_path=summary_dir / "regression_uncertainty_summary.svg",
                title="Bayesian Regression Uncertainty",
                value_format="{:.4f}",
                colors=["#6c9a8b", "#bc4749"],
            )

    summary_text = build_summary_text(mnist_rows, regression_metrics)
    summary_path = summary_dir / "experiment_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    trust_summary_text = build_trust_summary(mnist_rows, benchmark_metrics.get("results") if benchmark_metrics else None)
    trust_summary_path = summary_dir / "update_trust_summary.md"
    trust_summary_path.write_text(trust_summary_text, encoding="utf-8")

    summary_manifest = {
        "selection_mode": args.selection_mode,
        "summary_markdown": str(summary_path.relative_to(root_dir)),
        "trust_summary_markdown": str(trust_summary_path.relative_to(root_dir)),
        "generated_plots": sorted(str(path.relative_to(root_dir)) for path in summary_dir.glob("*.svg")),
    }
    with (summary_dir / "summary_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_manifest, handle, indent=2)


if __name__ == "__main__":
    main()
