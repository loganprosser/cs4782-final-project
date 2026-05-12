from __future__ import annotations

import re
from html import escape
from pathlib import Path


RUN_RE = re.compile(r"^- `(?P<label>.+?)`: accuracy `(?P<accuracy>[0-9.]+)`, error `(?P<error>[0-9.]+)`")
COMPUTE_RE = re.compile(
    r"^- `(?P<label>.+?)`: accuracy `(?P<accuracy>[0-9.]+)`, params `(?P<params>[0-9,]+)`, "
    r"train step `(?P<train>[0-9.]+) ms`, eval step `(?P<eval>[0-9.]+) ms`, "
    r"parameter memory `(?P<memory>[0-9.]+) MB`"
)

POSTER_MODELS = [
    {
        "source": "Standard MLP",
        "compute_source": "Standard MLP (no trust scaling)",
        "label": "Standard\nMLP",
        "group": "Paper recreation",
        "color": "#3b6fb6",
    },
    {
        "source": "Dropout MLP",
        "compute_source": "Dropout MLP (no trust scaling)",
        "label": "Dropout\nMLP",
        "group": "Paper recreation",
        "color": "#5b8fd1",
    },
    {
        "source": "Bayes by Backprop",
        "compute_source": "Bayes by Backprop (no trust scaling)",
        "label": "BBB",
        "group": "Paper recreation",
        "color": "#7aa7df",
    },
    {
        "source": "Bayesian + Dropout",
        "compute_source": "Bayesian + Dropout (no trust scaling)",
        "label": "BBB +\nDropout",
        "group": "Extension",
        "color": "#6f5aa8",
    },
    {
        "source": "Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)",
        "compute_source": "Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)",
        "label": "BBB\nKalman",
        "group": "Kalman extension",
        "color": "#d1842f",
    },
    {
        "source": "Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)",
        "compute_source": "Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)",
        "label": "BBB + Dropout\nKalman",
        "group": "Kalman extension",
        "color": "#c94f4f",
    },
]

PAPER_COMPARISON_MODELS = [
    {
        "source": "Standard MLP",
        "label": "Standard\nMLP",
        "paper_accuracy": 0.9812,
    },
    {
        "source": "Dropout MLP",
        "label": "Dropout\nMLP",
        "paper_accuracy": 0.9864,
    },
    {
        "source": "Bayes by Backprop",
        "label": "BBB,\nscale-mixture prior",
        "paper_accuracy": 0.9868,
    },
]

FINAL_COMPARISON_COLORS = {
    "Standard MLP h400/L2": "#0067c5",
    "Standard MLP + Kalman h400/L2": "#003f88",
    "Dropout MLP h400/L2": "#00a88f",
    "Dropout MLP + Kalman h400/L2": "#00705e",
    "BBB h400/L2": "#8a3ffc",
    "BBB + Kalman h400/L2": "#5b21b6",
    "BBB + Dropout h400/L2": "#f15a24",
    "BBB + Dropout + Kalman h400/L2": "#c62828",
    "Standard MLP h800/L2": "#4f9cf9",
    "Standard MLP + Kalman h800/L2": "#2563eb",
}


def parse_mnist_rows(summary_path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    in_mnist = False
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "## MNIST Classification":
            in_mnist = True
            continue
        if in_mnist and stripped.startswith("## "):
            break
        if not in_mnist:
            continue
        match = RUN_RE.match(line)
        if match:
            rows[match.group("label")] = float(match.group("accuracy"))
    return rows


def parse_train_step_rows(compute_path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in compute_path.read_text(encoding="utf-8").splitlines():
        match = COMPUTE_RE.match(line)
        if match:
            rows[match.group("label")] = float(match.group("train"))
    return rows


def parse_final_comparison_rows(summary_path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Model ") or line.startswith("|---"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) not in {6, 7}:
            continue
        std_accuracy = parts[5] if len(parts) == 7 else parts[3]
        best_accuracy = parts[6] if len(parts) == 7 else parts[5]
        rows.append(
            {
                "model": parts[0],
                "runs": int(parts[1]),
                "mean_error": float(parts[2]),
                "std_error": float(parts[3]),
                "mean_accuracy": float(parts[4]),
                "std_accuracy": float(std_accuracy),
                "best_accuracy": float(best_accuracy),
            }
        )
    return rows


def compact_final_label(model: str) -> str:
    if "kalmanalgo" in model:
        compact = model.replace(" MLP", "").replace(" h400/L2", "").replace(" h800/L2", "")
        compact = compact.replace("Standard + kalmanalgo", "Standard\nkalmanalgo")
        compact = compact.replace("BBB + Dropout + kalmanalgo", "BBB + Dropout\nkalmanalgo")
        compact = compact.replace("Dropout + kalmanalgo", "Dropout\nkalmanalgo")
        compact = compact.replace("BBB + kalmanalgo", "BBB\nkalmanalgo")
        return compact
    compact = model.replace(" MLP", "").replace(" h400/L2", "").replace(" h800/L2", " h800")
    compact = compact.replace("Standard", "Standard MLP")
    return compact


def final_color(model: str) -> str:
    return FINAL_COMPARISON_COLORS.get(model, "#555555")


def add_text(
    svg: list[str],
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    size: int = 14,
    weight: int | str = 400,
    fill: str = "#222222",
    underline: bool = False,
) -> None:
    lines = text.split("\n")
    decoration = ' text-decoration="underline"' if underline else ""
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{decoration}>'
    )
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size * 1.18
        svg.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>')
    svg.append("</text>")


def add_title(svg: list[str], x: float, y: float, title: str) -> None:
    if "Noisy" not in title:
        add_text(svg, x, y, title, size=25, weight=700)
        return
    before, after = title.split("Noisy", maxsplit=1)
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="25" font-weight="700" fill="#222222">'
    )
    svg.append(f"<tspan>{escape(before)}</tspan>")
    svg.append('<tspan font-weight="900">Noisy</tspan>')
    svg.append(f"<tspan>{escape(after)}</tspan>")
    svg.append("</text>")


def write_poster_accuracy_plot(summary_path: Path, output_path: Path) -> None:
    values_by_label = parse_mnist_rows(summary_path)
    missing = [model["source"] for model in POSTER_MODELS if model["source"] not in values_by_label]
    if missing:
        raise ValueError(f"Missing requested model rows: {missing}")

    rows = [
        {
            **model,
            "accuracy": values_by_label[model["source"]],
        }
        for model in POSTER_MODELS
    ]

    width = 1080
    height = 620
    left = 92
    right = 48
    top = 82
    bottom = 154
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = 0.975
    y_max = 0.983
    span = y_max - y_min
    category_width = plot_width / len(rows)
    bar_width = category_width * 0.56
    baseline_y = top + plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
    ]
    add_text(svg, width / 2, 34, "MNIST Test Accuracy", size=26, weight=700)
    add_text(svg, width / 2, 60, "Best validation checkpoint; paper-recreation baselines grouped in blue", size=14, fill="#555555")

    for tick_idx in range(5):
        tick = y_min + span * tick_idx / 4
        y = baseline_y - plot_height * tick_idx / 4
        svg.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" stroke="#e4ded3" stroke-width="1"/>')
        svg.append(f'<line x1="{left - 6:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#333333" stroke-width="1.2"/>')
        add_text(svg, left - 12, y + 5, f"{tick:.3f}", anchor="end", size=13, fill="#444444")

    svg.append(f'<line x1="{left:.1f}" y1="{baseline_y:.1f}" x2="{left + plot_width:.1f}" y2="{baseline_y:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{baseline_y:.1f}" stroke="#2b2b2b" stroke-width="2"/>')

    group_ranges = [
        ("Paper recreation", 0, 2, "#3b6fb6"),
        ("Extension", 3, 3, "#6f5aa8"),
        ("Kalman extension", 4, 5, "#c94f4f"),
    ]
    for label, start, end, color in group_ranges:
        x1 = left + start * category_width + category_width * 0.18
        x2 = left + (end + 1) * category_width - category_width * 0.18
        y = baseline_y + 102
        svg.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        add_text(svg, (x1 + x2) / 2, y + 24, label, size=13, weight=700, fill=color)

    for idx, row in enumerate(rows):
        x = left + (idx + 0.5) * category_width
        value = float(row["accuracy"])
        bar_height = ((value - y_min) / span) * plot_height
        bar_top = baseline_y - bar_height
        color = str(row["color"])
        svg.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="{color}" rx="7"/>'
        )
        add_text(svg, x, bar_top - 11, f"{value:.4f}", size=15, weight=700)
        add_text(svg, x, baseline_y + 28, str(row["label"]), size=14, weight=700)

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_paper_comparison_plot(summary_path: Path, output_path: Path) -> None:
    values_by_label = parse_mnist_rows(summary_path)
    missing = [model["source"] for model in PAPER_COMPARISON_MODELS if model["source"] not in values_by_label]
    if missing:
        raise ValueError(f"Missing requested model rows: {missing}")

    rows = [
        {
            **model,
            "ours_accuracy": values_by_label[model["source"]],
        }
        for model in PAPER_COMPARISON_MODELS
    ]

    width = 1040
    height = 600
    left = 96
    right = 56
    top = 110
    bottom = 132
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = 0.975
    y_max = 0.989
    span = y_max - y_min
    category_width = plot_width / len(rows)
    bar_width = category_width * 0.26
    baseline_y = top + plot_height
    paper_color = "#2f6fb3"
    ours_color = "#c8564b"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
    ]
    add_text(svg, width / 2, 34, "Paper vs. Our Reproduction", size=26, weight=700)
    add_text(svg, width / 2, 60, "MNIST test accuracy for directly comparable baseline models", size=14, fill="#555555")

    legend_x = width - right - 238
    legend_y = 82
    svg.append(f'<rect x="{legend_x - 18:.1f}" y="{legend_y - 19:.1f}" width="256" height="34" fill="#fff8ee" stroke="#ded5c8" rx="7"/>')
    svg.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 10:.1f}" width="16" height="16" fill="{paper_color}" rx="3"/>')
    add_text(svg, legend_x + 24, legend_y + 3, "Paper", anchor="start", size=14, weight=700)
    svg.append(f'<rect x="{legend_x + 122:.1f}" y="{legend_y - 10:.1f}" width="16" height="16" fill="{ours_color}" rx="3"/>')
    add_text(svg, legend_x + 146, legend_y + 3, "Ours", anchor="start", size=14, weight=700)

    for tick_idx in range(5):
        tick = y_min + span * tick_idx / 4
        y = baseline_y - plot_height * tick_idx / 4
        svg.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" stroke="#e4ded3" stroke-width="1"/>')
        svg.append(f'<line x1="{left - 6:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#333333" stroke-width="1.2"/>')
        add_text(svg, left - 12, y + 5, f"{tick:.3f}", anchor="end", size=13, fill="#444444")

    svg.append(f'<line x1="{left:.1f}" y1="{baseline_y:.1f}" x2="{left + plot_width:.1f}" y2="{baseline_y:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{baseline_y:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    add_text(svg, left, top - 18, "Accuracy", anchor="start", size=14, weight=700, fill="#333333")

    for idx, row in enumerate(rows):
        center_x = left + (idx + 0.5) * category_width
        for offset, key, color in [(-0.58, "paper_accuracy", paper_color), (0.58, "ours_accuracy", ours_color)]:
            value = float(row[key])
            x = center_x + offset * bar_width
            bar_height = ((value - y_min) / span) * plot_height
            bar_top = baseline_y - bar_height
            svg.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{color}" rx="7"/>'
            )
            add_text(svg, x, bar_top - 10, f"{value * 100:.2f}%", size=14, weight=700)
        delta = float(row["ours_accuracy"]) - float(row["paper_accuracy"])
        add_text(svg, center_x, baseline_y + 30, str(row["label"]), size=14, weight=700)
        add_text(svg, center_x, baseline_y + 80, f"{delta * 100:+.2f} pp", size=13, weight=700, fill="#6b6258")

    add_text(svg, width / 2, height - 18, "Numbers below each model show ours minus paper in percentage points", size=13, fill="#555555")
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_accuracy_vs_train_cost_plot(summary_path: Path, compute_path: Path, output_path: Path) -> None:
    accuracies_by_label = parse_mnist_rows(summary_path)
    train_steps_by_label = parse_train_step_rows(compute_path)
    missing_accuracy = [model["source"] for model in POSTER_MODELS if model["source"] not in accuracies_by_label]
    missing_train = [model["compute_source"] for model in POSTER_MODELS if model["compute_source"] not in train_steps_by_label]
    if missing_accuracy or missing_train:
        raise ValueError(f"Missing accuracy rows: {missing_accuracy}; missing train-step rows: {missing_train}")

    rows = [
        {
            **model,
            "accuracy": accuracies_by_label[model["source"]],
            "train_step_ms": train_steps_by_label[model["compute_source"]],
        }
        for model in POSTER_MODELS
    ]

    width = 1280
    height = 680
    left = 96
    right = 380
    top = 96
    bottom = 104
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min = 0.0
    x_max = 10.2
    y_min = 0.975
    y_max = 0.983
    x_span = x_max - x_min
    y_span = y_max - y_min

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / x_span) * plot_width

    def y_coord(value: float) -> float:
        return top + plot_height - ((value - y_min) / y_span) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
    ]
    add_text(svg, width / 2, 34, "Accuracy vs. Training Cost", size=26, weight=700)
    add_text(svg, width / 2, 60, "MNIST best-val accuracy against average train-step time", size=14, fill="#555555")

    for tick in [0, 2, 4, 6, 8, 10]:
        x = x_coord(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + plot_height:.1f}" stroke="#eee6db" stroke-width="1"/>')
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height:.1f}" x2="{x:.1f}" y2="{top + plot_height + 6:.1f}" stroke="#333333" stroke-width="1.2"/>')
        add_text(svg, x, top + plot_height + 26, f"{tick:.1f}", size=13, fill="#444444")

    for tick_idx in range(5):
        tick = y_min + y_span * tick_idx / 4
        y = y_coord(tick)
        svg.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" stroke="#e4ded3" stroke-width="1"/>')
        svg.append(f'<line x1="{left - 6:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#333333" stroke-width="1.2"/>')
        add_text(svg, left - 12, y + 5, f"{tick:.3f}", anchor="end", size=13, fill="#444444")

    svg.append(f'<line x1="{left:.1f}" y1="{top + plot_height:.1f}" x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_height:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    add_text(svg, left, top - 20, "Accuracy", anchor="start", size=18, weight=700, fill="#333333")
    add_text(svg, left + plot_width / 2, height - 22, "Average train step time (ms)", size=19, weight=700, fill="#333333")

    reference_row = rows[3]
    reference_y = y_coord(float(reference_row["accuracy"]))
    svg.append(
        f'<line x1="{left:.1f}" y1="{reference_y:.1f}" x2="{left + plot_width:.1f}" y2="{reference_y:.1f}" '
        'stroke="#6f5aa8" stroke-width="2.5" stroke-dasharray="8 7" opacity="0.75"/>'
    )
    add_text(
        svg,
        left + plot_width * 0.58,
        reference_y - 10,
        "Point 4 accuracy",
        anchor="middle",
        size=13,
        weight=700,
        fill="#6f5aa8",
    )

    for idx, row in enumerate(rows, start=1):
        x = x_coord(float(row["train_step_ms"]))
        y = y_coord(float(row["accuracy"]))
        color = str(row["color"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{color}" stroke="#fffdf8" stroke-width="4"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16.5" fill="none" stroke="#2b2b2b" stroke-width="1.2" opacity="0.3"/>')
        add_text(svg, x, y + 5, str(idx), size=14, weight=700, fill="#ffffff")

    panel_x = left + plot_width + 28
    panel_y = top + 8
    panel_width = right - 50
    panel_height = 420
    svg.append(
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" '
        'fill="#fff8ee" stroke="#ded5c8" rx="8"/>'
    )
    add_text(svg, panel_x + 18, panel_y + 30, "Models", anchor="start", size=18, weight=700)
    time_x = panel_x + panel_width - 110
    acc_x = panel_x + panel_width - 18
    add_text(svg, time_x, panel_y + 30, "ms", anchor="end", size=16, weight=700, fill="#5a5249")
    add_text(svg, acc_x, panel_y + 30, "acc.", anchor="end", size=16, weight=700, fill="#5a5249")

    for idx, row in enumerate(rows, start=1):
        row_y = panel_y + 58 + (idx - 1) * 56
        if idx > 1:
            svg.append(f'<line x1="{panel_x + 18:.1f}" y1="{row_y - 26:.1f}" x2="{panel_x + panel_width - 18:.1f}" y2="{row_y - 26:.1f}" stroke="#e4d9ca" stroke-width="1"/>')
        color = str(row["color"])
        label = str(row["label"]).replace("\n", " ")
        svg.append(f'<circle cx="{panel_x + 26:.1f}" cy="{row_y:.1f}" r="12" fill="{color}"/>')
        add_text(svg, panel_x + 26, row_y + 5, str(idx), size=11, weight=700, fill="#ffffff")
        label_x = panel_x + 48
        if label == "BBB + Dropout Kalman":
            add_text(svg, label_x, row_y - 2, "BBB + Dropout", anchor="start", size=15, weight=700)
            add_text(svg, label_x, row_y + 16, "Kalman", anchor="start", size=15, weight=700)
        else:
            add_text(svg, label_x, row_y + 5, label, anchor="start", size=15, weight=700)
        add_text(svg, time_x, row_y + 5, f'{float(row["train_step_ms"]):.2f}', anchor="end", size=15, fill="#222222")
        add_text(svg, acc_x, row_y + 5, f'{float(row["accuracy"]) * 100:.2f}%', anchor="end", size=15, fill="#222222")

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_final_multiseed_plot(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
    *,
    metric_key: str,
    spread_key: str,
    title: str,
    x_label: str,
    value_format: str,
    x_min: float,
    x_max: float,
    lower_is_better: bool = False,
    subtitle: str = "Mean over 4 seeds with one standard deviation",
) -> None:
    width = 1280
    height = max(560, 168 + 56 * len(rows))
    left = 330
    right = 235
    top = 74
    bottom = 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    row_gap = plot_height / len(rows)
    span = x_max - x_min

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / span) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
    ]
    add_title(svg, width / 2, 34, title)
    add_text(svg, width / 2, 58, subtitle, size=14, fill="#555555")

    group_ranges: list[tuple[int, int, str]] = []
    current_label = ""
    current_start = 0
    for idx, row in enumerate(rows):
        model = str(row["model"])
        label = "h800/L2" if "h800/L2" in model else "h400/L2"
        if idx == 0:
            current_label = label
            current_start = idx
            continue
        if label != current_label:
            group_ranges.append((current_start, idx - 1, current_label))
            current_start = idx
            current_label = label
    if rows:
        group_ranges.append((current_start, len(rows) - 1, current_label))

    for start, end, label in group_ranges:
        y1 = top + start * row_gap + 4
        y2 = top + (end + 1) * row_gap - 4
        fill = "#f4e4cf" if label == "h400/L2" else "#f4efe7"
        svg.append(f'<rect x="{left - 252:.1f}" y="{y1:.1f}" width="{plot_width + right + 232:.1f}" height="{y2 - y1:.1f}" fill="{fill}" rx="8"/>')
        add_text(svg, left - 238, y1 + 22, label, anchor="start", size=13, weight=700, fill="#6b6258")

    major_tick_count = 8
    for tick_idx in range(major_tick_count + 1):
        tick = x_min + span * tick_idx / major_tick_count
        x = x_coord(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + plot_height:.1f}" stroke="#d3c5b5" stroke-width="1.4"/>')
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height:.1f}" x2="{x:.1f}" y2="{top + plot_height + 6:.1f}" stroke="#333333" stroke-width="1.1"/>')
        add_text(svg, x, top + plot_height + 25, value_format.format(tick), size=13, fill="#444444")
        if tick_idx < major_tick_count:
            minor_tick = tick + span / major_tick_count / 2
            minor_x = x_coord(minor_tick)
            svg.append(f'<line x1="{minor_x:.1f}" y1="{top:.1f}" x2="{minor_x:.1f}" y2="{top + plot_height:.1f}" stroke="#dfd2c1" stroke-width="1" stroke-dasharray="3 5"/>')

    svg.append(f'<line x1="{left:.1f}" y1="{top + plot_height:.1f}" x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}" stroke="#2b2b2b" stroke-width="2"/>')
    add_text(svg, left + plot_width / 2, height - 12, x_label, size=15, weight=700, fill="#333333")

    best_value = min(float(row[metric_key]) for row in rows) if lower_is_better else max(float(row[metric_key]) for row in rows)
    best_x = x_coord(best_value)
    svg.append(
        f'<line x1="{best_x:.1f}" y1="{top:.1f}" x2="{best_x:.1f}" y2="{top + plot_height:.1f}" '
        'stroke="#6f5aa8" stroke-width="2" stroke-dasharray="7 6" opacity="0.68"/>'
    )

    add_text(svg, left + plot_width + 22, top - 8, "mean +/- std", anchor="start", size=13, weight=700, fill="#5a5249")
    for idx, row in enumerate(rows):
        y = top + (idx + 0.5) * row_gap
        value = float(row[metric_key])
        spread = float(row[spread_key])
        color = final_color(str(row["model"]))
        is_best = abs(value - best_value) < 1e-12
        low = max(x_min, value - spread)
        high = min(x_max, value + spread)
        low_x = x_coord(low)
        high_x = x_coord(high)
        x = x_coord(value)
        label = compact_final_label(str(row["model"]))
        label_weight = 800 if is_best else 700
        label_size = 14 if is_best else 13
        label_y = y - 8 if "\n" in label else y + 5
        add_text(svg, left - 18, label_y, label, anchor="end", size=label_size, weight=label_weight, fill="#222222", underline=is_best)
        stroke_width = 5 if is_best else 4
        cap_height = 8 if is_best else 7
        circle_radius = 8.5 if is_best else 7.5
        svg.append(f'<line x1="{low_x:.1f}" y1="{y:.1f}" x2="{high_x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round"/>')
        svg.append(f'<line x1="{low_x:.1f}" y1="{y - cap_height:.1f}" x2="{low_x:.1f}" y2="{y + cap_height:.1f}" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<line x1="{high_x:.1f}" y1="{y - cap_height:.1f}" x2="{high_x:.1f}" y2="{y + cap_height:.1f}" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{circle_radius}" fill="{color}" stroke="#fffdf8" stroke-width="3"/>')
        add_text(
            svg,
            left + plot_width + 22,
            y + 5,
            f"{value_format.format(value)} +/- {value_format.format(spread)}",
            anchor="start",
            size=15 if is_best else 14,
            weight=800 if is_best else 700,
            fill="#222222",
            underline=is_best,
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "classe_plots" / "summary_best_val" / "experiment_summary.md"
    compute_path = root / "classe_plots" / "summary_best_val" / "compute_summary.md"
    final_comparison_path = root / "classe_plots" / "final_comparison" / "final_error_comparison.md"
    final_noisy_comparison_path = root / "classe_plots" / "final_noisy_comparison" / "final_noisy_comparison.md"
    output_dir = root / "final_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_poster_accuracy_plot(summary_path, output_dir / "mnist_accuracy_poster.svg")
    write_paper_comparison_plot(summary_path, output_dir / "mnist_paper_vs_ours.svg")
    write_accuracy_vs_train_cost_plot(summary_path, compute_path, output_dir / "mnist_accuracy_vs_train_cost.svg")
    final_rows = parse_final_comparison_rows(final_comparison_path)
    write_final_multiseed_plot(
        final_rows,
        output_dir / "final_multiseed_accuracy_comparison.svg",
        metric_key="mean_accuracy",
        spread_key="std_error",
        title="Final Multi-Seed Accuracy",
        x_label="Mean test accuracy",
        value_format="{:.4f}",
        x_min=0.977,
        x_max=0.984,
        subtitle="Mean over 4 seeds with one standard deviation",
    )
    write_final_multiseed_plot(
        final_rows,
        output_dir / "final_multiseed_error_comparison.svg",
        metric_key="mean_error",
        spread_key="std_error",
        title="Final Multi-Seed Error",
        x_label="Mean test error",
        value_format="{:.4f}",
        x_min=0.017,
        x_max=0.023,
        lower_is_better=True,
        subtitle="Mean over 4 seeds with one standard deviation",
    )
    final_noisy_rows = parse_final_comparison_rows(final_noisy_comparison_path)
    write_final_multiseed_plot(
        final_noisy_rows,
        output_dir / "final_noisy_multiseed_accuracy_comparison.svg",
        metric_key="mean_accuracy",
        spread_key="std_accuracy",
        title="Final Noisy Multi-Seed Accuracy",
        x_label="Mean noisy FashionMNIST test accuracy",
        value_format="{:.4f}",
        x_min=0.862,
        x_max=0.884,
        subtitle="Mean over 3 seeds with one standard deviation",
    )
    write_final_multiseed_plot(
        final_noisy_rows,
        output_dir / "final_noisy_multiseed_error_comparison.svg",
        metric_key="mean_error",
        spread_key="std_error",
        title="Final Noisy Multi-Seed Error",
        x_label="Mean noisy FashionMNIST test error",
        value_format="{:.4f}",
        x_min=0.117,
        x_max=0.137,
        lower_is_better=True,
        subtitle="Mean over 3 seeds with one standard deviation",
    )


if __name__ == "__main__":
    main()
