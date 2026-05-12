from __future__ import annotations

import re
from pathlib import Path

from summarize_results import write_svg_bar_chart, write_svg_grouped_bar_chart


RUN_RE = re.compile(r"^- `(?P<label>.+?)`: accuracy `(?P<accuracy>[0-9.]+)`, error `(?P<error>[0-9.]+)`")
COMPUTE_RE = re.compile(
    r"^- `(?P<label>.+?)`: accuracy `(?P<accuracy>[0-9.]+)`, params `(?P<params>[0-9,]+)`, train step `(?P<train>[0-9.]+) ms`, eval step `(?P<eval>[0-9.]+) ms`, parameter memory `(?P<memory>[0-9.]+) MB`"
)


def parse_runs(summary_path: Path, start_heading: str, end_heading: str | None = None) -> list[dict[str, float | str]]:
    rows = []
    in_section = False
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == start_heading:
            in_section = True
            continue
        if in_section and end_heading and line.strip() == end_heading:
            break
        if not in_section:
            continue
        match = RUN_RE.match(line)
        if match:
            rows.append(
                {
                    "label": match.group("label"),
                    "accuracy": float(match.group("accuracy")),
                    "error": float(match.group("error")),
                }
            )
    return rows


def parse_trust_sections(summary_path: Path) -> dict[str, list[dict[str, float | str]]]:
    sections: dict[str, list[dict[str, float | str]]] = {}
    current = ""
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line.removeprefix("## ").strip()
            sections[current] = []
            continue
        if not current:
            continue
        match = RUN_RE.match(line)
        if match:
            sections[current].append(
                {
                    "label": match.group("label"),
                    "accuracy": float(match.group("accuracy")),
                    "error": float(match.group("error")),
                }
            )
    return sections


def make_accuracy_error_plots(rows: list[dict[str, float | str]], output_dir: Path, prefix: str = "mnist") -> None:
    labels = [str(row["label"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    errors = [float(row["error"]) for row in rows]
    write_svg_bar_chart(
        labels,
        accuracies,
        output_dir / f"{prefix}_accuracy_summary.svg",
        "MNIST Test Accuracy",
        "{:.4f}",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.97 if min(accuracies) > 0.96 else 0.9,
        y_max=1.0,
    )
    write_svg_bar_chart(
        labels,
        errors,
        output_dir / f"{prefix}_error_summary.svg",
        "MNIST Test Error",
        "{:.4f}",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )


def make_trust_plot(rows: list[dict[str, float | str]], output_dir: Path, output_stub: str, title: str, color: str) -> None:
    labels = [str(row["label"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    errors = [float(row["error"]) for row in rows]
    write_svg_grouped_bar_chart(
        labels,
        [{"label": "Accuracy", "values": accuracies, "color": color}],
        output_dir / f"{output_stub}_accuracy_by_trust.svg",
        f"{title} Accuracy by Trust Mode",
        "{:.4f}",
        y_min=0.97,
        y_max=1.0,
    )
    write_svg_grouped_bar_chart(
        labels,
        [{"label": "Error", "values": errors, "color": color}],
        output_dir / f"{output_stub}_error_by_trust.svg",
        f"{title} Error by Trust Mode",
        "{:.4f}",
        y_min=0.0,
    )


def parse_compute(summary_path: Path) -> list[dict[str, float | str]]:
    rows = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        match = COMPUTE_RE.match(line)
        if match:
            rows.append(
                {
                    "label": match.group("label"),
                    "train_step_ms": float(match.group("train")),
                    "eval_step_ms": float(match.group("eval")),
                    "parameter_memory_mb": float(match.group("memory")),
                }
            )
    return rows


def make_compute_plots(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    write_svg_bar_chart(
        labels,
        [float(row["train_step_ms"]) for row in rows],
        output_dir / "mnist_train_step_cost.svg",
        "MNIST Train Step Time",
        "{:.2f} ms",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )
    write_svg_bar_chart(
        labels,
        [float(row["eval_step_ms"]) for row in rows],
        output_dir / "mnist_eval_step_cost.svg",
        "MNIST Eval Step Time",
        "{:.2f} ms",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )
    write_svg_bar_chart(
        labels,
        [float(row["parameter_memory_mb"]) for row in rows],
        output_dir / "mnist_parameter_memory.svg",
        "MNIST Parameter Memory",
        "{:.2f} MB",
        ["#2a6f97", "#4d908e", "#bc4749", "#7a3e9d"],
        y_min=0.0,
    )


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    classe_dir = root_dir / "classe_plots"
    main_dir = classe_dir / "summary_best_val"
    dropout_dir = classe_dir / "prop_dropout_summary"

    main_summary = main_dir / "experiment_summary.md"
    if main_summary.exists():
        rows = parse_runs(main_summary, "## MNIST Classification", "## Synthetic Regression")
        if rows:
            make_accuracy_error_plots(rows, main_dir)

    trust_summary = main_dir / "update_trust_summary.md"
    if trust_summary.exists():
        sections = parse_trust_sections(trust_summary)
        if "Bayes by Backprop" in sections:
            make_trust_plot(sections["Bayes by Backprop"], main_dir, "bayesian_trust", "Bayes by Backprop", "#bc4749")
        if "Bayesian + Dropout" in sections:
            make_trust_plot(
                sections["Bayesian + Dropout"],
                main_dir,
                "bayesian_dropout_trust",
                "Bayesian + Dropout",
                "#7a3e9d",
            )

    compute_summary = main_dir / "compute_summary.md"
    if compute_summary.exists():
        rows = parse_compute(compute_summary)
        if rows:
            make_compute_plots(rows, main_dir)

    dropout_summary = dropout_dir / "experiment_summary.md"
    if dropout_summary.exists():
        rows = parse_runs(dropout_summary, "## Runs", "## Best Result")
        if rows:
            make_accuracy_error_plots(rows, dropout_dir)

    print(f"Regenerated readable bar plots under {classe_dir}")


if __name__ == "__main__":
    main()
