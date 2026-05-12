from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def grouped_rows(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def best_row(rows: list[dict[str, str]], key: str = "best_test_acc") -> dict[str, str]:
    return max(rows, key=lambda row: as_float(row, key))


def make_summary_doc(rows: list[dict[str, str]], summary_csv: Path, runs_dir: Path, plots_dir: Path, run_prefix: str) -> str:
    rows = sorted(rows, key=lambda row: as_float(row, "best_test_acc"), reverse=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"# Sample Trust Run Summary\n"
    prefix_text = run_prefix or "all runs in summary.csv"

    lines = [
        title,
        f"Generated: {generated}",
        "",
        f"Run filter: `{prefix_text}`",
        f"Runs summarized: {len(rows)}",
        "",
        "Outputs:",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Per-run metrics and plots: `{runs_dir / '<run_name>'}`",
        f"- Aggregate plots: `{plots_dir}`",
        "",
    ]

    if not rows:
        lines.extend(["No matching runs were found.", ""])
        return "\n".join(lines)

    top = rows[: min(15, len(rows))]
    lines.extend(
        [
            "## Top Runs by Best Test Accuracy",
            "",
            markdown_table(
                [
                    "rank",
                    "run_name",
                    "model",
                    "mode",
                    "batch_size",
                    "seed",
                    "best_test_acc",
                    "final_test_acc",
                    "final_trust_mean",
                ],
                [
                    [
                        str(index),
                        row.get("run_name", ""),
                        row.get("model", ""),
                        row.get("sample_trust_mode", ""),
                        row.get("batch_size", ""),
                        row.get("seed", ""),
                        f"{as_float(row, 'best_test_acc'):.4f}",
                        f"{as_float(row, 'final_test_acc'):.4f}",
                        f"{as_float(row, 'final_trust_mean'):.4f}",
                    ]
                    for index, row in enumerate(top, start=1)
                ],
            ),
            "",
        ]
    )

    model_mode = grouped_rows(rows, ("model", "sample_trust_mode"))
    model_mode_table: list[list[str]] = []
    for (model, mode), group in sorted(model_mode.items()):
        best = best_row(group)
        model_mode_table.append(
            [
                model,
                mode,
                str(len(group)),
                f"{mean([as_float(row, 'best_test_acc') for row in group]):.4f}",
                f"{as_float(best, 'best_test_acc'):.4f}",
                best.get("batch_size", ""),
                best.get("run_name", ""),
            ]
        )
    lines.extend(
        [
            "## By Model and Trust Mode",
            "",
            markdown_table(
                [
                    "model",
                    "mode",
                    "runs",
                    "mean_best_test_acc",
                    "best_test_acc",
                    "best_batch_size",
                    "best_run",
                ],
                model_mode_table,
            ),
            "",
        ]
    )

    batch_mode = grouped_rows(rows, ("batch_size", "sample_trust_mode"))
    batch_mode_table: list[list[str]] = []
    for (batch_size, mode), group in sorted(
        batch_mode.items(),
        key=lambda item: (int(item[0][0]) if item[0][0].isdigit() else item[0][0], item[0][1]),
    ):
        batch_mode_table.append(
            [
                batch_size,
                mode,
                str(len(group)),
                f"{mean([as_float(row, 'best_test_acc') for row in group]):.4f}",
                f"{mean([as_float(row, 'final_test_acc') for row in group]):.4f}",
                f"{mean([as_float(row, 'final_trust_mean') for row in group]):.4f}",
            ]
        )
    lines.extend(
        [
            "## By Batch Size and Trust Mode",
            "",
            markdown_table(
                [
                    "batch_size",
                    "mode",
                    "runs",
                    "mean_best_test_acc",
                    "mean_final_test_acc",
                    "mean_final_trust_mean",
                ],
                batch_mode_table,
            ),
            "",
        ]
    )

    baseline_groups = grouped_rows(
        [row for row in rows if row.get("sample_trust_mode") == "none"],
        ("model", "batch_size", "seed"),
    )
    comparisons: list[list[str]] = []
    for row in rows:
        if row.get("sample_trust_mode") == "none":
            continue
        baseline = baseline_groups.get((row.get("model", ""), row.get("batch_size", ""), row.get("seed", "")))
        if not baseline:
            continue
        delta = as_float(row, "best_test_acc") - as_float(baseline[0], "best_test_acc")
        comparisons.append(
            [
                row.get("model", ""),
                row.get("batch_size", ""),
                row.get("seed", ""),
                row.get("sample_trust_mode", ""),
                f"{as_float(row, 'best_test_acc'):.4f}",
                f"{as_float(baseline[0], 'best_test_acc'):.4f}",
                f"{delta:+.4f}",
            ]
        )
    comparisons.sort(key=lambda row: float(row[-1]), reverse=True)
    lines.extend(
        [
            "## Delta vs Matched `none` Baseline",
            "",
            markdown_table(
                ["model", "batch_size", "seed", "mode", "best_test_acc", "baseline_best_test_acc", "delta"],
                comparisons[: min(30, len(comparisons))],
            )
            if comparisons
            else "No matched baseline comparisons were available.",
            "",
            "Interpretation note: this report is descriptive. Treat any improvement as a hypothesis until it survives repeated seeds and a full-data run.",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a markdown report from sample trust summary.csv.")
    parser.add_argument("--summary-csv", type=str, default="results/sample_trust/reports/summary.csv")
    parser.add_argument("--output-md", type=str, default="results/sample_trust/reports/experiment_summary.md")
    parser.add_argument("--runs-dir", type=str, default="results/sample_trust/runs")
    parser.add_argument("--plots-dir", type=str, default="results/sample_trust/reports/plots")
    parser.add_argument("--run-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_md = Path(args.output_md)
    rows = read_rows(summary_csv) if summary_csv.exists() else []
    if args.run_prefix:
        rows = [row for row in rows if row.get("run_name", "").startswith(args.run_prefix)]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        make_summary_doc(rows, summary_csv, Path(args.runs_dir), Path(args.plots_dir), args.run_prefix),
        encoding="utf-8",
    )
    print(f"Summary doc saved to {output_md}")


if __name__ == "__main__":
    main()
