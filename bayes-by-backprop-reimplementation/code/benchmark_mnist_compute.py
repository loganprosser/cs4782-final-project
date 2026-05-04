from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from models import BayesianMLP, DropoutMLP, StandardMLP
from summarize_results import compact_plot_label, svg_multiline_text, wrap_label
from utils import set_seed


DISPLAY_NAMES = {
    "standard": "Standard MLP",
    "dropout": "Dropout MLP",
    "bayesian": "Bayes by Backprop",
    "bayesian_dropout_0p1": "Bayesian + Dropout",
}

COLORS = {
    "standard": "#2a6f97",
    "dropout": "#4d908e",
    "bayesian": "#bc4749",
    "bayesian_dropout_0p1": "#7a3e9d",
}

BENCHMARK_CONFIG = {
    "batch_size": 128,
    "train_steps": 20,
    "eval_steps": 30,
    "eval_mc_samples": 10,
    "seed": 0,
}


def display_name(model_name: str) -> str:
    if model_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[model_name]
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(run_name: str) -> torch.nn.Module:
    if "_trust_" in run_name:
        run_name = run_name.split("_trust_", maxsplit=1)[0]
    if run_name == "standard":
        return StandardMLP()
    if run_name == "dropout":
        return DropoutMLP()
    if run_name == "bayesian":
        return BayesianMLP(dropout=0.0)
    if run_name == "bayesian_dropout_0p1":
        return BayesianMLP(dropout=0.1)
    raise ValueError(f"Unsupported run name: {run_name}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parameter_memory_mb(model: torch.nn.Module) -> float:
    return sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)


def get_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 ** 2)


def benchmark_train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    run_name: str,
    steps: int,
) -> tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    start_rss = get_rss_mb()
    start_time = time.perf_counter()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if run_name.startswith("bayesian"):
            loss = F.cross_entropy(logits, targets)
            loss = loss + (model.log_variational_posterior() - model.log_prior()) / inputs.size(0)
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start_time
    rss_delta = max(get_rss_mb() - start_rss, 0.0)
    return (elapsed / steps) * 1000.0, rss_delta


@torch.no_grad()
def benchmark_eval_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    run_name: str,
    steps: int,
    mc_samples: int,
) -> float:
    model.eval()
    start_time = time.perf_counter()
    for _ in range(steps):
        if run_name.startswith("bayesian"):
            _ = torch.stack([model(inputs) for _ in range(mc_samples)], dim=0).mean(dim=0)
        else:
            _ = model(inputs)
    elapsed = time.perf_counter() - start_time
    return (elapsed / steps) * 1000.0


def write_svg_scatter(results: list[dict], output_path: Path) -> None:
    width = 1100
    height = 520
    left = 90
    right = 240
    top = 70
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [item["train_step_ms"] for item in results]
    y_values = [item["test_accuracy"] for item in results]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    def x_coord(value: float) -> float:
        return left + ((value - x_min) / x_span) * plot_width

    def y_coord(value: float) -> float:
        return top + plot_height - ((value - y_min) / y_span) * plot_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">Accuracy vs Training Cost</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#333333">Average train step time (ms)</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#333333" transform="rotate(-90 24 {top + plot_height / 2})">Test accuracy</text>',
    ]

    for item in results:
        cx = x_coord(item["train_step_ms"])
        cy = y_coord(item["test_accuracy"])
        color = COLORS.get(item["run_name"], "#555555")
        label = compact_plot_label(display_name(item["run_name"]))
        svg_lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{color}"/>')
        svg_lines.extend(svg_multiline_text(cx + 10, cy - 10, wrap_label(label, 18), anchor="start", font_size=12))

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_svg_bar(results: list[dict], output_path: Path, key: str, title: str, value_fmt: str) -> None:
    width = 1320
    height = max(540, 150 + 52 * len(results))
    left = 500
    right = 120
    top = 70
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [item[key] for item in results]
    max_value = max(values) * 1.15
    bar_height = plot_height / max(len(results) * 1.35, 1)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]

    for idx, item in enumerate(results):
        value = item[key]
        y = top + (idx + 0.5) * (plot_height / len(results))
        bar_width = (value / max_value) * plot_width if max_value > 0 else 0.0
        bar_top = y - bar_height / 2
        color = COLORS.get(item["run_name"], "#555555")
        svg_lines.append(
            f'<rect x="{left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="6"/>'
        )
        label_lines = wrap_label(display_name(item["run_name"]), 42)
        svg_lines.extend(svg_multiline_text(left - 18, y - ((len(label_lines) - 1) * 8), label_lines, anchor="end", font_size=13))
        svg_lines.append(
            f'<text x="{left + bar_width + 10:.1f}" y="{y + 5:.1f}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#222222">{value_fmt.format(value)}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def build_summary_text(results: list[dict], device_info: dict) -> str:
    sorted_by_accuracy = sorted(results, key=lambda row: row["test_accuracy"], reverse=True)
    fastest_train = min(results, key=lambda row: row["train_step_ms"])
    fastest_eval = min(results, key=lambda row: row["eval_step_ms"])
    cheapest_params = min(results, key=lambda row: row["parameter_count"])
    best_accuracy = sorted_by_accuracy[0]

    lines = [
        "# Compute Benchmark Summary",
        "",
        "## Environment",
        f"- Device reported by benchmark: `{device_info['device']}`.",
        f"- CUDA available: `{device_info['cuda_available']}`.",
        f"- MPS available: `{device_info['mps_available']}`.",
        "- GPU utilization percentage was not measured here because the current Python environment does not expose a working CUDA or MPS backend.",
        "",
        "## Model Tradeoffs",
    ]

    for row in results:
        lines.append(
            f"- `{display_name(row['run_name'])}`: accuracy `{row['test_accuracy']:.4f}`, params `{row['parameter_count']:,}`, train step `{row['train_step_ms']:.2f} ms`, eval step `{row['eval_step_ms']:.2f} ms`, parameter memory `{row['parameter_memory_mb']:.2f} MB`."
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            f"- Highest accuracy: `{display_name(best_accuracy['run_name'])}` at `{best_accuracy['test_accuracy']:.4f}`.",
            f"- Fastest training step: `{display_name(fastest_train['run_name'])}` at `{fastest_train['train_step_ms']:.2f} ms`.",
            f"- Fastest evaluation step: `{display_name(fastest_eval['run_name'])}` at `{fastest_eval['eval_step_ms']:.2f} ms`.",
            f"- Smallest parameter footprint: `{display_name(cheapest_params['run_name'])}` with `{cheapest_params['parameter_memory_mb']:.2f} MB` of weights.",
        ]
    )

    baseline = next((row for row in results if row["run_name"] == "dropout"), None)
    tuned = next((row for row in results if row["run_name"] == "bayesian_dropout_0p1"), None)
    if baseline and tuned:
        acc_gain = tuned["test_accuracy"] - baseline["test_accuracy"]
        train_cost_ratio = tuned["train_step_ms"] / baseline["train_step_ms"] if baseline["train_step_ms"] > 0 else float("inf")
        eval_cost_ratio = tuned["eval_step_ms"] / baseline["eval_step_ms"] if baseline["eval_step_ms"] > 0 else float("inf")
        lines.extend(
            [
                f"- Compared with Dropout MLP, `Bayesian + Dropout` gained `{acc_gain:.4f}` accuracy points.",
                f"- That gain cost about `{train_cost_ratio:.2f}x` training-time per batch and `{eval_cost_ratio:.2f}x` evaluation-time per batch in this benchmark.",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MNIST model compute cost.")
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--summary-dir", type=str, default="")
    parser.add_argument("--benchmark-output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(BENCHMARK_CONFIG["seed"])
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = root_dir / "results"
    if args.summary_dir:
        summary_dir = ensure_dir(root_dir / args.summary_dir)
    else:
        default_dir = "summary_best_val" if args.selection_mode == "best_val" else "summary_last_epoch"
        summary_dir = ensure_dir(results_dir / default_dir)
    benchmark_dir = ensure_dir(results_dir / "benchmark")

    metrics = load_json(results_dir / "mnist" / "test_accuracy.json")
    preferred_order = ["standard", "dropout", "bayesian", "bayesian_dropout_0p1"]
    run_names = [name for name in preferred_order if name in metrics]
    run_names.extend(name for name in metrics if name not in run_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    inputs = torch.randn(BENCHMARK_CONFIG["batch_size"], 1, 28, 28, device=device)
    targets = torch.randint(0, 10, (BENCHMARK_CONFIG["batch_size"],), device=device)

    results = []
    for run_name in run_names:
        model = build_model(run_name).to(device)
        train_step_ms, rss_delta_mb = benchmark_train_step(
            model, inputs, targets, run_name, BENCHMARK_CONFIG["train_steps"]
        )
        eval_step_ms = benchmark_eval_step(
            model, inputs, run_name, BENCHMARK_CONFIG["eval_steps"], BENCHMARK_CONFIG["eval_mc_samples"]
        )
        selected_metrics = metrics[run_name].get("test_metrics") if args.selection_mode == "best_val" else metrics[run_name].get("last_epoch_test_metrics", metrics[run_name].get("test_metrics"))
        results.append(
            {
                "run_name": run_name,
                "display_name": display_name(run_name),
                "parameter_count": parameter_count(model),
                "parameter_memory_mb": parameter_memory_mb(model),
                "train_step_ms": train_step_ms,
                "eval_step_ms": eval_step_ms,
                "rss_delta_mb": rss_delta_mb,
                "test_accuracy": selected_metrics["accuracy"],
                "test_error": selected_metrics["error_rate"],
            }
        )

    benchmark_name = args.benchmark_output or f"mnist_compute_benchmark_{args.selection_mode}.json"
    with (benchmark_dir / benchmark_name).open("w", encoding="utf-8") as handle:
        json.dump({"config": BENCHMARK_CONFIG, "device_info": device_info, "selection_mode": args.selection_mode, "results": results}, handle, indent=2)

    write_svg_scatter(results, summary_dir / "mnist_accuracy_vs_train_cost.svg")
    write_svg_bar(results, summary_dir / "mnist_train_step_cost.svg", "train_step_ms", "MNIST Train Step Time", "{:.2f} ms")
    write_svg_bar(results, summary_dir / "mnist_eval_step_cost.svg", "eval_step_ms", "MNIST Eval Step Time", "{:.2f} ms")
    write_svg_bar(results, summary_dir / "mnist_parameter_memory.svg", "parameter_memory_mb", "MNIST Parameter Memory", "{:.2f} MB")

    summary_text = build_summary_text(results, device_info)
    (summary_dir / "compute_summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
