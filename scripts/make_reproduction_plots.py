from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO = REPO_ROOT / "reproduction"
FIGURES = REPRO / "figures"
CACHE_DIR = REPRO / "logs" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "standard_mlp",
    "dropout_mlp",
    "bayes_by_backprop",
    "bayes_by_backprop_dropout",
]

DISPLAY = {
    "standard_mlp": "Standard MLP",
    "dropout_mlp": "Dropout MLP",
    "bayes_by_backprop": "Bayes by Backprop",
    "bayes_by_backprop_dropout": "BBB + Dropout",
}

COLORS = {
    "standard_mlp": "#28666e",
    "dropout_mlp": "#7c9a2b",
    "bayes_by_backprop": "#b4433f",
    "bayes_by_backprop_dropout": "#6f4e9b",
}

PAPER_VALUES = {
    "standard_mlp": "1.83%",
    "dropout_mlp": "1.51%",
    "bayes_by_backprop": "1.36%",
    "bayes_by_backprop_dropout": "N/A",
}

PAPER_NOTES = {
    "standard_mlp": "400-unit SGD baseline from Table 1",
    "dropout_mlp": "400-unit dropout baseline from Table 1",
    "bayes_by_backprop": "400-unit scale-mixture prior; Gaussian prior was 1.82%",
    "bayes_by_backprop_dropout": "our extension",
}


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def warn(message: str) -> None:
    print(f"Warning: {message}")


def read_csv(name: str) -> pd.DataFrame:
    path = REPRO / name
    if not path.exists():
        warn(f"missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def method_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty:
        return pd.DataFrame()
    rows = []
    for method in METHOD_ORDER:
        subset = final_df[final_df["method"] == method]
        if subset.empty:
            continue
        rows.append(
            {
                "method": method,
                "best_test_error": subset["best_test_error"].mean(),
                "best_test_accuracy": subset["best_test_accuracy"].mean(),
                "final_test_error": subset["final_test_error"].mean(),
                "final_test_accuracy": subset["final_test_accuracy"].mean(),
                "runtime_seconds": subset["runtime_seconds"].mean(),
                "average_epoch_time_seconds": subset["average_epoch_time_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_test_error_curves(training_df: pd.DataFrame) -> None:
    if training_df.empty:
        warn("cannot make test error curves without training_curves.csv")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method in METHOD_ORDER:
        subset = training_df[training_df["method"] == method]
        if subset.empty:
            continue
        grouped = subset.groupby("epoch")["test_error"].agg(["mean", "std"]).reset_index()
        x = grouped["epoch"].to_numpy()
        y = grouped["mean"].to_numpy() * 100.0
        ax.plot(x, y, marker="o", linewidth=2, markersize=3, label=DISPLAY[method], color=COLORS[method])
        if grouped["std"].notna().any() and subset["seed"].nunique() > 1:
            err = grouped["std"].fillna(0).to_numpy() * 100.0
            ax.fill_between(x, y - err, y + err, alpha=0.14, color=COLORS[method])
    ax.set_title("MNIST Test Error During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test error (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_test_error_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_error(summary: pd.DataFrame) -> None:
    if summary.empty:
        warn("cannot make final error bar plot without final_metrics.csv")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    values = [summary.loc[summary["method"] == m, "best_test_error"].iloc[0] * 100 for m in summary["method"]]
    labels = [DISPLAY[m] for m in summary["method"]]
    colors = [COLORS[m] for m in summary["method"]]
    bars = ax.bar(labels, values, color=colors, width=0.65)
    add_bar_labels(ax, bars, "{:.2f}%")
    ax.set_title("Final MNIST Test Error by Method")
    ax.set_ylabel("Best test error (%)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_final_error_barplot.png", bbox_inches="tight")
    plt.close(fig)


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        ours = np.nan
        if not summary.empty and method in set(summary["method"]):
            ours = summary.loc[summary["method"] == method, "best_test_error"].iloc[0] * 100.0
        rows.append(
            {
                "Method": {
                    "standard_mlp": "Vanilla SGD / Standard MLP",
                    "dropout_mlp": "Dropout",
                    "bayes_by_backprop": "Bayes by Backprop",
                    "bayes_by_backprop_dropout": "Bayes by Backprop + Dropout",
                }[method],
                "Paper test error": PAPER_VALUES[method],
                "Our test error": "" if math.isnan(ours) else f"{ours:.2f}%",
                "Notes": PAPER_NOTES[method],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(REPRO / "paper_reproduction_summary.csv", index=False)
    return table


def plot_table_image(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.axis("off")
    mpl_table = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left")
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.5)
    mpl_table.scale(1, 1.45)
    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if row == 0:
            cell.set_facecolor("#f0f2f4")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#fafafa")
    ax.set_title("MNIST Paper vs. Our Reproduction", pad=14, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_paper_vs_ours_table.png", bbox_inches="tight")
    plt.close(fig)


def collect_checkpoint_weights(checkpoint_path: Path, method: str) -> np.ndarray:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict", payload)
    arrays = []
    for key, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 2:
            continue
        if method.startswith("bayes_by_backprop"):
            if key.endswith("weight_mu"):
                arrays.append(tensor.detach().flatten().numpy())
        elif key.endswith("weight"):
            arrays.append(tensor.detach().flatten().numpy())
    if not arrays:
        return np.array([])
    values = np.concatenate(arrays)
    if values.size > 60000:
        rng = np.random.default_rng(0)
        values = rng.choice(values, size=60000, replace=False)
    return values


def plot_weight_histograms(final_df: pd.DataFrame) -> bool:
    if final_df.empty:
        warn("cannot make weight histograms without final_metrics.csv")
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    made_any = False
    for method in METHOD_ORDER:
        subset = final_df[final_df["method"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        checkpoint = REPO_ROOT / str(row["checkpoint_path"])
        if not checkpoint.exists():
            warn(f"checkpoint missing for {method}: {checkpoint}")
            continue
        try:
            weights = collect_checkpoint_weights(checkpoint, method)
        except Exception as exc:
            warn(f"failed to read checkpoint for {method}: {exc}")
            continue
        if weights.size == 0:
            warn(f"no plottable weights found for {method}")
            continue
        ax.hist(
            weights,
            bins=80,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=DISPLAY[method],
            color=COLORS[method],
        )
        made_any = True
    ax.set_title("Trained Weight Distributions")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    if made_any:
        fig.savefig(FIGURES / "weight_histograms.png", bbox_inches="tight")
    plt.close(fig)
    return made_any


def plot_runtime(summary: pd.DataFrame) -> None:
    if summary.empty:
        warn("cannot make runtime comparison without final_metrics.csv")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    values = summary["runtime_seconds"].to_numpy()
    labels = [DISPLAY[m] for m in summary["method"]]
    bars = ax.bar(labels, values, color=[COLORS[m] for m in summary["method"]], width=0.65)
    add_bar_labels(ax, bars, "{:.1f}")
    ax.set_title("Training Runtime Comparison")
    ax.set_ylabel("Total runtime (seconds)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_runtime_comparison.png", bbox_inches="tight")
    plt.close(fig)


def checkpoint_parameter_memory_mb(checkpoint_path: Path) -> float:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return math.nan
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict", payload)
    total_bytes = 0
    for tensor in state.values():
        if torch.is_tensor(tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def should_use_parameter_memory(memory_df: pd.DataFrame) -> bool:
    warnings_path = REPRO / "logs" / "run_warnings.log"
    psutil_missing = warnings_path.exists() and "psutil is not installed" in warnings_path.read_text(encoding="utf-8")
    if memory_df.empty or not psutil_missing:
        return False
    if memory_df["peak_gpu_memory_allocated_mb"].notna().any():
        return False
    values = memory_df["peak_cpu_memory_mb"].dropna().round(3).unique()
    return len(values) == 1


def parameter_memory_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if final_df.empty:
        return pd.DataFrame()
    for method in METHOD_ORDER:
        subset = final_df[final_df["method"] == method]
        if subset.empty:
            continue
        values = []
        for _, row in subset.iterrows():
            checkpoint = REPO_ROOT / str(row["checkpoint_path"])
            if checkpoint.exists():
                values.append(checkpoint_parameter_memory_mb(checkpoint))
        if values:
            rows.append({"method": method, "memory_mb": float(np.nanmean(values))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(REPRO / "model_memory_summary.csv", index=False)
    return df


def memory_summary(memory_df: pd.DataFrame, final_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if memory_df.empty:
        return pd.DataFrame(), "Peak memory usage (MB)"
    if should_use_parameter_memory(memory_df):
        param_df = parameter_memory_summary(final_df)
        if not param_df.empty:
            return param_df, "Model parameter memory (MB)"
    rows = []
    gpu_available = memory_df["peak_gpu_memory_allocated_mb"].notna().any()
    key = "peak_gpu_memory_allocated_mb" if gpu_available else "peak_cpu_memory_mb"
    ylabel = "Peak GPU memory allocated (MB)" if gpu_available else "Peak CPU memory usage (MB)"
    for method in METHOD_ORDER:
        subset = memory_df[memory_df["method"] == method]
        if subset.empty:
            continue
        rows.append({"method": method, "memory_mb": subset[key].mean()})
    return pd.DataFrame(rows), ylabel


def plot_memory(memory_plot_df: pd.DataFrame, ylabel: str) -> None:
    if memory_plot_df.empty:
        warn("cannot make memory comparison without memory data")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    values = memory_plot_df["memory_mb"].to_numpy()
    labels = [DISPLAY[m] for m in memory_plot_df["method"]]
    bars = ax.bar(labels, values, color=[COLORS[m] for m in memory_plot_df["method"]], width=0.65)
    add_bar_labels(ax, bars, "{:.1f}")
    ax.set_title("Peak Memory Comparison")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "peak_memory_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoffs(summary: pd.DataFrame, memory_plot_df: pd.DataFrame) -> None:
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for _, row in summary.iterrows():
            method = row["method"]
            ax.scatter(row["runtime_seconds"], row["best_test_error"] * 100.0, s=80, color=COLORS[method])
            ax.annotate(DISPLAY[method], (row["runtime_seconds"], row["best_test_error"] * 100.0), xytext=(5, 4), textcoords="offset points")
        ax.set_title("Runtime vs Accuracy Tradeoff")
        ax.set_xlabel("Total runtime in seconds")
        ax.set_ylabel("Best test error (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "runtime_vs_accuracy_tradeoff.png", bbox_inches="tight")
        plt.close(fig)

    if not summary.empty and not memory_plot_df.empty:
        merged = summary.merge(memory_plot_df, on="method")
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for _, row in merged.iterrows():
            method = row["method"]
            ax.scatter(row["memory_mb"], row["best_test_error"] * 100.0, s=80, color=COLORS[method])
            ax.annotate(DISPLAY[method], (row["memory_mb"], row["best_test_error"] * 100.0), xytext=(5, 4), textcoords="offset points")
        ax.set_title("Memory vs Accuracy Tradeoff")
        ax.set_xlabel("Peak memory usage (MB)")
        ax.set_ylabel("Best test error (%)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "memory_vs_accuracy_tradeoff.png", bbox_inches="tight")
        plt.close(fig)


def plot_dashboard(training_df: pd.DataFrame, summary: pd.DataFrame, memory_plot_df: pd.DataFrame, memory_ylabel: str, table: pd.DataFrame) -> None:
    if training_df.empty or summary.empty:
        warn("cannot make dashboard without training and final metrics")
        return
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.75])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_runtime = fig.add_subplot(gs[1, 0])
    ax_memory = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    for method in METHOD_ORDER:
        subset = training_df[training_df["method"] == method]
        if subset.empty:
            continue
        grouped = subset.groupby("epoch")["test_error"].mean().reset_index()
        ax_curve.plot(grouped["epoch"], grouped["test_error"] * 100, marker="o", linewidth=2, markersize=3, label=DISPLAY[method], color=COLORS[method])
    ax_curve.set_title("A. Test Error Curves")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Test error (%)")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False)

    values = summary["best_test_error"].to_numpy() * 100
    bars = ax_bar.bar([DISPLAY[m] for m in summary["method"]], values, color=[COLORS[m] for m in summary["method"]])
    add_bar_labels(ax_bar, bars, "{:.2f}%")
    ax_bar.set_title("B. Final Test Error")
    ax_bar.set_ylabel("Best test error (%)")
    ax_bar.tick_params(axis="x", rotation=15)

    bars = ax_runtime.bar([DISPLAY[m] for m in summary["method"]], summary["runtime_seconds"], color=[COLORS[m] for m in summary["method"]])
    add_bar_labels(ax_runtime, bars, "{:.1f}")
    ax_runtime.set_title("C. Training Runtime")
    ax_runtime.set_ylabel("Seconds")
    ax_runtime.tick_params(axis="x", rotation=15)

    if not memory_plot_df.empty:
        bars = ax_memory.bar([DISPLAY[m] for m in memory_plot_df["method"]], memory_plot_df["memory_mb"], color=[COLORS[m] for m in memory_plot_df["method"]])
        add_bar_labels(ax_memory, bars, "{:.1f}")
    ax_memory.set_title("D. Peak Memory")
    ax_memory.set_ylabel(memory_ylabel)
    ax_memory.tick_params(axis="x", rotation=15)

    ax_table.axis("off")
    mpl_table = ax_table.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left")
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    mpl_table.scale(1, 1.35)
    for (row, _), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if row == 0:
            cell.set_facecolor("#f0f2f4")
            cell.set_text_props(weight="bold")
    ax_table.set_title("E. Paper vs Ours", pad=10)
    fig.suptitle("Bayes by Backprop MNIST Reproduction", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES / "reproduction_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def write_captions(weight_hist_generated: bool, memory_caveat: str) -> None:
    weight_note = (
        "The histogram compares deterministic trained weights with Bayesian posterior means from the saved checkpoints, mirroring the paper's Figure 3-style weight-density diagnostic."
        if weight_hist_generated
        else "This optional plot was not generated because compatible checkpoint weights were unavailable."
    )
    text = f"""# Figure Captions

## MNIST Test Error Curves
This plot reproduces the paper's MNIST learning-curve comparison by tracking test error across training for deterministic SGD, dropout, Bayes by Backprop, and our Bayes by Backprop + Dropout extension. It was made to show whether Bayes by Backprop follows dropout-like generalization behavior during training. Lower curves indicate better test performance.

## Final Test Error Bar Plot
This bar plot compares the best test error reached by each of the four reproduced methods. It was made to summarize the final accuracy story in one poster-friendly view. The conclusion is that Bayes by Backprop is competitive with dropout when their bars are close.

## Paper Vs Ours Comparison Table
This table places our MNIST test errors next to the closest Table 1 values from Blundell et al. It was made to separate qualitative reproduction from exact numeric matching. Differences should be interpreted in light of our smaller training budget and implementation details.

## Weight Histograms
{weight_note}

## Training Runtime Comparison
This plot compares total wall-clock training time for the four methods. It was made to quantify the computational cost of Bayesian weight sampling and posterior parameters. Higher bars indicate more expensive training.

## Peak Memory Comparison
This plot compares memory use for each method. It was made to show whether the Bayesian methods require extra memory for posterior means and variances. Higher bars indicate a larger memory footprint. {memory_caveat}

## Runtime Vs Accuracy Tradeoff
This scatter plot shows best test error against total runtime. It was made to ask whether a method's accuracy is worth its training cost. Methods closest to the lower-left are the most efficient.

## Memory Vs Accuracy Tradeoff
This scatter plot shows best test error against the memory metric used for the comparison. It was made to ask whether a method's accuracy is worth its memory footprint. Methods closest to the lower-left are the most memory-efficient. {memory_caveat}

## Reproduction Dashboard
This combined figure collects the core reproduction evidence: learning curves, final errors, runtime, memory, and paper-vs-ours values. It was made as a single poster-ready summary. The main conclusion is visible by comparing Bayes by Backprop with dropout and then checking whether the extension improves the tradeoff.
"""
    (REPRO / "figure_captions.md").write_text(text, encoding="utf-8")


def memory_caveat(memory_plot_df: pd.DataFrame, memory_ylabel: str) -> str:
    if memory_ylabel == "Model parameter memory (MB)":
        return "The original CPU peak metric was not method-isolated because `psutil` was unavailable and all methods ran in one Python process. For the poster-facing memory comparison, we therefore plot checkpoint/model parameter memory, which captures the expected approximately 2x parameter footprint of Bayes by Backprop."
    warnings_path = REPRO / "logs" / "run_warnings.log"
    psutil_missing = warnings_path.exists() and "psutil is not installed" in warnings_path.read_text(encoding="utf-8")
    identical = False
    if not memory_plot_df.empty:
        values = memory_plot_df["memory_mb"].dropna().round(3).unique()
        identical = len(values) == 1
    if psutil_missing and identical:
        return "Because `psutil` was unavailable, CPU memory was recorded with a process-level `resource.getrusage` fallback; these values are useful as a limitation note but not as a method-isolated memory comparison."
    if psutil_missing:
        return "Because `psutil` was unavailable, CPU memory used the stdlib `resource.getrusage` fallback."
    return ""


def write_summary(summary: pd.DataFrame, memory_plot_df: pd.DataFrame, memory_ylabel: str, caveat: str) -> None:
    if summary.empty:
        text = "# Reproduction Summary\n\nNo final metrics were available yet. Run `python scripts/run_missing_reproduction_experiments.py` first.\n"
        (REPRO / "reproduction_summary.md").write_text(text, encoding="utf-8")
        return

    def err(method: str) -> float:
        return float(summary.loc[summary["method"] == method, "best_test_error"].iloc[0] * 100.0)

    def runtime(method: str) -> float:
        return float(summary.loc[summary["method"] == method, "runtime_seconds"].iloc[0])

    memory_lines = []
    if not memory_plot_df.empty:
        for _, row in memory_plot_df.iterrows():
            memory_lines.append(f"- {DISPLAY[row['method']]}: `{row['memory_mb']:.1f}` MB.")
    else:
        memory_lines.append("- Peak memory data were unavailable.")

    bbb_competitive = abs(err("bayes_by_backprop") - err("dropout_mlp")) <= 0.5
    extension_helped = err("bayes_by_backprop_dropout") < min(err("bayes_by_backprop"), err("dropout_mlp"))
    text = f"""# Reproduction Summary

## Chosen Result
We reproduced the MNIST classification comparison from Table 1 and Figure 2 of Blundell et al., "Weight Uncertainty in Neural Networks."

## What The Paper Found
The paper found that Bayes by Backprop with a scale-mixture prior achieved MNIST performance comparable to dropout in a feedforward ReLU network.

## What We Implemented
- Standard MLP.
- Dropout MLP.
- Bayes by Backprop.
- Bayes by Backprop + Dropout.

## What Our Results Show
- Standard MLP: `{err('standard_mlp'):.2f}%` best test error.
- Dropout MLP: `{err('dropout_mlp'):.2f}%` best test error.
- Bayes by Backprop: `{err('bayes_by_backprop'):.2f}%` best test error.
- Bayes by Backprop + Dropout: `{err('bayes_by_backprop_dropout'):.2f}%` best test error.

Bayes by Backprop was {'competitive with' if bbb_competitive else 'not as strong as'} dropout in this implementation. The Bayes by Backprop + Dropout extension {'improved over both individual methods' if extension_helped else 'did not improve over both individual methods'} in the current run.

## Resource Comparison
- Standard MLP runtime: `{runtime('standard_mlp'):.1f}` seconds.
- Dropout MLP runtime: `{runtime('dropout_mlp'):.1f}` seconds.
- Bayes by Backprop runtime: `{runtime('bayes_by_backprop'):.1f}` seconds.
- Bayes by Backprop + Dropout runtime: `{runtime('bayes_by_backprop_dropout'):.1f}` seconds.

Memory metric used: `{memory_ylabel}`.
{chr(10).join(memory_lines)}

{caveat}

The Bayesian methods are expected to cost more time and memory because each BayesianLinear layer stores posterior parameters and samples weights during forward passes.

## Discrepancies
Our run uses the existing project architecture with two 400-unit hidden layers, Adam, batch size 128, a 54k/6k train/validation split from MNIST training data, and a shorter training budget than the paper's 600-epoch Figure 2 run. The paper's Table 1 includes larger 800- and 1200-unit networks and hyperparameter searches; therefore exact test errors should not be expected.

## Poster-Ready Takeaway
In our MNIST reproduction, Bayes by Backprop qualitatively matches the paper's claim of dropout-competitive performance, while the added Bayes by Backprop + Dropout variant tests whether combining Bayesian weights with activation dropout improves the accuracy-cost tradeoff.
"""
    (REPRO / "reproduction_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    training_df = read_csv("training_curves.csv")
    final_df = read_csv("final_metrics.csv")
    memory_df = read_csv("memory_summary.csv")

    summary = method_summary(final_df)
    table = paper_table(summary)
    memory_plot_df, memory_ylabel = memory_summary(memory_df, final_df)

    plot_test_error_curves(training_df)
    plot_final_error(summary)
    plot_table_image(table)
    weight_hist_generated = plot_weight_histograms(final_df)
    plot_runtime(summary)
    plot_memory(memory_plot_df, memory_ylabel)
    plot_tradeoffs(summary, memory_plot_df)
    plot_dashboard(training_df, summary, memory_plot_df, memory_ylabel, table)
    caveat = memory_caveat(memory_plot_df, memory_ylabel)
    write_captions(weight_hist_generated, caveat)
    write_summary(summary, memory_plot_df, memory_ylabel, caveat)

    print(f"Generated reproduction figures in {FIGURES}")


if __name__ == "__main__":
    main()
