from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label(row: pd.Series) -> str:
    return f"{row['model']} {row['method']}"


def _plot_epoch_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    for (model, method), group in df.groupby(["model", "method"], sort=False):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group[metric], marker="o", linewidth=1.8, label=f"{model} {method}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _layer_metric_columns(df: pd.DataFrame, metric: str) -> list[str]:
    return [col for col in df.columns if col.startswith("layer:") and col.endswith(f":{metric}")]


def _layer_name(column: str, metric: str) -> str:
    return column.removeprefix("layer:").removesuffix(f":{metric}").replace("/", ".")


def _plot_kalman_layer_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path) -> None:
    cols = _layer_metric_columns(df, metric)
    kalman_df = df[df["method"].isin(["kalman", "dropout_kalman"])]
    if kalman_df.empty or not cols:
        return
    plt.figure(figsize=(12, 7))
    for (model, method), group in kalman_df.groupby(["model", "method"], sort=False):
        group = group.sort_values("epoch")
        for col in cols:
            if group[col].notna().any():
                label = f"{model} {method} {_layer_name(col, metric)}"
                plt.plot(group["epoch"], group[col], linewidth=1.2, label=label)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_raw_vs_filtered(df: pd.DataFrame, output: Path) -> None:
    raw_cols = _layer_metric_columns(df, "raw_grad_norm")
    filt_cols = _layer_metric_columns(df, "filtered_grad_norm")
    kalman_df = df[df["method"].isin(["kalman", "dropout_kalman"])]
    if kalman_df.empty or not raw_cols or not filt_cols:
        return
    plt.figure(figsize=(12, 7))
    for (model, method), group in kalman_df.groupby(["model", "method"], sort=False):
        group = group.sort_values("epoch")
        for raw_col in raw_cols:
            layer = _layer_name(raw_col, "raw_grad_norm")
            filt_col = f"layer:{layer.replace('.', '/')}:filtered_grad_norm"
            if filt_col not in group:
                continue
            if group[raw_col].notna().any():
                plt.plot(group["epoch"], group[raw_col], linewidth=1.0, alpha=0.65, label=f"{model} {method} {layer} raw")
            if group[filt_col].notna().any():
                plt.plot(
                    group["epoch"],
                    group[filt_col],
                    linewidth=1.5,
                    linestyle="--",
                    label=f"{model} {method} {layer} filtered",
                )
    plt.xlabel("epoch")
    plt.ylabel("gradient norm")
    plt.title("Raw vs filtered gradient norm by layer")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_summary_bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path) -> None:
    labels = summary.apply(_label, axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, summary[metric])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, max(1.0, float(summary[metric].max()) * 1.05))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_all(metrics_csv: Path, summary_csv: Path, output_dir: Path) -> None:
    output_dir = _ensure_dir(output_dir)
    df = pd.read_csv(metrics_csv)
    summary = pd.read_csv(summary_csv)

    _plot_epoch_metric(df, "test_accuracy", "test accuracy", "Test accuracy vs epoch", output_dir / "test_accuracy_vs_epoch.png")
    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        _plot_epoch_metric(
            df,
            "val_accuracy",
            "validation accuracy",
            "Validation accuracy vs epoch",
            output_dir / "validation_accuracy_vs_epoch.png",
        )
    _plot_epoch_metric(df, "test_loss", "test loss", "Test loss vs epoch", output_dir / "test_loss_vs_epoch.png")
    _plot_epoch_metric(df, "train_loss", "train loss", "Train loss vs epoch", output_dir / "train_loss_vs_epoch.png")
    _plot_epoch_metric(
        df,
        "generalization_gap",
        "train accuracy - test accuracy",
        "Generalization gap vs epoch",
        output_dir / "generalization_gap_vs_epoch.png",
    )
    _plot_kalman_layer_metric(
        df,
        "kalman_gain",
        "average Kalman gain",
        "Average Kalman gain by layer",
        output_dir / "kalman_gain_by_layer.png",
    )
    _plot_kalman_layer_metric(
        df,
        "measurement_R",
        "average measurement uncertainty R",
        "Average measurement uncertainty R by layer",
        output_dir / "measurement_uncertainty_by_layer.png",
    )
    _plot_raw_vs_filtered(df, output_dir / "raw_vs_filtered_grad_norm.png")
    _plot_summary_bar(summary, "final_test_acc", "final test accuracy", "Final test accuracy", output_dir / "final_test_accuracy_bar.png")
    if "best_val_acc" in summary.columns:
        _plot_summary_bar(summary, "best_val_acc", "best validation accuracy", "Best validation accuracy", output_dir / "best_validation_accuracy_bar.png")
    _plot_summary_bar(
        summary,
        "best_test_acc",
        "test accuracy at best validation epoch",
        "Test accuracy at best validation epoch",
        output_dir / "best_test_accuracy_bar.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Kalman gradient trust experiment results.")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    plot_all(
        args.result_dir / "metrics" / "metrics.csv",
        args.result_dir / "summaries" / "final_summary.csv",
        args.result_dir / "plots",
    )
