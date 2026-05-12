from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_MAIN_RUNS = [
    "standard",
    "dropout",
    "bayesian",
    "bayesian_dropout_0p1",
    "bayesian_trust_none",
    "bayesian_dropout_0p1_trust_none",
    "bayesian_trust_depth_lambda0.15",
    "bayesian_trust_gradnorm",
    "bayesian_trust_gradvar_beta0.95",
    "bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0",
    "bayesian_trust_propagatedUncertainty_beta0.95_Q0.0001_P1.0_R1.0_lambda0.15",
    "bayesian_dropout_0p1_trust_depth_lambda0.15",
    "bayesian_dropout_0p1_trust_gradnorm",
    "bayesian_dropout_0p1_trust_gradvar_beta0.95",
    "bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0",
    "bayesian_dropout_0p1_trust_propagatedUncertainty_beta0.95_Q0.0001_P1.0_R1.0_lambda0.15",
]

EXPECTED_DROPOUT_PROP_RUNS = [
    "standard_trust_none",
    "dropout_trust_none",
    "dropout_trust_propagatedUncertainty_beta0.95_Q0.0001_P1.0_R1.0_lambda0.15",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_mnist_metrics(mnist_dir: Path, expected_runs: list[str]) -> tuple[dict, list[str], list[str]]:
    metrics_path = mnist_dir / "test_accuracy.json"
    if not metrics_path.exists():
        return {}, [str(metrics_path)], expected_runs

    metrics = load_json(metrics_path)
    missing_files = []
    missing_runs = [run_name for run_name in expected_runs if run_name not in metrics]
    for run_name in expected_runs:
        if run_name not in metrics:
            continue
        payload = metrics[run_name]
        for key in ["history", "test_metrics", "last_epoch_test_metrics"]:
            if key not in payload:
                missing_files.append(f"{metrics_path}:{run_name}.{key}")
    return metrics, missing_files, missing_runs


def check_checkpoints(mnist_dir: Path, run_names: list[str], selection_mode: str) -> list[str]:
    suffix = "_best.pt" if selection_mode == "best_val" else ".pt"
    checkpoint_dir = mnist_dir / "checkpoints"
    return [
        str(checkpoint_dir / f"{run_name}{suffix}")
        for run_name in run_names
        if not (checkpoint_dir / f"{run_name}{suffix}").exists()
    ]


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(">>> " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check finished experiment artifacts and regenerate plots without training.")
    parser.add_argument("--experiment", choices=["main", "dropout_prop", "both"], default="both")
    parser.add_argument("--selection-mode", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--include-shift-check", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    python_bin = sys.executable
    all_ok = True

    if args.experiment in {"main", "both"}:
        print("\n[main] Checking results/bayes_by_backprop/mnist")
        mnist_dir = root_dir / "results" / "bayes_by_backprop" / "mnist"
        metrics, missing_files, missing_runs = check_mnist_metrics(mnist_dir, EXPECTED_MAIN_RUNS)
        checkpoint_missing = check_checkpoints(mnist_dir, [name for name in EXPECTED_MAIN_RUNS if name in metrics], args.selection_mode)
        regression_metrics = root_dir / "results" / "bayes_by_backprop" / "regression" / "regression_metrics.json"

        if missing_runs:
            print("Missing runs:")
            for item in missing_runs:
                print(f"  - {item}")
        if missing_files:
            print("Missing metric fields/files:")
            for item in missing_files:
                print(f"  - {item}")
        if checkpoint_missing:
            print("Missing checkpoints:")
            for item in checkpoint_missing:
                print(f"  - {item}")
        if regression_metrics.exists():
            print(f"Regression metrics: found {regression_metrics.relative_to(root_dir)}")
        else:
            print(f"Regression metrics: missing {regression_metrics.relative_to(root_dir)}")

        main_ok = not missing_files and not missing_runs
        all_ok = all_ok and main_ok
        if main_ok:
            print("Main metrics look complete enough for summary plots.")
        if args.generate and main_ok:
            summary_dir = "results/bayes_by_backprop/summary_best_val" if args.selection_mode == "best_val" else "results/bayes_by_backprop/summary_last_epoch"
            run_command(
                [python_bin, "code/evaluation/summarize_results.py", "--selection-mode", args.selection_mode, "--summary-dir", summary_dir],
                root_dir,
                args.dry_run,
            )

    if args.experiment in {"dropout_prop", "both"}:
        print("\n[dropout_prop] Checking results/prop_dropout/mnist")
        mnist_dir = root_dir / "results" / "prop_dropout" / "mnist"
        metrics, missing_files, missing_runs = check_mnist_metrics(mnist_dir, EXPECTED_DROPOUT_PROP_RUNS)
        checkpoint_missing = check_checkpoints(mnist_dir, [name for name in EXPECTED_DROPOUT_PROP_RUNS if name in metrics], args.selection_mode)

        if missing_runs:
            print("Missing runs:")
            for item in missing_runs:
                print(f"  - {item}")
        if missing_files:
            print("Missing metric fields/files:")
            for item in missing_files:
                print(f"  - {item}")
        if checkpoint_missing:
            print("Missing checkpoints:")
            for item in checkpoint_missing:
                print(f"  - {item}")

        dropout_ok = not missing_files and not missing_runs
        all_ok = all_ok and dropout_ok
        if dropout_ok:
            print("Dropout-propagated metrics look complete enough for summary plots.")
        if args.generate and dropout_ok:
            run_command(
                [
                    python_bin,
                    "code/evaluation/summarize_mnist_dir.py",
                    "--mnist-dir",
                    "results/prop_dropout/mnist",
                    "--summary-dir",
                    "results/prop_dropout/summary",
                    "--selection-mode",
                    args.selection_mode,
                ],
                root_dir,
                args.dry_run,
            )

    if args.include_shift_check:
        shift_metrics = root_dir / "results" / "bayes_by_backprop" / "uncertainty_shift" / "uncertainty_shift_metrics.json"
        print("\n[shift] Checking uncertainty-shift metrics")
        if shift_metrics.exists():
            print(f"Shift metrics: found {shift_metrics.relative_to(root_dir)}")
        else:
            print(f"Shift metrics: missing {shift_metrics.relative_to(root_dir)}")
            all_ok = False

    if all_ok:
        print("\nAll requested checks passed.")
    else:
        print("\nSome requested artifacts are missing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
