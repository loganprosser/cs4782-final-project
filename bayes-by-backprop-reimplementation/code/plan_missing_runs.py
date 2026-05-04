from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MNIST_ALIASES = {
    "standard": "standard_trust_none",
    "dropout": "dropout_trust_none",
    "bayesian": "bayesian_trust_none",
    "bayesian_dropout_0p1": "bayesian_dropout_0p1_trust_none",
}

MNIST_RUN_COMMANDS = {
    "standard_trust_none": 'run_cmd "${PYTHON_BIN}" code/train_mnist.py --model standard --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"',
    "dropout_trust_none": 'run_cmd "${PYTHON_BIN}" code/train_mnist.py --model dropout --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"',
    "bayesian_trust_none": 'run_cmd "${PYTHON_BIN}" code/train_mnist.py --model bayesian --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"',
    "bayesian_dropout_0p1_trust_none": 'run_cmd "${PYTHON_BIN}" code/train_mnist.py --model bayesian --bayesian-dropout 0.1 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"',
}

REGRESSION_ALIAS = {"bayesian": "bayesian_trust_none"}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def materialize_aliases(root_dir: Path) -> list[str]:
    changed = []

    mnist_metrics_path = root_dir / "results" / "mnist" / "test_accuracy.json"
    mnist_metrics = load_json(mnist_metrics_path)
    for alias, source in MNIST_ALIASES.items():
        if alias not in mnist_metrics and source in mnist_metrics:
            aliased = dict(mnist_metrics[source])
            aliased["run_name"] = alias
            mnist_metrics[alias] = aliased
            changed.append(f"results/mnist/test_accuracy.json:{alias} <- {source}")
    if changed:
        save_json(mnist_metrics_path, mnist_metrics)

    regression_metrics_path = root_dir / "results" / "regression" / "regression_metrics.json"
    regression_metrics = load_json(regression_metrics_path)
    for alias, source in REGRESSION_ALIAS.items():
        if alias not in regression_metrics and source in regression_metrics:
            aliased = dict(regression_metrics[source])
            aliased["run_name"] = alias
            regression_metrics[alias] = aliased
            changed.append(f"results/regression/regression_metrics.json:{alias} <- {source}")
    if regression_metrics and any(item.startswith("results/regression") for item in changed):
        save_json(regression_metrics_path, regression_metrics)

    return changed


def missing_mnist_sources(root_dir: Path) -> list[str]:
    metrics = load_json(root_dir / "results" / "mnist" / "test_accuracy.json")
    missing = []
    for alias, source in MNIST_ALIASES.items():
        if alias in metrics or source in metrics:
            continue
        missing.append(source)
    return missing


def needs_regression(root_dir: Path) -> bool:
    metrics = load_json(root_dir / "results" / "regression" / "regression_metrics.json")
    return not ("bayesian" in metrics or "bayesian_trust_none" in metrics)


def write_run_script(root_dir: Path, output_path: Path, missing_mnist: list[str], regression_missing: bool) -> None:
    commands = [MNIST_RUN_COMMANDS[name] for name in missing_mnist]
    if regression_missing:
        commands.append(
            'run_cmd "${PYTHON_BIN}" code/train_regression.py --model bayesian --update-trust-mode none --epochs "${REGRESSION_EPOCHS}" --batch-size "${REGRESSION_BATCH_SIZE}" --seed "${SEED}"'
        )

    lines = [
        "#!/usr/bin/env bash",
        "",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PYTHON_BIN="${ROOT_DIR}/venv/bin/python"',
        "",
        'MNIST_EPOCHS="${MNIST_EPOCHS:-10}"',
        'REGRESSION_EPOCHS="${REGRESSION_EPOCHS:-2000}"',
        'MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-128}"',
        'REGRESSION_BATCH_SIZE="${REGRESSION_BATCH_SIZE:-64}"',
        'SEED="${SEED:-0}"',
        "",
        "run_cmd() {",
        "  echo",
        '  echo ">>> $*"',
        '  "$@"',
        "}",
        "",
        'if [[ ! -x "${PYTHON_BIN}" ]]; then',
        '  echo "Expected project environment at ${PYTHON_BIN}"',
        "  exit 1",
        "fi",
        "",
        'cd "${ROOT_DIR}"',
        "",
    ]

    if commands:
        lines.extend(commands)
        lines.append("")
    else:
        lines.append('echo "No training runs are missing; only aliases/plots will be refreshed."')
        lines.append("")

    lines.extend(
        [
            'run_cmd "${PYTHON_BIN}" code/plan_missing_runs.py --materialize-aliases',
            'run_cmd "${PYTHON_BIN}" code/check_and_generate_plots.py --experiment both --selection-mode best_val --generate',
            "",
            'echo',
            'echo "Missing-run backfill and plot generation completed."',
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(output_path, 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan exactly which missing runs are needed for plot generation.")
    parser.add_argument("--write-script", type=str, default="temp_run_missing_and_plot.sh")
    parser.add_argument("--materialize-aliases", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]

    if args.materialize_aliases:
        changed = materialize_aliases(root_dir)
        if changed:
            print("Materialized compatibility aliases:")
            for item in changed:
                print(f"  - {item}")
        else:
            print("No compatibility aliases needed.")
        return

    changed = materialize_aliases(root_dir)
    missing_mnist = missing_mnist_sources(root_dir)
    regression_missing = needs_regression(root_dir)
    script_path = root_dir / args.write_script
    write_run_script(root_dir, script_path, missing_mnist, regression_missing)

    if changed:
        print("Materialized compatibility aliases:")
        for item in changed:
            print(f"  - {item}")

    if missing_mnist:
        print("MNIST runs that still need training:")
        for item in missing_mnist:
            print(f"  - {item}")
    else:
        print("MNIST runs needed for the main plots are present.")

    if regression_missing:
        print("Regression run still needs training:")
        print("  - bayesian_trust_none, then alias to bayesian")
    else:
        print("Regression metrics needed for the main plots are present.")

    print(f"Wrote runnable script: {script_path.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
