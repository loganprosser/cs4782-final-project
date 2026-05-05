from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO = REPO_ROOT / "reproduction"
BBB = REPO_ROOT / "bayes-by-backprop-reimplementation"


def find_files(patterns: list[str], roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            paths.extend(root.rglob(pattern))
    return sorted({p for p in paths if "venv" not in p.parts})


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_accuracy_table() -> list[dict[str, str]]:
    table = BBB / "results" / "mnist" / "accuracy_table.csv"
    if not table.exists():
        return []
    with table.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def inspect_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def existing_core_results() -> list[str]:
    rows = read_accuracy_table()
    wanted = {
        "standard": "standard_mlp",
        "dropout": "dropout_mlp",
        "bayesian": "bayes_by_backprop",
        "bayesian_dropout_0p1": "bayes_by_backprop_dropout",
    }
    lines = []
    for row in rows:
        model = row.get("Model", "")
        if model in wanted:
            lines.append(
                f"- `{wanted[model]}` exists as old run `{model}`: test accuracy `{float(row['Test Accuracy']):.4f}`, test error `{100 * float(row['Test Error']):.2f}%`."
            )
    return lines


def reproduction_status() -> list[str]:
    required = [
        "training_curves.csv",
        "final_metrics.csv",
        "resource_metrics.csv",
        "runtime_summary.csv",
        "memory_summary.csv",
        "paper_reproduction_summary.csv",
        "reproduction_summary.md",
        "figure_captions.md",
    ]
    lines = []
    for name in required:
        path = REPRO / name
        status = "present" if path.exists() else "missing"
        lines.append(f"- `{rel(path)}`: {status}.")
    return lines


def checkpoint_status() -> list[str]:
    old_ckpts = find_files(["*.pt", "*.pth", "*.ckpt"], [BBB / "results" / "mnist" / "checkpoints"])
    repro_ckpts = find_files(["*.pt", "*.pth", "*.ckpt"], [REPRO / "checkpoints"])
    lines = [
        f"- Old MNIST checkpoints found: `{len(old_ckpts)}` in `bayes-by-backprop-reimplementation/results/mnist/checkpoints/`.",
        f"- Reproduction checkpoints found: `{len(repro_ckpts)}` in `reproduction/checkpoints/`.",
    ]
    for path in repro_ckpts[:8]:
        lines.append(f"  - `{rel(path)}`")
    return lines


def dependency_notes() -> list[str]:
    notes = []
    for package in ["torch", "torchvision", "pandas", "matplotlib", "numpy", "psutil"]:
        installed = importlib.util.find_spec(package) is not None
        notes.append(f"- `{package}` importable in the current interpreter: `{installed}`.")
    notes.append("- The checked project virtual environment is `bayes-by-backprop-reimplementation/venv/bin/python`; use it if the system `python3` lacks ML dependencies.")
    if importlib.util.find_spec("psutil") is None:
        notes.append("- `psutil` is not installed in the current interpreter. The reproduction runner falls back to `resource.getrusage` for CPU memory when possible and logs a warning.")
    return notes


def build_markdown() -> str:
    script_files = find_files(["*.py", "*.sh"], [BBB, REPO_ROOT / "scripts", REPO_ROOT / "kalmanalgo", REPO_ROOT / "loganalgo"])
    data_files = find_files(["*.csv", "*.json", "*.log", "*.txt"], [BBB / "results", REPO_ROOT / "loganalgo" / "results", REPRO])
    plot_files = find_files(["*.png", "*.svg", "*.pdf"], [BBB / "results", BBB / "poster", BBB / "report", REPRO / "figures"])
    md_files = find_files(["*.md"], [REPO_ROOT])
    rows = read_accuracy_table()
    aggregate = inspect_json(BBB / "results" / "mnist" / "test_accuracy.json")

    history_note = "missing"
    if aggregate:
        core = [aggregate.get(k, {}) for k in ["standard", "dropout", "bayesian", "bayesian_dropout_0p1"]]
        if all(item.get("history") for item in core):
            history_note = "train/validation histories exist for the four old core runs, but per-epoch test metrics and runtime/memory metrics do not."

    old_results = existing_core_results()
    if not old_results:
        old_results = ["- No complete old four-method accuracy table was found."]

    text = f"""# Reproduction Audit

## Scope
This audit covers the repository rooted at `{REPO_ROOT}` with special attention to `bayes-by-backprop-reimplementation/`, which contains the Bayes by Backprop MNIST implementation.

## Existing Training Scripts
Key scripts already present:
{chr(10).join(f"- `{rel(path)}`" for path in script_files[:60])}

Important existing MNIST entry points:
- `bayes-by-backprop-reimplementation/code/train_mnist.py` trains `standard`, `dropout`, and `bayesian` models. `bayesian` with `--bayesian-dropout 0.1` implements Bayes by Backprop + Dropout.
- `bayes-by-backprop-reimplementation/code/benchmark_mnist_compute.py` benchmarks per-step compute cost.
- `bayes-by-backprop-reimplementation/code/summarize_results.py` builds the old result summaries.
- `scripts/run_missing_reproduction_experiments.py` is the new reproduction runner that writes the requested CSVs, logs, and checkpoints under `reproduction/`.
- `scripts/make_reproduction_plots.py` is the new plot/report generator for the requested figures and markdown.

## Existing Model Implementations
- Vanilla/standard MLP: `StandardMLP` in `bayes-by-backprop-reimplementation/code/models.py`.
- Dropout MLP: `DropoutMLP` in `bayes-by-backprop-reimplementation/code/models.py`.
- Bayes by Backprop MLP: `BayesianMLP` using `BayesianLinear` in `bayes-by-backprop-reimplementation/code/models.py` and `bayes-by-backprop-reimplementation/code/bayesian_layers.py`.
- Bayes by Backprop + Dropout MLP: `BayesianMLP(dropout=0.1)` in the same model file.

## Existing Saved Checkpoints
{chr(10).join(checkpoint_status())}

## Existing Logs, CSVs, JSONs
- Found `{len(data_files)}` CSV/JSON/log/txt-style result files outside virtual environments.
- Core old MNIST files include `bayes-by-backprop-reimplementation/results/mnist/accuracy_table.csv` and `bayes-by-backprop-reimplementation/results/mnist/test_accuracy.json`.
- Existing history status: {history_note}

## Existing Plots
- Found `{len(plot_files)}` plot/PDF artifacts outside virtual environments.
- Old MNIST/summary plots are mostly under `bayes-by-backprop-reimplementation/results/mnist/` and `bayes-by-backprop-reimplementation/results/summary*`.
- New reproduction figures are generated under `reproduction/figures/`.

## Existing Result Summaries And README Files
- Found `{len(md_files)}` markdown files outside virtual environments.
- Notable existing summaries: `bayes-by-backprop-reimplementation/results/summary_best_val/experiment_summary.md`, `bayes-by-backprop-reimplementation/results/summary_best_val/compute_summary.md`, and `bayes-by-backprop-reimplementation/README.md`.

## What Experiments Already Exist?
{chr(10).join(old_results)}

The old project also contains update-trust and uncertainty-shift experiments, but those are outside the requested final four-model comparison.

## What Results Are Already Usable?
- Old final test accuracies are usable as historical evidence that the project already trained all four requested model families.
- Old checkpoints are usable for weight histograms if checkpoint formats remain compatible.
- Old train/validation histories are useful context, but they do not satisfy the requested `training_curves.csv` schema because they lack per-epoch test metrics and resource measurements.

## What Results Were Missing Before The New Reproduction Layer?
- Per-epoch `test_loss`, `test_accuracy`, and `test_error` for all four final methods.
- Per-epoch runtime and cumulative runtime.
- Per-run runtime summary.
- Peak memory summary in the requested CSV format.
- Organized poster/report-ready PNG figures under a top-level `reproduction/` directory.
- A paper-vs-ours table image and markdown summary using exactly the four requested methods.

## Scripts That Produce Each Result
- `reproduction/training_curves.csv`: `python scripts/run_missing_reproduction_experiments.py --epochs 20 --batch-size 128 --lr 1e-3 --seeds 0 --device auto`
- `reproduction/final_metrics.csv`: same runner.
- `reproduction/resource_metrics.csv`: same runner.
- `reproduction/runtime_summary.csv`: same runner.
- `reproduction/memory_summary.csv`: same runner.
- `reproduction/paper_reproduction_summary.csv`: `python scripts/make_reproduction_plots.py`
- `reproduction/figures/*.png`: `python scripts/make_reproduction_plots.py`
- `reproduction/reproduction_summary.md`: `python scripts/make_reproduction_plots.py`
- `reproduction/figure_captions.md`: `python scripts/make_reproduction_plots.py`

## Exact Commands
Use the project virtual environment when running from this checkout:

```bash
bayes-by-backprop-reimplementation/venv/bin/python scripts/audit_reproduction_results.py
bayes-by-backprop-reimplementation/venv/bin/python scripts/run_missing_reproduction_experiments.py --epochs 20 --batch-size 128 --lr 1e-3 --seeds 0 --device auto
bayes-by-backprop-reimplementation/venv/bin/python scripts/make_reproduction_plots.py
```

For a quick sanity check:

```bash
bayes-by-backprop-reimplementation/venv/bin/python scripts/run_missing_reproduction_experiments.py --quick --epochs 2 --batch-size 128 --lr 1e-3 --seeds 0 --device auto --force
```

For a fuller multi-seed run:

```bash
bayes-by-backprop-reimplementation/venv/bin/python scripts/run_missing_reproduction_experiments.py --epochs 100 --batch-size 128 --lr 1e-3 --seeds 0 1 2 --device auto
```

## Time And Memory Measurements
{chr(10).join(dependency_notes())}

The new runner uses `time.perf_counter()` for per-epoch and total runtime. It records CUDA memory fields when CUDA is available. If CUDA is unavailable, GPU fields are left empty/NaN. CPU memory is measured with `psutil` when installed and otherwise with a `resource.getrusage` fallback when possible.

## Weight Histograms
Weight histograms can be generated from reproduction checkpoints in `reproduction/checkpoints/` and from compatible old checkpoints. For Bayesian models, the plotting script uses posterior mean weights (`weight_mu`) as a stable approximation to the trained weight distribution. The optional plot is skipped with a warning if checkpoint formats are incompatible.

## Current Reproduction Output Status
{chr(10).join(reproduction_status())}

## Paper Values Used For Comparison
The paper Table 1 values used in the new table are the closest 400-unit feedforward comparisons: SGD `1.83%`, dropout `1.51%`, Bayes by Backprop Gaussian `1.82%`, and Bayes by Backprop scale mixture `1.36%`. Our Bayesian implementation uses the scale-mixture prior, so the main paper-vs-ours table compares against `1.36%` while noting the Gaussian value.
"""
    if rows:
        text += "\n## Old Accuracy Table Snapshot\n\n"
        for row in rows:
            text += f"- `{row.get('Model')}`: accuracy `{row.get('Test Accuracy')}`, error `{row.get('Test Error')}`.\n"
    return text


def main() -> None:
    REPRO.mkdir(parents=True, exist_ok=True)
    (REPRO / "figures").mkdir(parents=True, exist_ok=True)
    (REPRO / "logs").mkdir(parents=True, exist_ok=True)
    (REPRO / "checkpoints").mkdir(parents=True, exist_ok=True)
    output = REPRO / "reproduction_audit.md"
    output.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
