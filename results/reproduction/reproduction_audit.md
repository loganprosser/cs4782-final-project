# Reproduction Audit

## Scope
This audit covers the repository rooted at `/Users/logan/Documents/dev/cornellspring2026/DeepLearning/cs4782-final-project` with special attention to `bayes-by-backprop-reimplementation/`, which contains the Bayes by Backprop MNIST implementation.

## Existing Training Scripts
Key scripts already present:
- `bayes-by-backprop-reimplementation/code/bayesian_layers.py`
- `bayes-by-backprop-reimplementation/code/benchmark_mnist_compute.py`
- `bayes-by-backprop-reimplementation/code/benchmark_uncertainty_shift.py`
- `bayes-by-backprop-reimplementation/code/evaluate.py`
- `bayes-by-backprop-reimplementation/code/losses.py`
- `bayes-by-backprop-reimplementation/code/models.py`
- `bayes-by-backprop-reimplementation/code/summarize_results.py`
- `bayes-by-backprop-reimplementation/code/train_mnist.py`
- `bayes-by-backprop-reimplementation/code/train_regression.py`
- `bayes-by-backprop-reimplementation/code/update_trust.py`
- `bayes-by-backprop-reimplementation/code/utils.py`
- `bayes-by-backprop-reimplementation/kalman_run.sh`
- `bayes-by-backprop-reimplementation/run.sh`
- `bayes-by-backprop-reimplementation/run_all_trust_experiments.sh`
- `bayes-by-backprop-reimplementation/run_uncertainty_shift.sh`
- `bayes-by-backprop-reimplementation/tests/test_update_trust.py`
- `kalmanalgo/__init__.py`
- `kalmanalgo/kalman_optimizer.py`
- `kalmanalgo/models.py`
- `kalmanalgo/plot_results.py`
- `kalmanalgo/tests/test_smoke.py`
- `kalmanalgo/train.py`
- `kalmanalgo/utils.py`
- `loganalgo/models.py`
- `loganalgo/plot_sample_trust_findings.py`
- `loganalgo/plot_sample_trust_results.py`
- `loganalgo/run.sh`
- `loganalgo/run_sample_trust_experiments.sh`
- `loganalgo/sample_trust.py`
- `loganalgo/summarize_sample_trust_results.py`
- `loganalgo/tests/test_models.py`
- `loganalgo/tests/test_sample_trust.py`
- `loganalgo/train_sample_trust_mnist.py`
- `scripts/audit_reproduction_results.py`
- `scripts/make_reproduction_plots.py`
- `scripts/run_missing_reproduction_experiments.py`

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
- Old MNIST checkpoints found: `24` in `bayes-by-backprop-reimplementation/results/mnist/checkpoints/`.
- Reproduction checkpoints found: `4` in `reproduction/checkpoints/`.
  - `reproduction/checkpoints/bayes_by_backprop_dropout_seed0.pt`
  - `reproduction/checkpoints/bayes_by_backprop_seed0.pt`
  - `reproduction/checkpoints/dropout_mlp_seed0.pt`
  - `reproduction/checkpoints/standard_mlp_seed0.pt`

## Existing Logs, CSVs, JSONs
- Found `1872` CSV/JSON/log/txt-style result files outside virtual environments.
- Core old MNIST files include `bayes-by-backprop-reimplementation/results/mnist/accuracy_table.csv` and `bayes-by-backprop-reimplementation/results/mnist/test_accuracy.json`.
- Existing history status: train/validation histories exist for the four old core runs, but per-epoch test metrics and runtime/memory metrics do not.

## Existing Plots
- Found `69` plot/PDF artifacts outside virtual environments.
- Old MNIST/summary plots are mostly under `bayes-by-backprop-reimplementation/results/mnist/` and `bayes-by-backprop-reimplementation/results/summary*`.
- New reproduction figures are generated under `reproduction/figures/`.

## Existing Result Summaries And README Files
- Found `20` markdown files outside virtual environments.
- Notable existing summaries: `bayes-by-backprop-reimplementation/results/summary_best_val/experiment_summary.md`, `bayes-by-backprop-reimplementation/results/summary_best_val/compute_summary.md`, and `bayes-by-backprop-reimplementation/README.md`.

## What Experiments Already Exist?
- `bayes_by_backprop` exists as old run `bayesian`: test accuracy `0.9797`, test error `2.03%`.
- `standard_mlp` exists as old run `standard`: test accuracy `0.9782`, test error `2.18%`.
- `dropout_mlp` exists as old run `dropout`: test accuracy `0.9812`, test error `1.88%`.
- `bayes_by_backprop_dropout` exists as old run `bayesian_dropout_0p1`: test accuracy `0.9821`, test error `1.79%`.

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
- `torch` importable in the current interpreter: `True`.
- `torchvision` importable in the current interpreter: `True`.
- `pandas` importable in the current interpreter: `True`.
- `matplotlib` importable in the current interpreter: `True`.
- `numpy` importable in the current interpreter: `True`.
- `psutil` importable in the current interpreter: `False`.
- The checked project virtual environment is `bayes-by-backprop-reimplementation/venv/bin/python`; use it if the system `python3` lacks ML dependencies.
- `psutil` is not installed in the current interpreter. The reproduction runner falls back to `resource.getrusage` for CPU memory when possible and logs a warning.

The new runner uses `time.perf_counter()` for per-epoch and total runtime. It records CUDA memory fields when CUDA is available. If CUDA is unavailable, GPU fields are left empty/NaN. CPU memory is measured with `psutil` when installed and otherwise with a `resource.getrusage` fallback when possible.

## Weight Histograms
Weight histograms can be generated from reproduction checkpoints in `reproduction/checkpoints/` and from compatible old checkpoints. For Bayesian models, the plotting script uses posterior mean weights (`weight_mu`) as a stable approximation to the trained weight distribution. The optional plot is skipped with a warning if checkpoint formats are incompatible.

## Current Reproduction Output Status
- `reproduction/training_curves.csv`: present.
- `reproduction/final_metrics.csv`: present.
- `reproduction/resource_metrics.csv`: present.
- `reproduction/runtime_summary.csv`: present.
- `reproduction/memory_summary.csv`: present.
- `reproduction/paper_reproduction_summary.csv`: present.
- `reproduction/reproduction_summary.md`: present.
- `reproduction/figure_captions.md`: present.

## Paper Values Used For Comparison
The paper Table 1 values used in the new table are the closest 400-unit feedforward comparisons: SGD `1.83%`, dropout `1.51%`, Bayes by Backprop Gaussian `1.82%`, and Bayes by Backprop scale mixture `1.36%`. Our Bayesian implementation uses the scale-mixture prior, so the main paper-vs-ours table compares against `1.36%` while noting the Gaussian value.

## Old Accuracy Table Snapshot

- `bayesian`: accuracy `0.9797`, error `0.0202999999999999`.
- `standard`: accuracy `0.9782`, error `0.0218`.
- `dropout`: accuracy `0.9812`, error `0.0188`.
- `bayesian_dropout_0p1`: accuracy `0.9821`, error `0.0179`.
- `standard_trust_none`: accuracy `0.98`, error `0.02`.
- `dropout_trust_none`: accuracy `0.9812`, error `0.0188`.
- `bayesian_trust_none`: accuracy `0.9768`, error `0.0232`.
- `bayesian_dropout_0p1_trust_none`: accuracy `0.9808`, error `0.0191999999999999`.
- `bayesian_trust_depth_lambda0.15`: accuracy `0.9772`, error `0.0228`.
- `bayesian_dropout_0p1_trust_depth_lambda0.15`: accuracy `0.9805`, error `0.0194999999999999`.
- `bayesian_trust_gradnorm`: accuracy `0.9798`, error `0.0201999999999999`.
- `bayesian_dropout_0p1_trust_gradnorm`: accuracy `0.9805`, error `0.0194999999999999`.
- `bayesian_trust_gradvar_beta0.95`: accuracy `0.978`, error `0.022`.
- `bayesian_dropout_0p1_trust_gradvar_beta0.95`: accuracy `0.9815`, error `0.0184999999999999`.
- `bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0`: accuracy `0.9806`, error `0.0193999999999999`.
- `bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0`: accuracy `0.9811`, error `0.018900000000000028`.
