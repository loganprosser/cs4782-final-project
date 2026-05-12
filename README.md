# Bayes by Backprop Reimplementation

CS 4782 final project repository for reproducing and extending Bayes by Backprop from *Weight Uncertainty in Neural Networks*.

This project implements Bayesian neural network layers trained with the Bayes by Backprop objective, compares them against standard and dropout baselines on MNIST-style classification, and includes extensions around update trust, Kalman-style optimizer behavior, uncertainty shift, and noisy-label experiments.

## Repository Layout

This repository is organized around the course-required submission structure:

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── code/
├── data/
├── results/
├── poster/
└── report/
```

Important subdirectories:

- `code/bayes_by_backprop/`: core implementation, including Bayesian layers, models, losses, evaluation helpers, update-trust logic, and utilities.
- `code/experiments/`: training scripts, benchmark scripts, reproduction scripts, and shell runners.
- `code/evaluation/`: plotting, summary, audit, and final-figure generation scripts.
- `code/kalman_trust/`: Kalman/trust-mechanism extensions and earlier exploratory implementations.
- `code/tests/`: focused unit tests for update-trust behavior.
- `results/bayes_by_backprop/`: preserved outputs from the core implementation.
- `results/reproduction/`: final reproduction CSVs, figures, logs, and summaries.
- `results/figures_final/`: final poster/report-ready plots.
- `poster/`: poster PDF candidate.
- `report/`: report PDF candidate and related notes/references.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

The main dependencies are PyTorch, torchvision, numpy, pandas, and matplotlib.

## Main Experiments

The core MNIST training entry point is:

```bash
python code/experiments/train_mnist.py --model bayesian --epochs 10 --mc-samples 10
```

Useful baseline and extension commands:

```bash
python code/experiments/train_mnist.py --model standard --epochs 10
python code/experiments/train_mnist.py --model dropout --epochs 10
python code/experiments/train_mnist.py --model bayesian --epochs 10 --mc-samples 10
python code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.1 --epochs 10 --mc-samples 10
```

Regression uncertainty experiment:

```bash
python code/experiments/train_regression.py --model bayesian --epochs 2000 --mc-samples 100
```

By default, new core outputs are written under `results/bayes_by_backprop/`.

## Reproduction Outputs

The final reproduction artifacts are preserved under `results/reproduction/`, including:

- `training_curves.csv`
- `final_metrics.csv`
- `resource_metrics.csv`
- `runtime_summary.csv`
- `memory_summary.csv`
- `paper_reproduction_summary.csv`
- `reproduction_summary.md`
- `figure_captions.md`
- `figures/*.png`

The reproduction runner is:

```bash
python code/experiments/run_missing_reproduction_experiments.py --epochs 20 --batch-size 128 --lr 1e-3 --seeds 0 --device auto
```

The reproduction plot/report generator is:

```bash
python code/evaluation/make_reproduction_plots.py
```

## Results And Large Files

Small summaries, figures, tables, and markdown reports are kept in `results/`.

Large or regenerable files are intentionally ignored by git:

- raw MNIST/Fashion-MNIST downloads in `data/`
- model checkpoints and weights such as `*.pt`, `*.pth`, and `*.ckpt`
- checkpoint directories under `results/`
- bulky archived training dumps
- virtual environments and generated caches

This keeps the GitHub submission small while preserving local work on disk.

## Verification

After reorganizing the repository, these checks were run:

```bash
python3 -m compileall -q code
```

Result: passed. This verifies that the Python files under `code/` are syntactically valid after the directory move and import-path cleanup.

The unit test suite was also attempted:

```bash
python3 -m unittest discover -s code/tests
```

Result in the current shell: failed because PyTorch was not installed in that interpreter:

```text
ModuleNotFoundError: No module named 'torch'
```

This is an environment/dependency issue, not a syntax failure from the reorganization. After installing dependencies, rerun:

```bash
source .venv/bin/activate
pip install -r code/requirements.txt
python3 -m unittest discover -s code/tests
```

Ignore rules were also spot-checked with `git check-ignore` to confirm that raw datasets, checkpoints, and bulky archived runs are excluded from the GitHub submission.

## Notes On Reorganization

The most complete implementation originally lived in `bayes-by-backprop-reimplementation/`. Its core code was moved into `code/bayes_by_backprop/`, runnable scripts were moved into `code/experiments/`, plotting and summary utilities were moved into `code/evaluation/`, and related Kalman/trust work was moved into `code/kalman_trust/`.

The old local virtual environment, bytecode caches, macOS metadata, generated plotting caches, and LaTeX auxiliary files were removed from the working tree.
