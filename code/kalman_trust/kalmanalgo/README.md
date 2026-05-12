# Kalman Gradient Trust MNIST Experiments

This directory implements deterministic MNIST classifiers trained with a Kalman-style gradient trust filter. It is inspired by Bayesian uncertainty ideas, but it is not Bayes by Backprop: there are no sampled weights and no distributions over weights.

## What It Does

For each trainable parameter tensor, the filter keeps:

- `m`: filtered gradient estimate
- `P`: covariance/uncertainty of the filtered gradient estimate
- `grad_mean`: EMA of observed raw gradients
- `grad_var`: EMA of observed gradient variance

After `loss.backward()`, the raw gradient `g` is replaced by:

```text
K = P / (P + R + eps)
m = m + K * (g - m)
P = (1 - K) * P + Q
```

where `R = grad_var + R_floor`.

## Device Order

`--device auto` chooses:

```text
mps -> cuda -> cpu
```

## Run One Experiment

Install dependencies if your environment does not already have them:

```bash
python3 -m pip install -r kalmanalgo/requirements.txt
```

```bash
cd kalmanalgo
python3 train.py --model mlp --method kalman --epochs 10 --batch-size 128 --lr 0.01
```

Methods:

```text
baseline dropout kalman dropout_kalman
```

Models:

```text
mlp cnn
```

## Run All 8 Comparisons

```bash
cd kalmanalgo
python3 train.py --run-all --epochs 10 --batch-size 128 --lr 0.01 --optimizer sgd
```

Outputs are written under `kalmanalgo/results/<timestamp>/` by default, even when launched with `python3 -m kalmanalgo.train` from the repo root:

```text
results/<timestamp>/
  hyperparameters.json
  metrics/metrics.csv
  summaries/final_summary.csv
  summaries/report.md
  checkpoints/*_best.pt
  plots/*.png
```

`results/` and `data/` are ignored by git in this directory.

## Useful Quick Checks

Run fast smoke tests without downloading MNIST:

```bash
python3 -m pytest kalmanalgo/tests
```

Run a tiny fake-data experiment manually:

```bash
python3 -m kalmanalgo.train --model mlp --method kalman --epochs 1 --fake-data --train-subset 128 --test-subset 64 --no-plots
```

## Main CLI Arguments

```text
--model mlp|cnn
--method baseline|dropout|kalman|dropout_kalman
--optimizer sgd|adam
--epochs
--batch-size
--lr
--seed
--hidden-size
--P0
--Q
--R-floor
--beta-R
--diag-mode tensor|scalar
```
