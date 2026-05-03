# Bayes by Backprop Reimplementation

This project re-implements the core idea from *Weight Uncertainty in Neural Networks*: Bayes by Backprop. Instead of learning a single fixed value for each neural-network weight, the Bayesian models learn a variational posterior over weights parameterized by a mean `mu` and an uncertainty parameter `rho`. During each forward pass, the model samples weights with the reparameterization trick:

```python
w = mu + softplus(rho) * epsilon
```

where `epsilon ~ N(0, 1)`. This keeps the standard deviation positive and allows gradients to flow through sampling. The training objective combines a data-fit term with a complexity cost that encourages the learned posterior to stay close to a prior over weights.

We reproduced two results from the paper: MNIST classification performance compared against standard and dropout neural networks, and a synthetic regression experiment showing increased predictive uncertainty in regions without training data.

## Repository Layout

```text
bayes-by-backprop-reimplementation/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── bayesian_layers.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── models.py
│   ├── train_mnist.py
│   ├── train_regression.py
│   └── utils.py
├── results/
│   ├── mnist/
│   └── regression/
├── poster/
└── report/
```

Place your final poster PDF in `poster/` and your final report PDF in `report/`.

## Setup

```bash
cd bayes-by-backprop-reimplementation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## MNIST Experiment

The tunable defaults live in `DEFAULT_CONFIG` at the top of `code/train_mnist.py`. Edit that block, then run:

```bash
python3 code/train_mnist.py
```

You can still override from the command line if you want:

```bash
python3 code/train_mnist.py --model standard --epochs 10
python3 code/train_mnist.py --model dropout --epochs 10
python3 code/train_mnist.py --model bayesian --epochs 10 --mc-samples 10
python3 code/train_mnist.py --model bayesian --bayesian-dropout 0.1 --epochs 10 --mc-samples 10
```

If you try `--bayesian-dropout`, treat it as an extension for model tuning rather than the pure paper architecture. The script saves that run under a separate name such as `bayesian_dropout_0p1` so it does not overwrite the core Bayes-by-Backprop result.

Outputs are written under `results/mnist/`, including:

- `accuracy_table.csv`
- `<model>_training_curves.png`
- `<model>_test_accuracy.json`

## Regression Experiment

The tunable defaults live in `DEFAULT_CONFIG` at the top of `code/train_regression.py`. Edit that block, then run:

```bash
python3 code/train_regression.py
```

You can still override from the command line if needed:

```bash
python3 code/train_regression.py --model standard --epochs 2000
python3 code/train_regression.py --model bayesian --epochs 2000 --mc-samples 100
```

Outputs are written under `results/regression/`, including:

- `standard_regression.png`
- `bayesian_regression_uncertainty.png`
- `regression_metrics.json`

## Implementation Notes

- `BayesianLinear` learns `weight_mu`, `weight_rho`, `bias_mu`, and `bias_rho`.
- The standard deviation is computed with `softplus(rho)`.
- The variational posterior is diagonal Gaussian.
- The prior is a scale-mixture Gaussian with defaults:
  - `pi = 0.5`
  - `sigma1 = 1.0`
  - `sigma2 = 0.002`
- The Bayesian loss is:

```python
loss = data_term + kl_weight * (log_q - log_p) / dataset_size
```

- For Bayesian evaluation, multiple stochastic forward passes are averaged to estimate predictive behavior.

## Suggested Analysis

For the write-up, compare:

- test accuracy and error rate across standard, dropout, and Bayesian MNIST models
- training stability and convergence behavior
- whether the Bayesian regression model shows wider uncertainty away from observed data
- discrepancies between your results and the paper due to compute budget, architecture simplifications, KL weighting, and implementation choices
