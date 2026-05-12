# Samplewise Gradient Trust Notes

This directory implements a standalone samplewise gradient trust experiment for ordinary neural network training. It is intentionally separate from Bayes-by-Backprop, Bayesian layers, MC sampling, and posterior uncertainty. The question here is narrower: can a standard MLP or CNN train better if mini-batch examples are weighted by a detached trust score?

The training loss is:

```text
L_trust = sum_i tau_i l_i / (sum_i tau_i + eps)
```

Here `l_i` is the unreduced per-example cross entropy and `tau_i` is detached before weighting the loss, so the model cannot directly optimize the trust rule.

Implemented modes:

- `none`: all examples get trust 1.0, matching ordinary mean cross entropy.
- `confidence`: trust is the model probability assigned to the true class.
- `entropy`: trust is `exp(-lambda_entropy * predictive_entropy)`.
- `inverse_loss`: trust is `exp(-lambda_loss * per_example_loss)`.
- `final_layer_align`: trust is positive cosine alignment between each sample's approximate final-layer gradient and the batch gradient.
- `final_layer_align_mag`: final-layer alignment multiplied by a gradient magnitude gate.
- `ema_final_layer_align`: final-layer alignment against a running EMA reference gradient.
- `combined_simple`: final-layer alignment times magnitude gate times inverse loss.

Run a single experiment:

```bash
python train_sample_trust_mnist.py --model mlp --sample-trust-mode final_layer_align --batch-size 32 --epochs 5
```

Run the small sweep:

```bash
./run.sh
```

Outputs:

- Per-run metrics: `results/sample_trust/runs/<run_name>/metrics.csv`
- Per-run plots: `results/sample_trust/runs/<run_name>/plots/`
- Summary table: `results/sample_trust/reports/summary.csv`
- Summary markdown: `results/sample_trust/reports/<run_tag>_summary.md`
- Aggregate and combined plots: `results/sample_trust/reports/plots/`

The default sweep covers `mlp`, `mlp_wide`, `mlp_deep`, `cnn`, `cnn_deep`, `cnn_bn`, and `resnet_tiny`, batch sizes `16 32 64 128 256`, and seeds `0 1 2` for 10 epochs. To narrow it:

```bash
MODELS="cnn_bn resnet_tiny" SEEDS="0 1" ./run.sh
```

Inspect first:

- `best_test_acc` and `final_test_acc` in `summary.csv`
- train/test accuracy curves for stability
- `trust_mean`, `trust_std`, `trust_min`, and `trust_max`
- `mean_alignment` and `mean_grad_norm` for final-layer alignment modes

Do not read improved trust metrics alone as success. The method only helps if held-out accuracy or loss improves under a fair comparison.
