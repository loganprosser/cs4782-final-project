# Uncertainty Shift Benchmark

## Motivation

Clean MNIST accuracy is a weak test of uncertainty-aware networks because most trained models are confident and correct on ordinary test images. If Bayesian weights, dropout sampling, or Kalman-style update trust are useful, the benefit should become clearer when the input distribution drifts away from training data.

This benchmark asks:

- Do uncertainty-aware models retain more accuracy under corruption?
- Does predictive uncertainty increase when the model is wrong?
- Do Monte Carlo disagreement metrics identify unreliable predictions better than deterministic confidence?
- Do Kalman-layer trust updates improve robustness or uncertainty quality relative to the same Bayesian model without trust scaling?

## Test Design

The script evaluates saved MNIST checkpoints on the same held-out test images under seven conditions:

- `clean`: unmodified MNIST test images.
- `gaussian_0.15`: additive Gaussian pixel noise with standard deviation `0.15`.
- `gaussian_0.3`: additive Gaussian pixel noise with standard deviation `0.30`.
- `gaussian_0.45`: additive Gaussian pixel noise with standard deviation `0.45`.
- `occlusion_8`: an `8 x 8` center square is masked out.
- `occlusion_12`: a `12 x 12` center square is masked out.
- `occlusion_16`: a `16 x 16` center square is masked out.

For deterministic models, the benchmark uses one forward pass. For dropout and Bayesian models, it uses Monte Carlo prediction samples and averages the predictive probabilities.

## Metrics

- `accuracy`: classification accuracy under each condition.
- `nll`: negative log likelihood of the true class.
- `brier`: multiclass Brier score.
- `entropy_mean`: average predictive entropy.
- `entropy_error_gap`: mean entropy on wrong predictions minus mean entropy on correct predictions. Positive is better.
- `entropy_error_auc`: AUROC for using entropy to rank wrong predictions above correct predictions. `0.5` is random, `1.0` is ideal.
- `mutual_information_mean`: MC predictive entropy minus expected per-sample entropy. This approximates epistemic disagreement.
- `mutual_information_error_auc`: AUROC for using MC disagreement to rank wrong predictions above correct predictions.

## Expected Interpretation

The clearest evidence for useful uncertainty is not necessarily the highest clean accuracy. Stronger evidence would look like:

- Smaller accuracy degradation as noise or occlusion increases.
- Higher `entropy_error_auc` under shifted conditions.
- Positive `entropy_error_gap`, especially on severe corruptions.
- Higher mutual-information error ranking for Bayesian/dropout models than deterministic baselines.
- Kalman-trust models matching or improving shifted accuracy without collapsing uncertainty rankings.

## Outputs

Running `run_uncertainty_shift.sh` creates:

- `results/uncertainty_shift/uncertainty_shift_metrics.json`
- `results/uncertainty_shift/uncertainty_shift_metrics.csv`
- `results/uncertainty_shift/uncertainty_shift_summary.md`
- `results/uncertainty_shift/uncertainty_shift_manifest.json`
- `results/uncertainty_shift/uncertainty_shift_accuracy.png`
- `results/uncertainty_shift/uncertainty_shift_entropy_auc.png`

By default, `run_uncertainty_shift.sh` includes all saved trust-mode checkpoints, including Kalman-layer runs. Use `INCLUDE_TRUST_RUNS=false ./run_uncertainty_shift.sh` for only the four no-trust baseline checkpoints.
