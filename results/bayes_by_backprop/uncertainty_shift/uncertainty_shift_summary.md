# Uncertainty Shift Benchmark Summary

## Test
This benchmark evaluates trained MNIST checkpoints on clean inputs plus Gaussian noise and center occlusion corruptions.
It is designed to expose the regime where Bayesian, dropout, and Kalman-trust variants should help most: not just average accuracy, but knowing when predictions have become unreliable.

## Configuration
- Selection mode: `best_val`.
- Examples per condition: `2000`.
- MC samples for stochastic models: `20`.
- Device: `mps`.

## Key Findings
- Best clean accuracy in this run: `Bayesian + Dropout (none)` at `0.9775`.
- Best average shifted accuracy: `Bayes by Backprop (none)` at `0.5972`.
- Best uncertainty/error ranking under shift: `Bayesian + Dropout (none)` at `0.8345` mean entropy AUROC.
- Top shifted-accuracy runners: `Bayes by Backprop (none)` (`0.5972`), `Bayesian + Dropout (none)` (`0.5958`), `Bayes by Backprop (depth lambda0.15)` (`0.5903`).
- Top uncertainty-ranking runners: `Bayesian + Dropout (none)` (`0.8345`), `Bayesian + Dropout (kalman-layer beta0.95 Q0.0001 P1.0 R1.0)` (`0.8324`), `Bayesian + Dropout (depth lambda0.15)` (`0.8313`).

## Kalman Readout
- `Bayes by Backprop (kalman-layer beta0.95 Q0.0001 P1.0 R1.0)` vs `Bayes by Backprop (none)`: clean accuracy delta `+0.0020`, shifted accuracy delta `-0.0190`, entropy AUROC delta `-0.0053`.
- `Bayesian + Dropout (kalman-layer beta0.95 Q0.0001 P1.0 R1.0)` vs `Bayesian + Dropout (none)`: clean accuracy delta `-0.0010`, shifted accuracy delta `-0.0142`, entropy AUROC delta `-0.0021`.

## Reading The Metrics
- `entropy_error_auc` measures whether higher predictive entropy ranks wrong predictions above correct predictions; `0.5` is random and `1.0` is ideal.
- `entropy_error_gap` is mean entropy on wrong predictions minus mean entropy on correct predictions; positive values mean uncertainty rises on mistakes.
- `mutual_information_*` uses MC disagreement, so it is most meaningful for Bayesian and dropout-style stochastic models.
