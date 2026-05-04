# Compute Benchmark Summary

## Environment
- Device reported by benchmark: `cuda`.
- CUDA available: `True`.
- MPS available: `False`.
- GPU utilization percentage was not measured here because the current Python environment does not expose a working CUDA or MPS backend.

## Model Tradeoffs
- `Standard MLP (no trust scaling)`: accuracy `0.9776`, params `478,410`, train step `6.70 ms`, eval step `0.06 ms`, parameter memory `1.82 MB`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9810`, params `478,410`, train step `1.16 ms`, eval step `0.08 ms`, parameter memory `1.82 MB`.
- `Bayes by Backprop (no trust scaling)`: accuracy `0.9801`, params `956,820`, train step `9.70 ms`, eval step `38.15 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (no trust scaling)`: accuracy `0.9818`, params `956,820`, train step `8.20 ms`, eval step `38.83 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (depth-decay lambda=0.15)`: accuracy `0.9805`, params `956,820`, train step `9.03 ms`, eval step `38.80 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (depth-decay lambda=0.15)`: accuracy `0.9808`, params `956,820`, train step `10.01 ms`, eval step `39.02 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (grad-norm scaling)`: accuracy `0.9791`, params `956,820`, train step `9.26 ms`, eval step `39.11 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (grad-norm scaling)`: accuracy `0.9792`, params `956,820`, train step `9.05 ms`, eval step `38.48 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (running-grad-var beta=0.95)`: accuracy `0.9797`, params `956,820`, train step `9.07 ms`, eval step `38.51 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (running-grad-var beta=0.95)`: accuracy `0.9790`, params `956,820`, train step `9.08 ms`, eval step `38.54 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9816`, params `956,820`, train step `9.36 ms`, eval step `38.36 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9818`, params `956,820`, train step `9.06 ms`, eval step `36.30 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15)`: accuracy `0.9809`, params `956,820`, train step `9.22 ms`, eval step `27.60 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15)`: accuracy `0.9808`, params `956,820`, train step `8.09 ms`, eval step `25.17 ms`, parameter memory `3.65 MB`.

## Key Findings
- Highest accuracy: `Bayesian + Dropout (no trust scaling)` at `0.9818`.
- Fastest training step: `Dropout MLP (no trust scaling)` at `1.16 ms`.
- Fastest evaluation step: `Standard MLP (no trust scaling)` at `0.06 ms`.
- Smallest parameter footprint: `Standard MLP (no trust scaling)` with `1.82 MB` of weights.
