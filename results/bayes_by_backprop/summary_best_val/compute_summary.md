# Compute Benchmark Summary

## Environment
- Device reported by benchmark: `cpu`.
- CUDA available: `False`.
- MPS available: `True`.
- GPU utilization percentage was not measured here because the current Python environment does not expose a working CUDA or MPS backend.

## Model Tradeoffs
- `Standard MLP`: accuracy `0.9782`, params `478,410`, train step `1.85 ms`, eval step `0.35 ms`, parameter memory `1.82 MB`.
- `Dropout MLP`: accuracy `0.9812`, params `478,410`, train step `2.33 ms`, eval step `0.35 ms`, parameter memory `1.82 MB`.
- `Bayes by Backprop`: accuracy `0.9797`, params `956,820`, train step `16.13 ms`, eval step `87.19 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout`: accuracy `0.9821`, params `956,820`, train step `15.86 ms`, eval step `86.79 ms`, parameter memory `3.65 MB`.
- `Standard MLP (no trust scaling)`: accuracy `0.9800`, params `478,410`, train step `1.69 ms`, eval step `0.33 ms`, parameter memory `1.82 MB`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9812`, params `478,410`, train step `2.29 ms`, eval step `0.33 ms`, parameter memory `1.82 MB`.
- `Bayes by Backprop (no trust scaling)`: accuracy `0.9768`, params `956,820`, train step `15.43 ms`, eval step `87.36 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (no trust scaling)`: accuracy `0.9808`, params `956,820`, train step `16.91 ms`, eval step `91.21 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (depth-decay lambda=0.15)`: accuracy `0.9772`, params `956,820`, train step `16.64 ms`, eval step `85.87 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (grad-norm scaling)`: accuracy `0.9798`, params `956,820`, train step `15.89 ms`, eval step `86.49 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (running-grad-var beta=0.95)`: accuracy `0.9780`, params `956,820`, train step `14.98 ms`, eval step `86.75 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (depth-decay lambda=0.15)`: accuracy `0.9805`, params `956,820`, train step `15.59 ms`, eval step `85.19 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (grad-norm scaling)`: accuracy `0.9805`, params `956,820`, train step `15.52 ms`, eval step `90.16 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (running-grad-var beta=0.95)`: accuracy `0.9815`, params `956,820`, train step `15.59 ms`, eval step `85.35 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9806`, params `956,820`, train step `14.85 ms`, eval step `87.66 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9811`, params `956,820`, train step `15.82 ms`, eval step `83.69 ms`, parameter memory `3.65 MB`.

## Key Findings
- Highest accuracy: `Bayesian + Dropout` at `0.9821`.
- Fastest training step: `Standard MLP (no trust scaling)` at `1.69 ms`.
- Fastest evaluation step: `Standard MLP (no trust scaling)` at `0.33 ms`.
- Smallest parameter footprint: `Standard MLP` with `1.82 MB` of weights.
- Compared with Dropout MLP, `Bayesian + Dropout` gained `0.0009` accuracy points.
- That gain cost about `6.80x` training-time per batch and `245.80x` evaluation-time per batch in this benchmark.
