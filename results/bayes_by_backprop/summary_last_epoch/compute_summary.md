# Compute Benchmark Summary

## Environment
- Device reported by benchmark: `cpu`.
- CUDA available: `False`.
- MPS available: `True`.
- GPU utilization percentage was not measured here because the current Python environment does not expose a working CUDA or MPS backend.

## Model Tradeoffs
- `Standard MLP`: accuracy `0.9782`, params `478,410`, train step `1.89 ms`, eval step `0.35 ms`, parameter memory `1.82 MB`.
- `Dropout MLP`: accuracy `0.9812`, params `478,410`, train step `2.36 ms`, eval step `0.34 ms`, parameter memory `1.82 MB`.
- `Bayes by Backprop`: accuracy `0.9797`, params `956,820`, train step `15.03 ms`, eval step `86.23 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout`: accuracy `0.9821`, params `956,820`, train step `16.03 ms`, eval step `87.91 ms`, parameter memory `3.65 MB`.
- `Standard MLP (no trust scaling)`: accuracy `0.9782`, params `478,410`, train step `1.71 ms`, eval step `0.33 ms`, parameter memory `1.82 MB`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9812`, params `478,410`, train step `2.23 ms`, eval step `0.34 ms`, parameter memory `1.82 MB`.
- `Bayes by Backprop (no trust scaling)`: accuracy `0.9797`, params `956,820`, train step `15.03 ms`, eval step `85.78 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (no trust scaling)`: accuracy `0.9829`, params `956,820`, train step `15.90 ms`, eval step `88.25 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (depth-decay lambda=0.15)`: accuracy `0.9800`, params `956,820`, train step `15.03 ms`, eval step `89.25 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (grad-norm scaling)`: accuracy `0.9806`, params `956,820`, train step `15.43 ms`, eval step `86.23 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (running-grad-var beta=0.95)`: accuracy `0.9793`, params `956,820`, train step `14.71 ms`, eval step `85.65 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (depth-decay lambda=0.15)`: accuracy `0.9806`, params `956,820`, train step `15.38 ms`, eval step `87.29 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (grad-norm scaling)`: accuracy `0.9807`, params `956,820`, train step `15.90 ms`, eval step `85.85 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (running-grad-var beta=0.95)`: accuracy `0.9800`, params `956,820`, train step `15.81 ms`, eval step `85.62 ms`, parameter memory `3.65 MB`.
- `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9814`, params `956,820`, train step `14.71 ms`, eval step `86.44 ms`, parameter memory `3.65 MB`.
- `Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9826`, params `956,820`, train step `15.84 ms`, eval step `83.57 ms`, parameter memory `3.65 MB`.

## Key Findings
- Highest accuracy: `Bayesian + Dropout (no trust scaling)` at `0.9829`.
- Fastest training step: `Standard MLP (no trust scaling)` at `1.71 ms`.
- Fastest evaluation step: `Standard MLP (no trust scaling)` at `0.33 ms`.
- Smallest parameter footprint: `Standard MLP` with `1.82 MB` of weights.
- Compared with Dropout MLP, `Bayesian + Dropout` gained `0.0009` accuracy points.
- That gain cost about `6.78x` training-time per batch and `259.67x` evaluation-time per batch in this benchmark.
