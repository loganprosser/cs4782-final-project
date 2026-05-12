# Kalman Sweep Summary

## Ranked Mean Accuracy
- `Standard MLP + Kalman h800/L2`: mean accuracy `0.9800` +/- `0.0019` over `5` run(s); best `0.9829`.
- `Standard MLP no trust h800/L2`: mean accuracy `0.9799` +/- `0.0018` over `5` run(s); best `0.9824`.

## Reading
- Multi-seed rows are the strongest evidence; single-seed deep/wide rows are architecture probes.
- This sweep intentionally drops the underperforming grad-norm, grad-var, depth-decay, and propagated-uncertainty modes.
