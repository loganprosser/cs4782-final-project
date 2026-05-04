# Update Trust Summary

## Bayes by Backprop
- Best trust-mode run: `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)` at `0.9806` accuracy.
- Relative to `Bayes by Backprop (no trust scaling)`, the best trust variant changed accuracy by `0.0038`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9806`, error `0.0194`, train step `15.19 ms`, eval step `90.79 ms`.
- `Grad Norm`: accuracy `0.9798`, error `0.0202`, train step `14.88 ms`, eval step `87.25 ms`.
- `Grad Var`: accuracy `0.9780`, error `0.0220`, train step `15.04 ms`, eval step `88.45 ms`.
- `Depth Decay`: accuracy `0.9772`, error `0.0228`, train step `15.59 ms`, eval step `88.49 ms`.
- `None`: accuracy `0.9768`, error `0.0232`, train step `15.21 ms`, eval step `85.80 ms`.

## Bayesian + Dropout
- Best trust-mode run: `Bayesian + Dropout (running-grad-var beta=0.95)` at `0.9815` accuracy.
- Relative to `Bayesian + Dropout (no trust scaling)`, the best trust variant changed accuracy by `0.0007`.
- `Grad Var`: accuracy `0.9815`, error `0.0185`, train step `15.69 ms`, eval step `86.77 ms`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9811`, error `0.0189`, train step `16.04 ms`, eval step `87.88 ms`.
- `None`: accuracy `0.9808`, error `0.0192`, train step `15.86 ms`, eval step `87.62 ms`.
- `Depth Decay`: accuracy `0.9805`, error `0.0195`, train step `15.92 ms`, eval step `85.51 ms`.
- `Grad Norm`: accuracy `0.9805`, error `0.0195`, train step `15.55 ms`, eval step `89.90 ms`.
