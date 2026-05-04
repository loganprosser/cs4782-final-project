# Update Trust Summary

## Bayes by Backprop
- Best trust-mode run: `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)` at `0.9816` accuracy.
- Relative to `Bayes by Backprop (no trust scaling)`, the best trust variant changed accuracy by `0.0015`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9816`, error `0.0184`, train step `9.36 ms`, eval step `38.36 ms`.
- `Propagateduncertainty Beta0.95 Q0.0001 P1.0 R1.0 Lambda0.15`: accuracy `0.9809`, error `0.0191`, train step `9.22 ms`, eval step `27.60 ms`.
- `Depth Decay`: accuracy `0.9805`, error `0.0195`, train step `9.03 ms`, eval step `38.80 ms`.
- `None`: accuracy `0.9801`, error `0.0199`, train step `9.70 ms`, eval step `38.15 ms`.
- `Grad Var`: accuracy `0.9797`, error `0.0203`, train step `9.07 ms`, eval step `38.51 ms`.
- `Grad Norm`: accuracy `0.9791`, error `0.0209`, train step `9.26 ms`, eval step `39.11 ms`.

## Bayesian + Dropout
- Best trust-mode run: `Bayesian + Dropout (no trust scaling)` at `0.9818` accuracy.
- Relative to `Bayesian + Dropout (no trust scaling)`, the best trust variant changed accuracy by `0.0000`.
- `None`: accuracy `0.9818`, error `0.0182`, train step `8.20 ms`, eval step `38.83 ms`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9818`, error `0.0182`, train step `9.06 ms`, eval step `36.30 ms`.
- `Depth Decay`: accuracy `0.9808`, error `0.0192`, train step `10.01 ms`, eval step `39.02 ms`.
- `Propagateduncertainty Beta0.95 Q0.0001 P1.0 R1.0 Lambda0.15`: accuracy `0.9808`, error `0.0192`, train step `8.09 ms`, eval step `25.17 ms`.
- `Grad Norm`: accuracy `0.9792`, error `0.0208`, train step `9.05 ms`, eval step `38.48 ms`.
- `Grad Var`: accuracy `0.9790`, error `0.0210`, train step `9.08 ms`, eval step `38.54 ms`.
