# Update Trust Summary

## Bayes by Backprop
- Best trust-mode run: `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)` at `0.9814` accuracy.
- Relative to `Bayes by Backprop (no trust scaling)`, the best trust variant changed accuracy by `0.0017`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9814`, error `0.0186`, train step `15.19 ms`, eval step `90.79 ms`.
- `Grad Norm`: accuracy `0.9806`, error `0.0194`, train step `14.88 ms`, eval step `87.25 ms`.
- `Depth Decay`: accuracy `0.9800`, error `0.0200`, train step `15.59 ms`, eval step `88.49 ms`.
- `None`: accuracy `0.9797`, error `0.0203`, train step `15.21 ms`, eval step `85.80 ms`.
- `Grad Var`: accuracy `0.9793`, error `0.0207`, train step `15.04 ms`, eval step `88.45 ms`.

## Bayesian + Dropout
- Best trust-mode run: `Bayesian + Dropout (no trust scaling)` at `0.9829` accuracy.
- Relative to `Bayesian + Dropout (no trust scaling)`, the best trust variant changed accuracy by `0.0000`.
- `None`: accuracy `0.9829`, error `0.0171`, train step `15.86 ms`, eval step `87.62 ms`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9826`, error `0.0174`, train step `16.04 ms`, eval step `87.88 ms`.
- `Grad Norm`: accuracy `0.9807`, error `0.0193`, train step `15.55 ms`, eval step `89.90 ms`.
- `Depth Decay`: accuracy `0.9806`, error `0.0194`, train step `15.92 ms`, eval step `85.51 ms`.
- `Grad Var`: accuracy `0.9800`, error `0.0200`, train step `15.69 ms`, eval step `86.77 ms`.
