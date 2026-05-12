# Update Trust Summary

## Bayes by Backprop
- Best trust-mode run: `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)` at `0.9814` accuracy.
- Relative to `Bayes by Backprop (no trust scaling)`, the best trust variant changed accuracy by `0.0017`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9814`, error `0.0186`.
- `Grad Norm`: accuracy `0.9806`, error `0.0194`, train step `15.76 ms`, eval step `89.31 ms`.
- `Depth Decay`: accuracy `0.9800`, error `0.0200`, train step `15.67 ms`, eval step `87.59 ms`.
- `None`: accuracy `0.9797`, error `0.0203`, train step `15.67 ms`, eval step `87.33 ms`.
- `Grad Var`: accuracy `0.9793`, error `0.0207`, train step `15.36 ms`, eval step `87.82 ms`.

## Bayesian + Dropout
- Best trust-mode run: `Bayesian + Dropout (no trust scaling)` at `0.9829` accuracy.
- Relative to `Bayesian + Dropout (no trust scaling)`, the best trust variant changed accuracy by `0.0000`.
- `None`: accuracy `0.9829`, error `0.0171`, train step `15.98 ms`, eval step `88.29 ms`.
- `Kalmanlayer Beta0.95 Q0.0001 P1.0 R1.0`: accuracy `0.9811`, error `0.0189`.
- `Grad Norm`: accuracy `0.9807`, error `0.0193`, train step `16.71 ms`, eval step `89.81 ms`.
- `Depth Decay`: accuracy `0.9806`, error `0.0194`, train step `15.86 ms`, eval step `92.58 ms`.
- `Grad Var`: accuracy `0.9800`, error `0.0200`, train step `16.30 ms`, eval step `93.28 ms`.
