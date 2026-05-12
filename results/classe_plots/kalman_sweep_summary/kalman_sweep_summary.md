# Kalman Sweep Summary

## Ranked Mean Accuracy
- `Standard MLP + Kalman h800/L2`: mean accuracy `0.9829` +/- `0.0000` over `1` run(s); best `0.9829`.
- `Dropout MLP + Kalman h800/L2`: mean accuracy `0.9821` +/- `0.0000` over `1` run(s); best `0.9821`.
- `BBB + Dropout + Kalman`: mean accuracy `0.9818` +/- `0.0003` over `3` run(s); best `0.9821`.
- `BBB + Dropout no trust`: mean accuracy `0.9818` +/- `0.0004` over `3` run(s); best `0.9821`.
- `Standard MLP no trust h800/L2`: mean accuracy `0.9813` +/- `0.0000` over `1` run(s); best `0.9813`.
- `BBB + Kalman h400/L3`: mean accuracy `0.9806` +/- `0.0000` over `1` run(s); best `0.9806`.
- `Dropout MLP no trust`: mean accuracy `0.9805` +/- `0.0006` over `3` run(s); best `0.9810`.
- `BBB + Kalman`: mean accuracy `0.9805` +/- `0.0010` over `3` run(s); best `0.9816`.
- `BBB + Kalman h800/L2`: mean accuracy `0.9804` +/- `0.0000` over `1` run(s); best `0.9804`.
- `Dropout MLP no trust h400/L3`: mean accuracy `0.9803` +/- `0.0000` over `1` run(s); best `0.9803`.
- `BBB no trust`: mean accuracy `0.9802` +/- `0.0004` over `3` run(s); best `0.9807`.
- `Dropout MLP no trust h800/L2`: mean accuracy `0.9802` +/- `0.0000` over `1` run(s); best `0.9802`.
- `BBB no trust h800/L2`: mean accuracy `0.9802` +/- `0.0000` over `1` run(s); best `0.9802`.
- `Dropout MLP + Kalman h400/L3`: mean accuracy `0.9800` +/- `0.0000` over `1` run(s); best `0.9800`.
- `Dropout MLP + Kalman`: mean accuracy `0.9798` +/- `0.0011` over `3` run(s); best `0.9810`.
- `Standard MLP + Kalman h400/L3`: mean accuracy `0.9797` +/- `0.0000` over `1` run(s); best `0.9797`.
- `Standard MLP + Kalman`: mean accuracy `0.9794` +/- `0.0010` over `3` run(s); best `0.9803`.
- `Standard MLP no trust`: mean accuracy `0.9793` +/- `0.0018` over `3` run(s); best `0.9811`.
- `BBB no trust h400/L3`: mean accuracy `0.9786` +/- `0.0000` over `1` run(s); best `0.9786`.
- `Standard MLP no trust h400/L3`: mean accuracy `0.9778` +/- `0.0000` over `1` run(s); best `0.9778`.

## Reading
- Multi-seed rows are the strongest evidence; single-seed deep/wide rows are architecture probes.
- This sweep intentionally drops the underperforming grad-norm, grad-var, depth-decay, and propagated-uncertainty modes.
