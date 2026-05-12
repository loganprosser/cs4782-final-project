# Kalman Sweep Summary

## Ranked Mean Accuracy
- `Standard MLP no trust`: mean accuracy `0.8808` +/- `0.0028` over `3` run(s); best `0.8828`.
- `Standard MLP + Kalman h800/L2`: mean accuracy `0.8808` +/- `0.0017` over `3` run(s); best `0.8821`.
- `Standard MLP no trust h800/L2`: mean accuracy `0.8801` +/- `0.0034` over `3` run(s); best `0.8824`.
- `Standard MLP + Kalman`: mean accuracy `0.8777` +/- `0.0037` over `3` run(s); best `0.8799`.
- `BBB no trust`: mean accuracy `0.8769` +/- `0.0027` over `3` run(s); best `0.8787`.
- `BBB + Kalman`: mean accuracy `0.8763` +/- `0.0008` over `3` run(s); best `0.8772`.
- `BBB + Dropout no trust`: mean accuracy `0.8761` +/- `0.0008` over `3` run(s); best `0.8769`.
- `BBB + Dropout + Kalman`: mean accuracy `0.8752` +/- `0.0002` over `3` run(s); best `0.8753`.
- `Dropout MLP no trust`: mean accuracy `0.8677` +/- `0.0014` over `3` run(s); best `0.8686`.
- `Dropout MLP + Kalman`: mean accuracy `0.8654` +/- `0.0015` over `3` run(s); best `0.8671`.

## Reading
- Multi-seed rows are the strongest evidence; single-seed deep/wide rows are architecture probes.
- This sweep intentionally drops the underperforming grad-norm, grad-var, depth-decay, and propagated-uncertainty modes.
