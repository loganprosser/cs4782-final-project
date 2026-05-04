# Dropout Propagated-Uncertainty Summary

## Runs
- `Standard MLP (no trust scaling)`: accuracy `0.9776`, error `0.0224`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9810`, error `0.0190`.
- `Dropout MLP (propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15)`: accuracy `0.9807`, error `0.0193`.

## Best Result
- Best saved run: `Dropout MLP (no trust scaling)` at `0.9810` accuracy.

## Interpretation
- These runs do not use Bayes by Backprop. They use deterministic/dropout MLP weights, then apply propagated-uncertainty gradient trust before the optimizer step.
