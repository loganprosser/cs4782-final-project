# Experiment Summary

## Overall Verdict
Strong qualitative replication: Bayes by Backprop is competitive with dropout on MNIST and reproduces the expected regression uncertainty pattern.

## MNIST Classification
The paper reports that Bayes by Backprop is competitive with dropout on MNIST, rather than requiring an exact numerical match in a small-scale reproduction.

- `Standard MLP`: accuracy `0.9782`, error `0.0218`.
- `Dropout MLP`: accuracy `0.9812`, error `0.0188`.
- `Bayes by Backprop`: accuracy `0.9797`, error `0.0203`.
- `Bayesian + Dropout`: accuracy `0.9821`, error `0.0179`.
- `Standard MLP (no trust scaling)`: accuracy `0.9800`, error `0.0200`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9812`, error `0.0188`.
- `Bayes by Backprop (no trust scaling)`: accuracy `0.9768`, error `0.0232`.
- `Bayesian + Dropout (no trust scaling)`: accuracy `0.9808`, error `0.0192`.
- `Bayes by Backprop (depth-decay lambda=0.15)`: accuracy `0.9772`, error `0.0228`.
- `Bayes by Backprop (grad-norm scaling)`: accuracy `0.9798`, error `0.0202`.
- `Bayes by Backprop (running-grad-var beta=0.95)`: accuracy `0.9780`, error `0.0220`.
- `Bayesian + Dropout (depth-decay lambda=0.15)`: accuracy `0.9805`, error `0.0195`.
- `Bayesian + Dropout (grad-norm scaling)`: accuracy `0.9805`, error `0.0195`.
- `Bayesian + Dropout (running-grad-var beta=0.95)`: accuracy `0.9815`, error `0.0185`.
- `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9806`, error `0.0194`.
- `Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9811`, error `0.0189`.
- Best saved MNIST result: `Bayesian + Dropout` with `0.9821` accuracy.
- Bayes by Backprop is within `0.0015` accuracy points of dropout and exceeds the standard MLP by `0.0015`.
- The Bayesian+dropout extension reached `0.9821` accuracy, which is `0.0009` above plain dropout.
- This supports the paper's claim that Bayes by Backprop can be competitive with dropout on MNIST in a smaller-scale reproduction.

## Synthetic Regression
- Bayesian train MSE: `0.004032`; full-grid MSE: `0.007593`.
- Mean predictive std in observed regions: `0.0161`.
- Mean predictive std in the gap / extrapolation region: `0.0229`.
- Uncertainty ratio (gap / observed): `1.42x`.
- This qualitatively matches the paper: the Bayesian model is more uncertain away from observed data.

## Key Findings
- The saved Bayesian MNIST model reached `97.97%` test accuracy after 10 epochs.
- The best saved MNIST model was `Bayesian + Dropout` at `98.21%` accuracy.
- The original Bayes by Backprop model stayed within `0.24%` of the best classifier.
- Regression uncertainty was higher away from observed data (`0.0229` vs `0.0161` predictive std).
