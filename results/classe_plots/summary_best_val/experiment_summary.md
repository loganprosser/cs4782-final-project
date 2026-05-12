# Experiment Summary

## Overall Verdict
Strong qualitative replication: Bayes by Backprop is competitive with dropout on MNIST and reproduces the expected regression uncertainty pattern.

## MNIST Classification
The paper reports that Bayes by Backprop is competitive with dropout on MNIST, rather than requiring an exact numerical match in a small-scale reproduction.

- `Standard MLP`: accuracy `0.9776`, error `0.0224`.
- `Dropout MLP`: accuracy `0.9810`, error `0.0190`.
- `Bayes by Backprop`: accuracy `0.9801`, error `0.0199`.
- `Bayesian + Dropout`: accuracy `0.9818`, error `0.0182`.
- `Standard MLP (no trust scaling)`: accuracy `0.9776`, error `0.0224`.
- `Dropout MLP (no trust scaling)`: accuracy `0.9810`, error `0.0190`.
- `Bayes by Backprop (no trust scaling)`: accuracy `0.9801`, error `0.0199`.
- `Bayesian + Dropout (no trust scaling)`: accuracy `0.9818`, error `0.0182`.
- `Bayes by Backprop (depth-decay lambda=0.15)`: accuracy `0.9805`, error `0.0195`.
- `Bayesian + Dropout (depth-decay lambda=0.15)`: accuracy `0.9808`, error `0.0192`.
- `Bayes by Backprop (grad-norm scaling)`: accuracy `0.9791`, error `0.0209`.
- `Bayesian + Dropout (grad-norm scaling)`: accuracy `0.9792`, error `0.0208`.
- `Bayes by Backprop (running-grad-var beta=0.95)`: accuracy `0.9797`, error `0.0203`.
- `Bayesian + Dropout (running-grad-var beta=0.95)`: accuracy `0.9790`, error `0.0210`.
- `Bayes by Backprop (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9816`, error `0.0184`.
- `Bayesian + Dropout (kalmanLayer beta0.95 Q0.0001 P1.0 R1.0)`: accuracy `0.9818`, error `0.0182`.
- `Bayes by Backprop (propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15)`: accuracy `0.9809`, error `0.0191`.
- `Bayesian + Dropout (propagatedUncertainty beta0.95 Q0.0001 P1.0 R1.0 lambda0.15)`: accuracy `0.9808`, error `0.0192`.
- Best saved MNIST result: `Bayesian + Dropout` with `0.9818` accuracy.
- Bayes by Backprop is within `0.0009` accuracy points of dropout and exceeds the standard MLP by `0.0025`.
- The Bayesian+dropout extension reached `0.9818` accuracy, which is `0.0008` above plain dropout.
- This supports the paper's claim that Bayes by Backprop can be competitive with dropout on MNIST in a smaller-scale reproduction.

## Synthetic Regression
- Bayesian train MSE: `0.004618`; full-grid MSE: `0.005812`.
- Mean predictive std in observed regions: `0.0161`.
- Mean predictive std in the gap / extrapolation region: `0.0228`.
- Uncertainty ratio (gap / observed): `1.41x`.
- This qualitatively matches the paper: the Bayesian model is more uncertain away from observed data.

## Key Findings
- The saved Bayesian MNIST model reached `98.01%` test accuracy after 10 epochs.
- The best saved MNIST model was `Bayesian + Dropout` at `98.18%` accuracy.
- The original Bayes by Backprop model stayed within `0.17%` of the best classifier.
- Regression uncertainty was higher away from observed data (`0.0228` vs `0.0161` predictive std).
