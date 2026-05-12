# Reproduction Summary

## Chosen Result
We reproduced the MNIST classification comparison from Table 1 and Figure 2 of Blundell et al., "Weight Uncertainty in Neural Networks."

## What The Paper Found
The paper found that Bayes by Backprop with a scale-mixture prior achieved MNIST performance comparable to dropout in a feedforward ReLU network.

## What We Implemented
- Standard MLP.
- Dropout MLP.
- Bayes by Backprop.
- Bayes by Backprop + Dropout.

## What Our Results Show
- Standard MLP: `1.80%` best test error.
- Dropout MLP: `1.54%` best test error.
- Bayes by Backprop: `1.74%` best test error.
- Bayes by Backprop + Dropout: `1.68%` best test error.

Bayes by Backprop was competitive with dropout in this implementation. The Bayes by Backprop + Dropout extension did not improve over both individual methods in the current run.

## Resource Comparison
- Standard MLP runtime: `69.6` seconds.
- Dropout MLP runtime: `72.2` seconds.
- Bayes by Backprop runtime: `481.8` seconds.
- Bayes by Backprop + Dropout runtime: `406.1` seconds.

Memory metric used: `Model parameter memory (MB)`.
- Standard MLP: `1.8` MB.
- Dropout MLP: `1.8` MB.
- Bayes by Backprop: `3.6` MB.
- BBB + Dropout: `3.6` MB.

The original CPU peak metric was not method-isolated because `psutil` was unavailable and all methods ran in one Python process. For the poster-facing memory comparison, we therefore plot checkpoint/model parameter memory, which captures the expected approximately 2x parameter footprint of Bayes by Backprop.

The Bayesian methods are expected to cost more time and memory because each BayesianLinear layer stores posterior parameters and samples weights during forward passes.

## Discrepancies
Our run uses the existing project architecture with two 400-unit hidden layers, Adam, batch size 128, a 54k/6k train/validation split from MNIST training data, and a shorter training budget than the paper's 600-epoch Figure 2 run. The paper's Table 1 includes larger 800- and 1200-unit networks and hyperparameter searches; therefore exact test errors should not be expected.

## Poster-Ready Takeaway
In our MNIST reproduction, Bayes by Backprop qualitatively matches the paper's claim of dropout-competitive performance, while the added Bayes by Backprop + Dropout variant tests whether combining Bayesian weights with activation dropout improves the accuracy-cost tradeoff.
