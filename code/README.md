# Code Layout

- `bayes_by_backprop/`: core model, Bayesian layers, losses, evaluation helpers, update-trust logic, and plotting utilities.
- `experiments/`: training, benchmarking, reproduction, and shell entry points.
- `evaluation/`: summary, plotting, audit, and final-figure generation scripts.
- `kalman_trust/`: Kalman-style optimizer/trust extensions, including earlier standalone Kalman and sample-trust explorations.
- `tests/`: focused tests for the core update-trust behavior.

Most scripts are intended to be run from the repository root, for example:

```bash
python code/experiments/train_mnist.py
python code/evaluation/summarize_results.py
python -m unittest discover -s code/tests
```
