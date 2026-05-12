#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SUMMARY_PYTHON_BIN="${PYTHON_BIN}"

# Toggle sections on or off here.
RUN_UNIT_TESTS=true
RUN_MNIST_BASELINES=true
RUN_MNIST_BAYESIAN_TRUST=true
RUN_MNIST_BAYESIAN_DROPOUT_TRUST=true
RUN_REGRESSION_TRUST=false
RUN_SUMMARY=true
RUN_COMPUTE_BENCHMARK=true

# Common hyperparameters.
MNIST_EPOCHS=10
REGRESSION_EPOCHS=2000
MNIST_BATCH_SIZE=128
REGRESSION_BATCH_SIZE=64
SEED=0

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected project environment at ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_DIR}"

if [[ "${RUN_UNIT_TESTS}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" -m unittest discover -s code/tests
fi

if [[ "${RUN_MNIST_BASELINES}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model standard \
    --bayesian-dropout 0.0 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"

  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model dropout \
    --bayesian-dropout 0.0 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"

  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.0 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"

  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.1 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_MNIST_BAYESIAN_TRUST}" == "true" ]]; then
  for mode in depth_decay grad_norm running_grad_var; do
    run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
      --model bayesian \
      --bayesian-dropout 0.0 \
      --update-trust-mode "${mode}" \
      --epochs "${MNIST_EPOCHS}" \
      --batch-size "${MNIST_BATCH_SIZE}" \
      --seed "${SEED}"
  done
fi

if [[ "${RUN_MNIST_BAYESIAN_DROPOUT_TRUST}" == "true" ]]; then
  for mode in depth_decay grad_norm running_grad_var; do
    run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
      --model bayesian \
      --bayesian-dropout 0.1 \
      --update-trust-mode "${mode}" \
      --epochs "${MNIST_EPOCHS}" \
      --batch-size "${MNIST_BATCH_SIZE}" \
      --seed "${SEED}"
  done
fi

if [[ "${RUN_REGRESSION_TRUST}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py \
    --model bayesian \
    --update-trust-mode none \
    --epochs "${REGRESSION_EPOCHS}" \
    --batch-size "${REGRESSION_BATCH_SIZE}" \
    --seed "${SEED}"

  for mode in depth_decay grad_norm running_grad_var; do
    run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py \
      --model bayesian \
      --update-trust-mode "${mode}" \
      --epochs "${REGRESSION_EPOCHS}" \
      --batch-size "${REGRESSION_BATCH_SIZE}" \
      --seed "${SEED}"
  done
fi

if [[ "${RUN_SUMMARY}" == "true" ]]; then
  run_cmd "${SUMMARY_PYTHON_BIN}" code/evaluation/summarize_results.py
fi

if [[ "${RUN_COMPUTE_BENCHMARK}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/benchmark_mnist_compute.py
fi

echo
echo "All requested experiment stages completed."
