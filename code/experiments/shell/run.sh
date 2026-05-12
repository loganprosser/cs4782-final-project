#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

RUN_UNIT_TESTS=true
RUN_BASELINES=true
RUN_TRUST=true
RUN_KALMAN=true
RUN_REGRESSION=false
BUILD_SUMMARIES=true
RUN_BENCHMARKS=true

MNIST_EPOCHS=10
REGRESSION_EPOCHS=2000
MNIST_BATCH_SIZE=128
REGRESSION_BATCH_SIZE=64
SEED=0

KALMAN_BETA=0.95
KALMAN_PROCESS_NOISE=1e-4
KALMAN_INITIAL_P=1.0
KALMAN_INITIAL_R=1.0
KALMAN_EPS=1e-8
KALMAN_CLIP_MIN=0.05
KALMAN_CLIP_MAX=1.0

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

if [[ "${RUN_BASELINES}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model standard --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model dropout --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.1 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
fi

if [[ "${RUN_TRUST}" == "true" ]]; then
  for mode in depth_decay grad_norm running_grad_var; do
    run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.0 --update-trust-mode "${mode}" --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
    run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.1 --update-trust-mode "${mode}" --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  done
fi

if [[ "${RUN_KALMAN}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.0 --update-trust-mode kalman_layer --kalman-beta "${KALMAN_BETA}" --kalman-process-noise "${KALMAN_PROCESS_NOISE}" --kalman-initial-P "${KALMAN_INITIAL_P}" --kalman-initial-R "${KALMAN_INITIAL_R}" --kalman-eps "${KALMAN_EPS}" --kalman-clip-min "${KALMAN_CLIP_MIN}" --kalman-clip-max "${KALMAN_CLIP_MAX}" --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.1 --update-trust-mode kalman_layer --kalman-beta "${KALMAN_BETA}" --kalman-process-noise "${KALMAN_PROCESS_NOISE}" --kalman-initial-P "${KALMAN_INITIAL_P}" --kalman-initial-R "${KALMAN_INITIAL_R}" --kalman-eps "${KALMAN_EPS}" --kalman-clip-min "${KALMAN_CLIP_MIN}" --kalman-clip-max "${KALMAN_CLIP_MAX}" --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
fi

if [[ "${RUN_REGRESSION}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py --model bayesian --update-trust-mode none --epochs "${REGRESSION_EPOCHS}" --batch-size "${REGRESSION_BATCH_SIZE}" --seed "${SEED}"
  for mode in depth_decay grad_norm running_grad_var kalman_layer; do
    if [[ "${mode}" == "kalman_layer" ]]; then
      run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py --model bayesian --update-trust-mode "${mode}" --kalman-beta "${KALMAN_BETA}" --kalman-process-noise "${KALMAN_PROCESS_NOISE}" --kalman-initial-P "${KALMAN_INITIAL_P}" --kalman-initial-R "${KALMAN_INITIAL_R}" --kalman-eps "${KALMAN_EPS}" --kalman-clip-min "${KALMAN_CLIP_MIN}" --kalman-clip-max "${KALMAN_CLIP_MAX}" --epochs "${REGRESSION_EPOCHS}" --batch-size "${REGRESSION_BATCH_SIZE}" --seed "${SEED}"
    else
      run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py --model bayesian --update-trust-mode "${mode}" --epochs "${REGRESSION_EPOCHS}" --batch-size "${REGRESSION_BATCH_SIZE}" --seed "${SEED}"
    fi
  done
fi

if [[ "${BUILD_SUMMARIES}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/evaluation/summarize_results.py --selection-mode best_val --summary-dir results/bayes_by_backprop/summary_best_val
  run_cmd "${PYTHON_BIN}" code/evaluation/summarize_results.py --selection-mode last_epoch --summary-dir results/bayes_by_backprop/summary_last_epoch
fi

if [[ "${RUN_BENCHMARKS}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/benchmark_mnist_compute.py --selection-mode best_val --summary-dir results/bayes_by_backprop/summary_best_val --benchmark-output mnist_compute_benchmark_best_val.json
  run_cmd "${PYTHON_BIN}" code/experiments/benchmark_mnist_compute.py --selection-mode last_epoch --summary-dir results/bayes_by_backprop/summary_last_epoch --benchmark-output mnist_compute_benchmark_last_epoch.json
fi

echo
echo "Full experiment suite completed."
