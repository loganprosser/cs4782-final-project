#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SUMMARY_PYTHON_BIN="${PYTHON_BIN}"

RUN_UNIT_TESTS=true
RUN_MNIST_BASELINES=true
RUN_MNIST_BAYESIAN_KALMAN=true
RUN_MNIST_BAYESIAN_DROPOUT_KALMAN=true
RUN_REGRESSION_KALMAN=false
RUN_SUMMARY=true
RUN_COMPUTE_BENCHMARK=true

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

if [[ "${RUN_MNIST_BASELINES}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model standard --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model dropout --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.0 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py --model bayesian --bayesian-dropout 0.1 --update-trust-mode none --epochs "${MNIST_EPOCHS}" --batch-size "${MNIST_BATCH_SIZE}" --seed "${SEED}"
fi

if [[ "${RUN_MNIST_BAYESIAN_KALMAN}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.0 \
    --update-trust-mode kalman_layer \
    --kalman-beta "${KALMAN_BETA}" \
    --kalman-process-noise "${KALMAN_PROCESS_NOISE}" \
    --kalman-initial-P "${KALMAN_INITIAL_P}" \
    --kalman-initial-R "${KALMAN_INITIAL_R}" \
    --kalman-eps "${KALMAN_EPS}" \
    --kalman-clip-min "${KALMAN_CLIP_MIN}" \
    --kalman-clip-max "${KALMAN_CLIP_MAX}" \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_MNIST_BAYESIAN_DROPOUT_KALMAN}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.1 \
    --update-trust-mode kalman_layer \
    --kalman-beta "${KALMAN_BETA}" \
    --kalman-process-noise "${KALMAN_PROCESS_NOISE}" \
    --kalman-initial-P "${KALMAN_INITIAL_P}" \
    --kalman-initial-R "${KALMAN_INITIAL_R}" \
    --kalman-eps "${KALMAN_EPS}" \
    --kalman-clip-min "${KALMAN_CLIP_MIN}" \
    --kalman-clip-max "${KALMAN_CLIP_MAX}" \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_REGRESSION_KALMAN}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_regression.py \
    --model bayesian \
    --update-trust-mode kalman_layer \
    --kalman-beta "${KALMAN_BETA}" \
    --kalman-process-noise "${KALMAN_PROCESS_NOISE}" \
    --kalman-initial-P "${KALMAN_INITIAL_P}" \
    --kalman-initial-R "${KALMAN_INITIAL_R}" \
    --kalman-eps "${KALMAN_EPS}" \
    --kalman-clip-min "${KALMAN_CLIP_MIN}" \
    --kalman-clip-max "${KALMAN_CLIP_MAX}" \
    --epochs "${REGRESSION_EPOCHS}" \
    --batch-size "${REGRESSION_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_SUMMARY}" == "true" ]]; then
  run_cmd "${SUMMARY_PYTHON_BIN}" code/evaluation/summarize_results.py
fi

if [[ "${RUN_COMPUTE_BENCHMARK}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/benchmark_mnist_compute.py
fi

echo
echo "Kalman experiment stages completed."
