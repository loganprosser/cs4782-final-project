#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
SUMMARY_PYTHON_BIN="${PYTHON_BIN}"

# This script only trains the new propagated-uncertainty variants. It reuses the
# existing MNIST/regression data and appends new result rows/checkpoints.
RUN_UNIT_TESTS=true
RUN_MNIST_BAYESIAN_PROPAGATED=true
RUN_MNIST_BAYESIAN_DROPOUT_PROPAGATED=true
RUN_REGRESSION_PROPAGATED=false
RUN_SUMMARY=true
RUN_COMPUTE_BENCHMARK=true

MNIST_EPOCHS=10
REGRESSION_EPOCHS=2000
MNIST_BATCH_SIZE=128
REGRESSION_BATCH_SIZE=64
SEED=0

PROPAGATED_BETA=0.95
PROPAGATED_PROCESS_NOISE=1e-4
PROPAGATED_INITIAL_P=1.0
PROPAGATED_INITIAL_R=1.0
PROPAGATED_EPS=1e-8
PROPAGATED_CLIP_MIN=0.05
PROPAGATED_CLIP_MAX=1.0
PROPAGATED_DEPTH_LAMBDA=0.15

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
  run_cmd "${PYTHON_BIN}" -m unittest tests.test_update_trust
fi

if [[ "${RUN_MNIST_BAYESIAN_PROPAGATED}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.0 \
    --update-trust-mode propagated_uncertainty \
    --depth-decay-lambda "${PROPAGATED_DEPTH_LAMBDA}" \
    --kalman-beta "${PROPAGATED_BETA}" \
    --kalman-process-noise "${PROPAGATED_PROCESS_NOISE}" \
    --kalman-initial-P "${PROPAGATED_INITIAL_P}" \
    --kalman-initial-R "${PROPAGATED_INITIAL_R}" \
    --kalman-eps "${PROPAGATED_EPS}" \
    --kalman-clip-min "${PROPAGATED_CLIP_MIN}" \
    --kalman-clip-max "${PROPAGATED_CLIP_MAX}" \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_MNIST_BAYESIAN_DROPOUT_PROPAGATED}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/train_mnist.py \
    --model bayesian \
    --bayesian-dropout 0.1 \
    --update-trust-mode propagated_uncertainty \
    --depth-decay-lambda "${PROPAGATED_DEPTH_LAMBDA}" \
    --kalman-beta "${PROPAGATED_BETA}" \
    --kalman-process-noise "${PROPAGATED_PROCESS_NOISE}" \
    --kalman-initial-P "${PROPAGATED_INITIAL_P}" \
    --kalman-initial-R "${PROPAGATED_INITIAL_R}" \
    --kalman-eps "${PROPAGATED_EPS}" \
    --kalman-clip-min "${PROPAGATED_CLIP_MIN}" \
    --kalman-clip-max "${PROPAGATED_CLIP_MAX}" \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_REGRESSION_PROPAGATED}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/train_regression.py \
    --model bayesian \
    --update-trust-mode propagated_uncertainty \
    --depth-decay-lambda "${PROPAGATED_DEPTH_LAMBDA}" \
    --kalman-beta "${PROPAGATED_BETA}" \
    --kalman-process-noise "${PROPAGATED_PROCESS_NOISE}" \
    --kalman-initial-P "${PROPAGATED_INITIAL_P}" \
    --kalman-initial-R "${PROPAGATED_INITIAL_R}" \
    --kalman-eps "${PROPAGATED_EPS}" \
    --kalman-clip-min "${PROPAGATED_CLIP_MIN}" \
    --kalman-clip-max "${PROPAGATED_CLIP_MAX}" \
    --epochs "${REGRESSION_EPOCHS}" \
    --batch-size "${REGRESSION_BATCH_SIZE}" \
    --seed "${SEED}"
fi

if [[ "${RUN_SUMMARY}" == "true" ]]; then
  run_cmd "${SUMMARY_PYTHON_BIN}" code/summarize_results.py
fi

if [[ "${RUN_COMPUTE_BENCHMARK}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/benchmark_mnist_compute.py
fi

echo
echo "Propagated-uncertainty experiment stages completed."
