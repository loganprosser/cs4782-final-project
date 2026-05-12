#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

# Standalone experiment lane:
#   no Bayes by Backprop, just deterministic/dropout MLPs plus propagated
#   uncertainty gradient trust. All outputs go under results/prop_dropout so
#   this can run while train_prop.sh is also training.
RUN_STANDARD_BASELINE=true
RUN_DROPOUT_BASELINE=true
RUN_DROPOUT_PROPAGATED=true
RUN_SUMMARY=true

MNIST_EPOCHS=10
MNIST_BATCH_SIZE=128
SEED=0
DATA_DIR="data"
OUTPUT_DIR="results/prop_dropout/mnist"
SUMMARY_DIR="results/prop_dropout/summary"

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

if [[ "${RUN_STANDARD_BASELINE}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model standard \
    --bayesian-dropout 0.0 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"
fi

if [[ "${RUN_DROPOUT_BASELINE}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model dropout \
    --bayesian-dropout 0.0 \
    --update-trust-mode none \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"
fi

if [[ "${RUN_DROPOUT_PROPAGATED}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/experiments/train_mnist.py \
    --model dropout \
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
    --seed "${SEED}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"
fi

if [[ "${RUN_SUMMARY}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/evaluation/summarize_mnist_dir.py \
    --mnist-dir "${OUTPUT_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --selection-mode best_val
fi

echo
echo "Dropout propagated-uncertainty experiment completed."
