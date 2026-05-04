#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-true}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-false}"
RUN_KALMAN_IF_MISSING="${RUN_KALMAN_IF_MISSING:-false}"
INCLUDE_TRUST_RUNS="${INCLUDE_TRUST_RUNS:-true}"

MNIST_EPOCHS="${MNIST_EPOCHS:-10}"
MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-128}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-256}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
MC_SAMPLES="${MC_SAMPLES:-20}"
SEED="${SEED:-0}"
SELECTION_MODE="${SELECTION_MODE:-best_val}"

KALMAN_BETA="${KALMAN_BETA:-0.95}"
KALMAN_PROCESS_NOISE="${KALMAN_PROCESS_NOISE:-1e-4}"
KALMAN_INITIAL_P="${KALMAN_INITIAL_P:-1.0}"
KALMAN_INITIAL_R="${KALMAN_INITIAL_R:-1.0}"
KALMAN_EPS="${KALMAN_EPS:-1e-8}"
KALMAN_CLIP_MIN="${KALMAN_CLIP_MIN:-0.05}"
KALMAN_CLIP_MAX="${KALMAN_CLIP_MAX:-1.0}"

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

checkpoint_exists() {
  local run_name="$1"
  [[ -f "${ROOT_DIR}/results/mnist/checkpoints/${run_name}_best.pt" ]]
}

train_mnist_run() {
  local model="$1"
  local dropout="$2"
  local trust_mode="$3"
  run_cmd "${PYTHON_BIN}" code/train_mnist.py \
    --model "${model}" \
    --bayesian-dropout "${dropout}" \
    --update-trust-mode "${trust_mode}" \
    --epochs "${MNIST_EPOCHS}" \
    --batch-size "${MNIST_BATCH_SIZE}" \
    --seed "${SEED}"
}

train_kalman_run() {
  local dropout="$1"
  run_cmd "${PYTHON_BIN}" code/train_mnist.py \
    --model bayesian \
    --bayesian-dropout "${dropout}" \
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
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected project environment at ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/results/.matplotlib"
mkdir -p "${ROOT_DIR}/results/.cache"
export MPLCONFIGDIR="${ROOT_DIR}/results/.matplotlib"
export XDG_CACHE_HOME="${ROOT_DIR}/results/.cache"

if [[ "${RUN_UNIT_TESTS}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" -m unittest tests.test_update_trust
fi

if [[ "${TRAIN_IF_MISSING}" == "true" ]]; then
  checkpoint_exists standard_trust_none || train_mnist_run standard 0.0 none
  checkpoint_exists dropout_trust_none || train_mnist_run dropout 0.0 none
  checkpoint_exists bayesian_trust_none || train_mnist_run bayesian 0.0 none
  checkpoint_exists bayesian_dropout_0p1_trust_none || train_mnist_run bayesian 0.1 none
fi

if [[ "${RUN_KALMAN_IF_MISSING}" == "true" ]]; then
  checkpoint_exists bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0 || train_kalman_run 0.0
  checkpoint_exists bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0 || train_kalman_run 0.1
fi

if [[ "${TRAIN_IF_MISSING}" != "true" ]]; then
  missing=()
  checkpoint_exists standard_trust_none || missing+=(standard_trust_none)
  checkpoint_exists dropout_trust_none || missing+=(dropout_trust_none)
  checkpoint_exists bayesian_trust_none || missing+=(bayesian_trust_none)
  checkpoint_exists bayesian_dropout_0p1_trust_none || missing+=(bayesian_dropout_0p1_trust_none)
  if [[ "${INCLUDE_TRUST_RUNS}" == "true" ]]; then
    checkpoint_exists bayesian_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0 || missing+=(bayesian_trust_kalmanLayer)
    checkpoint_exists bayesian_dropout_0p1_trust_kalmanLayer_beta0.95_Q0.0001_P1.0_R1.0 || missing+=(bayesian_dropout_0p1_trust_kalmanLayer)
  fi
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Missing required checkpoints: ${missing[*]}"
    echo "Set TRAIN_IF_MISSING=true or RUN_KALMAN_IF_MISSING=true if you want this script to train missing runs."
    exit 1
  fi
fi

if [[ "${INCLUDE_TRUST_RUNS}" == "true" ]]; then
  run_cmd "${PYTHON_BIN}" code/benchmark_uncertainty_shift.py \
    --selection-mode "${SELECTION_MODE}" \
    --summary-dir results/uncertainty_shift \
    --output-dir results/uncertainty_shift \
    --batch-size "${BENCHMARK_BATCH_SIZE}" \
    --max-examples "${MAX_EXAMPLES}" \
    --mc-samples "${MC_SAMPLES}" \
    --seed "${SEED}" \
    --include-trust-runs
else
  run_cmd "${PYTHON_BIN}" code/benchmark_uncertainty_shift.py \
    --selection-mode "${SELECTION_MODE}" \
    --summary-dir results/uncertainty_shift \
    --output-dir results/uncertainty_shift \
    --batch-size "${BENCHMARK_BATCH_SIZE}" \
    --max-examples "${MAX_EXAMPLES}" \
    --mc-samples "${MC_SAMPLES}" \
    --seed "${SEED}"
fi

echo
echo "Uncertainty shift benchmark completed."
echo "Summary: ${ROOT_DIR}/results/uncertainty_shift/uncertainty_shift_summary.md"
