#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

# Finalist sweep: compare Kalman against no-trust baselines across seeds, and
# do a small deep/wide architecture probe. This intentionally drops depth_decay,
# grad_norm, grad_var, and propagated_uncertainty.
SEEDS="${SEEDS:-0 1 2}"
ARCH_PROBE_SEED="${ARCH_PROBE_SEED:-0}"
RUN_ARCH_PROBES="${RUN_ARCH_PROBES:-true}"

MNIST_EPOCHS="${MNIST_EPOCHS:-10}"
MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-results/kalman_sweep/mnist}"
SUMMARY_DIR="${SUMMARY_DIR:-results/kalman_sweep/summary}"

KALMAN_BETA="${KALMAN_BETA:-0.95}"
KALMAN_PROCESS_NOISE="${KALMAN_PROCESS_NOISE:-1e-4}"
KALMAN_INITIAL_P="${KALMAN_INITIAL_P:-1.0}"
KALMAN_INITIAL_R="${KALMAN_INITIAL_R:-1.0}"
KALMAN_EPS="${KALMAN_EPS:-1e-8}"
KALMAN_CLIP_MIN="${KALMAN_CLIP_MIN:-0.05}"
KALMAN_CLIP_MAX="${KALMAN_CLIP_MAX:-1.0}"

run_cmd() {
  echo
  echo ">>> $1"
  shift
  "$@"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected project environment at ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_DIR}"

train_one() {
  local model="$1"
  local dropout="$2"
  local trust="$3"
  local seed="$4"
  local hidden_dim="$5"
  local hidden_layers="$6"
  local tag="$7"

  local args=(
    code/train_mnist.py
    --model "${model}"
    --bayesian-dropout "${dropout}"
    --update-trust-mode "${trust}"
    --epochs "${MNIST_EPOCHS}"
    --batch-size "${MNIST_BATCH_SIZE}"
    --seed "${seed}"
    --hidden-dim "${hidden_dim}"
    --hidden-layers "${hidden_layers}"
    --output-dir "${OUTPUT_DIR}"
    --run-tag "${tag}"
    --quiet
    --no-log-update-trust
    --no-log-kalman-trust
  )

  if [[ "${trust}" == "kalman_layer" ]]; then
    args+=(
      --kalman-beta "${KALMAN_BETA}"
      --kalman-process-noise "${KALMAN_PROCESS_NOISE}"
      --kalman-initial-P "${KALMAN_INITIAL_P}"
      --kalman-initial-R "${KALMAN_INITIAL_R}"
      --kalman-eps "${KALMAN_EPS}"
      --kalman-clip-min "${KALMAN_CLIP_MIN}"
      --kalman-clip-max "${KALMAN_CLIP_MAX}"
    )
  fi

  run_cmd "${model} trust=${trust} seed=${seed} h=${hidden_dim} layers=${hidden_layers} tag=${tag}" "${PYTHON_BIN}" "${args[@]}"
}

for seed in ${SEEDS}; do
  tag="s${seed}"

  train_one standard 0.0 none "${seed}" 400 2 "${tag}"
  train_one standard 0.0 kalman_layer "${seed}" 400 2 "${tag}"

  train_one dropout 0.0 none "${seed}" 400 2 "${tag}"
  train_one dropout 0.0 kalman_layer "${seed}" 400 2 "${tag}"

  train_one bayesian 0.0 none "${seed}" 400 2 "${tag}"
  train_one bayesian 0.0 kalman_layer "${seed}" 400 2 "${tag}"

  train_one bayesian 0.1 none "${seed}" 400 2 "${tag}"
  train_one bayesian 0.1 kalman_layer "${seed}" 400 2 "${tag}"
done

if [[ "${RUN_ARCH_PROBES}" == "true" ]]; then
  for arch in wide deep; do
    if [[ "${arch}" == "wide" ]]; then
      hidden_dim=800
      hidden_layers=2
    else
      hidden_dim=400
      hidden_layers=3
    fi
    tag="${arch}_s${ARCH_PROBE_SEED}"

    train_one standard 0.0 none "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"
    train_one standard 0.0 kalman_layer "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"

    train_one dropout 0.0 none "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"
    train_one dropout 0.0 kalman_layer "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"

    train_one bayesian 0.0 none "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"
    train_one bayesian 0.0 kalman_layer "${ARCH_PROBE_SEED}" "${hidden_dim}" "${hidden_layers}" "${tag}"
  done
fi

run_cmd "summarize kalman sweep" "${PYTHON_BIN}" code/summarize_kalman_sweep.py \
  --mnist-dir "${OUTPUT_DIR}" \
  --summary-dir "${SUMMARY_DIR}" \
  --selection-mode best_val

echo
echo "Kalman finalist sweep completed. Summary: ${SUMMARY_DIR}"
