#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

# Confirmation run for the best lead from the Kalman sweep:
# wide Standard MLP h800/L2, no trust vs Kalman, across more seeds.
SEEDS="${SEEDS:-0 1 2 3 4}"
PARALLEL_GPUS="${PARALLEL_GPUS:-false}"
GPU_IDS="${GPU_IDS:-}"

MNIST_EPOCHS="${MNIST_EPOCHS:-10}"
MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-800}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/wide_standard_confirm/mnist}"
SUMMARY_DIR="${SUMMARY_DIR:-results/wide_standard_confirm/summary}"

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

if [[ "${PARALLEL_GPUS}" == "true" && -z "${GPU_IDS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  else
    GPU_IDS="0"
  fi
fi

if [[ "${PARALLEL_GPUS}" == "true" ]]; then
  read -r -a GPU_LIST <<< "${GPU_IDS}"
  NEXT_GPU_INDEX=0
  PIDS=()
  echo "Parallel GPU mode enabled on GPU ids: ${GPU_IDS}"
fi

train_one() {
  local trust="$1"
  local seed="$2"
  local tag="$3"

  local args=(
    code/train_mnist.py
    --model standard
    --bayesian-dropout 0.0
    --update-trust-mode "${trust}"
    --epochs "${MNIST_EPOCHS}"
    --batch-size "${MNIST_BATCH_SIZE}"
    --seed "${seed}"
    --hidden-dim "${HIDDEN_DIM}"
    --hidden-layers "${HIDDEN_LAYERS}"
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

  if [[ "${PARALLEL_GPUS}" == "true" ]]; then
    local gpu="${GPU_LIST[${NEXT_GPU_INDEX}]}"
    NEXT_GPU_INDEX=$(( (NEXT_GPU_INDEX + 1) % ${#GPU_LIST[@]} ))
    echo
    echo ">>> gpu=${gpu} standard trust=${trust} seed=${seed} h=${HIDDEN_DIM} layers=${HIDDEN_LAYERS}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" &
    PIDS+=("$!")
    if [[ "${#PIDS[@]}" -ge "${#GPU_LIST[@]}" ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  else
    run_cmd "standard trust=${trust} seed=${seed} h=${HIDDEN_DIM} layers=${HIDDEN_LAYERS}" "${PYTHON_BIN}" "${args[@]}"
  fi
}

wait_for_parallel_jobs() {
  if [[ "${PARALLEL_GPUS}" != "true" ]]; then
    return
  fi
  local pid
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
  PIDS=()
}

for seed in ${SEEDS}; do
  train_one none "${seed}" "s${seed}"
  train_one kalman_layer "${seed}" "s${seed}"
done

wait_for_parallel_jobs

run_cmd "summarize wide standard confirmation" "${PYTHON_BIN}" code/summarize_kalman_sweep.py \
  --mnist-dir "${OUTPUT_DIR}" \
  --summary-dir "${SUMMARY_DIR}" \
  --selection-mode best_val

echo
echo "Wide Standard MLP confirmation completed. Summary: ${SUMMARY_DIR}"
