#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

# Finalist sweep: compare Kalman against no-trust baselines across seeds, and
# do a small deep/wide architecture probe. This intentionally drops depth_decay,
# grad_norm, grad_var, and propagated_uncertainty.
SEEDS="${SEEDS:-0 1 2}"
ARCH_PROBE_SEED="${ARCH_PROBE_SEED:-0}"
RUN_ARCH_PROBES="${RUN_ARCH_PROBES:-true}"
PARALLEL_GPUS="${PARALLEL_GPUS:-false}"
GPU_IDS="${GPU_IDS:-}"

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

if [[ "${PARALLEL_GPUS}" == "true" && -z "${GPU_IDS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  else
    GPU_IDS="0"
  fi
fi

if [[ "${PARALLEL_GPUS}" == "true" ]]; then
  read -r -a GPU_LIST <<< "${GPU_IDS}"
  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "PARALLEL_GPUS=true but no GPU ids were found."
    exit 1
  fi
  NEXT_GPU_INDEX=0
  PIDS=()
  echo "Parallel GPU mode enabled on GPU ids: ${GPU_IDS}"
fi

train_one() {
  local model="$1"
  local dropout="$2"
  local trust="$3"
  local seed="$4"
  local hidden_dim="$5"
  local hidden_layers="$6"
  local tag="$7"

  local args=(
    code/experiments/train_mnist.py
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

  if [[ "${PARALLEL_GPUS}" == "true" ]]; then
    local gpu="${GPU_LIST[${NEXT_GPU_INDEX}]}"
    NEXT_GPU_INDEX=$(( (NEXT_GPU_INDEX + 1) % ${#GPU_LIST[@]} ))
    echo
    echo ">>> gpu=${gpu} ${model} trust=${trust} seed=${seed} h=${hidden_dim} layers=${hidden_layers} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" &
    PIDS+=("$!")
    if [[ "${#PIDS[@]}" -ge "${#GPU_LIST[@]}" ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  else
    run_cmd "${model} trust=${trust} seed=${seed} h=${hidden_dim} layers=${hidden_layers} tag=${tag}" "${PYTHON_BIN}" "${args[@]}"
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

wait_for_parallel_jobs

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

wait_for_parallel_jobs

run_cmd "summarize kalman sweep" "${PYTHON_BIN}" code/evaluation/summarize_kalman_sweep.py \
  --mnist-dir "${OUTPUT_DIR}" \
  --summary-dir "${SUMMARY_DIR}" \
  --selection-mode best_val

echo
echo "Kalman finalist sweep completed. Summary: ${SUMMARY_DIR}"
