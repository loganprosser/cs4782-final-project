#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

SEEDS="${SEEDS:-0 1 2}"
DATASET="${DATASET:-fashion_mnist}"
LABEL_NOISE="${LABEL_NOISE:-0.10}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-results/final_noisy_bbb_kalman/runs}"
SUMMARY_DIR="${SUMMARY_DIR:-results/final_noisy_bbb_kalman/summary}"

PARALLEL_GPUS="${PARALLEL_GPUS:-false}"
GPU_IDS="${GPU_IDS:-}"

KALMAN_BETA="${KALMAN_BETA:-0.95}"
KALMAN_PROCESS_NOISE="${KALMAN_PROCESS_NOISE:-1e-4}"
KALMAN_INITIAL_P="${KALMAN_INITIAL_P:-1.0}"
KALMAN_INITIAL_R="${KALMAN_INITIAL_R:-1.0}"
KALMAN_EPS="${KALMAN_EPS:-1e-8}"
KALMAN_CLIP_MIN="${KALMAN_CLIP_MIN:-0.05}"
KALMAN_CLIP_MAX="${KALMAN_CLIP_MAX:-1.0}"

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
  local model="$1"
  local bayesian_dropout="$2"
  local trust="$3"
  local seed="$4"
  local hidden_dim="$5"
  local hidden_layers="$6"
  local tag="$7"

  local args=(
    code/experiments/train_mnist.py
    --model "${model}"
    --bayesian-dropout "${bayesian_dropout}"
    --update-trust-mode "${trust}"
    --dataset "${DATASET}"
    --label-noise "${LABEL_NOISE}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
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
    echo
    echo ">>> ${model} trust=${trust} seed=${seed} h=${hidden_dim} layers=${hidden_layers} tag=${tag}"
    "${PYTHON_BIN}" "${args[@]}"
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

  train_one standard 0.0 none "${seed}" 800 2 "wide_${tag}"
  train_one standard 0.0 kalman_layer "${seed}" 800 2 "wide_${tag}"

  train_one dropout 0.0 none "${seed}" 400 2 "${tag}"
  train_one dropout 0.0 kalman_layer "${seed}" 400 2 "${tag}"

  train_one bayesian 0.0 none "${seed}" 400 2 "${tag}"
  train_one bayesian 0.0 kalman_layer "${seed}" 400 2 "${tag}"

  train_one bayesian 0.1 none "${seed}" 400 2 "${tag}"
  train_one bayesian 0.1 kalman_layer "${seed}" 400 2 "${tag}"
done

wait_for_parallel_jobs

echo
echo ">>> summarize final noisy BBB/Kalman sweep"
"${PYTHON_BIN}" code/evaluation/summarize_kalman_sweep.py \
  --mnist-dir "${OUTPUT_DIR}" \
  --summary-dir "${SUMMARY_DIR}" \
  --selection-mode best_val

echo
echo "Final noisy BBB/Kalman experiment completed. Summary: ${SUMMARY_DIR}"
