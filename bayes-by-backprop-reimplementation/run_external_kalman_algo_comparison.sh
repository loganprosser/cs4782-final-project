#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

SEEDS="${SEEDS:-0}"
DIAG_MODES="${DIAG_MODES:-tensor scalar}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-results/external_kalman_algo}"
PLOT_DIR="${PLOT_DIR:-final_plots_with_external_kalman}"

PARALLEL_GPUS="${PARALLEL_GPUS:-false}"
GPU_IDS="${GPU_IDS:-}"

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
  local dataset="$1"
  local label_noise="$2"
  local model="$3"
  local bayesian_dropout="$4"
  local hidden_dim="$5"
  local hidden_layers="$6"
  local seed="$7"
  local diag_mode="$8"

  echo
  echo ">>> external kalmanalgo dataset=${dataset} noise=${label_noise} model=${model} h=${hidden_dim}/L${hidden_layers} seed=${seed} diag=${diag_mode}"
  local args=(
    code/train_external_kalman_mnist.py
    --dataset "${dataset}" \
    --label-noise "${label_noise}" \
    --model "${model}" \
    --bayesian-dropout "${bayesian_dropout}" \
    --hidden-dim "${hidden_dim}" \
    --hidden-layers "${hidden_layers}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${seed}" \
    --diag-mode "${diag_mode}" \
    --output-dir "${OUTPUT_DIR}" \
    --quiet
  )

  if [[ "${PARALLEL_GPUS}" == "true" ]]; then
    local gpu="${GPU_LIST[${NEXT_GPU_INDEX}]}"
    NEXT_GPU_INDEX=$(( (NEXT_GPU_INDEX + 1) % ${#GPU_LIST[@]} ))
    echo "    gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" &
    PIDS+=("$!")
    if [[ "${#PIDS[@]}" -ge "${#GPU_LIST[@]}" ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  else
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
  for diag_mode in ${DIAG_MODES}; do
    train_one mnist 0.0 bayesian 0.1 400 2 "${seed}" "${diag_mode}"
    train_one mnist 0.0 standard 0.0 800 2 "${seed}" "${diag_mode}"
    train_one fashion_mnist 0.10 standard 0.0 400 2 "${seed}" "${diag_mode}"
    train_one fashion_mnist 0.10 standard 0.0 800 2 "${seed}" "${diag_mode}"
  done
done

wait_for_parallel_jobs

echo
echo ">>> regenerate copied final plots with external kalmanalgo rows"
"${PYTHON_BIN}" code/make_external_kalman_final_plots.py \
  --external-json "${OUTPUT_DIR}/test_accuracy.json" \
  --output-dir "${PLOT_DIR}"

echo
echo "External kalmanalgo comparison complete: ${PLOT_DIR}"
