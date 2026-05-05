#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-adamw adamw_clip kalman_lr_controller direction_aware_trust per_unit_filter_trust combined_kalman_trust old_kalman_grad_scaling}"
MODEL="${MODEL:-standard}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-kalman_trust_results}"

PARALLEL_GPUS="${PARALLEL_GPUS:-false}"
GPU_IDS="${GPU_IDS:-}"

KALMAN_BETA="${KALMAN_BETA:-0.95}"
KALMAN_Q="${KALMAN_Q:-1e-4}"
KALMAN_P0="${KALMAN_P0:-1.0}"
KALMAN_R0="${KALMAN_R0:-1.0}"
TRUST_MIN="${TRUST_MIN:-0.05}"
TRUST_MAX="${TRUST_MAX:-1.0}"

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

run_one() {
  local variant="$1"
  local seed="$2"
  local args=(
    code/train_kalman_trust_mnist.py
    --optimizer_variant "${variant}"
    --model "${MODEL}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --seed "${seed}"
    --output-dir "${OUTPUT_DIR}"
    --kalman_beta "${KALMAN_BETA}"
    --kalman_Q "${KALMAN_Q}"
    --kalman_P0 "${KALMAN_P0}"
    --kalman_R0 "${KALMAN_R0}"
    --trust_min "${TRUST_MIN}"
    --trust_max "${TRUST_MAX}"
    --quiet
  )

  if [[ "${PARALLEL_GPUS}" == "true" ]]; then
    local gpu="${GPU_LIST[${NEXT_GPU_INDEX}]}"
    NEXT_GPU_INDEX=$(( (NEXT_GPU_INDEX + 1) % ${#GPU_LIST[@]} ))
    echo
    echo ">>> gpu=${gpu} variant=${variant} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" &
    PIDS+=("$!")
    if [[ "${#PIDS[@]}" -ge "${#GPU_LIST[@]}" ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  else
    echo
    echo ">>> variant=${variant} seed=${seed}"
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
  for variant in ${VARIANTS}; do
    run_one "${variant}" "${seed}"
  done
done

wait_for_parallel_jobs

echo
echo ">>> summarize optimizer variants"
"${PYTHON_BIN}" code/summarize_kalman_trust_results.py --results-dir "${OUTPUT_DIR}"

echo
echo "Kalman trust optimizer experiment completed. Results: ${OUTPUT_DIR}"
