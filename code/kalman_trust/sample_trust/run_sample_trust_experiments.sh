#!/usr/bin/env bash
set -euo pipefail

TRUST_MODES=(
  none
  confidence
  entropy
  inverse_loss
  final_layer_align
  final_layer_align_mag
  ema_final_layer_align
  combined_simple
)

BATCH_SIZES=(16 32 64 128 256)
SEEDS=(0)

EPOCHS="${EPOCHS:-5}"
TRAIN_SUBSET="${TRAIN_SUBSET:-10000}"
TEST_SUBSET="${TEST_SUBSET:-2000}"
DATA_DIR="${DATA_DIR:-../../../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../../../results/sample_trust}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_ROOT}/runs}"
REPORTS_DIR="${REPORTS_DIR:-${OUTPUT_ROOT}/reports}"
PLOTS_DIR="${PLOTS_DIR:-${REPORTS_DIR}/plots}"
SUMMARY_CSV="${SUMMARY_CSV:-${REPORTS_DIR}/summary.csv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

for seed in "${SEEDS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for mode in "${TRUST_MODES[@]}"; do
      python3 train_sample_trust_mnist.py \
        --model mlp \
        --sample-trust-mode "${mode}" \
        --batch-size "${batch_size}" \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --train-subset "${TRAIN_SUBSET}" \
        --test-subset "${TEST_SUBSET}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${RUNS_DIR}" \
        --summary-csv "${SUMMARY_CSV}"
    done
  done
done

python3 plot_sample_trust_results.py --summary-csv "${SUMMARY_CSV}" --output-dir "${PLOTS_DIR}" --aggregate-only
