#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -x "../bayes-by-backprop-reimplementation/venv/bin/python" ]; then
  PYTHON="${PYTHON:-../bayes-by-backprop-reimplementation/venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

MODELS=(${MODELS:-mlp mlp_wide mlp_deep cnn cnn_deep cnn_bn resnet_tiny})
TRUST_MODES=(${TRUST_MODES:-none confidence entropy inverse_loss final_layer_align final_layer_align_mag ema_final_layer_align combined_simple})
BATCH_SIZES=(${BATCH_SIZES:-16 32 64 128 256})
SEEDS=(${SEEDS:-0 1 2})

EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-400}"
TRAIN_SUBSET="${TRAIN_SUBSET:-10000}"
TEST_SUBSET="${TEST_SUBSET:-2000}"
DATA_DIR="${DATA_DIR:-../bayes-by-backprop-reimplementation/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sample_trust}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_ROOT}/runs}"
REPORTS_DIR="${REPORTS_DIR:-${OUTPUT_ROOT}/reports}"
PLOTS_DIR="${PLOTS_DIR:-${REPORTS_DIR}/plots}"
SUMMARY_CSV="${SUMMARY_CSV:-${REPORTS_DIR}/summary.csv}"
DEVICE="${DEVICE:-auto}"
RUN_TAG="${RUN_TAG:-sample_trust_$(date +%Y%m%d_%H%M%S)}"
NO_DOWNLOAD="${NO_DOWNLOAD:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Sample trust sweep"
echo "  run tag: ${RUN_TAG}"
echo "  python: ${PYTHON}"
echo "  models: ${MODELS[*]}"
echo "  modes: ${TRUST_MODES[*]}"
echo "  batch sizes: ${BATCH_SIZES[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "  epochs: ${EPOCHS}"
echo "  run outputs: ${RUNS_DIR}"
echo "  reports: ${REPORTS_DIR}"
echo "  total runs: $((${#MODELS[@]} * ${#TRUST_MODES[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]}))"

download_args=()
if [ "${NO_DOWNLOAD}" = "1" ]; then
  download_args+=(--no-download)
fi

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for mode in "${TRUST_MODES[@]}"; do
        lr_tag="${LR//./p}"
        run_name="${RUN_TAG}_${model}_${mode}_bs${batch_size}_lr${lr_tag}_seed${seed}"
        echo
        echo "Running ${run_name}"
        "${PYTHON}" train_sample_trust_mnist.py \
          --model "${model}" \
          --hidden-dim "${HIDDEN_DIM}" \
          --sample-trust-mode "${mode}" \
          --batch-size "${batch_size}" \
          --seed "${seed}" \
          --epochs "${EPOCHS}" \
          --lr "${LR}" \
          --train-subset "${TRAIN_SUBSET}" \
          --test-subset "${TEST_SUBSET}" \
          --data-dir "${DATA_DIR}" \
          --output-dir "${RUNS_DIR}" \
          --summary-csv "${SUMMARY_CSV}" \
          --device "${DEVICE}" \
          --run-name "${run_name}" \
          "${download_args[@]}"
      done
    done
  done
done

"${PYTHON}" plot_sample_trust_results.py \
  --summary-csv "${SUMMARY_CSV}" \
  --output-dir "${PLOTS_DIR}" \
  --aggregate-only

"${PYTHON}" summarize_sample_trust_results.py \
  --summary-csv "${SUMMARY_CSV}" \
  --output-md "${REPORTS_DIR}/${RUN_TAG}_summary.md" \
  --runs-dir "${RUNS_DIR}" \
  --plots-dir "${PLOTS_DIR}" \
  --run-prefix "${RUN_TAG}_"

"${PYTHON}" plot_sample_trust_findings.py \
  --summary-csv "${SUMMARY_CSV}" \
  --run-prefix "${RUN_TAG}" \
  --output-dir "${PLOTS_DIR}"

echo
echo "Done."
echo "  Run outputs: ${RUNS_DIR}/"
echo "  Summary CSV: ${SUMMARY_CSV}"
echo "  Summary doc: ${REPORTS_DIR}/${RUN_TAG}_summary.md"
echo "  Combined findings: ${PLOTS_DIR}/${RUN_TAG}_combined_findings.png"
echo "  Plots: ${PLOTS_DIR}/"
