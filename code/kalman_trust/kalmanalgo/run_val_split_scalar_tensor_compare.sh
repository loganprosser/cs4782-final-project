#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRIPT_DIR/.cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
RESULT_ROOT="../../../results/kalmanalgo_val_split_scalar_tensor_compare"
DATA_DIR="${DATA_DIR:-../../../data}"

COMMON_ARGS=(
  --run-all
  --epochs 10
  --batch-size 128
  --lr 0.01
  --optimizer sgd
  --momentum 0.9
  --hidden-size 400
  --val-split 0.1
  --device mps
  --data-dir "$DATA_DIR"
  --no-download
  --run-name ""
  --no-timestamped-result-dir
)

"$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" --diag-mode tensor --results-dir "$RESULT_ROOT/tensor"
"$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" --diag-mode scalar --results-dir "$RESULT_ROOT/scalar"
