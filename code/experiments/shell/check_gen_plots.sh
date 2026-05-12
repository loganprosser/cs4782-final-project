#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

EXPERIMENT="${EXPERIMENT:-both}"
SELECTION_MODE="${SELECTION_MODE:-best_val}"
GENERATE="${GENERATE:-false}"
INCLUDE_SHIFT_CHECK="${INCLUDE_SHIFT_CHECK:-false}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected project environment at ${PYTHON_BIN}"
  exit 1
fi

ARGS=(
  code/evaluation/check_and_generate_plots.py
  --experiment "${EXPERIMENT}"
  --selection-mode "${SELECTION_MODE}"
)

if [[ "${GENERATE}" == "true" ]]; then
  ARGS+=(--generate)
fi

if [[ "${INCLUDE_SHIFT_CHECK}" == "true" ]]; then
  ARGS+=(--include-shift-check)
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ARGS[@]}"
