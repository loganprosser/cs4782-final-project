#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
OUTPUT_SCRIPT="${OUTPUT_SCRIPT:-temp_run_missing_and_plot.sh}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected project environment at ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" code/plan_missing_runs.py --write-script "${OUTPUT_SCRIPT}"
