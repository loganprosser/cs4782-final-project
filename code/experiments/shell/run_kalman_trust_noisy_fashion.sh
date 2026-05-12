#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export DATASET="${DATASET:-fashion_mnist}"
export LABEL_NOISE="${LABEL_NOISE:-0.10}"
export OUTPUT_DIR="${OUTPUT_DIR:-kalman_trust_results_fashion_noise10}"
export EPOCHS="${EPOCHS:-10}"
export SEEDS="${SEEDS:-0 1 2}"
export VARIANTS="${VARIANTS:-adamw adamw_clip kalman_lr_controller direction_aware_trust per_unit_filter_trust combined_kalman_trust old_kalman_grad_scaling}"

exec "${ROOT_DIR}/run_kalman_trust_optimizers.sh"
