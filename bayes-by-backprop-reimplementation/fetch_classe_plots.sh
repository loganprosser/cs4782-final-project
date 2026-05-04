#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lap284@lnx4555.classe.cornell.edu}"
REMOTE_REPO="${REMOTE_REPO:-/nfs/cms/tracktrigger/logan/cs4782/final_project/cs4782-final-project}"
REMOTE_PROJECT="${REMOTE_PROJECT:-${REMOTE_REPO}/bayes-by-backprop-reimplementation}"
LOCAL_DIR="${LOCAL_DIR:-classe_plots}"

mkdir -p "${LOCAL_DIR}"

echo "Fetching main summary plots..."
scp -r "${REMOTE_HOST}:${REMOTE_PROJECT}/results/summary_best_val" "${LOCAL_DIR}/"

echo "Fetching dropout propagated summary plots..."
scp -r "${REMOTE_HOST}:${REMOTE_PROJECT}/results/prop_dropout/summary" "${LOCAL_DIR}/prop_dropout_summary"

echo
echo "Plots copied into ${LOCAL_DIR}/"
