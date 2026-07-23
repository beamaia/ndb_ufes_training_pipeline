#!/bin/bash

set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO}/logs/pndb_ufes_batches"
LOG_FILE="${LOG_DIR}/mps_validation_external.log"

mkdir -p "${LOG_DIR}"
cd "${REPO}" || exit 1

{
  echo "MPS validation started: $(date)"
  ./.venv/bin/python scripts/check_mps.py
  device_status=$?
  if [ "${device_status}" -eq 0 ]; then
    ./.venv/bin/python scripts/check_models_mps.py
    model_status=$?
  else
    echo "Model checks skipped because the device check failed."
    model_status="${device_status}"
  fi
  echo "MPS validation finished: $(date)"
  echo "device_status=${device_status} model_status=${model_status}"
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Log saved to ${LOG_FILE}"
read -r -n 1 -p "Press any key to close this Terminal window."
