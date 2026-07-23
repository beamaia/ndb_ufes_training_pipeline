#!/usr/bin/env bash
set -euo pipefail

# Archived three-batch launcher. Batch 3 was not part of the final v1.0.0
# thesis comparison and must be requested explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
THESIS_REPO="${NDB_UFES_THESIS_REPO:-${TRAINING_REPO}/../ndb_ufes_data_organizer}"

export NDB_UFES_MLFLOW_ENV="${NDB_UFES_MLFLOW_ENV:-${TRAINING_REPO}/../ndb_ufes_mlflow/.env}"
export NDB_UFES_ORGANIZER_DATA="${NDB_UFES_ORGANIZER_DATA:-${THESIS_REPO}/data}"
export NDB_UFES_OUTPUT_ROOT="${NDB_UFES_OUTPUT_ROOT:-${THESIS_REPO}/results/training_runs}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:8000/}"
export TORCH_HOME="${TORCH_HOME:-${TRAINING_REPO}/.cache/torch}"

cd "${TRAINING_REPO}"

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-30}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MOMENTUM="${MOMENTUM:-0.9}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-10}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.1}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-0.000001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
TORCH_THREADS="${TORCH_THREADS:-1}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-full_$(date +%Y%m%d_%H%M%S)}"

read -r -a MODELS <<< "${MODELS:-mobilenetv2 densenet121 resnet50}"

for batch in batch1 batch2 batch3; do
echo "Starting ${batch}: models=${#MODELS[@]}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, device=${DEVICE}, torch_threads=${TORCH_THREADS}"
  uv run python -u scripts/run_pndb_ufes_batch.py \
    --batch "${batch}" \
    --models "${MODELS[@]}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --optimizer "${OPTIMIZER}" \
    --learning-rate "${LEARNING_RATE}" \
    --momentum "${MOMENTUM}" \
    --scheduler-patience "${SCHEDULER_PATIENCE}" \
    --scheduler-factor "${SCHEDULER_FACTOR}" \
    --scheduler-min-lr "${SCHEDULER_MIN_LR}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --device "${DEVICE}" \
    --torch-threads "${TORCH_THREADS}" \
    --tracking-uri "${MLFLOW_TRACKING_URI}" \
    --run-suffix "${RUN_SUFFIX}" \
    --seed "${SEED}" \
    --allow-exploratory-batch3
done

echo "All three P-NDB-UFES batches completed, including exploratory Batch 3."
