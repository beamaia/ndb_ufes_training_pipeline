#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_REPO="${NDB_UFES_THESIS_REPO:-${TRAINING_REPO}/../ndb_ufes_data_organizer}"
ORGANIZER_DATA="${NDB_UFES_ORGANIZER_DATA:-${THESIS_REPO}/data}"
MLFLOW_ENV="${NDB_UFES_MLFLOW_ENV:-${TRAINING_REPO}/../ndb_ufes_mlflow/.env}"
OUTPUT_ROOT="${NDB_UFES_OUTPUT_ROOT:-${THESIS_REPO}/results/training_runs}"
LOG_DIR="${NDB_UFES_QUEUE_LOG_DIR:-${TRAINING_REPO}/logs/pndb_ufes_batches}"
PID_FILE="${LOG_DIR}/queue.pid"
LAUNCHER_PID_FILE="${LOG_DIR}/queue_launcher.pid"

mkdir -p "${LOG_DIR}"
cd "${TRAINING_REPO}" || exit 1
echo "$$" > "${PID_FILE}"
cleanup_queue_pid() {
  if [ -f "${PID_FILE}" ] && [ "$(cat "${PID_FILE}")" = "$$" ]; then
    rm -f "${PID_FILE}"
  fi
  if [ -f "${LAUNCHER_PID_FILE}" ] && [ "$(cat "${LAUNCHER_PID_FILE}")" = "$$" ]; then
    rm -f "${LAUNCHER_PID_FILE}"
  fi
}
trap cleanup_queue_pid EXIT INT TERM

export NDB_UFES_THESIS_REPO="${THESIS_REPO}"
export NDB_UFES_TRAINING_REPO="${TRAINING_REPO}"
export NDB_UFES_ORGANIZER_DATA="${ORGANIZER_DATA}"
export NDB_UFES_MLFLOW_ENV="${MLFLOW_ENV}"
export NDB_UFES_OUTPUT_ROOT="${OUTPUT_ROOT}"
export UV_CACHE_DIR="${TRAINING_REPO}/.uv-cache"
export DVC_SITE_CACHE_DIR="${TRAINING_REPO}/.dvc/tmp/site_cache"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

read -r -a MODELS <<< "${MODELS:-mobilenetv2 densenet121 resnet50}"
read -r -a BATCHES <<< "${BATCHES:-batch1 batch2}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-30}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MOMENTUM="${MOMENTUM:-0.9}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-10}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.1}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-0.000001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
DEVICE="${DEVICE:-auto}"
NODE_TYPE="${NODE_TYPE:-float32}"
TORCH_THREADS="${TORCH_THREADS:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-full_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"
ALLOW_EXPLORATORY_BATCH3="${ALLOW_EXPLORATORY_BATCH3:-0}"

for batch in "${BATCHES[@]}"; do
  if [[ "${batch}" == "batch3" && "${ALLOW_EXPLORATORY_BATCH3}" != "1" ]]; then
    echo "Batch 3 is exploratory. Set ALLOW_EXPLORATORY_BATCH3=1 explicitly." >&2
    exit 2
  fi
done

resume_args=()
case "${RESUME}" in
  1|true|TRUE|yes|YES) resume_args+=(--resume) ;;
esac

echo "Experiment queue started at $(date)"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} optimizer=${OPTIMIZER} learning_rate=${LEARNING_RATE}"
echo "scheduler_patience=${SCHEDULER_PATIENCE} early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
echo "device=${DEVICE} node_type=${NODE_TYPE} torch_threads=${TORCH_THREADS}"
echo "run_suffix=${RUN_SUFFIX} seed=${SEED}"
echo "resume=${RESUME}"
echo "training_repo=${TRAINING_REPO}"
echo "thesis_repo=${THESIS_REPO}"
echo "logs=${LOG_DIR}"

for batch in "${BATCHES[@]}"; do
  for model in "${MODELS[@]}"; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/${stamp}_${batch}_${model}.log"
    echo "START ${batch} ${model} $(date)" | tee -a "${LOG_DIR}/queue_status.log"
    (
      echo "START ${batch} ${model} $(date)"
      exploratory_args=()
      if [[ "${batch}" == "batch3" ]]; then
        exploratory_args+=(--allow-exploratory-batch3)
      fi
      uv run python "${TRAINING_REPO}/scripts/run_pndb_ufes_batch.py" \
        --batch "${batch}" \
        --models "${model}" \
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
        --node-type "${NODE_TYPE}" \
        --run-suffix "${RUN_SUFFIX}" \
        --seed "${SEED}" \
        "${exploratory_args[@]}" \
        "${resume_args[@]}"
      status=$?
      echo "END ${batch} ${model} status=${status} $(date)"
      exit "${status}"
    ) 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"
    echo "END ${batch} ${model} status=${status} log=${log_file} $(date)" | tee -a "${LOG_DIR}/queue_status.log"
    if [ "${status}" -eq 130 ] || [ "${status}" -eq 143 ]; then
      echo "Experiment queue interrupted after ${batch} ${model}." | tee -a "${LOG_DIR}/queue_status.log"
      exit "${status}"
    fi
    if [ "${status}" -ne 0 ]; then
      echo "Experiment queue stopped after failed ${batch} ${model}; inspect ${log_file}." | tee -a "${LOG_DIR}/queue_status.log"
      exit "${status}"
    fi
  done
done

echo "Experiment queue finished at $(date)" | tee -a "${LOG_DIR}/queue_status.log"
