#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_REPO="${NDB_UFES_THESIS_REPO:-${TRAINING_REPO}/../ndb_ufes_data_organizer}"
ORGANIZER_DATA="${NDB_UFES_ORGANIZER_DATA:-${THESIS_REPO}/data}"
MLFLOW_ENV="${NDB_UFES_MLFLOW_ENV:-${TRAINING_REPO}/../ndb_ufes_mlflow/.env}"
OUTPUT_ROOT="${NDB_UFES_OUTPUT_ROOT:-${THESIS_REPO}/results/training_runs}"
LOG_DIR="${NDB_UFES_QUEUE_LOG_DIR:-${TRAINING_REPO}/logs/pndb_ufes_batches}"
PID_FILE="${LOG_DIR}/queue.pid"
LAUNCHER_PID_FILE="${LOG_DIR}/queue_launcher.pid"
DEVICE="${DEVICE:-auto}"
MPS_PREFLIGHT="${NDB_UFES_MPS_PREFLIGHT:-1}"
BATCH_SIZE="${BATCH_SIZE:-30}"
EPOCHS="${EPOCHS:-150}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MOMENTUM="${MOMENTUM:-0.9}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-10}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.1}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-0.000001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
RUN_SUFFIX="${RUN_SUFFIX:-full_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
# Estimated shortest-to-longest order for a single batch on Apple Silicon.
DEFAULT_MODELS="mobilenetv2 densenet121 resnet50"
MODELS="${MODELS:-${DEFAULT_MODELS}}"
read -r -a REQUESTED_MODELS <<< "${MODELS}"

for required in "${THESIS_REPO}" "${ORGANIZER_DATA}" "${MLFLOW_ENV}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
for pid_file in "${PID_FILE}" "${LAUNCHER_PID_FILE}"; do
  if [ -f "${pid_file}" ]; then
    recorded_pid="$(tr -d '[:space:]' < "${pid_file}")"
    if [[ "${recorded_pid}" =~ ^[0-9]+$ ]] && kill -0 "${recorded_pid}" 2>/dev/null; then
      recorded_command="$(ps -p "${recorded_pid}" -o command= 2>/dev/null || true)"
      case "${recorded_command}" in
        *run_pndb_ufes_queue.sh*|*start_pndb_ufes_queue.sh*)
          echo "A P-NDB-UFES queue process is already running (PID ${recorded_pid})." >&2
          echo "Run scripts/stop_pndb_ufes_queue.sh first." >&2
          exit 1
          ;;
        *)
          echo "Removing a reused stale PID file: ${pid_file} (PID ${recorded_pid})." >&2
          rm -f "${pid_file}"
          ;;
      esac
    else
      rm -f "${pid_file}"
    fi
  fi
done

export NDB_UFES_THESIS_REPO="${THESIS_REPO}"
export NDB_UFES_TRAINING_REPO="${TRAINING_REPO}"
export NDB_UFES_ORGANIZER_DATA="${ORGANIZER_DATA}"
export NDB_UFES_MLFLOW_ENV="${MLFLOW_ENV}"
export NDB_UFES_OUTPUT_ROOT="${OUTPUT_ROOT}"
export NDB_UFES_QUEUE_LOG_DIR="${LOG_DIR}"

if [ "${DEVICE}" = "mps" ] && [ "${MPS_PREFLIGHT}" = "1" ]; then
  echo "Running MPS device preflight before starting the experiment queue."
  "${TRAINING_REPO}/.venv/bin/python" "${TRAINING_REPO}/scripts/check_mps.py"
  "${TRAINING_REPO}/.venv/bin/python" "${TRAINING_REPO}/scripts/check_models_mps.py" \
    --models "${REQUESTED_MODELS[@]}" --batch-size "${BATCH_SIZE}"
fi

export BATCH_SIZE EPOCHS OPTIMIZER LEARNING_RATE MOMENTUM \
  SCHEDULER_PATIENCE SCHEDULER_FACTOR SCHEDULER_MIN_LR \
  EARLY_STOPPING_PATIENCE RUN_SUFFIX SEED
nohup /bin/bash "${TRAINING_REPO}/scripts/run_pndb_ufes_queue.sh" \
  > "${LOG_DIR}/queue_stdout.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${LAUNCHER_PID_FILE}"
echo "Started P-NDB-UFES queue launcher with PID ${pid}."
echo "Models: ${MODELS}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} optimizer=${OPTIMIZER} learning_rate=${LEARNING_RATE}"
echo "momentum=${MOMENTUM} scheduler_patience=${SCHEDULER_PATIENCE} scheduler_factor=${SCHEDULER_FACTOR} min_lr=${SCHEDULER_MIN_LR}"
echo "early_stopping_patience=${EARLY_STOPPING_PATIENCE}; run suffix=${RUN_SUFFIX}; seed=${SEED}"
echo "Follow it with: tail -F ${LOG_DIR}/queue_stdout.log"
