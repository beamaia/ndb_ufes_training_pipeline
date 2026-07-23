#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${NDB_UFES_QUEUE_LOG_DIR:-${TRAINING_REPO}/logs/pndb_ufes_batches}"
PID_FILE="${LOG_DIR}/queue.pid"
LAUNCHER_PID_FILE="${LOG_DIR}/queue_launcher.pid"

pid=""
for pid_file in "${PID_FILE}" "${LAUNCHER_PID_FILE}"; do
  if [ -f "${pid_file}" ]; then
    candidate="$(tr -d '[:space:]' < "${pid_file}")"
    if [[ "${candidate}" =~ ^[0-9]+$ ]] && ps -p "${candidate}" >/dev/null 2>&1; then
      pid="${candidate}"
      break
    fi
    rm -f "${pid_file}"
  fi
done

if [ -z "${pid}" ]; then
  echo "No running P-NDB-UFES queue found."
  rm -f "${PID_FILE}" "${LAUNCHER_PID_FILE}"
  exit 0
fi

command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
if [ -z "${command_line}" ]; then
  echo "Recorded queue PID ${pid} is no longer running. Removing stale PID file."
  rm -f "${PID_FILE}"
  exit 0
fi

case "${command_line}" in
  *run_pndb_ufes_queue.sh*|*start_pndb_ufes_queue.sh*) ;;
  *)
    echo "Refusing to kill PID ${pid}: it is not the P-NDB-UFES queue." >&2
    echo "Current command: ${command_line}" >&2
    exit 1
    ;;
esac

descendants=()
collect_descendants() {
  local parent="$1"
  local child
  while read -r child; do
    [ -n "${child}" ] || continue
    collect_descendants "${child}"
    descendants+=("${child}")
  done < <(ps -axo pid=,ppid= | awk -v parent="${parent}" '$2 == parent {print $1}')
}

collect_descendants "${pid}"
echo "Stopping P-NDB-UFES queue ${pid} and ${#descendants[@]} child process(es)."
for child in "${descendants[@]}"; do
  kill -TERM "${child}" 2>/dev/null || true
done
kill -TERM "${pid}" 2>/dev/null || true
sleep 2
for child in "${descendants[@]}"; do
  kill -KILL "${child}" 2>/dev/null || true
done
kill -KILL "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}" "${LAUNCHER_PID_FILE}"
echo "P-NDB-UFES queue stopped."
