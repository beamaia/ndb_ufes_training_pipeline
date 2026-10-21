#!/usr/bin/env bash
# Sets up and runs the training on a RunPod pod. The network volume (mounted at /workspace) holds this repository, the organizer's files, the
# patch images, the Python environment and the checkpoints, so a new pod finds all of it and only the first one downloads anything.
#   ./pod.sh --params configs/timing.yaml        everything after the script name goes to main.py
# When main.py ends, or fails, the pod stops itself so that it does not keep billing. KEEP_POD=1 leaves it running.
set -euo pipefail
cd "$(dirname "$0")"
VOLUME=$(cd .. && pwd)
ORGANIZER="$VOLUME/ndb_ufes_data_organizer"
export UV_LINK_MODE=copy  # the package cache stays on the pod's own disk (the volume is billed per GB); only the environment, .venv, is kept on the volume

[ -f .env ] || { echo "No .env here: it needs the MLflow address, user and password (MLFLOW_*) and HF_TOKEN. See the README." >&2; exit 1; }
for file in results/phase1/folds/fold_assignments.csv results/phase0/dataset_tables/patch_table_public.csv; do
    [ -f "$ORGANIZER/$file" ] || { echo "Missing $ORGANIZER/$file: send it from the laptop (see the README)." >&2; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v uv >/dev/null || pip install --quiet uv
uv sync

stop_pod() {
    if [ -n "${RUNPOD_POD_ID:-}" ] && ! mountpoint -q "$VOLUME"; then
        echo "Nothing is mounted at $VOLUME, so stopping the pod would erase everything on it: the pod is NOT stopped. Copy what you need, then stop it yourself." >&2
    elif [ -n "${RUNPOD_POD_ID:-}" ] && [ -z "${KEEP_POD:-}" ]; then
        echo "Stopping pod $RUNPOD_POD_ID"
        runpodctl stop pod "$RUNPOD_POD_ID" || echo "The pod did not stop: stop it in the RunPod console." >&2
    fi
}
trap stop_pod EXIT  # from here on, whatever happens, the pod stops

uv run python main.py --download-data "$@"
