#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage: ./train.sh [--pull|--no-pull] [--check-env] [--check-data]

Options:
  --pull       Pull DVC data before training. This is the default.
  --no-pull    Skip DVC pull and use the current local data checkout.
  --check-env  Print the resolved uv/DVC/MLflow environment and exit.
  --check-data Validate params.yaml data paths and split counts, then exit.
  -h, --help   Show this help.
EOF
}

PULL_DATA=1
CHECK_ENV=0
CHECK_DATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)
      PULL_DATA=1
      shift
      ;;
    --no-pull)
      PULL_DATA=0
      shift
      ;;
    --check-env)
      CHECK_ENV=1
      shift
      ;;
    --check-data)
      CHECK_DATA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export DVC_SITE_CACHE_DIR="${DVC_SITE_CACHE_DIR:-.dvc/tmp/site_cache}"
MLFLOW_ENV_FILE="${MLFLOW_ENV_FILE:-${SCRIPT_DIR}/../ndb_ufes_mlflow/.env}"

if [[ -f "${MLFLOW_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${MLFLOW_ENV_FILE}"
  set +a
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:8000/}"

PROJECT_VENV="$(pwd)/.venv"
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${PROJECT_VENV}" ]]; then
  OLD_VENV_BIN="${VIRTUAL_ENV}/bin"
  CLEAN_PATH=""
  IFS=":" read -r -a PATH_PARTS <<< "${PATH}"
  for path_part in "${PATH_PARTS[@]}"; do
    if [[ "${path_part}" != "${OLD_VENV_BIN}" ]]; then
      if [[ -z "${CLEAN_PATH}" ]]; then
        CLEAN_PATH="${path_part}"
      else
        CLEAN_PATH="${CLEAN_PATH}:${path_part}"
      fi
    fi
  done
  PATH="${CLEAN_PATH}"
  unset VIRTUAL_ENV
fi

if [[ "${CHECK_ENV}" == "1" ]]; then
  echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
  echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
  echo "DVC_SITE_CACHE_DIR=${DVC_SITE_CACHE_DIR}"
  echo "MLFLOW_ENV_FILE=${MLFLOW_ENV_FILE}"
  echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
  echo "MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-}"
  uv run python -c "import sys; print(sys.executable)"
  exit 0
fi

if [[ "${PULL_DATA}" == "1" ]]; then
  uv run python -m dvc pull
fi

uv run python - <<'PY'
import yaml
from pathlib import Path

from src.validate_input.params import Parameters
from src.dataset import LeakageSafeNDBUfesOrganizer

params = Parameters.model_validate(yaml.safe_load(Path("params.yaml").read_text()))
organizer = LeakageSafeNDBUfesOrganizer(params)
(train_paths, _, train_meta), (val_paths, _, val_meta) = organizer.data_per_fold(params.dataset.cv_folds[0], train=True)
test_paths, _, test_meta = organizer.data_per_fold(params.dataset.test_fold, train=False)
missing = [str(path) for path in list(train_paths[:10]) + list(val_paths[:10]) + list(test_paths[:10]) if not path.exists()]
if missing:
    raise FileNotFoundError(f"Configured data paths are not available. First missing examples: {missing[:5]}")
print("Data check OK")
print(f"Fold CSV: {params.dataset.fold_assignments_path}")
print(f"Patch root: {Path(params.dataset.root) / params.dataset.patch}")
print(f"CV folds: {params.dataset.cv_folds}; held-out test fold: {params.dataset.test_fold}")
print(f"First split sizes: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
print(f"Train folds in first split: {sorted({row['fold'] for row in train_meta})}")
print(f"Validation folds in first split: {sorted({row['fold'] for row in val_meta})}")
print(f"Test folds: {sorted({row['fold'] for row in test_meta})}")
PY

if [[ "${CHECK_DATA}" == "1" ]]; then
  exit 0
fi

uv run python main.py
