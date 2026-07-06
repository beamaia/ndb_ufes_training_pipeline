#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export DVC_SITE_CACHE_DIR="${DVC_SITE_CACHE_DIR:-.dvc/tmp/site_cache}"

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

if [[ "${1:-}" == "--check-env" ]]; then
  echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
  echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
  echo "DVC_SITE_CACHE_DIR=${DVC_SITE_CACHE_DIR}"
  uv run python -c "import sys; print(sys.executable)"
  exit 0
fi

uv run python -m dvc "$@"
