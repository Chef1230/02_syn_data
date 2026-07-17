#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v1.yaml}}"

# A non-option first argument is a convenient config-path override.
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  schema
  --config "${CONFIG_PATH}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${NUM_SCHEMAS:-}" ]]; then
  ARGS+=(--count "${NUM_SCHEMAS}")
fi
if [[ -n "${BASE_SEED:-}" ]]; then
  ARGS+=(--seed "${BASE_SEED}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${SAMPLE_ID_PREFIX:-}" ]]; then
  ARGS+=(--sample-id-prefix "${SAMPLE_ID_PREFIX}")
fi
if [[ -n "${PROGRESS_EVERY:-}" ]]; then
  ARGS+=(--progress-every "${PROGRESS_EVERY}")
fi

case "${OVERWRITE:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--overwrite)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-overwrite)
    ;;
  "")
    ;;
  *)
    echo "OVERWRITE must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

if [[ "${VALIDATE_CONFIG_ONLY:-0}" == "1" ]]; then
  ARGS+=(--validate-config-only)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m rdb_prior.cli "${ARGS[@]}" "$@"
