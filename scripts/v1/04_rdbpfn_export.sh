#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v1.yaml}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  rdbpfn-export
  --config "${CONFIG_PATH}"
)

if [[ -n "${TASK_MANIFEST:-}" ]]; then
  ARGS+=(--task-manifest "${TASK_MANIFEST}")
fi
if [[ -n "${RDBPFN_OUTPUT_DIR:-${OUTPUT_DIR:-}}" ]]; then
  ARGS+=(--output-dir "${RDBPFN_OUTPUT_DIR:-${OUTPUT_DIR}}")
fi
if [[ -n "${NUM_EXPORTS:-}" ]]; then
  ARGS+=(--count "${NUM_EXPORTS}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${SHARD_ID:-}" ]]; then
  ARGS+=(--shard-id "${SHARD_ID}")
fi
if [[ -n "${NUM_SHARDS:-}" ]]; then
  ARGS+=(--num-shards "${NUM_SHARDS}")
fi
if [[ -n "${VALIDATION_FRACTION:-}" ]]; then
  ARGS+=(--validation-fraction "${VALIDATION_FRACTION}")
fi
if [[ -n "${MIN_VALIDATION_ROWS:-}" ]]; then
  ARGS+=(--min-validation-rows "${MIN_VALIDATION_ROWS}")
fi
if [[ -n "${PROGRESS_EVERY:-}" ]]; then
  ARGS+=(--progress-every "${PROGRESS_EVERY}")
fi
if [[ -n "${LOG_LEVEL:-}" ]]; then
  ARGS+=(--log-level "${LOG_LEVEL}")
fi
if [[ -n "${LOG_FILE:-}" ]]; then
  ARGS+=(--log-file "${LOG_FILE}")
fi
if [[ -n "${PROGRESS_WIDTH:-}" ]]; then
  ARGS+=(--progress-width "${PROGRESS_WIDTH}")
fi

case "${COMPRESS:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--compress)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-compress)
    ;;
  "")
    ;;
  *)
    echo "COMPRESS must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

case "${PROGRESS_BAR:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--progress)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-progress)
    ;;
  "")
    ;;
  *)
    echo "PROGRESS_BAR must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

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
