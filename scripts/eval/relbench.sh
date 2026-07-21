#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-convert}"

RELBENCH_DATASET="${RELBENCH_DATASET:-rel-amazon}"
RELBENCH_TASK="${RELBENCH_TASK:-user-churn}"
RELBENCH_OUTPUT="${RELBENCH_OUTPUT:-${PROJECT_ROOT}/outputs/relbench/${RELBENCH_DATASET}/${RELBENCH_TASK}}"
TASK_MANIFEST="${TASK_MANIFEST:-${RELBENCH_OUTPUT}/task/manifest.json}"
ROUTER_EVAL_OUTPUT="${ROUTER_EVAL_OUTPUT:-${RELBENCH_OUTPUT}/router_eval}"
RELBENCH_METADATA="${RELBENCH_METADATA:-${RELBENCH_OUTPUT}/relbench_metadata.json}"
RELBENCH_METRICS_OUTPUT="${RELBENCH_METRICS_OUTPUT:-${ROUTER_EVAL_OUTPUT}/relbench_metrics.json}"
ROUTED_H5_OUTPUT="${ROUTED_H5_OUTPUT:-${RELBENCH_OUTPUT}/routed.h5}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

bool_arg() {
  local value="$1"
  local positive="$2"
  local negative="$3"
  case "${value}" in
    1|true|TRUE|yes|YES) printf '%s\n' "${positive}" ;;
    0|false|FALSE|no|NO) printf '%s\n' "${negative}" ;;
    *) echo "Expected true/false, got: ${value}" >&2; exit 2 ;;
  esac
}

require_checkpoint() {
  if [[ -z "${ROUTER_CHECKPOINT:-}" ]]; then
    echo "Set ROUTER_CHECKPOINT to router/checkpoints/best.pt or last.pt." >&2
    exit 2
  fi
  if [[ ! -f "${ROUTER_CHECKPOINT}" ]]; then
    echo "Router checkpoint does not exist: ${ROUTER_CHECKPOINT}" >&2
    exit 2
  fi
}

run_convert() {
  local args=(
    relbench-import
    --dataset "${RELBENCH_DATASET}"
    --task "${RELBENCH_TASK}"
    --output-dir "${RELBENCH_OUTPUT}"
    --seed "${SEED:-0}"
    --max-rows-per-task "${MAX_ROWS_PER_TASK:-600}"
    --query-rows-per-task "${QUERY_ROWS_PER_TASK:-256}"
    --max-classes "${MAX_CLASSES:-16}"
    --max-text-length "${MAX_TEXT_LENGTH:-256}"
    "$(bool_arg "${DOWNLOAD:-1}" --download --no-download)"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
    --progress
  )
  [[ -n "${SUPPORT_ROWS:-}" ]] && args+=(--support-rows "${SUPPORT_ROWS}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

run_eval() {
  require_checkpoint
  if [[ ! -f "${TASK_MANIFEST}" ]]; then
    echo "RelBench task manifest does not exist: ${TASK_MANIFEST}" >&2
    exit 2
  fi
  local args=(
    router-eval
    --task-manifest "${TASK_MANIFEST}"
    --checkpoint "${ROUTER_CHECKPOINT}"
    --output-dir "${ROUTER_EVAL_OUTPUT}"
    --device "${DEVICE:-auto}"
    --mixed-precision "${MIXED_PRECISION:-none}"
    --artifact-cache-size "${ARTIFACT_CACHE_SIZE:-4}"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
    --progress
  )
  [[ -n "${NUM_TASKS:-}" ]] && args+=(--count "${NUM_TASKS}")
  [[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
  if [[ -n "${NUM_TASKS:-}" || ( -n "${START_INDEX:-}" && "${START_INDEX}" != "0" ) ]]; then
    echo "Skipping official RelBench metrics for a partial task selection."
  else
    run_score
  fi
}

run_score() {
  if [[ ! -f "${RELBENCH_METADATA}" ]]; then
    echo "RelBench metadata does not exist: ${RELBENCH_METADATA}" >&2
    exit 2
  fi
  if [[ ! -f "${ROUTER_EVAL_OUTPUT}/predictions.jsonl" ]]; then
    echo "Router predictions do not exist: ${ROUTER_EVAL_OUTPUT}/predictions.jsonl" >&2
    exit 2
  fi
  local args=(
    relbench-score
    --metadata "${RELBENCH_METADATA}"
    --predictions "${ROUTER_EVAL_OUTPUT}/predictions.jsonl"
    --output "${RELBENCH_METRICS_OUTPUT}"
    "$(bool_arg "${SCORE_DOWNLOAD:-0}" --download --no-download)"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
  )
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

run_h5() {
  require_checkpoint
  if [[ ! -f "${TASK_MANIFEST}" ]]; then
    echo "RelBench task manifest does not exist: ${TASK_MANIFEST}" >&2
    exit 2
  fi
  local args=(
    routed-h5
    --config "${CONFIG_PATH}"
    --task-manifest "${TASK_MANIFEST}"
    --checkpoint "${ROUTER_CHECKPOINT}"
    --output "${ROUTED_H5_OUTPUT}"
    --device "${DEVICE:-auto}"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
  )
  [[ -n "${NUM_TASKS:-}" ]] && args+=(--count "${NUM_TASKS}")
  [[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

cd "${PROJECT_ROOT}"
case "${ACTION}" in
  convert) run_convert ;;
  eval) run_eval ;;
  score) run_score ;;
  h5) run_h5 ;;
  all)
    run_convert
    run_eval
    run_h5
    ;;
  *)
    echo "Usage: bash scripts/eval/relbench.sh [convert|eval|score|h5|all]" >&2
    exit 2
    ;;
esac
