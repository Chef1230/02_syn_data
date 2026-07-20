#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}/../RDBPFN/model_pretrain"
exec "${PYTHON_BIN}" -m accelerate.commands.launch -m src.train --num_processes 1 --config-name RDBPFN_routed
