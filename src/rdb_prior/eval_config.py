"""Load shell-facing evaluation settings from a strict YAML file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import yaml


_FIELDS: dict[tuple[str, str], tuple[str, str]] = {
    ("relbench", "dataset"): ("RELBENCH_DATASET", "string"),
    ("relbench", "task"): ("RELBENCH_TASK", "string"),
    ("relbench", "tasks"): ("RELBENCH_TASKS", "list"),
    ("relbench", "cache_dir"): ("RELBENCH_CACHE_DIR", "path"),
    ("relbench", "output"): ("RELBENCH_OUTPUT", "path"),
    ("relbench", "output_root"): ("RELBENCH_OUTPUT_ROOT", "path"),
    ("relbench", "metadata"): ("RELBENCH_METADATA", "path"),
    ("relbench", "download"): ("DOWNLOAD", "boolean"),
    ("relbench", "score_download"): ("SCORE_DOWNLOAD", "boolean"),
    ("relbench", "reuse_converted"): ("REUSE_CONVERTED", "boolean"),
    ("relbench", "seed"): ("SEED", "integer"),
    ("relbench", "max_rows_per_task"): ("MAX_ROWS_PER_TASK", "integer"),
    ("relbench", "query_rows_per_task"): (
        "QUERY_ROWS_PER_TASK",
        "integer",
    ),
    ("relbench", "support_rows"): ("SUPPORT_ROWS", "integer"),
    ("relbench", "max_classes"): ("MAX_CLASSES", "integer"),
    ("relbench", "max_text_length"): ("MAX_TEXT_LENGTH", "integer"),
    ("router", "checkpoint"): ("ROUTER_CHECKPOINT", "path"),
    ("router", "config"): ("CONFIG_PATH", "path"),
    ("router", "eval_output"): ("ROUTER_EVAL_OUTPUT", "path"),
    ("router", "artifact_cache_size"): (
        "ARTIFACT_CACHE_SIZE",
        "integer",
    ),
    ("h5", "output"): ("ROUTED_H5_OUTPUT", "path"),
    ("tfm", "rdbpfn_root"): ("RDBPFN_ROOT", "path"),
    ("tfm", "checkpoint"): ("TFM_CHECKPOINT", "path"),
    ("tfm", "model_config"): ("TFM_MODEL_CONFIG", "path"),
    ("tfm", "predictions_output"): ("TFM_PREDICTIONS_OUTPUT", "path"),
    ("tfm", "metrics_output"): ("TFM_METRICS_OUTPUT", "path"),
    ("runtime", "device"): ("DEVICE", "string"),
    ("runtime", "cuda_visible_devices"): ("CUDA_VISIBLE_DEVICES", "string"),
    ("runtime", "mixed_precision"): ("MIXED_PRECISION", "string"),
    ("runtime", "overwrite"): ("OVERWRITE", "boolean"),
    ("runtime", "num_tasks"): ("NUM_TASKS", "integer"),
    ("runtime", "start_index"): ("START_INDEX", "integer"),
    ("runtime", "progress_every"): ("PROGRESS_EVERY", "integer"),
    ("runtime", "progress_width"): ("PROGRESS_WIDTH", "integer"),
    ("runtime", "log_level"): ("LOG_LEVEL", "string"),
    ("runtime", "log_file"): ("LOG_FILE", "path"),
}


def load_eval_environment(path: Path, *, project_root: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("evaluation config root must be a mapping")

    environment: dict[str, str] = {}
    for section, raw_section in payload.items():
        if not isinstance(section, str) or not isinstance(raw_section, dict):
            raise ValueError(f"evaluation config section {section!r} must be a mapping")
        for name, value in raw_section.items():
            key = (section, name)
            if key not in _FIELDS:
                raise ValueError(f"unknown evaluation config key: {section}.{name}")
            if value is None:
                continue
            variable, kind = _FIELDS[key]
            environment[variable] = _convert_value(
                value,
                kind=kind,
                field=f"{section}.{name}",
                project_root=project_root,
            )
    return environment


def _convert_value(
    value: Any,
    *,
    kind: str,
    field: str,
    project_root: Path,
) -> str:
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be true or false")
        return "1" if value else "0"
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
        return str(value)
    if kind == "list":
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            items = [item.strip() for item in value]
        else:
            raise ValueError(f"{field} must be a string list")
        if not items or any(not item for item in items):
            raise ValueError(f"{field} cannot contain empty task names")
        return ",".join(items)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    text = value.strip()
    if kind == "path":
        expanded = Path(os.path.expandvars(text)).expanduser()
        if not expanded.is_absolute():
            expanded = project_root / expanded
        return str(expanded.resolve())
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("project_root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    environment = load_eval_environment(
        args.config.resolve(), project_root=args.project_root.resolve()
    )
    output = sys.stdout.buffer
    for key, value in environment.items():
        output.write(key.encode("utf-8") + b"\0")
        output.write(value.encode("utf-8") + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_eval_environment"]
