"""Reassemble chunked router predictions and run official RelBench metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class RelBenchScoreConfig:
    metadata_path: Path
    predictions_path: Path
    output_path: Path
    download: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("metadata_path", "predictions_path", "output_path"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path")


@dataclass(frozen=True, slots=True, kw_only=True)
class RelBenchScoreResult:
    output_path: Path
    prediction_count: int
    metrics: Mapping[str, float]


def score_relbench_predictions(
    config: RelBenchScoreConfig,
    *,
    task: Any | None = None,
) -> RelBenchScoreResult:
    """Score a complete prediction JSONL in original RelBench test order."""
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"RelBench metric output already exists: {config.output_path}; "
            "use overwrite=True"
        )
    metadata = _load_json(config.metadata_path)
    if metadata.get("format") != "rdb-prior-relbench-import-v1":
        raise ValueError("input metadata is not a RelBench import artifact")
    if task is None:
        try:
            from relbench.tasks import get_task
        except ImportError as error:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "RelBench is required for official scoring; install with "
                "`pip install -e '.[relbench]'`."
            ) from error
        task = get_task(
            metadata["dataset"],
            metadata["task"],
            download=config.download,
        )

    mapping = metadata.get("prediction_mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("RelBench metadata has no prediction mapping")
    test_start = int(mapping["test_anchor_start"])
    test_count = int(mapping["test_row_count"])
    rows: list[dict[str, Any] | None] = [None] * test_count
    with config.predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"prediction line {line_number} must be a JSON object"
                )
            position = int(row["row_id"]) - test_start
            if position < 0 or position >= test_count:
                raise ValueError(
                    f"prediction row_id {row['row_id']} is outside RelBench test rows"
                )
            if rows[position] is not None:
                raise ValueError(
                    f"duplicate prediction for RelBench test position {position}"
                )
            rows[position] = row
    missing = [index for index, row in enumerate(rows) if row is None]
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        raise ValueError(
            f"predictions are incomplete: {len(missing)} RelBench test rows "
            f"are missing (first positions: {preview})"
        )

    complete_rows = [row for row in rows if row is not None]
    prediction = _prediction_array(metadata, complete_rows)
    raw_metrics = task.evaluate(prediction)
    metrics = {
        str(name): float(value) for name, value in raw_metrics.items()
    }
    payload = {
        "format": "rdb-prior-relbench-official-metrics-v1",
        "dataset": metadata["dataset"],
        "task": metadata["task"],
        "relbench_task_type": metadata["relbench_task_type"],
        "prediction_count": test_count,
        "metrics": metrics,
        "metadata": str(config.metadata_path),
        "predictions": str(config.predictions_path),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(config.output_path)
    return RelBenchScoreResult(
        output_path=config.output_path,
        prediction_count=test_count,
        metrics=metrics,
    )


def _prediction_array(
    metadata: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> np.ndarray:
    task_type = metadata["relbench_task_type"]
    if task_type == "regression":
        return np.asarray([row["y_pred"] for row in rows], dtype=np.float64)
    class_values = metadata.get("class_values")
    if not isinstance(class_values, list) or len(class_values) < 2:
        raise ValueError("classification metadata has fewer than two classes")
    probabilities = np.asarray(
        [row["probabilities"] for row in rows], dtype=np.float64
    )
    if probabilities.shape != (len(rows), len(class_values)):
        raise ValueError(
            "prediction probability width does not match imported class values"
        )
    if task_type == "binary_classification":
        positive = next(
            (
                index
                for index, value in enumerate(class_values)
                if value is True or value == 1
            ),
            None,
        )
        if positive is None:
            raise ValueError(
                "binary RelBench labels must include the positive value 1"
            )
        return probabilities[:, positive]
    if task_type == "multiclass_classification":
        class_indices: list[int] = []
        for value in class_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "multiclass RelBench labels must be non-negative integers"
                )
            index = int(value)
            if index < 0 or float(index) != float(value):
                raise ValueError(
                    "multiclass RelBench labels must be non-negative integers"
                )
            class_indices.append(index)
        result = np.zeros(
            (len(rows), max(class_indices) + 1), dtype=np.float64
        )
        result[:, class_indices] = probabilities
        return result
    raise ValueError(f"unsupported RelBench task type {task_type!r}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


__all__ = [
    "RelBenchScoreConfig",
    "RelBenchScoreResult",
    "score_relbench_predictions",
]
