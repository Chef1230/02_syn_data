"""Leakage-safe evaluation of a router checkpoint on relational tasks."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from rdb_prior.routing.checkpoint import load_router_checkpoint
from rdb_prior.routing.config import RouterEvaluationConfig
from rdb_prior.routing.data import RoutingTaskStore, RoutingTaskTensorizer
from rdb_prior.routing.trainer import resolve_device
from rdb_prior.task.model import PredictionType


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterEvaluationResult:
    output_root: Path
    metrics_path: Path
    predictions_path: Path
    task_count: int
    query_row_count: int


def evaluate_router_checkpoint(
    config: RouterEvaluationConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> RouterEvaluationResult:
    """Evaluate query rows without ever exposing their labels to the model."""
    metrics_path = config.output_root / "metrics.json"
    predictions_path = config.output_root / "predictions.jsonl"
    if not config.overwrite:
        existing = [path for path in (metrics_path, predictions_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"router evaluation output already exists: {existing[0]}; "
                "use overwrite=True"
            )
    config.output_root.mkdir(parents=True, exist_ok=True)
    metrics_temporary = metrics_path.with_suffix(".json.tmp")
    predictions_temporary = predictions_path.with_suffix(".jsonl.tmp")

    device = resolve_device(config.device)
    precision = _resolve_precision(config.mixed_precision, device)
    model, checkpoint = load_router_checkpoint(config.checkpoint, device=device)
    model.eval()
    tensorizer = RoutingTaskTensorizer(model.config)
    store = RoutingTaskStore(
        config.task_manifest,
        start_index=config.start_index,
        task_count=config.task_count,
        cache_size=config.artifact_cache_size,
    )
    references = store.references
    if not references:
        raise ValueError("no benchmark tasks selected")

    per_task: list[dict[str, object]] = []
    classification = _ClassificationAggregate()
    regression = _RegressionAggregate()
    query_row_count = 0
    try:
        with predictions_temporary.open("w", encoding="utf-8") as predictions:
            with torch.inference_mode():
                for completed, reference in enumerate(references, start=1):
                    raw = store.load(reference)
                    descriptors = tensorizer.tensorize_descriptors(raw).to(device)
                    with _autocast(device, precision):
                        selection = model.select(descriptors)
                    materialized = tensorizer.materialize_selected(
                        raw,
                        selected_path_mask=(
                            selection.route_selection.hard_mask.detach().cpu()
                        ),
                        selected_column_mask=(
                            selection.column_hard_mask.detach().cpu()
                        ),
                    ).to(device)
                    with _autocast(device, precision):
                        output = model(materialized, selection)

                    query = materialized.query_mask
                    query_count = int(query.sum().item())
                    support_count = int(materialized.support_mask.sum().item())
                    if support_count < 1 or query_count < 1:
                        raise ValueError(
                            f"task {materialized.task_id} must contain support and query rows"
                        )
                    query_row_count += query_count
                    route = _route_summary(materialized, output)
                    if materialized.prediction_type is PredictionType.CLASSIFICATION:
                        summary, rows = _classification_result(
                            materialized,
                            output.classification_logits,
                            query,
                        )
                        classification.add(summary)
                    else:
                        summary, rows = _regression_result(
                            materialized,
                            output.regression_prediction,
                            query,
                        )
                        regression.add(summary)
                    summary.update(route)
                    per_task.append(summary)
                    for row in rows:
                        predictions.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                    if progress is not None:
                        progress(completed, len(references), materialized.task_id)

        payload = {
            "format": "rdb-prior-router-benchmark-eval-v1",
            "configuration": config.to_dict(),
            "checkpoint": {
                "format": checkpoint.get("format"),
                "epoch": checkpoint.get("epoch"),
                "validation_loss": checkpoint.get("validation_loss"),
            },
            "task_count": len(references),
            "query_row_count": query_row_count,
            "classification": classification.result(),
            "regression": regression.result(),
            "per_task": per_task,
        }
        metrics_temporary.write_text(
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
        predictions_temporary.replace(predictions_path)
        metrics_temporary.replace(metrics_path)
    except Exception:
        for temporary in (metrics_temporary, predictions_temporary):
            if temporary.exists():
                temporary.unlink()
        raise

    return RouterEvaluationResult(
        output_root=config.output_root,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        task_count=len(references),
        query_row_count=query_row_count,
    )


def _classification_result(task, logits, query):
    num_classes = int(task.num_classes)
    if num_classes < 2:
        raise ValueError(f"classification task {task.task_id} has fewer than two classes")
    if num_classes > logits.shape[-1]:
        raise ValueError(
            f"task {task.task_id} has {num_classes} classes but checkpoint supports "
            f"only {logits.shape[-1]}"
        )
    probabilities = torch.softmax(logits[query, :num_classes].float(), dim=-1)
    y_true = task.labels[query].long()
    if bool(((y_true < 0) | (y_true >= num_classes)).any()):
        raise ValueError(f"task {task.task_id} contains invalid encoded query labels")
    y_pred = probabilities.argmax(dim=-1)
    y_np = y_true.cpu().numpy()
    pred_np = y_pred.cpu().numpy()
    prob_np = probabilities.cpu().numpy()
    log_loss = float(
        -np.log(np.clip(prob_np[np.arange(len(y_np)), y_np], 1e-12, 1.0)).mean()
    )
    accuracy = float(np.mean(pred_np == y_np))
    balanced = _balanced_accuracy(y_np, pred_np)
    roc_auc = _multiclass_roc_auc(y_np, prob_np)
    class_values = tuple(task.class_values)
    row_ids = task.row_ids[query].cpu().numpy()
    rows = []
    for index in range(len(y_np)):
        rows.append(
            {
                "task_id": task.task_id,
                "row_id": int(row_ids[index]),
                "prediction_type": "classification",
                "y_true_encoded": int(y_np[index]),
                "y_pred_encoded": int(pred_np[index]),
                "y_true": _class_value(class_values, int(y_np[index])),
                "y_pred": _class_value(class_values, int(pred_np[index])),
                "probabilities": [float(value) for value in prob_np[index]],
            }
        )
    return (
        {
            "task_id": task.task_id,
            "prediction_type": "classification",
            "support_rows": int(task.support_mask.sum().item()),
            "query_rows": len(y_np),
            "num_classes": num_classes,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "log_loss": log_loss,
            "roc_auc_ovr": roc_auc,
            "_correct": int(np.sum(pred_np == y_np)),
        },
        rows,
    )


def _regression_result(task, prediction, query):
    scale = float(task.label_scale)
    center = float(task.label_center)
    y_true = task.labels[query].float().cpu().numpy() * scale + center
    y_pred = prediction[query].float().cpu().numpy() * scale + center
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = None if denominator <= 0 else float(1.0 - np.sum(np.square(residual)) / denominator)
    row_ids = task.row_ids[query].cpu().numpy()
    rows = [
        {
            "task_id": task.task_id,
            "row_id": int(row_ids[index]),
            "prediction_type": "regression",
            "y_true": float(y_true[index]),
            "y_pred": float(y_pred[index]),
        }
        for index in range(len(y_true))
    ]
    return (
        {
            "task_id": task.task_id,
            "prediction_type": "regression",
            "support_rows": int(task.support_mask.sum().item()),
            "query_rows": len(y_true),
            "mae": mae,
            "rmse": math.sqrt(mse),
            "r2": r2,
            "_absolute_error": float(np.sum(np.abs(residual))),
            "_squared_error": float(np.sum(np.square(residual))),
            "_sum_y": float(np.sum(y_true)),
            "_sum_y_squared": float(np.sum(np.square(y_true))),
        },
        rows,
    )


def _route_summary(task, output) -> dict[str, object]:
    path_mask = output.route_selection.hard_mask.detach().cpu().bool()
    column_mask = output.column_hard_mask.detach().cpu().bool()
    selected_indices = torch.nonzero(path_mask, as_tuple=False).flatten().tolist()
    selected_paths = [task.path_signatures[index] for index in selected_indices]
    selected_columns = {
        task.path_signatures[path_index]: [
            task.source_column_ids[path_index][column_index]
            for column_index in torch.nonzero(
                column_mask[path_index], as_tuple=False
            ).flatten().tolist()
            if column_index < len(task.source_column_ids[path_index])
        ]
        for path_index in selected_indices
    }
    return {
        "selected_path_count": len(selected_paths),
        "selected_paths": selected_paths,
        "selected_columns": selected_columns,
    }


class _ClassificationAggregate:
    def __init__(self) -> None:
        self.tasks = 0
        self.rows = 0
        self.correct = 0
        self.weighted_log_loss = 0.0
        self.balanced: list[float] = []
        self.auc: list[float] = []

    def add(self, summary: dict[str, object]) -> None:
        rows = int(summary["query_rows"])
        self.tasks += 1
        self.rows += rows
        self.correct += int(summary.pop("_correct"))
        self.weighted_log_loss += float(summary["log_loss"]) * rows
        self.balanced.append(float(summary["balanced_accuracy"]))
        if summary["roc_auc_ovr"] is not None:
            self.auc.append(float(summary["roc_auc_ovr"]))

    def result(self) -> dict[str, object]:
        return {
            "task_count": self.tasks,
            "query_row_count": self.rows,
            "accuracy": None if not self.rows else self.correct / self.rows,
            "log_loss": None if not self.rows else self.weighted_log_loss / self.rows,
            "macro_balanced_accuracy": _mean_or_none(self.balanced),
            "macro_roc_auc_ovr": _mean_or_none(self.auc),
        }


class _RegressionAggregate:
    def __init__(self) -> None:
        self.tasks = 0
        self.rows = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.sum_y = 0.0
        self.sum_y_squared = 0.0
        self.r2: list[float] = []

    def add(self, summary: dict[str, object]) -> None:
        self.tasks += 1
        self.rows += int(summary["query_rows"])
        self.absolute_error += float(summary.pop("_absolute_error"))
        self.squared_error += float(summary.pop("_squared_error"))
        self.sum_y += float(summary.pop("_sum_y"))
        self.sum_y_squared += float(summary.pop("_sum_y_squared"))
        if summary["r2"] is not None:
            self.r2.append(float(summary["r2"]))

    def result(self) -> dict[str, object]:
        denominator = (
            self.sum_y_squared - self.sum_y * self.sum_y / self.rows
            if self.rows
            else 0.0
        )
        return {
            "task_count": self.tasks,
            "query_row_count": self.rows,
            "mae": None if not self.rows else self.absolute_error / self.rows,
            "rmse": None if not self.rows else math.sqrt(self.squared_error / self.rows),
            "r2": None if denominator <= 0 else 1.0 - self.squared_error / denominator,
            "macro_r2": _mean_or_none(self.r2),
        }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for value in np.unique(y_true):
        selected = y_true == value
        recalls.append(float(np.mean(y_pred[selected] == value)))
    return float(np.mean(recalls))


def _multiclass_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    aucs = []
    for class_index in range(probabilities.shape[1]):
        binary = (y_true == class_index).astype(np.int8)
        auc = _binary_roc_auc(binary, probabilities[:, class_index])
        if auc is not None:
            aucs.append(auc)
    return _mean_or_none(aucs)


def _binary_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _class_value(values: tuple, index: int):
    value = values[index] if index < len(values) else index
    return value.item() if isinstance(value, np.generic) else value


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _resolve_precision(value: str, device: torch.device) -> str:
    if value == "none":
        return value
    if device.type != "cuda":
        return "none"
    if value == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed_precision=bf16 requires a BF16-capable CUDA GPU")
    return value


def _autocast(device: torch.device, precision: str):
    if precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


__all__ = ["RouterEvaluationResult", "evaluate_router_checkpoint"]
