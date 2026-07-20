"""Schema-split training loop for the sparse relational PFN."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch

from .checkpoint import save_router_checkpoint
from .config import RouterTrainingConfig
from .data import RawRoutingTask, RoutingTaskTensorizer, load_routing_tasks
from .losses import sparse_router_loss
from .network import SparseRelationalPFN


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterTrainingResult:
    output_root: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_path: Path
    train_task_count: int
    validation_task_count: int
    best_validation_loss: float


def train_sparse_router(
    config: RouterTrainingConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> RouterTrainingResult:
    _set_seed(config.seed)
    device = resolve_device(config.device)
    raw_tasks = load_routing_tasks(
        config.task_manifest,
        start_index=config.start_index,
        task_count=config.task_count,
    )
    if not raw_tasks:
        raise ValueError("no routing tasks selected")
    train_tasks, validation_tasks = _split_by_schema(
        raw_tasks,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model = SparseRelationalPFN(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tensorizer = RoutingTaskTensorizer(config.model)
    output_root = config.output_root
    checkpoints = output_root / "checkpoints"
    best_path = checkpoints / "best.pt"
    last_path = checkpoints / "last.pt"
    if output_root.exists() and best_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"router output already exists: {output_root}; use overwrite=True"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    total_steps = config.epochs * len(train_tasks)
    completed_steps = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = list(train_tasks)
        random.Random(config.seed + epoch).shuffle(order)
        train_metrics = _run_epoch(
            model=model,
            tasks=tuple(order),
            tensorizer=tensorizer,
            config=config,
            device=device,
            optimizer=optimizer,
            progress=(
                None
                if progress is None
                else lambda done, _total, task_id: progress(
                    completed_steps + done, total_steps, task_id
                )
            ),
        )
        completed_steps += len(order)
        model.eval()
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model=model,
                tasks=validation_tasks or train_tasks[:1],
                tensorizer=tensorizer,
                config=config,
                device=device,
                optimizer=None,
                progress=None,
            )
        validation_loss = validation_metrics["loss"]
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        save_router_checkpoint(
            last_path,
            model=model,
            epoch=epoch,
            validation_loss=validation_loss,
            metrics=epoch_record,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_router_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                validation_loss=validation_loss,
                metrics=epoch_record,
            )
        _LOGGER.info(
            "router epoch=%d train=%.6f validation=%.6f",
            epoch,
            train_metrics["loss"],
            validation_loss,
        )

    metrics_path = output_root / "metrics.json"
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "format": "rdb-prior-sparse-router-metrics-v1",
                "configuration": config.to_dict(),
                "device": str(device),
                "train_task_count": len(train_tasks),
                "validation_task_count": len(validation_tasks),
                "best_validation_loss": best_validation,
                "history": history,
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(metrics_path)
    return RouterTrainingResult(
        output_root=output_root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        metrics_path=metrics_path,
        train_task_count=len(train_tasks),
        validation_task_count=len(validation_tasks),
        best_validation_loss=best_validation,
    )


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    *,
    model: SparseRelationalPFN,
    tasks: tuple[RawRoutingTask, ...],
    tensorizer: RoutingTaskTensorizer,
    config: RouterTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    progress: Callable[[int, int, str], None] | None,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "query_prediction": 0.0,
        "route": 0.0,
        "cost": 0.0,
        "sparse": 0.0,
        "diversity": 0.0,
    }
    for completed, raw in enumerate(tasks, start=1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        descriptors = tensorizer.tensorize_descriptors(raw).to(device)
        selection = model.select(descriptors)
        batch = tensorizer.materialize_selected(
            raw,
            selected_path_mask=selection.route_selection.hard_mask,
            selected_column_mask=selection.column_hard_mask,
        ).to(device)
        output = model(batch, selection)
        losses = sparse_router_loss(output, batch, config)
        if optimizer is not None:
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
        for name, value in losses.detached_metrics().items():
            totals[name] += value
        if progress is not None:
            progress(completed, len(tasks), batch.task_id)
    denominator = max(1, len(tasks))
    return {name: value / denominator for name, value in totals.items()}


def _split_by_schema(
    tasks: tuple[RawRoutingTask, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[RawRoutingTask, ...], tuple[RawRoutingTask, ...]]:
    if validation_fraction <= 0 or len(tasks) < 2:
        return tasks, ()
    train: list[RawRoutingTask] = []
    validation: list[RawRoutingTask] = []
    threshold = round(validation_fraction * 10_000)
    for task in tasks:
        schema_id = task.task_artifact.task.plan.schema_id
        digest = hashlib.sha256(f"{seed}:{schema_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (validation if bucket < threshold else train).append(task)
    if not validation:
        schema_id = tasks[-1].task_artifact.task.plan.schema_id
        validation_schema_ids = {schema_id}
        validation = [
            task
            for task in tasks
            if task.task_artifact.task.plan.schema_id == schema_id
        ]
        train = [
            task
            for task in tasks
            if task.task_artifact.task.plan.schema_id
            not in validation_schema_ids
        ]
    if not train:
        return tasks, ()
    return tuple(train), tuple(validation)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "RouterTrainingResult",
    "train_sparse_router",
    "resolve_device",
]
