"""Schema-split training loop for the sparse relational PFN."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Callable, Iterator

import numpy as np
import torch

from .checkpoint import save_router_checkpoint
from .config import RouterTrainingConfig
from .data import (
    RawRoutingTask,
    RoutingTaskReference,
    RoutingTaskStore,
    RoutingTaskTensorizer,
    collate_routed_tasks,
)
from .losses import sparse_router_batch_loss
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
    store = RoutingTaskStore(
        config.task_manifest,
        start_index=config.start_index,
        task_count=config.task_count,
        cache_size=config.artifact_cache_size,
    )
    references = store.references
    if not references:
        raise ValueError("no routing tasks selected")
    train_tasks, validation_tasks = _split_by_schema(
        references,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    precision = _resolve_mixed_precision(config.mixed_precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    model = SparseRelationalPFN(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tensorizer = RoutingTaskTensorizer(config.model)
    scaler = _make_grad_scaler(enabled=precision == "fp16")
    _LOGGER.info(
        "router runtime device=%s tasks=%d batch=%d workers=%d prefetch=%d precision=%s cache=%d",
        device,
        len(references),
        config.batch_size,
        config.num_workers,
        config.prefetch_factor,
        precision,
        config.artifact_cache_size,
    )
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
            store=store,
            tensorizer=tensorizer,
            config=config,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            precision=precision,
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
                store=store,
                tensorizer=tensorizer,
                config=config,
                device=device,
                optimizer=None,
                scaler=None,
                precision=precision,
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
    tasks: tuple[RoutingTaskReference, ...],
    store: RoutingTaskStore,
    tensorizer: RoutingTaskTensorizer,
    config: RouterTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: object | None,
    precision: str,
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
    preload_workers = (
        max(1, config.num_workers // 2) if config.num_workers else 0
    )
    materialize_workers = (
        max(1, config.num_workers - preload_workers)
        if config.num_workers > 1
        else 0
    )
    materialize_pool = (
        ThreadPoolExecutor(
            max_workers=materialize_workers,
            thread_name_prefix="router-materialize",
        )
        if materialize_workers
        else None
    )
    completed = 0
    try:
        prepared = _iter_prepared_batches(
            tasks,
            store=store,
            tensorizer=tensorizer,
            batch_size=config.batch_size,
            num_workers=preload_workers,
            prefetch_factor=config.prefetch_factor,
        )
        for raw_tasks, descriptors in prepared:
            current_size = len(raw_tasks)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            descriptor_batch = _move_batch(
                collate_routed_tasks(descriptors), device
            )
            with _autocast(device, precision):
                selection = model.select(descriptor_batch)
            selected_paths = selection.route_selection.hard_mask.detach().cpu()
            selected_columns = selection.column_hard_mask.detach().cpu()
            materialize_args = [
                (
                    raw,
                    selected_paths[index, : descriptor.path_features.shape[0]],
                    selected_columns[
                        index,
                        : descriptor.path_features.shape[0],
                        : descriptor.source_values.shape[2],
                    ],
                )
                for index, (raw, descriptor) in enumerate(
                    zip(raw_tasks, descriptors)
                )
            ]
            if materialize_pool is None:
                materialized = tuple(
                    _materialize_task(tensorizer, *arguments)
                    for arguments in materialize_args
                )
            else:
                materialized = tuple(
                    materialize_pool.map(
                        lambda arguments: _materialize_task(
                            tensorizer, *arguments
                        ),
                        materialize_args,
                    )
                )
            batch = _move_batch(collate_routed_tasks(materialized), device)
            with _autocast(device, precision):
                output = model(batch, selection)
                losses = sparse_router_batch_loss(output, batch, config)
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses.total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                    optimizer.step()
            for name, value in losses.detached_metrics().items():
                totals[name] += value * current_size
            completed += current_size
            if progress is not None:
                progress(completed, len(tasks), batch.task_ids[-1])
    finally:
        if materialize_pool is not None:
            materialize_pool.shutdown(wait=True)
    denominator = max(1, len(tasks))
    return {name: value / denominator for name, value in totals.items()}


def _iter_prepared_batches(
    references: tuple[RoutingTaskReference, ...],
    *,
    store: RoutingTaskStore,
    tensorizer: RoutingTaskTensorizer,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> Iterator[tuple[tuple[RawRoutingTask, ...], tuple]]:
    groups = tuple(
        references[index : index + batch_size]
        for index in range(0, len(references), batch_size)
    )
    if not num_workers:
        for group in groups:
            prepared = tuple(
                _load_and_describe(store, tensorizer, reference)
                for reference in group
            )
            yield (
                tuple(item[0] for item in prepared),
                tuple(item[1] for item in prepared),
            )
        return

    with ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="router-prefetch",
    ) as executor:
        group_iterator = iter(groups)
        pending: deque[tuple[Future, ...]] = deque()

        def submit_next() -> bool:
            try:
                group = next(group_iterator)
            except StopIteration:
                return False
            pending.append(
                tuple(
                    executor.submit(
                        _load_and_describe,
                        store,
                        tensorizer,
                        reference,
                    )
                    for reference in group
                )
            )
            return True

        for _ in range(max(1, prefetch_factor)):
            if not submit_next():
                break
        while pending:
            futures = pending.popleft()
            submit_next()
            prepared = tuple(future.result() for future in futures)
            yield (
                tuple(item[0] for item in prepared),
                tuple(item[1] for item in prepared),
            )


def _load_and_describe(
    store: RoutingTaskStore,
    tensorizer: RoutingTaskTensorizer,
    reference: RoutingTaskReference,
):
    raw = store.load(reference)
    return raw, tensorizer.tensorize_descriptors(raw)


def _materialize_task(
    tensorizer: RoutingTaskTensorizer,
    raw: RawRoutingTask,
    selected_paths: torch.Tensor,
    selected_columns: torch.Tensor,
):
    return tensorizer.materialize_selected(
        raw,
        selected_path_mask=selected_paths,
        selected_column_mask=selected_columns,
    )


def _move_batch(batch, device: torch.device):
    if device.type == "cuda":
        return batch.pin_memory().to(device, non_blocking=True)
    return batch.to(device)


def _autocast(device: torch.device, precision: str):
    if precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _resolve_mixed_precision(value: str, device: torch.device) -> str:
    if value == "none":
        return value
    if device.type != "cuda":
        _LOGGER.warning(
            "mixed_precision=%s requested on %s; falling back to float32",
            value,
            device,
        )
        return "none"
    if value == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "mixed_precision=bf16 requires a BF16-capable CUDA GPU; use fp16"
        )
    return value


def _make_grad_scaler(*, enabled: bool):
    generic_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if generic_scaler is not None:
        return generic_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _split_by_schema(
    tasks: tuple[RoutingTaskReference, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[RoutingTaskReference, ...], tuple[RoutingTaskReference, ...]]:
    if validation_fraction <= 0 or len(tasks) < 2:
        return tasks, ()
    train: list[RoutingTaskReference] = []
    validation: list[RoutingTaskReference] = []
    threshold = round(validation_fraction * 10_000)
    for task in tasks:
        schema_id = task.schema_id
        digest = hashlib.sha256(f"{seed}:{schema_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (validation if bucket < threshold else train).append(task)
    if not validation:
        schema_id = tasks[-1].schema_id
        validation_schema_ids = {schema_id}
        validation = [
            task
            for task in tasks
            if task.schema_id == schema_id
        ]
        train = [
            task
            for task in tasks
            if task.schema_id not in validation_schema_ids
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
