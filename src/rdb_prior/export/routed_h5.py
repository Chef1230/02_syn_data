"""Export checkpoint-conditioned relation-column tokens to H5."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from rdb_prior.routing.checkpoint import load_router_checkpoint
from rdb_prior.routing.config import RoutedH5Config
from rdb_prior.routing.data import RoutingTaskStore, RoutingTaskTensorizer
from rdb_prior.routing.trainer import resolve_device


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutedH5Result:
    output_path: Path
    task_count: int


def export_routed_h5(
    config: RoutedH5Config,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> RoutedH5Result:
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"routed H5 already exists: {config.output_path}; use overwrite=True"
        )
    device = resolve_device(config.device)
    model, checkpoint = load_router_checkpoint(config.checkpoint, device=device)
    model.eval()
    tensorizer = RoutingTaskTensorizer(model.config)
    store = RoutingTaskStore(
        config.task_manifest,
        start_index=config.start_index,
        task_count=config.task_count,
    )
    references = store.references
    h5py = _require_h5py()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as handle, torch.no_grad():
            handle.attrs["format"] = "rdb-prior-routed-token-h5-v1"
            handle.attrs["configuration"] = json.dumps(
                config.to_dict(), sort_keys=True
            )
            handle.attrs["model_config"] = json.dumps(
                model.config.to_dict(), sort_keys=True
            )
            handle.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
            task_group = handle.create_group("tasks")
            for completed, reference in enumerate(references, start=1):
                raw = store.load(reference)
                descriptors = tensorizer.tensorize_descriptors(raw).to(device)
                selection = model.select(descriptors)
                batch = tensorizer.materialize_selected(
                    raw,
                    selected_path_mask=selection.route_selection.hard_mask,
                    selected_column_mask=selection.column_hard_mask,
                ).to(device)
                output = model(batch, selection)
                group = task_group.create_group(_safe_group_name(batch.task_id))
                group.attrs["task_id"] = batch.task_id
                group.attrs["prediction_type"] = batch.prediction_type.value
                group.attrs["num_classes"] = batch.num_classes
                group.attrs["label_center"] = batch.label_center
                group.attrs["label_scale"] = batch.label_scale
                group.attrs["class_values"] = json.dumps(batch.class_values)
                _dataset(group, "row_ids", batch.row_ids)
                _dataset(group, "target_tokens", output.target_tokens)
                _dataset(group, "relation_tokens", output.relation_tokens)
                _dataset(group, "relation_token_mask", output.relation_mask)
                _dataset(group, "support_mask", batch.support_mask)
                _dataset(group, "y", batch.labels)
                _dataset(
                    group, "route_logits", output.route_selection.logits
                )
                _dataset(
                    group, "selected_path_mask", output.route_selection.hard_mask
                )
                _dataset(group, "source_column_logits", output.column_logits)
                _dataset(
                    group, "selected_source_column_mask", output.column_hard_mask
                )
                _dataset(group, "path_features", batch.path_features)
                _dataset(group, "path_costs", batch.path_costs)
                selected_pairs = [
                    (path_index, column_index)
                    for path_index in range(
                        output.route_selection.hard_mask.shape[0]
                    )
                    if bool(output.route_selection.hard_mask[path_index])
                    for column_index in range(
                        output.column_hard_mask.shape[1]
                    )
                    if bool(output.column_hard_mask[path_index, column_index])
                ]
                group.create_dataset(
                    "relation_path_index",
                    data=np.asarray(
                        [pair[0] for pair in selected_pairs], dtype=np.int32
                    ),
                )
                group.create_dataset(
                    "relation_column_index",
                    data=np.asarray(
                        [pair[1] for pair in selected_pairs], dtype=np.int32
                    ),
                )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(
                    "path_signatures",
                    data=np.asarray(batch.path_signatures, dtype=object),
                    dtype=string_dtype,
                )
                group.attrs["source_column_ids"] = json.dumps(
                    batch.source_column_ids
                )
                if progress is not None:
                    progress(completed, len(references), batch.task_id)
        temporary.replace(config.output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return RoutedH5Result(
        output_path=config.output_path,
        task_count=len(references),
    )


def _dataset(group: object, name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().numpy()
    group.create_dataset(name, data=values, compression="lzf")


def _safe_group_name(value: str) -> str:
    return value.replace("/", "_")


def _require_h5py() -> object:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - declared dependency.
        raise RuntimeError("h5py is required for routed H5 export") from error
    return h5py


__all__ = [
    "RoutedH5Config",
    "RoutedH5Result",
    "export_routed_h5",
]
