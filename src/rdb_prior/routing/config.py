"""Validated configuration for the sparse MLP router."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterModelConfig:
    max_path_depth: int = 2
    max_candidates: int = 20
    top_k_paths: int = 3
    max_source_columns: int = 8
    rows_per_hop: int = 32
    min_rows_per_hop: int = 16
    max_target_columns: int = 32
    max_rows_per_task: int = 600
    token_dim: int = 64
    type_embedding_dim: int = 16
    path_feature_dim: int = 8
    column_feature_dim: int = 8
    router_hidden_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    max_classes: int = 16
    aggregation: str = "set"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_path_depth",
            "max_candidates",
            "top_k_paths",
            "max_source_columns",
            "rows_per_hop",
            "min_rows_per_hop",
            "max_target_columns",
            "max_rows_per_task",
            "token_dim",
            "type_embedding_dim",
            "path_feature_dim",
            "column_feature_dim",
            "router_hidden_dim",
            "transformer_heads",
            "transformer_layers",
            "max_classes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.top_k_paths > self.max_candidates:
            raise ValueError("top_k_paths cannot exceed max_candidates")
        if self.max_rows_per_task < 2:
            raise ValueError("max_rows_per_task must be at least two")
        if self.min_rows_per_hop > self.rows_per_hop:
            raise ValueError("min_rows_per_hop cannot exceed rows_per_hop")
        if self.token_dim % self.transformer_heads:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if self.aggregation not in {"set", "sequence"}:
            raise ValueError("aggregation must be 'set' or 'sequence'")
        for name in ("dropout", "temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterTrainingConfig:
    task_manifest: Path
    output_root: Path
    model: RouterModelConfig = RouterModelConfig()
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.1
    task_count: int | None = None
    start_index: int = 0
    seed: int = 42
    device: str = "auto"
    lambda_route: float = 1.0
    lambda_cost: float = 0.05
    lambda_sparse: float = 0.05
    lambda_diversity: float = 0.05
    overwrite: bool = False
    progress_every: int = 50
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    artifact_cache_size: int = 16
    mixed_precision: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.task_manifest, Path):
            raise TypeError("task_manifest must be pathlib.Path")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if not isinstance(self.model, RouterModelConfig):
            raise TypeError("model must be RouterModelConfig")
        for name in (
            "epochs",
            "start_index",
            "seed",
            "progress_every",
            "batch_size",
            "num_workers",
            "prefetch_factor",
            "artifact_cache_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or name in {"epochs", "batch_size", "prefetch_factor"} and value < 1:
                raise ValueError(f"{name} has an invalid value")
        if self.task_count is not None and (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise ValueError("task_count must be a positive integer or None")
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "validation_fraction",
            "lambda_route",
            "lambda_cost",
            "lambda_sparse",
            "lambda_diversity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not _valid_device(self.device):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["task_manifest"] = str(self.task_manifest)
        result["output_root"] = str(self.output_root)
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutedH5Config:
    task_manifest: Path
    checkpoint: Path
    output_path: Path
    task_count: int | None = None
    start_index: int = 0
    device: str = "auto"
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("task_manifest", "checkpoint", "output_path"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path")
        if self.task_count is not None and (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise ValueError("task_count must be a positive integer or None")
        if isinstance(self.start_index, bool) or not isinstance(
            self.start_index, int
        ) or self.start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        if not _valid_device(self.device):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("task_manifest", "checkpoint", "output_path"):
            result[name] = str(result[name])
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterEvaluationConfig:
    task_manifest: Path
    checkpoint: Path
    output_root: Path
    task_count: int | None = None
    start_index: int = 0
    device: str = "auto"
    mixed_precision: str = "none"
    artifact_cache_size: int = 16
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("task_manifest", "checkpoint", "output_root"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path")
        if self.task_count is not None and (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise ValueError("task_count must be a positive integer or None")
        for name in ("start_index", "artifact_cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not _valid_device(self.device):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("task_manifest", "checkpoint", "output_root"):
            result[name] = str(result[name])
        return result


def _valid_device(value: str) -> bool:
    if value in {"auto", "cpu", "cuda", "mps"}:
        return True
    if not isinstance(value, str) or not value.startswith("cuda:"):
        return False
    index = value.removeprefix("cuda:")
    return index.isdigit()


__all__ = [
    "RouterEvaluationConfig",
    "RouterModelConfig",
    "RouterTrainingConfig",
    "RoutedH5Config",
]
