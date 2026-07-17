# src/rdb_prior/pipeline.py
# -*- coding: utf-8 -*-
"""Schema-only pipeline: Blueprint sampling -> PhysicalSchema artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

from rdb_prior.artifacts import SchemaArtifactWriter
from rdb_prior.compilation.compiler import (
    PhysicalCompilerConfig,
    PhysicalSchemaCompiler,
)
from rdb_prior.runtime import RuntimeContext, digest_config
from rdb_prior.schema.sampler import (
    BlueprintSampler,
    BlueprintSamplerConfig,
)
from rdb_prior.schema.validation import validate_blueprint


_SAMPLE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaPipelineConfig:
    output_root: Path
    num_schemas: int = 100
    base_seed: int = 42
    start_index: int = 0
    sample_id_prefix: str = "sample"
    overwrite: bool = False
    progress_every: int = 100
    project_version: str = "schema-pipeline-v1"
    sampler: BlueprintSamplerConfig = BlueprintSamplerConfig()
    compiler: PhysicalCompilerConfig = PhysicalCompilerConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        for name, value in (
            ("num_schemas", self.num_schemas),
            ("base_seed", self.base_seed),
            ("start_index", self.start_index),
            ("progress_every", self.progress_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.num_schemas < 1:
            raise ValueError("num_schemas must be positive")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
        if not isinstance(self.sample_id_prefix, str) or not (
            _SAMPLE_PREFIX.fullmatch(self.sample_id_prefix)
        ):
            raise ValueError("sample_id_prefix is not artifact-safe")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if (
            not isinstance(self.project_version, str)
            or not self.project_version.strip()
        ):
            raise ValueError("project_version must not be empty")
        if not isinstance(self.sampler, BlueprintSamplerConfig):
            raise TypeError("sampler must be BlueprintSamplerConfig")
        if not isinstance(self.compiler, PhysicalCompilerConfig):
            raise TypeError("compiler must be PhysicalCompilerConfig")

    def to_dict(self) -> dict[str, object]:
        return {
            "num_schemas": self.num_schemas,
            "base_seed": self.base_seed,
            "start_index": self.start_index,
            "sample_id_prefix": self.sample_id_prefix,
            "overwrite": self.overwrite,
            "progress_every": self.progress_every,
            "project_version": self.project_version,
            "sampler": {
                "min_tables": self.sampler.min_tables,
                "max_tables": self.sampler.max_tables,
                "table_count_values": list(
                    self.sampler.table_count_values
                ),
                "table_count_weights": list(
                    self.sampler.table_count_weights
                ),
                "max_rank": self.sampler.max_rank,
                "max_extra_edges": self.sampler.max_extra_edges,
                "extra_edge_probability": (
                    self.sampler.extra_edge_probability
                ),
                "blueprint_id_prefix": (
                    self.sampler.blueprint_id_prefix
                ),
                "motif_weights": [
                    [motif_type, weight]
                    for motif_type, weight in self.sampler.motif_weights
                ],
            },
            "compiler": {
                "min_feature_columns": (
                    self.compiler.min_feature_columns
                ),
                "max_feature_columns": (
                    self.compiler.max_feature_columns
                ),
                "feature_columns_by_table_count": [
                    {
                        "table_count_min": rule.table_count_min,
                        "table_count_max": rule.table_count_max,
                        "min_columns": rule.min_columns,
                        "max_columns": rule.max_columns,
                    }
                    for rule in (
                        self.compiler.feature_columns_by_table_count
                    )
                ],
                "feature_columns_by_role": {
                    rule.role.value: {
                        "min_columns": rule.min_columns,
                        "max_columns": rule.max_columns,
                    }
                    for rule in self.compiler.feature_columns_by_role
                },
                "feature_nullable_probability": (
                    self.compiler.feature_nullable_probability
                ),
                "primary_key_names": list(
                    self.compiler.primary_key_names
                ),
                "schema_id_prefix": self.compiler.schema_id_prefix,
            },
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaPipelineResult:
    output_root: Path
    manifest_path: Path
    artifact_paths: tuple[Path, ...]

    @property
    def generated_count(self) -> int:
        return len(self.artifact_paths)


def generate_physical_schemas(
    config: SchemaPipelineConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> SchemaPipelineResult:
    """Generate a deterministic batch of validated physical schemas."""
    if not isinstance(config, SchemaPipelineConfig):
        raise TypeError("config must be SchemaPipelineConfig")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    configuration = config.to_dict()
    config_digest = digest_config(configuration)
    root_runtime = RuntimeContext(config.base_seed)
    blueprint_sampler = BlueprintSampler(config.sampler)
    compiler = PhysicalSchemaCompiler(config.compiler)
    writer = SchemaArtifactWriter(
        output_root=config.output_root,
        overwrite=config.overwrite,
    )

    paths: list[Path] = []
    entries: list[dict[str, object]] = []

    for offset in range(config.num_schemas):
        sample_index = config.start_index + offset
        sample_id = f"{config.sample_id_prefix}_{sample_index:06d}"
        runtime = root_runtime.for_sample(sample_id)
        blueprint = blueprint_sampler.sample(sample_id, runtime)
        report = validate_blueprint(blueprint, raise_on_error=True)
        physical_schema = compiler.compile(
            blueprint,
            sample_id,
            runtime,
        )
        runtime_record = runtime.record(
            project_version=config.project_version,
            config_digest=config_digest,
            metadata={
                "stage": "physical_schema",
                "sample_id": sample_id,
            },
        )
        artifact_path = writer.commit(
            sample_id=sample_id,
            runtime=runtime_record,
            blueprint=blueprint,
            physical_schema=physical_schema,
            report=report,
        )
        paths.append(artifact_path)

        role_counts = Counter(
            node.role.value
            for node in blueprint.nodes
        )
        entries.append(
            {
                "sample_id": sample_id,
                "artifact": artifact_path.relative_to(
                    config.output_root
                ).as_posix(),
                "derived_seed": runtime.seed(),
                "blueprint_id": blueprint.blueprint_id,
                "physical_schema_id": physical_schema.schema_id,
                "table_count": len(physical_schema.tables),
                "foreign_key_count": len(
                    physical_schema.foreign_keys
                ),
                "role_counts": dict(sorted(role_counts.items())),
            }
        )

        completed = offset + 1
        if (
            progress is not None
            and config.progress_every > 0
            and (
                completed % config.progress_every == 0
                or completed == config.num_schemas
            )
        ):
            progress(completed, config.num_schemas, sample_id)

    manifest_path = writer.write_manifest(
        configuration=configuration,
        entries=entries,
    )
    return SchemaPipelineResult(
        output_root=config.output_root,
        manifest_path=manifest_path,
        artifact_paths=tuple(paths),
    )


__all__ = [
    "SchemaPipelineConfig",
    "SchemaPipelineResult",
    "generate_physical_schemas",
]
