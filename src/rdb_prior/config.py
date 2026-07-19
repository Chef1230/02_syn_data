"""Strict configuration loading for the schema-generation stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - declared project dependency.
    yaml = None

from rdb_prior.compilation.compiler import (
    PhysicalCompilerConfig,
    RoleFeatureRule,
    TableCountFeatureRule,
)
from rdb_prior.export.pipeline import RDBPFNExportConfig
from rdb_prior.instance.plan import FeatureSCMFamily
from rdb_prior.instance.planner import InstancePlannerConfig
from rdb_prior.pipeline import InstancePipelineConfig, SchemaPipelineConfig
from rdb_prior.schema.graph import SchemaGraphConfig
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.model import TaskMechanism
from rdb_prior.task.pipeline import TaskPipelineConfig
from rdb_prior.task.planner import TaskPlannerConfig


_CONFIG_VERSION = 1


class SchemaConfigError(ValueError):
    """Raised when a schema pipeline configuration is malformed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaConfigOverrides:
    """Optional CLI overrides; ``None`` preserves the YAML value."""

    output_root: Path | None = None
    num_schemas: int | None = None
    base_seed: int | None = None
    start_index: int | None = None
    sample_id_prefix: str | None = None
    progress_every: int | None = None
    overwrite: bool | None = None
    min_tables: int | None = None
    max_tables: int | None = None
    max_rank: int | None = None
    max_extra_edges: int | None = None
    extra_edge_probability: float | None = None
    min_feature_columns: int | None = None
    max_feature_columns: int | None = None
    feature_nullable_probability: float | None = None
    blueprint_id_prefix: str | None = None
    schema_id_prefix: str | None = None
    write_schema_dot: bool | None = None
    schema_graph_format: str | None = None
    graphviz_command: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceConfigOverrides:
    schema_manifest: Path | None = None
    output_root: Path | None = None
    count: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    progress_every: int | None = None
    overwrite: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskConfigOverrides:
    instance_manifest: Path | None = None
    output_root: Path | None = None
    database_count: int | None = None
    tasks_per_database: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    progress_every: int | None = None
    overwrite: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNExportConfigOverrides:
    task_manifest: Path | None = None
    output_root: Path | None = None
    task_count: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    validation_fraction: float | None = None
    min_validation_rows: int | None = None
    compress: bool | None = None
    progress_every: int | None = None
    overwrite: bool | None = None
    h5_enabled: bool | None = None
    h5_output: Path | None = None
    rdbpfn_preprocessing_root: Path | None = None
    h5_run_dfs: bool | None = None
    dfs_depth: int | None = None
    dfs_jobs: int | None = None
    h5_total_rows: int | None = None
    h5_max_columns: int | None = None
    h5_seed: int | None = None


def load_schema_pipeline_config(
    path: str | Path,
    *,
    overrides: SchemaConfigOverrides | None = None,
) -> SchemaPipelineConfig:
    """Load one YAML/JSON file into validated immutable runtime config."""
    config_path = Path(path).resolve()
    document = _load_document(config_path)
    root = _mapping(document, "config")
    _reject_unknown(
        root,
        {
            "config_version",
            "seed",
            "paths",
            "generation",
            "schema",
            "schema_graph",
            "motifs",
            "physical_design",
            "instance",
            "instance_generation",
            "task",
            "task_generation",
            "rdbpfn_export",
        },
        "config",
    )

    version = root.get("config_version", _CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise SchemaConfigError("config.config_version must be an integer")
    if version != _CONFIG_VERSION:
        raise SchemaConfigError(
            f"Unsupported config_version {version!r}; expected "
            f"{_CONFIG_VERSION}"
        )

    paths = _section(
        root,
        "paths",
        {
            "schema_output_root",
            "schema_manifest",
            "instance_output_root",
            "instance_manifest",
            "task_output_root",
            "task_manifest",
            "rdbpfn_output_root",
        },
    )
    generation = _section(
        root,
        "generation",
        {
            "num_schemas",
            "start_index",
            "sample_id_prefix",
            "progress_every",
            "overwrite",
            "project_version",
        },
    )
    schema = _section(
        root,
        "schema",
        {
            "min_tables",
            "max_tables",
            "table_count_values",
            "table_count_weights",
            "max_rank",
            "max_extra_edges",
            "extra_edge_probability",
            "min_motif_occurrences",
            "max_motif_occurrences",
            "background_attachment_probability",
            "blueprint_id_prefix",
        },
    )
    schema_graph = _section(
        root,
        "schema_graph",
        {
            "write_dot",
            "render_format",
            "graphviz_command",
            "include_columns",
            "include_role_metadata",
        },
    )
    motifs = _section(root, "motifs", {"weights"})
    physical = _section(
        root,
        "physical_design",
        {
            "schema_id_prefix",
            "min_feature_columns",
            "max_feature_columns",
            "feature_nullable_probability",
            "primary_key_names",
            "feature_columns_by_table_count",
            "feature_columns_by_role",
        },
    )
    _section(root, "instance", _INSTANCE_OPTIONS)
    _section(root, "instance_generation", _INSTANCE_GENERATION_OPTIONS)
    _section(root, "task", _TASK_OPTIONS)
    _section(root, "task_generation", _TASK_GENERATION_OPTIONS)
    _section(root, "rdbpfn_export", _RDBPFN_EXPORT_OPTIONS)

    cli = overrides or SchemaConfigOverrides()
    if not isinstance(cli, SchemaConfigOverrides):
        raise TypeError("overrides must be SchemaConfigOverrides or None")

    sampler_defaults = BlueprintSamplerConfig()
    compiler_defaults = PhysicalCompilerConfig()
    graph_defaults = SchemaGraphConfig()

    distribution_overridden = (
        cli.min_tables is not None or cli.max_tables is not None
    )
    if distribution_overridden:
        table_count_values: tuple[int, ...] = ()
        table_count_weights: tuple[int | float, ...] = ()
    else:
        table_count_values = _integer_tuple(
            schema.get(
                "table_count_values",
                sampler_defaults.table_count_values,
            ),
            "config.schema.table_count_values",
        )
        table_count_weights = _numeric_tuple(
            schema.get(
                "table_count_weights",
                sampler_defaults.table_count_weights,
            ),
            "config.schema.table_count_weights",
        )

    motif_weights = _motif_weights(
        motifs.get("weights"),
        default=sampler_defaults.motif_weights,
    )

    feature_bounds_overridden = (
        cli.min_feature_columns is not None
        or cli.max_feature_columns is not None
    )
    if feature_bounds_overridden:
        table_feature_rules: tuple[TableCountFeatureRule, ...] = ()
        role_feature_rules: tuple[RoleFeatureRule, ...] = ()
    else:
        table_feature_rules = _table_feature_rules(
            physical.get("feature_columns_by_table_count", ())
        )
        role_feature_rules = _role_feature_rules(
            physical.get("feature_columns_by_role", {})
        )

    primary_key_names = _string_tuple(
        physical.get(
            "primary_key_names",
            compiler_defaults.primary_key_names,
        ),
        "config.physical_design.primary_key_names",
    )

    output_value = paths.get(
        "schema_output_root",
        "outputs/schema_v1",
    )
    if not isinstance(output_value, (str, Path)):
        raise SchemaConfigError(
            "config.paths.schema_output_root must be a path string"
        )
    output_root = _resolve_output_root(
        config_path=config_path,
        configured=Path(output_value),
        override=cli.output_root,
    )

    try:
        sampler = BlueprintSamplerConfig(
            min_tables=_override(
                cli.min_tables,
                schema.get("min_tables", sampler_defaults.min_tables),
            ),
            max_tables=_override(
                cli.max_tables,
                schema.get("max_tables", sampler_defaults.max_tables),
            ),
            table_count_values=table_count_values,
            table_count_weights=table_count_weights,
            max_rank=_override(
                cli.max_rank,
                schema.get("max_rank", sampler_defaults.max_rank),
            ),
            max_extra_edges=_override(
                cli.max_extra_edges,
                schema.get(
                    "max_extra_edges",
                    sampler_defaults.max_extra_edges,
                ),
            ),
            extra_edge_probability=_override(
                cli.extra_edge_probability,
                schema.get(
                    "extra_edge_probability",
                    sampler_defaults.extra_edge_probability,
                ),
            ),
            min_motif_occurrences=schema.get(
                "min_motif_occurrences",
                sampler_defaults.min_motif_occurrences,
            ),
            max_motif_occurrences=schema.get(
                "max_motif_occurrences",
                sampler_defaults.max_motif_occurrences,
            ),
            background_attachment_probability=schema.get(
                "background_attachment_probability",
                sampler_defaults.background_attachment_probability,
            ),
            blueprint_id_prefix=_override(
                cli.blueprint_id_prefix,
                schema.get(
                    "blueprint_id_prefix",
                    sampler_defaults.blueprint_id_prefix,
                ),
            ),
            motif_weights=motif_weights,
        )
        compiler = PhysicalCompilerConfig(
            min_feature_columns=_override(
                cli.min_feature_columns,
                physical.get(
                    "min_feature_columns",
                    compiler_defaults.min_feature_columns,
                ),
            ),
            max_feature_columns=_override(
                cli.max_feature_columns,
                physical.get(
                    "max_feature_columns",
                    compiler_defaults.max_feature_columns,
                ),
            ),
            feature_columns_by_table_count=table_feature_rules,
            feature_columns_by_role=role_feature_rules,
            feature_nullable_probability=_override(
                cli.feature_nullable_probability,
                physical.get(
                    "feature_nullable_probability",
                    compiler_defaults.feature_nullable_probability,
                ),
            ),
            primary_key_names=primary_key_names,
            schema_id_prefix=_override(
                cli.schema_id_prefix,
                physical.get(
                    "schema_id_prefix",
                    compiler_defaults.schema_id_prefix,
                ),
            ),
        )
        render_format = _override(
            cli.schema_graph_format,
            schema_graph.get(
                "render_format",
                graph_defaults.render_format,
            ),
        )
        if (
            isinstance(render_format, str)
            and render_format.strip().lower() == "none"
        ):
            render_format = None
        graph = SchemaGraphConfig(
            write_dot=_override(
                cli.write_schema_dot,
                schema_graph.get("write_dot", graph_defaults.write_dot),
            ),
            render_format=render_format,
            graphviz_command=_override(
                cli.graphviz_command,
                schema_graph.get(
                    "graphviz_command",
                    graph_defaults.graphviz_command,
                ),
            ),
            include_columns=schema_graph.get(
                "include_columns",
                graph_defaults.include_columns,
            ),
            include_role_metadata=schema_graph.get(
                "include_role_metadata",
                graph_defaults.include_role_metadata,
            ),
        )
        # Construction validates configured motif names against the active
        # library, so --validate-config-only fails before any artifact write.
        BlueprintSampler(sampler)
        return SchemaPipelineConfig(
            output_root=output_root,
            num_schemas=_override(
                cli.num_schemas,
                generation.get("num_schemas", 100),
            ),
            base_seed=_override(
                cli.base_seed,
                root.get("seed", 42),
            ),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            sample_id_prefix=_override(
                cli.sample_id_prefix,
                generation.get("sample_id_prefix", "sample"),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            project_version=generation.get(
                "project_version",
                "schema-pipeline-v1",
            ),
            sampler=sampler,
            compiler=compiler,
            graph=graph,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid schema pipeline config {config_path}: {error}"
        ) from error


_INSTANCE_OPTIONS = {
    "entity_rows_min",
    "entity_rows_max",
    "lookup_rows_min",
    "lookup_rows_max",
    "event_fanout_min",
    "event_fanout_max",
    "bridge_rows_factor_min",
    "bridge_rows_factor_max",
    "detail_fanout_min",
    "detail_fanout_max",
    "entity_child_factor_min",
    "entity_child_factor_max",
    "max_rows_per_table",
    "latent_dimension",
    "optional_rate_min",
    "optional_rate_max",
    "affinity_strength",
    "degree_strength",
    "feature_missing_rate_min",
    "feature_missing_rate_max",
    "feature_noise_scale_min",
    "feature_noise_scale_max",
    "categorical_cardinality_min",
    "categorical_cardinality_max",
    "time_scale_seconds_min",
    "time_scale_seconds_max",
    "scm_weights",
}

_INSTANCE_GENERATION_OPTIONS = {
    "count",
    "start_index",
    "shard_id",
    "num_shards",
    "progress_every",
    "overwrite",
    "project_version",
}

_TASK_OPTIONS = {
    "tasks_per_database",
    "mechanism_weights",
    "support_fraction",
    "min_support_rows",
    "min_query_rows",
    "min_class_count_per_split",
    "max_classification_categories",
    "cutoff_quantile_min",
    "cutoff_quantile_max",
    "horizon_fraction_min",
    "horizon_fraction_max",
    "max_attempts_per_database",
    "require_full_task_count",
}

_TASK_GENERATION_OPTIONS = {
    "database_count",
    "start_index",
    "shard_id",
    "num_shards",
    "progress_every",
    "overwrite",
    "project_version",
}

_RDBPFN_EXPORT_OPTIONS = {
    "task_count",
    "start_index",
    "shard_id",
    "num_shards",
    "validation_fraction",
    "min_validation_rows",
    "compress",
    "progress_every",
    "overwrite",
    "project_version",
    "h5_enabled",
    "h5_output",
    "rdbpfn_preprocessing_root",
    "h5_run_dfs",
    "dfs_depth",
    "dfs_jobs",
    "h5_total_rows",
    "h5_max_columns",
    "h5_seed",
}


def load_instance_pipeline_config(
    path: str | Path,
    *,
    overrides: InstanceConfigOverrides | None = None,
) -> InstancePipelineConfig:
    """Load and validate stage-02 instance generation configuration."""
    config_path = Path(path).resolve()
    # Validate the shared schema sections too: one file remains the source of
    # truth for the staged shell pipeline.
    load_schema_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        {
            "schema_output_root",
            "schema_manifest",
            "instance_output_root",
            "instance_manifest",
            "task_output_root",
            "task_manifest",
            "rdbpfn_output_root",
        },
    )
    instance = _section(root, "instance", _INSTANCE_OPTIONS)
    generation = _section(
        root,
        "instance_generation",
        _INSTANCE_GENERATION_OPTIONS,
    )
    cli = overrides or InstanceConfigOverrides()
    if not isinstance(cli, InstanceConfigOverrides):
        raise TypeError("overrides must be InstanceConfigOverrides or None")
    defaults = InstancePlannerConfig()

    raw_weights = instance.get("scm_weights")
    if raw_weights is None:
        scm_weights = defaults.scm_weights
    else:
        weights = _mapping(raw_weights, "config.instance.scm_weights")
        try:
            scm_weights = tuple(
                (FeatureSCMFamily(name), float(weight))
                for name, weight in sorted(weights.items())
            )
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(
                f"Invalid config.instance.scm_weights: {error}"
            ) from error

    schema_output = paths.get("schema_output_root", "outputs/schema_v1")
    schema_manifest_value = paths.get(
        "schema_manifest",
        str(Path(schema_output) / "manifest.json"),
    )
    output_value = paths.get("instance_output_root", "outputs/instance_v1")
    if not isinstance(schema_manifest_value, (str, Path)):
        raise SchemaConfigError("config.paths.schema_manifest must be a path string")
    if not isinstance(output_value, (str, Path)):
        raise SchemaConfigError(
            "config.paths.instance_output_root must be a path string"
        )

    try:
        planner_values = {
            name: instance.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
            if name != "scm_weights"
        }
        planner = InstancePlannerConfig(
            **planner_values,
            scm_weights=scm_weights,
        )
        return InstancePipelineConfig(
            schema_manifest=_resolve_output_root(
                config_path=config_path,
                configured=Path(schema_manifest_value),
                override=cli.schema_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=Path(output_value),
                override=cli.output_root,
            ),
            count=_override(cli.count, generation.get("count")),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, generation.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                generation.get("num_shards", 1),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            project_version=generation.get(
                "project_version", "instance-pipeline-v1"
            ),
            planner=planner,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid instance pipeline config {config_path}: {error}"
        ) from error


def load_task_pipeline_config(
    path: str | Path,
    *,
    overrides: TaskConfigOverrides | None = None,
) -> TaskPipelineConfig:
    """Load and validate stage-03 task generation configuration."""
    config_path = Path(path).resolve()
    load_instance_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        {
            "schema_output_root",
            "schema_manifest",
            "instance_output_root",
            "instance_manifest",
            "task_output_root",
            "task_manifest",
            "rdbpfn_output_root",
        },
    )
    task = _section(root, "task", _TASK_OPTIONS)
    generation = _section(
        root,
        "task_generation",
        _TASK_GENERATION_OPTIONS,
    )
    cli = overrides or TaskConfigOverrides()
    if not isinstance(cli, TaskConfigOverrides):
        raise TypeError("overrides must be TaskConfigOverrides or None")
    defaults = TaskPlannerConfig()

    raw_weights = task.get("mechanism_weights")
    if raw_weights is None:
        mechanism_weights = defaults.mechanism_weights
    else:
        weights = _mapping(raw_weights, "config.task.mechanism_weights")
        try:
            mechanism_weights = tuple(
                (TaskMechanism(name), float(weight))
                for name, weight in sorted(weights.items())
            )
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(
                f"Invalid config.task.mechanism_weights: {error}"
            ) from error

    instance_output = paths.get("instance_output_root", "outputs/instance_v1")
    manifest_value = paths.get(
        "instance_manifest",
        str(Path(instance_output) / "manifest.json"),
    )
    output_value = paths.get("task_output_root", "outputs/task_v1")
    if not isinstance(manifest_value, (str, Path)):
        raise SchemaConfigError(
            "config.paths.instance_manifest must be a path string"
        )
    if not isinstance(output_value, (str, Path)):
        raise SchemaConfigError(
            "config.paths.task_output_root must be a path string"
        )

    try:
        planner_values = {
            name: task.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
            if name not in {"mechanism_weights", "tasks_per_database"}
        }
        planner = TaskPlannerConfig(
            **planner_values,
            tasks_per_database=_override(
                cli.tasks_per_database,
                task.get("tasks_per_database", defaults.tasks_per_database),
            ),
            mechanism_weights=mechanism_weights,
        )
        return TaskPipelineConfig(
            instance_manifest=_resolve_output_root(
                config_path=config_path,
                configured=Path(manifest_value),
                override=cli.instance_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=Path(output_value),
                override=cli.output_root,
            ),
            database_count=_override(
                cli.database_count,
                generation.get("database_count"),
            ),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, generation.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                generation.get("num_shards", 1),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            project_version=generation.get(
                "project_version", "task-pipeline-v1"
            ),
            planner=planner,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid task pipeline config {config_path}: {error}"
        ) from error


def load_rdbpfn_export_config(
    path: str | Path,
    *,
    overrides: RDBPFNExportConfigOverrides | None = None,
) -> RDBPFNExportConfig:
    """Load and validate stage-04 RDBPFN export configuration."""
    config_path = Path(path).resolve()
    load_task_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        {
            "schema_output_root",
            "schema_manifest",
            "instance_output_root",
            "instance_manifest",
            "task_output_root",
            "task_manifest",
            "rdbpfn_output_root",
        },
    )
    export = _section(root, "rdbpfn_export", _RDBPFN_EXPORT_OPTIONS)
    cli = overrides or RDBPFNExportConfigOverrides()
    if not isinstance(cli, RDBPFNExportConfigOverrides):
        raise TypeError(
            "overrides must be RDBPFNExportConfigOverrides or None"
        )

    task_output = paths.get("task_output_root", "outputs/task_v1")
    manifest_value = paths.get(
        "task_manifest",
        str(Path(task_output) / "manifest.json"),
    )
    output_value = paths.get("rdbpfn_output_root", "outputs/rdbpfn_v1")
    h5_output_value = export.get("h5_output")
    preprocessing_value = export.get(
        "rdbpfn_preprocessing_root",
        "../RDBPFN/data_preprocessing",
    )
    if not isinstance(manifest_value, (str, Path)):
        raise SchemaConfigError("config.paths.task_manifest must be a path string")
    if not isinstance(output_value, (str, Path)):
        raise SchemaConfigError(
            "config.paths.rdbpfn_output_root must be a path string"
        )
    if h5_output_value is not None and not isinstance(
        h5_output_value, (str, Path)
    ):
        raise SchemaConfigError("config.rdbpfn_export.h5_output must be a path string")
    if not isinstance(preprocessing_value, (str, Path)):
        raise SchemaConfigError(
            "config.rdbpfn_export.rdbpfn_preprocessing_root must be a path string"
        )

    resolved_h5_output = None
    if cli.h5_output is not None or h5_output_value is not None:
        resolved_h5_output = _resolve_output_root(
            config_path=config_path,
            configured=Path(h5_output_value or "."),
            override=cli.h5_output,
        )

    try:
        return RDBPFNExportConfig(
            task_manifest=_resolve_output_root(
                config_path=config_path,
                configured=Path(manifest_value),
                override=cli.task_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=Path(output_value),
                override=cli.output_root,
            ),
            task_count=_override(cli.task_count, export.get("task_count")),
            start_index=_override(
                cli.start_index,
                export.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, export.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                export.get("num_shards", 1),
            ),
            validation_fraction=_override(
                cli.validation_fraction,
                export.get("validation_fraction", 0.2),
            ),
            min_validation_rows=_override(
                cli.min_validation_rows,
                export.get("min_validation_rows", 8),
            ),
            compress=_override(cli.compress, export.get("compress", True)),
            progress_every=_override(
                cli.progress_every,
                export.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                export.get("overwrite", False),
            ),
            project_version=export.get(
                "project_version", "rdbpfn-export-v1"
            ),
            h5_enabled=_override(
                cli.h5_enabled,
                export.get("h5_enabled", False),
            ),
            h5_output=resolved_h5_output,
            rdbpfn_preprocessing_root=_resolve_output_root(
                config_path=config_path,
                configured=Path(preprocessing_value),
                override=cli.rdbpfn_preprocessing_root,
            ),
            h5_run_dfs=_override(
                cli.h5_run_dfs,
                export.get("h5_run_dfs", True),
            ),
            dfs_depth=_override(cli.dfs_depth, export.get("dfs_depth", 1)),
            dfs_jobs=_override(cli.dfs_jobs, export.get("dfs_jobs", 4)),
            h5_total_rows=_override(
                cli.h5_total_rows,
                export.get("h5_total_rows", 600),
            ),
            h5_max_columns=_override(
                cli.h5_max_columns,
                export.get("h5_max_columns", 60),
            ),
            h5_seed=_override(cli.h5_seed, export.get("h5_seed", 42)),
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid RDBPFN export config {config_path}: {error}"
        ) from error


def _load_document(path: Path) -> Any:
    if not path.is_file():
        raise SchemaConfigError(f"Config file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        if yaml is None:
            raise SchemaConfigError(
                "PyYAML is required to load YAML configuration files"
            )
        return yaml.safe_load(text)
    except SchemaConfigError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaConfigError(f"Cannot read config {path}: {error}") from error
    except Exception as error:
        if yaml is not None and isinstance(error, yaml.YAMLError):
            raise SchemaConfigError(
                f"Cannot parse YAML config {path}: {error}"
            ) from error
        raise


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        raise SchemaConfigError(f"{path} must be a mapping, not null")
    if not isinstance(value, Mapping):
        raise SchemaConfigError(f"{path} must be a mapping")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise SchemaConfigError(f"{path} keys must be strings")
    return result


def _section(
    root: Mapping[str, Any],
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    value = root.get(name, {})
    section = _mapping(value, f"config.{name}")
    _reject_unknown(section, allowed, f"config.{name}")
    return section


def _reject_unknown(
    values: Mapping[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SchemaConfigError(
            f"{path} contains unknown option(s): {', '.join(unknown)}"
        )


def _sequence(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchemaConfigError(f"{path} must be a list")
    return tuple(value)


def _integer_tuple(value: Any, path: str) -> tuple[int, ...]:
    values = _sequence(value, path)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise SchemaConfigError(f"{path} items must be integers")
    return values


def _numeric_tuple(value: Any, path: str) -> tuple[int | float, ...]:
    values = _sequence(value, path)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in values
    ):
        raise SchemaConfigError(f"{path} items must be numeric")
    return values


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    values = _sequence(value, path)
    if any(not isinstance(item, str) for item in values):
        raise SchemaConfigError(f"{path} items must be strings")
    return values


def _motif_weights(
    value: Any,
    *,
    default: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, int | float], ...]:
    if value is None:
        return default
    weights = _mapping(value, "config.motifs.weights")
    result: list[tuple[str, int | float]] = []
    for motif_type, weight in sorted(weights.items()):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise SchemaConfigError(
                f"config.motifs.weights.{motif_type} must be numeric"
            )
        result.append((motif_type, weight))
    return tuple(result)


def _table_feature_rules(value: Any) -> tuple[TableCountFeatureRule, ...]:
    entries = _sequence(
        value,
        "config.physical_design.feature_columns_by_table_count",
    )
    rules: list[TableCountFeatureRule] = []
    allowed = {
        "table_count_min",
        "table_count_max",
        "min_columns",
        "max_columns",
    }
    for index, entry in enumerate(entries):
        path = (
            "config.physical_design.feature_columns_by_table_count"
            f"[{index}]"
        )
        data = _mapping(entry, path)
        _reject_unknown(data, allowed, path)
        missing = sorted(allowed - set(data))
        if missing:
            raise SchemaConfigError(
                f"{path} is missing: {', '.join(missing)}"
            )
        try:
            rules.append(TableCountFeatureRule(**data))
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(f"Invalid {path}: {error}") from error
    return tuple(rules)


def _role_feature_rules(value: Any) -> tuple[RoleFeatureRule, ...]:
    roles = _mapping(
        value,
        "config.physical_design.feature_columns_by_role",
    )
    rules: list[RoleFeatureRule] = []
    allowed = {"min_columns", "max_columns"}
    for role_name, entry in sorted(roles.items()):
        path = (
            "config.physical_design.feature_columns_by_role."
            f"{role_name}"
        )
        try:
            role = TableRole(role_name)
        except ValueError as error:
            allowed_roles = ", ".join(role.value for role in TableRole)
            raise SchemaConfigError(
                f"Unknown role {role_name!r}; expected one of: "
                f"{allowed_roles}"
            ) from error
        data = _mapping(entry, path)
        _reject_unknown(data, allowed, path)
        missing = sorted(allowed - set(data))
        if missing:
            raise SchemaConfigError(
                f"{path} is missing: {', '.join(missing)}"
            )
        try:
            rules.append(RoleFeatureRule(role=role, **data))
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(f"Invalid {path}: {error}") from error
    return tuple(rules)


def _resolve_output_root(
    *,
    config_path: Path,
    configured: Path,
    override: Path | None,
) -> Path:
    if override is not None:
        value = Path(override)
        return value.resolve()

    if configured.is_absolute():
        return configured.resolve()
    config_directory = next(
        (
            parent
            for parent in config_path.parents
            if parent.name == "configs"
        ),
        None,
    )
    project_root = (
        config_directory.parent
        if config_directory is not None
        else config_path.parent
    )
    return (project_root / configured).resolve()


def _override(override: Any, configured: Any) -> Any:
    return configured if override is None else override


__all__ = [
    "SchemaConfigError",
    "SchemaConfigOverrides",
    "InstanceConfigOverrides",
    "TaskConfigOverrides",
    "RDBPFNExportConfigOverrides",
    "load_schema_pipeline_config",
    "load_instance_pipeline_config",
    "load_task_pipeline_config",
    "load_rdbpfn_export_config",
]
