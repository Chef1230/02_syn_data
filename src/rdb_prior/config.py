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
from rdb_prior.pipeline import SchemaPipelineConfig
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole


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
            "motifs",
            "physical_design",
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
        {"schema_output_root"},
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
            "blueprint_id_prefix",
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

    cli = overrides or SchemaConfigOverrides()
    if not isinstance(cli, SchemaConfigOverrides):
        raise TypeError("overrides must be SchemaConfigOverrides or None")

    sampler_defaults = BlueprintSamplerConfig()
    compiler_defaults = PhysicalCompilerConfig()

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
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid schema pipeline config {config_path}: {error}"
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
    project_root = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else config_path.parent
    )
    return (project_root / configured).resolve()


def _override(override: Any, configured: Any) -> Any:
    return configured if override is None else override


__all__ = [
    "SchemaConfigError",
    "SchemaConfigOverrides",
    "load_schema_pipeline_config",
]
