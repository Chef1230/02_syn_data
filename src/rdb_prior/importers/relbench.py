"""Convert RelBench entity tasks into native schema/instance/task artifacts.

The imported target is a prediction-anchor table.  Every RelBench task row is
represented by a unique anchor row containing the entity FK, prediction time,
and masked target.  This preserves repeated entity/time examples and lets the
router apply a row-specific temporal cutoff while traversing historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

from rdb_prior.artifacts import (
    InstanceArtifactWriter,
    SchemaArtifactWriter,
)
from rdb_prior.compilation.model import (
    ColumnKind,
    CompilationResult,
    CompilationTrace,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalForeignKey,
    PhysicalSchema,
    PhysicalTable,
)
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.instance.plan import (
    FeatureSCMFamily,
    InstancePlan,
    PopulationPlan,
    RelationMechanismPlan,
    TableMechanismPlan,
    TemporalFamily,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import (
    BlueprintEdge,
    BlueprintNode,
    SchemaBlueprint,
)
from rdb_prior.schema.spec import (
    Cardinality,
    IdentityDependency,
    Optionality,
    TableRole,
)
from rdb_prior.schema.validation import ValidationReport
from rdb_prior.task.artifacts import TaskArtifactWriter
from rdb_prior.task.model import (
    ObservationRule,
    PlannedTask,
    PredictionType,
    RoutePathLabel,
    RouteRole,
    TaskData,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.validation import TaskValidationReport, validate_task
from rdb_prior.validation.checks import InstanceValidationReport


_SUPPORTED_TASK_TYPES = {
    "binary_classification": PredictionType.CLASSIFICATION,
    "multiclass_classification": PredictionType.CLASSIFICATION,
    "regression": PredictionType.REGRESSION,
}
_NON_SQL = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True, slots=True, kw_only=True)
class RelBenchImportConfig:
    dataset_name: str
    task_name: str
    output_root: Path
    download: bool = False
    overwrite: bool = False
    seed: int = 0
    max_rows_per_task: int = 600
    query_rows_per_task: int = 256
    support_rows: int | None = None
    max_classes: int = 16
    max_text_length: int = 256

    def __post_init__(self) -> None:
        for name in ("dataset_name", "task_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        for name in (
            "seed",
            "max_rows_per_task",
            "query_rows_per_task",
            "max_classes",
            "max_text_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "max_rows_per_task",
            "query_rows_per_task",
            "max_classes",
            "max_text_length",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.support_rows is not None:
            if isinstance(self.support_rows, bool) or not isinstance(
                self.support_rows, int
            ):
                raise TypeError("support_rows must be an integer or None")
            if self.support_rows < 1:
                raise ValueError("support_rows must be positive")
        support_budget = self.resolved_support_rows
        if support_budget < 1:
            raise ValueError(
                "max_rows_per_task must leave at least one support row"
            )
        if support_budget + self.query_rows_per_task > self.max_rows_per_task:
            raise ValueError(
                "support_rows + query_rows_per_task must not exceed "
                "max_rows_per_task"
            )

    @property
    def resolved_support_rows(self) -> int:
        if self.support_rows is not None:
            return self.support_rows
        return self.max_rows_per_task - self.query_rows_per_task

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "output_root": str(self.output_root),
            "download": self.download,
            "overwrite": self.overwrite,
            "seed": self.seed,
            "max_rows_per_task": self.max_rows_per_task,
            "query_rows_per_task": self.query_rows_per_task,
            "support_rows": self.resolved_support_rows,
            "max_classes": self.max_classes,
            "max_text_length": self.max_text_length,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RelBenchImportResult:
    output_root: Path
    schema_manifest: Path
    instance_manifest: Path
    task_manifest: Path
    metadata_path: Path
    task_count: int
    support_row_count: int
    query_row_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _ConvertedTable:
    original_name: str
    table: PhysicalTable
    data: TableData
    original_pkey: str | None
    pkey_values: Any
    time_column_id: str | None
    column_mapping: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _ConvertedDatabase:
    tables: tuple[_ConvertedTable, ...]
    foreign_keys: tuple[PhysicalForeignKey, ...]
    anchor_table: PhysicalTable
    anchor_data: TableData
    anchor_entity_foreign_key: PhysicalForeignKey
    anchor_time_column_id: str
    anchor_label_column_id: str
    dropped_relations: tuple[dict[str, str], ...]


def import_relbench(
    config: RelBenchImportConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> RelBenchImportResult:
    """Load a registered RelBench task and convert it to native artifacts."""
    try:
        from relbench.tasks import get_task
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "RelBench is required for relbench-import; install the optional "
            "dependency with `pip install -e '.[relbench]'`."
        ) from error

    task = get_task(
        config.dataset_name,
        config.task_name,
        download=config.download,
    )
    return convert_relbench_objects(
        config,
        dataset=task.dataset,
        task=task,
        progress=progress,
    )


def convert_relbench_objects(
    config: RelBenchImportConfig,
    *,
    dataset: Any,
    task: Any,
    progress: Callable[[int, int, str], None] | None = None,
) -> RelBenchImportResult:
    """Convert already loaded RelBench-compatible objects.

    This separate entry point keeps the conversion core testable without a
    network download and is also useful for locally registered RelBench tasks.
    """
    prediction_type, task_type = _prediction_type(task)
    entity_table_name = _required_task_attribute(task, "entity_table")
    entity_column = _required_task_attribute(task, "entity_col")
    time_column = _required_task_attribute(task, "time_col")
    target_column = _required_task_attribute(task, "target_col")
    metadata_path = config.output_root / "relbench_metadata.json"
    if metadata_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"RelBench metadata already exists: {metadata_path}; "
            "use overwrite=True"
        )

    split_frames: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        table = task.get_table(split, mask_input_cols=False)
        frame = table.df.reset_index(drop=True)
        for column in (entity_column, time_column, target_column):
            if column not in frame.columns:
                raise ValueError(
                    f"RelBench {split} table is missing required column "
                    f"{column!r}"
                )
        if not len(frame):
            raise ValueError(f"RelBench {split} split is empty")
        split_frames[split] = frame

    database = dataset.get_db(upto_test_timestamp=True)
    if entity_table_name not in database.table_dict:
        raise ValueError(
            f"RelBench entity table {entity_table_name!r} is absent from database"
        )

    combined = _concat_split_frames(split_frames)
    timestamp_origin_ns = _timestamp_origin_ns(
        database, anchor_series=combined[time_column]
    )
    anchor_times = _timestamp_array(
        combined[time_column], origin_ns=timestamp_origin_ns
    )
    if np.any(anchor_times == -1):
        raise ValueError("RelBench task timestamps cannot be missing")
    labels = _label_array(
        combined[target_column],
        prediction_type=prediction_type,
        max_text_length=config.max_text_length,
    )
    train_count = len(split_frames["train"])
    validation_count = len(split_frames["val"])
    support_total = train_count + validation_count
    query_total = len(split_frames["test"])
    support_labels_all = labels[:support_total]
    query_labels_all = labels[support_total:]
    support_indices = _sample_support_indices(
        support_labels_all,
        count=min(config.resolved_support_rows, support_total),
        prediction_type=prediction_type,
        seed=config.seed,
    )
    if prediction_type is PredictionType.CLASSIFICATION:
        support_classes = np.unique(support_labels_all[support_indices])
        query_classes = np.unique(query_labels_all)
        if len(support_classes) > config.max_classes:
            raise ValueError(
                f"RelBench task has {len(support_classes)} support classes, "
                f"exceeding max_classes={config.max_classes}"
            )
        missing_classes = np.setdiff1d(query_classes, support_classes)
        if len(missing_classes):
            raise ValueError(
                "RelBench test labels contain classes absent from the sampled "
                f"support set: {missing_classes.tolist()}"
            )

    sample_id = _artifact_id(
        f"relbench-{config.dataset_name}-{config.task_name}"
    )
    blueprint_id = f"{sample_id}-blueprint"
    schema_id = f"{sample_id}-schema"
    instance_id = f"{sample_id}-instance"
    plan_id = f"{sample_id}-instance-plan"
    runtime = RuntimeContext(config.seed).child(
        "relbench", config.dataset_name, config.task_name
    )
    converted = _convert_database(
        database,
        combined_task_frame=combined,
        entity_table_name=entity_table_name,
        entity_column=entity_column,
        anchor_times=anchor_times,
        labels=labels,
        max_text_length=config.max_text_length,
        timestamp_origin_ns=timestamp_origin_ns,
    )
    physical_tables = tuple(
        item.table for item in converted.tables
    ) + (converted.anchor_table,)
    table_data = tuple(
        item.data for item in converted.tables
    ) + (converted.anchor_data,)
    foreign_keys = converted.foreign_keys + (
        converted.anchor_entity_foreign_key,
    )
    schema = PhysicalSchema(
        schema_id=schema_id,
        blueprint_id=blueprint_id,
        tables=physical_tables,
        foreign_keys=foreign_keys,
    )
    blueprint, trace = _blueprint_and_trace(schema)
    compilation = CompilationResult(schema=schema, trace=trace)

    schema_root = config.output_root / "schema"
    instance_root = config.output_root / "instance"
    task_root = config.output_root / "task"
    schema_writer = SchemaArtifactWriter(
        output_root=schema_root,
        overwrite=config.overwrite,
    )
    schema_path = schema_writer.commit(
        sample_id=sample_id,
        runtime=runtime.child("schema").record(
            project_version="0.1.0",
            metadata={"source": "relbench"},
        ),
        blueprint=blueprint,
        compilation=compilation,
        report=ValidationReport(blueprint_id=blueprint_id, issues=()),
    )
    schema_manifest = schema_writer.write_manifest(
        configuration=config.to_dict(),
        entries=(
            {
                "sample_id": sample_id,
                "artifact": schema_path.relative_to(schema_root).as_posix(),
                "blueprint_id": blueprint_id,
                "physical_schema_id": schema_id,
                "table_count": len(schema.tables),
                "foreign_key_count": len(schema.foreign_keys),
                "source": "relbench",
            },
        ),
    )

    instance_plan = _instance_plan(
        schema,
        table_data,
        sample_id=sample_id,
        plan_id=plan_id,
        runtime=runtime,
    )
    native_database = DatabaseInstance(
        instance_id=instance_id,
        schema_id=schema_id,
        plan_id=plan_id,
        tables=table_data,
    )
    instance_writer = InstanceArtifactWriter(
        output_root=instance_root,
        overwrite=config.overwrite,
    )
    instance_target = instance_writer.instance_directory / sample_id / "artifact.json"
    instance_path = instance_writer.commit(
        sample_id=sample_id,
        schema_artifact=_relative_reference(
            instance_target.parent, schema_path
        ),
        runtime=runtime.child("instance").record(
            project_version="0.1.0",
            metadata={"source": "relbench"},
        ),
        schema=schema,
        plan=instance_plan,
        database=native_database,
        report=InstanceValidationReport(
            schema_id=schema_id,
            plan_id=plan_id,
        ),
    )
    instance_manifest = instance_writer.write_manifest(
        configuration=config.to_dict(),
        entries=(
            {
                "sample_id": sample_id,
                "artifact": instance_path.relative_to(instance_root).as_posix(),
                "schema_id": schema_id,
                "instance_id": instance_id,
                "table_count": len(schema.tables),
                "row_count": sum(table.row_count for table in table_data),
                "source": "relbench",
            },
        ),
    )

    observation_rules = tuple(
        ObservationRule(
            table_id=table.table.table_id,
            time_column_id=table.time_column_id,
            max_timestamp=int(np.max(anchor_times)),
        )
        for table in converted.tables
        if table.time_column_id is not None
    ) + (
        ObservationRule(
            table_id=converted.anchor_table.table_id,
            time_column_id=converted.anchor_time_column_id,
            max_timestamp=int(np.max(anchor_times)),
        ),
    )
    route_supervision = _route_supervision(converted)
    task_writer = TaskArtifactWriter(
        output_root=task_root,
        overwrite=config.overwrite,
    )
    task_entries: list[dict[str, Any]] = []
    query_chunks: list[dict[str, Any]] = []
    query_offsets = range(0, query_total, config.query_rows_per_task)
    total_chunks = (
        query_total + config.query_rows_per_task - 1
    ) // config.query_rows_per_task
    support_row_ids = support_indices.astype(np.int64, copy=False)
    support_labels = labels[support_row_ids]
    for chunk_index, query_start in enumerate(query_offsets):
        query_stop = min(
            query_total, query_start + config.query_rows_per_task
        )
        task_id = f"{sample_id}-chunk-{chunk_index:05d}"
        query_row_ids = np.arange(
            support_total + query_start,
            support_total + query_stop,
            dtype=np.int64,
        )
        task_plan = TaskPlan(
            task_id=task_id,
            sample_id=sample_id,
            instance_id=instance_id,
            schema_id=schema_id,
            mechanism=TaskMechanism.RELATION_ATTRIBUTE,
            prediction_type=prediction_type,
            target_table_id=converted.anchor_table.table_id,
            source_table_id=converted.anchor_table.table_id,
            target_column_id=converted.anchor_label_column_id,
            split_strategy="relbench_train_val_support_test_query",
            seed=runtime.uint32_seed("task", chunk_index),
            row_cutoff_time_column_id=converted.anchor_time_column_id,
            masked_column_ids=(converted.anchor_label_column_id,),
            observation_rules=observation_rules,
            route_supervision=route_supervision,
            parameters=(
                ("relbench_chunk_index", float(chunk_index)),
                ("relbench_test_start", float(query_start)),
            ),
        )
        planned = PlannedTask(
            plan=task_plan,
            data=TaskData(
                support_row_ids=support_row_ids,
                support_labels=support_labels,
                query_row_ids=query_row_ids,
                query_labels=labels[query_row_ids],
            ),
        )
        validation = validate_task(schema, native_database, planned)
        blocking = tuple(
            issue for issue in validation.issues if issue.code != "query_classes"
        )
        if blocking:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in blocking
            )
            raise ValueError(f"converted task validation failed: {details}")
        task_target = (
            task_writer.task_directory / sample_id / task_id / "artifact.json"
        )
        task_path = task_writer.commit(
            sample_id=sample_id,
            instance_artifact=_relative_reference(
                task_target.parent, instance_path
            ),
            schema_artifact=_relative_reference(task_target.parent, schema_path),
            runtime=runtime.child("task", chunk_index).record(
                project_version="0.1.0",
                metadata={"source": "relbench"},
            ),
            task=planned,
            report=TaskValidationReport(task_id=task_id),
        )
        task_entries.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "artifact": task_path.relative_to(task_root).as_posix(),
                "instance_artifact": _relative_reference(
                    task_root, instance_path
                ),
                "schema_id": schema_id,
                "mechanism": TaskMechanism.RELATION_ATTRIBUTE.value,
                "prediction_type": prediction_type.value,
                "support_count": len(support_row_ids),
                "query_count": len(query_row_ids),
                "relbench_test_start": query_start,
                "relbench_test_stop": query_stop,
            }
        )
        query_chunks.append(
            {
                "task_id": task_id,
                "test_start": query_start,
                "test_stop": query_stop,
                "anchor_row_start": support_total + query_start,
                "anchor_row_stop": support_total + query_stop,
            }
        )
        if progress is not None:
            progress(chunk_index + 1, total_chunks, task_id)

    task_manifest = task_writer.write_manifest(
        configuration={
            **config.to_dict(),
            "relbench_task_type": task_type,
            "entity_table": entity_table_name,
            "entity_column": entity_column,
            "time_column": time_column,
            "target_column": target_column,
            "timestamp_origin_ns": timestamp_origin_ns,
        },
        database_count=1,
        entries=task_entries,
    )
    _write_json(
        metadata_path,
        {
            "format": "rdb-prior-relbench-import-v1",
            "dataset": config.dataset_name,
            "task": config.task_name,
            "relbench_task_type": task_type,
            "prediction_type": prediction_type.value,
            "class_values": (
                [
                    _json_scalar(value)
                    for value in np.unique(support_labels_all).tolist()
                ]
                if prediction_type is PredictionType.CLASSIFICATION
                else []
            ),
            "entity_table": entity_table_name,
            "entity_column": entity_column,
            "time_column": time_column,
            "target_column": target_column,
            "timestamp_origin_ns": timestamp_origin_ns,
            "sample_id": sample_id,
            "schema_id": schema_id,
            "instance_id": instance_id,
            "schema_manifest": _relative_reference(
                config.output_root, schema_manifest
            ),
            "instance_manifest": _relative_reference(
                config.output_root, instance_manifest
            ),
            "task_manifest": _relative_reference(
                config.output_root, task_manifest
            ),
            "split_rows": {
                "train": train_count,
                "validation": validation_count,
                "test": query_total,
            },
            "prediction_mapping": {
                "test_anchor_start": support_total,
                "test_row_count": query_total,
                "test_position_formula": (
                    "test_position = row_id - test_anchor_start"
                ),
                "query_chunks": query_chunks,
            },
            "tables": [
                {
                    "original_name": table.original_name,
                    "table_id": table.table.table_id,
                    "physical_name": table.table.name,
                    "original_pkey": table.original_pkey,
                    "time_column_id": table.time_column_id,
                    "columns": list(table.column_mapping),
                }
                for table in converted.tables
            ],
            "anchor_table": {
                "table_id": converted.anchor_table.table_id,
                "physical_name": converted.anchor_table.name,
                "time_column_id": converted.anchor_time_column_id,
                "label_column_id": converted.anchor_label_column_id,
                "entity_foreign_key_id": (
                    converted.anchor_entity_foreign_key.foreign_key_id
                ),
            },
            "dropped_relations": list(converted.dropped_relations),
            "configuration": config.to_dict(),
        },
    )
    return RelBenchImportResult(
        output_root=config.output_root,
        schema_manifest=schema_manifest,
        instance_manifest=instance_manifest,
        task_manifest=task_manifest,
        metadata_path=metadata_path,
        task_count=len(task_entries),
        support_row_count=len(support_row_ids),
        query_row_count=query_total,
    )


def _prediction_type(task: Any) -> tuple[PredictionType, str]:
    raw = getattr(task, "task_type", None)
    value = getattr(raw, "value", raw)
    if not isinstance(value, str) or value not in _SUPPORTED_TASK_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_TASK_TYPES))
        raise ValueError(
            f"unsupported RelBench task type {value!r}; supported: {supported}"
        )
    return _SUPPORTED_TASK_TYPES[value], value


def _required_task_attribute(task: Any, name: str) -> str:
    value = getattr(task, name, None)
    if not isinstance(value, str) or not value:
        raise ValueError(
            "relbench-import currently supports EntityTask-compatible tasks; "
            f"missing {name}"
        )
    return value


def _concat_split_frames(split_frames: Mapping[str, Any]):
    import pandas as pd

    return pd.concat(
        [split_frames[name] for name in ("train", "val", "test")],
        ignore_index=True,
    )


def _convert_database(
    database: Any,
    *,
    combined_task_frame: Any,
    entity_table_name: str,
    entity_column: str,
    anchor_times: np.ndarray,
    labels: np.ndarray,
    max_text_length: int,
    timestamp_origin_ns: int,
) -> _ConvertedDatabase:
    original_names = tuple(sorted(database.table_dict))
    table_ids = {
        name: f"T{index:04d}" for index, name in enumerate(original_names)
    }
    used_table_names: set[str] = set()
    physical_names = {
        name: _unique_sql_name(name, "table", used_table_names)
        for name in original_names
    }
    parent_keys = {
        name: (
            table.df[table.pkey_col]
            if table.pkey_col is not None
            else np.arange(len(table.df), dtype=np.int64)
        )
        for name, table in database.table_dict.items()
    }
    relation_specs: list[tuple[str, str, str]] = []
    dropped_relations: list[dict[str, str]] = []
    for child_name in original_names:
        rel_table = database.table_dict[child_name]
        for child_column, parent_name in sorted(
            rel_table.fkey_col_to_pkey_table.items()
        ):
            if parent_name not in database.table_dict:
                raise ValueError(
                    f"RelBench FK {child_name}.{child_column} references "
                    f"unknown table {parent_name!r}"
                )
            if parent_name == child_name:
                dropped_relations.append(
                    {
                        "child_table": child_name,
                        "child_column": child_column,
                        "parent_table": parent_name,
                        "reason": "self-references are unsupported by PhysicalSchema",
                    }
                )
                continue
            relation_specs.append((child_name, child_column, parent_name))

    relation_ids = {
        spec: f"FK{index:05d}" for index, spec in enumerate(relation_specs)
    }
    converted_tables: list[_ConvertedTable] = []
    fk_arrays: dict[tuple[str, str, str], np.ndarray] = {}
    fk_column_ids: dict[tuple[str, str, str], str] = {}
    time_column_ids: dict[str, str | None] = {}
    for original_name in original_names:
        rel_table = database.table_dict[original_name]
        frame = rel_table.df.reset_index(drop=True)
        table_id = table_ids[original_name]
        used_columns: set[str] = set()
        columns: list[PhysicalColumn] = []
        arrays: dict[str, np.ndarray] = {}
        mapping: list[dict[str, Any]] = []
        primary_id = f"{table_id}_PK"
        columns.append(
            PhysicalColumn(
                column_id=primary_id,
                name=_unique_sql_name("row_id", "column", used_columns),
                data_type=PhysicalDataType.BIGINT,
                kind=ColumnKind.PRIMARY_KEY,
                ordinal=0,
                nullable=False,
                unique=True,
            )
        )
        arrays[primary_id] = np.arange(len(frame), dtype=np.int64)
        mapping.append(
            {
                "original_name": rel_table.pkey_col,
                "column_id": primary_id,
                "kind": ColumnKind.PRIMARY_KEY.value,
                "synthetic": rel_table.pkey_col is None,
            }
        )
        child_relations = [
            spec for spec in relation_specs if spec[0] == original_name
        ]
        for relation_index, spec in enumerate(child_relations):
            _child_name, original_column, parent_name = spec
            column_id = f"{table_id}_FK{relation_index:03d}"
            values = _map_foreign_keys(
                frame[original_column], parent_keys[parent_name]
            )
            fk_arrays[spec] = values
            fk_column_ids[spec] = column_id
            nullable = bool(np.any(values < 0))
            columns.append(
                PhysicalColumn(
                    column_id=column_id,
                    name=_unique_sql_name(
                        f"fk_{original_column}", "fk", used_columns
                    ),
                    data_type=PhysicalDataType.BIGINT,
                    kind=ColumnKind.FOREIGN_KEY,
                    ordinal=len(columns),
                    nullable=nullable,
                    unique=False,
                )
            )
            arrays[column_id] = values
            mapping.append(
                {
                    "original_name": original_column,
                    "column_id": column_id,
                    "kind": ColumnKind.FOREIGN_KEY.value,
                    "parent_table": parent_name,
                }
            )

        key_columns = {
            value
            for value in (
                rel_table.pkey_col,
                *rel_table.fkey_col_to_pkey_table.keys(),
            )
            if value is not None
        }
        data_columns = [
            name for name in frame.columns if name not in key_columns
        ]
        if (
            rel_table.time_col is not None
            and rel_table.time_col not in data_columns
        ):
            data_columns.append(rel_table.time_col)
        time_id: str | None = None
        for data_index, original_column in enumerate(data_columns):
            column_id = f"{table_id}_C{data_index:03d}"
            is_time = original_column == rel_table.time_col
            if is_time:
                data_type = PhysicalDataType.TIMESTAMP
                values = _timestamp_array(
                    frame[original_column], origin_ns=timestamp_origin_ns
                )
                kind = ColumnKind.TIME
                time_id = column_id
            else:
                data_type, values = _feature_array(
                    frame[original_column],
                    max_text_length=max_text_length,
                )
                kind = ColumnKind.FEATURE
            nullable = (
                bool(np.any(values < 0)) if is_time else _has_missing(values)
            )
            columns.append(
                PhysicalColumn(
                    column_id=column_id,
                    name=_unique_sql_name(
                        original_column, "column", used_columns
                    ),
                    data_type=data_type,
                    kind=kind,
                    ordinal=len(columns),
                    nullable=nullable,
                    unique=False,
                )
            )
            arrays[column_id] = values
            mapping.append(
                {
                    "original_name": original_column,
                    "column_id": column_id,
                    "kind": kind.value,
                }
            )
        time_column_ids[original_name] = time_id
        role = _table_role(rel_table)
        physical_table = PhysicalTable(
            table_id=table_id,
            name=physical_names[original_name],
            role=role,
            rank=0,
            columns=tuple(columns),
        )
        converted_tables.append(
            _ConvertedTable(
                original_name=original_name,
                table=physical_table,
                data=TableData(table_id=table_id, columns=arrays),
                original_pkey=rel_table.pkey_col,
                pkey_values=parent_keys[original_name],
                time_column_id=time_id,
                column_mapping=tuple(mapping),
            )
        )

    foreign_keys: list[PhysicalForeignKey] = []
    table_map = {table.original_name: table for table in converted_tables}
    for spec in relation_specs:
        child_name, _original_column, parent_name = spec
        values = fk_arrays[spec]
        valid = values[values >= 0]
        cardinality = (
            Cardinality.ONE_TO_ONE
            if len(np.unique(valid)) == len(valid)
            else Cardinality.ONE_TO_MANY
        )
        foreign_key_id = relation_ids[spec]
        foreign_keys.append(
            PhysicalForeignKey(
                foreign_key_id=foreign_key_id,
                name=f"fk_{len(foreign_keys):05d}",
                parent_table_id=table_map[parent_name].table.table_id,
                parent_column_id=table_map[parent_name].table.primary_key.column_id,
                child_table_id=table_map[child_name].table.table_id,
                child_column_id=fk_column_ids[spec],
                cardinality=cardinality,
                optionality=(
                    Optionality.OPTIONAL
                    if np.any(values < 0)
                    else Optionality.REQUIRED
                ),
                identity_dependency=IdentityDependency.INDEPENDENT,
                relation_strategy="relbench_import",
            )
        )

    entity_table = table_map[entity_table_name]
    anchor_table_id = f"T{len(original_names):04d}"
    anchor_pk_id = f"{anchor_table_id}_PK"
    anchor_fk_id = f"{anchor_table_id}_FK000"
    anchor_time_id = f"{anchor_table_id}_C000"
    anchor_label_id = f"{anchor_table_id}_C001"
    anchor_entity_values = _map_foreign_keys(
        combined_task_frame[entity_column], entity_table.pkey_values
    )
    if np.any(anchor_entity_values < 0):
        missing = int(np.count_nonzero(anchor_entity_values < 0))
        raise ValueError(
            f"RelBench task contains {missing} dangling entity references"
        )
    label_type = _array_physical_type(labels)
    anchor_columns = (
        PhysicalColumn(
            column_id=anchor_pk_id,
            name="anchor_id",
            data_type=PhysicalDataType.BIGINT,
            kind=ColumnKind.PRIMARY_KEY,
            ordinal=0,
            nullable=False,
            unique=True,
        ),
        PhysicalColumn(
            column_id=anchor_fk_id,
            name="entity_id",
            data_type=PhysicalDataType.BIGINT,
            kind=ColumnKind.FOREIGN_KEY,
            ordinal=1,
            nullable=False,
            unique=False,
        ),
        PhysicalColumn(
            column_id=anchor_time_id,
            name="prediction_time",
            data_type=PhysicalDataType.TIMESTAMP,
            kind=ColumnKind.TIME,
            ordinal=2,
            nullable=False,
            unique=False,
        ),
        PhysicalColumn(
            column_id=anchor_label_id,
            name="label",
            data_type=label_type,
            kind=ColumnKind.FEATURE,
            ordinal=3,
            nullable=False,
            unique=False,
        ),
    )
    anchor_table = PhysicalTable(
        table_id=anchor_table_id,
        name=_unique_sql_name("relbench_anchor", "anchor", used_table_names),
        role=TableRole.EVENT,
        rank=0,
        columns=anchor_columns,
    )
    anchor_data = TableData(
        table_id=anchor_table_id,
        columns={
            anchor_pk_id: np.arange(len(labels), dtype=np.int64),
            anchor_fk_id: anchor_entity_values,
            anchor_time_id: anchor_times,
            anchor_label_id: labels,
        },
    )
    anchor_relation_id = f"FK{len(relation_specs):05d}"
    anchor_relation = PhysicalForeignKey(
        foreign_key_id=anchor_relation_id,
        name=f"fk_{len(foreign_keys):05d}",
        parent_table_id=entity_table.table.table_id,
        parent_column_id=entity_table.table.primary_key.column_id,
        child_table_id=anchor_table_id,
        child_column_id=anchor_fk_id,
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.INDEPENDENT,
        relation_strategy="relbench_prediction_anchor",
    )
    return _ConvertedDatabase(
        tables=tuple(converted_tables),
        foreign_keys=tuple(foreign_keys),
        anchor_table=anchor_table,
        anchor_data=anchor_data,
        anchor_entity_foreign_key=anchor_relation,
        anchor_time_column_id=anchor_time_id,
        anchor_label_column_id=anchor_label_id,
        dropped_relations=tuple(dropped_relations),
    )


def _blueprint_and_trace(
    schema: PhysicalSchema,
) -> tuple[SchemaBlueprint, CompilationTrace]:
    node_ids = {
        table.table_id: f"N{index:04d}"
        for index, table in enumerate(schema.tables)
    }
    edge_ids = {
        foreign_key.foreign_key_id: f"E{index:05d}"
        for index, foreign_key in enumerate(schema.foreign_keys)
    }
    blueprint = SchemaBlueprint(
        blueprint_id=schema.blueprint_id,
        nodes=tuple(
            BlueprintNode(
                node_id=node_ids[table.table_id],
                role=table.role,
                rank=table.rank,
            )
            for table in schema.tables
        ),
        edges=tuple(
            BlueprintEdge(
                edge_id=edge_ids[foreign_key.foreign_key_id],
                parent_node_id=node_ids[foreign_key.parent_table_id],
                child_node_id=node_ids[foreign_key.child_table_id],
            )
            for foreign_key in schema.foreign_keys
        ),
    )
    trace = CompilationTrace(
        blueprint_id=schema.blueprint_id,
        schema_id=schema.schema_id,
        node_to_tables=tuple(
            (node_ids[table.table_id], (table.table_id,))
            for table in schema.tables
        ),
        edge_to_foreign_keys=tuple(
            (
                edge_ids[foreign_key.foreign_key_id],
                (foreign_key.foreign_key_id,),
            )
            for foreign_key in schema.foreign_keys
        ),
    )
    return blueprint, trace


def _instance_plan(
    schema: PhysicalSchema,
    table_data: tuple[TableData, ...],
    *,
    sample_id: str,
    plan_id: str,
    runtime: RuntimeContext,
) -> InstancePlan:
    data_map = {table.table_id: table for table in table_data}
    tables = tuple(
        TableMechanismPlan(
            table_id=table.table_id,
            role=table.role,
            population=PopulationPlan(
                strategy="relbench_import",
                row_count=max(1, data_map[table.table_id].row_count),
            ),
            latent_dimension=1,
            feature_family=FeatureSCMFamily.EXOGENOUS,
            temporal_family=(
                TemporalFamily.TIME_LAGGED
                if any(
                    column.kind is ColumnKind.TIME
                    for column in table.columns
                )
                else TemporalFamily.NONE
            ),
            latent_seed=runtime.uint63_seed("table", table.table_id, "latent"),
            feature_seed=runtime.uint63_seed(
                "table", table.table_id, "feature"
            ),
            temporal_seed=runtime.uint63_seed(
                "table", table.table_id, "temporal"
            ),
        )
        for table in schema.tables
    )
    relations = tuple(
        RelationMechanismPlan(
            relation_group_id=f"RG{index:05d}",
            foreign_key_ids=(foreign_key.foreign_key_id,),
            parent_table_ids=(foreign_key.parent_table_id,),
            child_table_id=foreign_key.child_table_id,
            family="relbench_import",
            optional_rates=(0.0,),
            seed=runtime.uint63_seed("relation", foreign_key.foreign_key_id),
        )
        for index, foreign_key in enumerate(schema.foreign_keys)
    )
    return InstancePlan(
        plan_id=plan_id,
        sample_id=sample_id,
        schema_id=schema.schema_id,
        blueprint_id=schema.blueprint_id,
        global_seed=runtime.uint63_seed("instance_plan"),
        generation_order=tuple(table.table_id for table in schema.tables),
        tables=tables,
        relations=relations,
    )


def _route_supervision(
    converted: _ConvertedDatabase,
) -> tuple[RoutePathLabel, ...]:
    anchor_fk = converted.anchor_entity_foreign_key
    labels = [
        RoutePathLabel(
            foreign_key_ids=(anchor_fk.foreign_key_id,),
            role=RouteRole.REQUIRED,
        )
    ]
    entity_id = anchor_fk.parent_table_id
    for foreign_key in converted.foreign_keys:
        if entity_id in {
            foreign_key.parent_table_id,
            foreign_key.child_table_id,
        }:
            labels.append(
                RoutePathLabel(
                    foreign_key_ids=(
                        anchor_fk.foreign_key_id,
                        foreign_key.foreign_key_id,
                    ),
                    role=RouteRole.OPTIONAL,
                )
            )
    return tuple(labels)


def _sample_support_indices(
    labels: np.ndarray,
    *,
    count: int,
    prediction_type: PredictionType,
    seed: int,
) -> np.ndarray:
    if count < 1:
        raise ValueError("RelBench support split must be non-empty")
    if count >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    mandatory: list[int] = []
    if prediction_type is PredictionType.CLASSIFICATION:
        classes = np.unique(labels)
        if len(classes) > count:
            raise ValueError(
                f"support_rows={count} cannot cover {len(classes)} classes"
            )
        for value in classes:
            candidates = np.flatnonzero(labels == value)
            mandatory.append(
                int(candidates[int(rng.integers(0, len(candidates)))])
            )
    remaining = np.setdiff1d(
        np.arange(len(labels), dtype=np.int64),
        np.asarray(mandatory, dtype=np.int64),
        assume_unique=False,
    )
    needed = count - len(mandatory)
    sampled = rng.choice(remaining, needed, replace=False).astype(np.int64)
    return np.sort(
        np.concatenate((np.asarray(mandatory, dtype=np.int64), sampled))
    )


def _map_foreign_keys(values: Any, parent_values: Any) -> np.ndarray:
    import pandas as pd

    parent_index = pd.Index(parent_values)
    if not parent_index.is_unique:
        raise ValueError("RelBench parent primary keys must be unique")
    mapped = parent_index.get_indexer(pd.Index(values))
    return mapped.astype(np.int64, copy=False)


def _timestamp_origin_ns(database: Any, *, anchor_series: Any) -> int:
    minimum = 0
    maximum = 0
    timestamp_series = [anchor_series]
    for table in database.table_dict.values():
        if table.time_col is not None and table.time_col in table.df.columns:
            timestamp_series.append(table.df[table.time_col])
    for series in timestamp_series:
        raw, missing = _raw_timestamp_array(series)
        valid = raw[~missing]
        if len(valid):
            minimum = min(minimum, int(np.min(valid)))
            maximum = max(maximum, int(np.max(valid)))
    origin = min(0, minimum)
    if maximum - origin > np.iinfo(np.int64).max:
        raise ValueError("RelBench timestamp range exceeds int64 capacity")
    return origin


def _raw_timestamp_array(series: Any) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    values = pd.to_datetime(series, errors="coerce", utc=True)
    missing = np.asarray(pd.isna(values), dtype=np.bool_)
    result = values.astype("int64").to_numpy(dtype=np.int64, copy=True)
    return result, missing


def _timestamp_array(series: Any, *, origin_ns: int = 0) -> np.ndarray:
    result, missing = _raw_timestamp_array(series)
    valid = ~missing
    if np.any(valid):
        minimum = int(np.min(result[valid]))
        maximum = int(np.max(result[valid]))
        if minimum < origin_ns:
            raise ValueError("timestamp origin is after a valid timestamp")
        if maximum - origin_ns > np.iinfo(np.int64).max:
            raise ValueError("timestamp range exceeds int64 capacity")
        result[valid] -= np.int64(origin_ns)
    result[missing] = -1
    return result


def _label_array(
    series: Any,
    *,
    prediction_type: PredictionType,
    max_text_length: int,
) -> np.ndarray:
    import pandas as pd

    if prediction_type is PredictionType.REGRESSION:
        values = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("RelBench regression labels must be finite")
        return values
    if bool(series.isna().any()):
        raise ValueError("RelBench classification labels cannot be missing")
    _data_type, values = _feature_array(
        series,
        max_text_length=max_text_length,
    )
    if values.dtype.kind == "f":
        if np.any(~np.isfinite(values)):
            raise ValueError("RelBench classification labels must be finite")
    return values


def _feature_array(
    series: Any,
    *,
    max_text_length: int,
) -> tuple[PhysicalDataType, np.ndarray]:
    import pandas as pd
    from pandas.api import types as ptypes

    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        if bool(series.isna().any()):
            return (
                PhysicalDataType.BOOLEAN,
                series.astype("Float64").to_numpy(
                    dtype=np.float64, na_value=np.nan
                ),
            )
        return PhysicalDataType.BOOLEAN, series.to_numpy(dtype=np.bool_)
    if ptypes.is_integer_dtype(dtype):
        if bool(series.isna().any()):
            return (
                PhysicalDataType.BIGINT,
                series.to_numpy(dtype=np.float64, na_value=np.nan),
            )
        return PhysicalDataType.BIGINT, series.to_numpy(dtype=np.int64)
    if ptypes.is_float_dtype(dtype) or ptypes.is_numeric_dtype(dtype):
        return (
            PhysicalDataType.DOUBLE,
            pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64),
        )
    if ptypes.is_datetime64_any_dtype(dtype):
        return PhysicalDataType.TIMESTAMP, _timestamp_array(series)
    encoded: list[str] = []
    for value in series.tolist():
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        elif isinstance(value, (list, tuple, dict, np.ndarray)):
            serializable = value.tolist() if isinstance(value, np.ndarray) else value
            text = json.dumps(
                serializable,
                ensure_ascii=False,
                sort_keys=isinstance(serializable, dict),
                default=str,
            )
        elif bool(pd.isna(value)):
            text = ""
        else:
            text = str(value)
        encoded.append(text[:max_text_length])
    width = max(1, max((len(value) for value in encoded), default=0))
    return PhysicalDataType.TEXT, np.asarray(encoded, dtype=f"<U{width}")


def _array_physical_type(values: np.ndarray) -> PhysicalDataType:
    if values.dtype.kind in {"U", "S"}:
        return PhysicalDataType.TEXT
    if values.dtype.kind == "b":
        return PhysicalDataType.BOOLEAN
    if values.dtype.kind in {"i", "u"}:
        return PhysicalDataType.BIGINT
    return PhysicalDataType.DOUBLE


def _has_missing(values: np.ndarray) -> bool:
    if values.dtype.kind == "f":
        return bool(np.any(~np.isfinite(values)))
    if values.dtype.kind in {"U", "S"}:
        return bool(np.any(values == ""))
    return False


def _table_role(table: Any) -> TableRole:
    if len(table.fkey_col_to_pkey_table) >= 2:
        return TableRole.BRIDGE
    if table.time_col is not None:
        return TableRole.EVENT
    if table.pkey_col is not None:
        return TableRole.ENTITY
    return TableRole.DETAIL


def _unique_sql_name(value: Any, prefix: str, used: set[str]) -> str:
    base = _NON_SQL.sub("_", str(value).lower()).strip("_")
    if not base or not base[0].isalpha():
        base = f"{prefix}_{base}" if base else prefix
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _artifact_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("dataset/task names do not form a safe artifact ID")
    return result


def _relative_reference(start: Path, target: Path) -> str:
    return os.path.relpath(target, start=start).replace(os.sep, "/")


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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
    temporary.replace(path)


__all__ = [
    "RelBenchImportConfig",
    "RelBenchImportResult",
    "convert_relbench_objects",
    "import_relbench",
]
