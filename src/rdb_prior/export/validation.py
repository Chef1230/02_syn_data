"""Structural and leakage checks for converted RDBPFN datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .model import RDBPFNDataset


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportValidationReport:
    issues: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_rdbpfn_dataset(dataset: RDBPFNDataset) -> ExportValidationReport:
    issues: list[str] = []
    metadata = dataset.metadata
    table_schemas = metadata.get("tables")
    task_schemas = metadata.get("tasks")
    if metadata.get("dataset_name") != dataset.dataset_name:
        issues.append("dataset_name does not match export identity")
    if not isinstance(table_schemas, list) or not isinstance(task_schemas, list):
        return ExportValidationReport(
            issues=("metadata tables/tasks must be lists",)
        )
    if len(task_schemas) != 1 or not isinstance(task_schemas[0], Mapping):
        return ExportValidationReport(
            issues=("export must contain exactly one task",)
        )

    schemas = {
        item.get("name"): item
        for item in table_schemas
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if set(schemas) != set(dataset.tables):
        issues.append("metadata table names do not match table payloads")
    for table_name, columns in dataset.tables.items():
        schema = schemas.get(table_name)
        if schema is None:
            continue
        column_schemas = schema.get("columns")
        if not isinstance(column_schemas, list):
            issues.append(f"{table_name}: columns metadata is not a list")
            continue
        names = {
            item.get("name")
            for item in column_schemas
            if isinstance(item, Mapping)
        }
        if names != set(columns):
            issues.append(f"{table_name}: metadata columns do not match payload")

    for table_name, column_name in dataset.masked_columns:
        if column_name in dataset.tables.get(table_name, {}):
            issues.append(f"masked target leaked into {table_name}.{column_name}")

    task_schema = task_schemas[0]
    if task_schema.get("name") != dataset.task_name:
        issues.append("task metadata name does not match export identity")
    task_columns = task_schema.get("columns")
    if not isinstance(task_columns, list):
        return ExportValidationReport(
            issues=tuple(issues + ["task columns must be a list"])
        )
    task_column_names = {
        item.get("name") for item in task_columns if isinstance(item, Mapping)
    }
    for split_name, columns in dataset.splits.items():
        if set(columns) != task_column_names:
            issues.append(f"{split_name}: split columns do not match metadata")
    target_column = task_schema.get("target_column")
    if target_column not in task_column_names:
        issues.append("target column is absent from task splits")

    target_table = task_schema.get("target_table")
    if target_table not in dataset.tables:
        issues.append("target table is absent from exported tables")
    else:
        key_columns = [
            item["name"]
            for item in schemas[target_table]["columns"]
            if item.get("dtype") == "primary_key"
        ]
        if len(key_columns) != 1:
            issues.append("target table must contain exactly one primary key")
        else:
            key_name = key_columns[0]
            target_keys = set(dataset.tables[target_table][key_name].tolist())
            for split_name, columns in dataset.splits.items():
                if key_name not in columns:
                    issues.append(f"{split_name}: target primary key is absent")
                elif not set(columns[key_name].tolist()) <= target_keys:
                    issues.append(f"{split_name}: task keys are outside target table")

    for table_name, schema in schemas.items():
        for column in schema.get("columns", []):
            if column.get("dtype") != "foreign_key":
                continue
            link_to = column.get("link_to")
            if not isinstance(link_to, str) or "." not in link_to:
                issues.append(f"{table_name}.{column.get('name')}: invalid link_to")
                continue
            parent_table, parent_column = link_to.split(".", 1)
            if parent_table not in dataset.tables or parent_column not in dataset.tables[parent_table]:
                issues.append(f"{table_name}.{column.get('name')}: missing parent")
                continue
            parent_values = set(dataset.tables[parent_table][parent_column].tolist())
            child_values = dataset.tables[table_name][column["name"]]
            valid_values = {
                item
                for item in child_values.tolist()
                if item is not None
                and not (isinstance(item, float) and np.isnan(item))
            }
            if not valid_values <= parent_values:
                issues.append(f"{table_name}.{column['name']}: orphan foreign key")
    return ExportValidationReport(issues=tuple(issues))


__all__ = ["ExportValidationReport", "validate_rdbpfn_dataset"]
