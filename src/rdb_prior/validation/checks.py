"""Hard checks for executable plans and generated database instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalDataType,
    PhysicalSchema,
)
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.schema.spec import Optionality


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceValidationIssue:
    code: str
    message: str
    table_id: str | None = None
    column_id: str | None = None
    foreign_key_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        for name in ("table_id", "column_id", "foreign_key_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceValidationIssue:
        return cls(
            code=data["code"],
            message=data["message"],
            table_id=data.get("table_id"),
            column_id=data.get("column_id"),
            foreign_key_id=data.get("foreign_key_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceValidationReport:
    schema_id: str
    plan_id: str
    issues: tuple[InstanceValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "plan_id": self.plan_id,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceValidationReport:
        return cls(
            schema_id=data["schema_id"],
            plan_id=data["plan_id"],
            issues=tuple(
                InstanceValidationIssue.from_dict(item)
                for item in data.get("issues", ())
            ),
        )


def validate_instance_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
) -> InstanceValidationReport:
    issues: list[InstanceValidationIssue] = []
    if schema.schema_id != plan.schema_id:
        issues.append(_issue("plan_schema_mismatch", "plan schema ID differs"))
    schema_tables = {table.table_id: table for table in schema.tables}
    plan_tables = {table.table_id: table for table in plan.tables}
    if set(schema_tables) != set(plan_tables):
        issues.append(_issue("plan_table_coverage", "plan table set differs"))
    for table_id in set(schema_tables) & set(plan_tables):
        if schema_tables[table_id].role is not plan_tables[table_id].role:
            issues.append(
                _issue(
                    "plan_role_mismatch",
                    "planned role differs from physical role",
                    table_id=table_id,
                )
            )

    positions = {table_id: index for index, table_id in enumerate(plan.generation_order)}
    expected_fks = {foreign_key.foreign_key_id for foreign_key in schema.foreign_keys}
    planned_fks: list[str] = []
    schema_fk_map = {
        foreign_key.foreign_key_id: foreign_key
        for foreign_key in schema.foreign_keys
    }
    for relation in plan.relations:
        planned_fks.extend(relation.foreign_key_ids)
        for fk_id, parent_id in zip(
            relation.foreign_key_ids,
            relation.parent_table_ids,
            strict=True,
        ):
            foreign_key = schema_fk_map.get(fk_id)
            if foreign_key is None:
                continue
            if (
                foreign_key.parent_table_id != parent_id
                or foreign_key.child_table_id != relation.child_table_id
            ):
                issues.append(
                    _issue(
                        "plan_relation_binding",
                        "relation group does not match physical FK",
                        foreign_key_id=fk_id,
                    )
                )
    if set(planned_fks) != expected_fks or len(planned_fks) != len(expected_fks):
        issues.append(
            _issue(
                "plan_relation_coverage",
                "every physical FK must occur in exactly one relation group",
            )
        )
    for foreign_key in schema.foreign_keys:
        if positions.get(foreign_key.parent_table_id, -1) >= positions.get(
            foreign_key.child_table_id, -1
        ):
            issues.append(
                _issue(
                    "plan_generation_order",
                    "parent table must be generated before child table",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )
    return InstanceValidationReport(
        schema_id=schema.schema_id,
        plan_id=plan.plan_id,
        issues=tuple(issues),
    )


def validate_database_instance(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
) -> InstanceValidationReport:
    plan_report = validate_instance_plan(schema, plan)
    issues = list(plan_report.issues)
    if database.schema_id != schema.schema_id or database.plan_id != plan.plan_id:
        issues.append(
            _issue("database_identity", "database IDs do not match schema and plan")
        )
    database_tables = {table.table_id: table for table in database.tables}
    schema_table_ids = {table.table_id for table in schema.tables}
    if set(database_tables) != schema_table_ids:
        issues.append(_issue("database_table_coverage", "database table set differs"))

    for table in schema.tables:
        data = database_tables.get(table.table_id)
        if data is None:
            continue
        expected_rows = plan.table(table.table_id).population.row_count
        if data.row_count != expected_rows:
            issues.append(
                _issue(
                    "row_count_mismatch",
                    f"expected {expected_rows} rows, found {data.row_count}",
                    table_id=table.table_id,
                )
            )
        expected_columns = {column.column_id for column in table.columns}
        if set(data.columns) != expected_columns:
            issues.append(
                _issue(
                    "column_coverage",
                    "generated columns differ from physical schema",
                    table_id=table.table_id,
                )
            )
            continue
        for column in table.columns:
            values = data.column(column.column_id)
            issues.extend(_validate_column(table.table_id, column, values))

    for foreign_key in schema.foreign_keys:
        child = database_tables.get(foreign_key.child_table_id)
        parent = database_tables.get(foreign_key.parent_table_id)
        if child is None or parent is None:
            continue
        values = child.column(foreign_key.child_column_id)
        invalid = (values < -1) | (values >= parent.row_count)
        if np.any(invalid):
            issues.append(
                _issue(
                    "foreign_key_bounds",
                    "foreign key contains an invalid parent row index",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )
        if foreign_key.optionality is Optionality.REQUIRED and np.any(values < 0):
            issues.append(
                _issue(
                    "required_foreign_key_missing",
                    "required foreign key contains missing values",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )

    for relation in plan.relations:
        if relation.family != "affinity_bridge" or any(relation.optional_rates):
            continue
        child = database_tables.get(relation.child_table_id)
        if child is None:
            continue
        fk_map = {
            foreign_key.foreign_key_id: foreign_key
            for foreign_key in schema.foreign_keys
        }
        matrix = np.column_stack(
            [child.column(fk_map[fk_id].child_column_id) for fk_id in relation.foreign_key_ids]
        )
        combinations = int(
            np.prod(
                [database_tables[parent_id].row_count for parent_id in relation.parent_table_ids]
            )
        )
        if len(matrix) <= combinations and len(np.unique(matrix, axis=0)) != len(matrix):
            issues.append(
                _issue(
                    "bridge_duplicate_tuple",
                    "bridge relation contains duplicate parent tuples",
                    table_id=relation.child_table_id,
                )
            )
    return InstanceValidationReport(
        schema_id=schema.schema_id,
        plan_id=plan.plan_id,
        issues=tuple(issues),
    )


def _validate_column(table_id: str, column: Any, values: np.ndarray) -> list[InstanceValidationIssue]:
    issues: list[InstanceValidationIssue] = []
    if values.dtype == object:
        issues.append(
            _issue(
                "object_dtype",
                "object arrays cannot be persisted safely",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
        return issues
    if column.kind is ColumnKind.PRIMARY_KEY:
        expected = np.arange(len(values), dtype=np.int64)
        if not np.array_equal(values, expected):
            issues.append(
                _issue(
                    "primary_key_sequence",
                    "primary key must be a contiguous row index",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
        return issues
    if column.kind is ColumnKind.FOREIGN_KEY:
        if values.dtype.kind not in {"i", "u"}:
            issues.append(
                _issue(
                    "foreign_key_dtype",
                    "foreign key must use an integer dtype",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
        return issues

    if values.dtype.kind == "f":
        present = values[~np.isnan(values)]
        if np.any(~np.isfinite(present)):
            issues.append(
                _issue(
                    "non_finite_feature",
                    "feature contains infinite values",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
    elif values.dtype.kind in {"U", "S"}:
        present = values[values != ""]
    else:
        present = values
    if len(present) == 0:
        issues.append(
            _issue(
                "all_missing_feature",
                "feature contains no observed values",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    if column.unique and len(np.unique(present)) != len(present):
        issues.append(
            _issue(
                "unique_feature_duplicate",
                "unique column contains duplicate observed values",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    if column.data_type is PhysicalDataType.TEXT and values.dtype.kind not in {"U", "S"}:
        issues.append(
            _issue(
                "text_dtype",
                "text column must use a fixed-width string dtype",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    return issues


def _issue(
    code: str,
    message: str,
    *,
    table_id: str | None = None,
    column_id: str | None = None,
    foreign_key_id: str | None = None,
) -> InstanceValidationIssue:
    return InstanceValidationIssue(
        code=code,
        message=message,
        table_id=table_id,
        column_id=column_id,
        foreign_key_id=foreign_key_id,
    )


__all__ = [
    "InstanceValidationIssue",
    "InstanceValidationReport",
    "validate_instance_plan",
    "validate_database_instance",
]
