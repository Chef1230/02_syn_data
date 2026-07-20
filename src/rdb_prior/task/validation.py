"""Hard correctness and leakage checks for generated task artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.mechanisms import future_event_labels
from rdb_prior.task.model import (
    PlannedTask,
    PredictionType,
    RoutePathLabel,
    TaskMechanism,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskValidationIssue:
        return cls(code=data["code"], message=data["message"])


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationReport:
    task_id: str
    issues: tuple[TaskValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskValidationReport:
        return cls(
            task_id=data["task_id"],
            issues=tuple(
                TaskValidationIssue.from_dict(item)
                for item in data.get("issues", ())
            ),
        )


def validate_task(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> TaskValidationReport:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    if plan.schema_id != schema.schema_id or plan.schema_id != database.schema_id:
        issues.append(_issue("schema_identity", "task schema identity mismatch"))
    if plan.instance_id != database.instance_id:
        issues.append(_issue("instance_identity", "task instance identity mismatch"))
    try:
        target_data = database.table(plan.target_table_id)
        schema.table(plan.source_table_id)
    except KeyError:
        issues.append(_issue("table_reference", "task references an unknown table"))
        return TaskValidationReport(task_id=plan.task_id, issues=tuple(issues))

    all_rows = np.concatenate([data.support_row_ids, data.query_row_ids])
    if np.any(all_rows >= target_data.row_count):
        issues.append(_issue("row_bounds", "task row ID is outside target table"))
    for labels in (data.support_labels, data.query_labels):
        if labels.dtype.kind == "f" and np.any(~np.isfinite(labels)):
            issues.append(_issue("non_finite_label", "task labels are not finite"))
        if labels.dtype.kind in {"U", "S"} and np.any(labels == ""):
            issues.append(_issue("empty_label", "task labels contain empty strings"))
    if plan.prediction_type is PredictionType.CLASSIFICATION:
        if len(np.unique(data.support_labels)) < 2:
            issues.append(_issue("support_classes", "support has fewer than two classes"))
        if len(np.unique(data.query_labels)) < 2:
            issues.append(_issue("query_classes", "query has fewer than two classes"))

    issues.extend(_validate_route_supervision(schema, plan.target_table_id, plan.route_supervision))

    if plan.mechanism is TaskMechanism.RELATION_ATTRIBUTE:
        issues.extend(_validate_relation_attribute(schema, database, task))
    elif plan.mechanism is TaskMechanism.FUTURE_EVENT_EXISTENCE:
        issues.extend(_validate_future_event(schema, database, task))
    else:  # pragma: no cover - enum protects construction.
        issues.append(_issue("unknown_mechanism", "unsupported task mechanism"))
    return TaskValidationReport(task_id=plan.task_id, issues=tuple(issues))


def _validate_relation_attribute(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    if plan.target_table_id != plan.source_table_id:
        issues.append(
            _issue("attribute_source", "attribute target and source table differ")
        )
        return issues
    try:
        column = schema.table(plan.target_table_id).column(
            plan.target_column_id or ""
        )
    except KeyError:
        return [_issue("target_column", "target column does not exist")]
    if column.kind is not ColumnKind.FEATURE:
        issues.append(_issue("target_kind", "attribute target is not a feature"))
    if plan.target_column_id not in plan.masked_column_ids:
        issues.append(_issue("target_leakage", "query target column is not masked"))
    values = database.table(plan.target_table_id).column(column.column_id)
    if not _arrays_equal(
        data.support_labels,
        values[data.support_row_ids],
    ) or not _arrays_equal(
        data.query_labels,
        values[data.query_row_ids],
    ):
        issues.append(_issue("attribute_labels", "labels differ from target column"))
    return issues


def _validate_future_event(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    expected = future_event_labels(schema, database, plan)
    if not np.array_equal(data.support_labels, expected[data.support_row_ids]):
        issues.append(_issue("future_support_labels", "support labels are incorrect"))
    if not np.array_equal(data.query_labels, expected[data.query_row_ids]):
        issues.append(_issue("future_query_labels", "query labels are incorrect"))
    source_rules = [
        rule
        for rule in plan.observation_rules
        if rule.table_id == plan.source_table_id
        and rule.time_column_id == plan.time_column_id
    ]
    if len(source_rules) != 1 or source_rules[0].max_timestamp != plan.cutoff_time:
        issues.append(
            _issue(
                "future_visibility",
                "source event rows are not cut off at label cutoff",
            )
        )
    expected_rules = {
        (table.table_id, column.column_id)
        for table in schema.tables
        if table.role is TableRole.EVENT
        for column in table.columns
        if column.kind is ColumnKind.TIME
    }
    actual_rules = {
        (rule.table_id, rule.time_column_id)
        for rule in plan.observation_rules
        if rule.max_timestamp == plan.cutoff_time
    }
    if actual_rules != expected_rules:
        issues.append(
            _issue(
                "global_future_visibility",
                "every Event table must use the common task cutoff",
            )
        )
    return issues


def _validate_route_supervision(
    schema: PhysicalSchema,
    target_table_id: str,
    labels: tuple[RoutePathLabel, ...],
) -> list[TaskValidationIssue]:
    issues: list[TaskValidationIssue] = []
    foreign_keys = {
        foreign_key.foreign_key_id: foreign_key
        for foreign_key in schema.foreign_keys
    }
    for label in labels:
        current = target_table_id
        visited = {current}
        for foreign_key_id in label.foreign_key_ids:
            foreign_key = foreign_keys.get(foreign_key_id)
            if foreign_key is None:
                issues.append(
                    _issue(
                        "route_foreign_key",
                        f"route supervision references unknown FK {foreign_key_id}",
                    )
                )
                break
            if current == foreign_key.parent_table_id:
                following = foreign_key.child_table_id
            elif current == foreign_key.child_table_id:
                following = foreign_key.parent_table_id
            else:
                issues.append(
                    _issue(
                        "route_continuity",
                        "route supervision is not contiguous from target table",
                    )
                )
                break
            if following in visited:
                issues.append(
                    _issue("route_cycle", "route supervision contains a cycle")
                )
                break
            visited.add(following)
            current = following
    return issues


def _issue(code: str, message: str) -> TaskValidationIssue:
    return TaskValidationIssue(code=code, message=message)


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind == "f" or second.dtype.kind == "f":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


__all__ = [
    "TaskValidationIssue",
    "TaskValidationReport",
    "validate_task",
]
