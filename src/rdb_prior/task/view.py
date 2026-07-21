"""Canonical database visibility and target masking for one task."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.task.model import TaskPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskView:
    row_masks: Mapping[str, np.ndarray]
    masked_columns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        values = dict(self.row_masks)
        if not values:
            raise ValueError("row_masks must not be empty")
        for table_id, mask in values.items():
            if not isinstance(table_id, str) or not table_id:
                raise ValueError("row mask table IDs must be non-empty")
            if not isinstance(mask, np.ndarray) or mask.ndim != 1:
                raise TypeError("row masks must be one-dimensional arrays")
            if mask.dtype != np.bool_:
                raise TypeError("row masks must use boolean dtype")
        object.__setattr__(self, "row_masks", MappingProxyType(values))

    def visible_rows(self, table_id: str) -> np.ndarray:
        try:
            return np.flatnonzero(self.row_masks[table_id]).astype(np.int64)
        except KeyError as error:
            raise KeyError(f"TaskView has no table {table_id!r}") from error

    def is_column_masked(self, table_id: str, column_id: str) -> bool:
        return (table_id, column_id) in self.masked_columns


def build_task_view(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    plan: TaskPlan,
) -> TaskView:
    masks = {
        table.table_id: np.ones(
            database.table(table.table_id).row_count,
            dtype=bool,
        )
        for table in schema.tables
    }
    for rule in plan.observation_rules:
        values = database.table(rule.table_id).column(rule.time_column_id)
        masks[rule.table_id] &= (values >= 0) & (
            values <= rule.max_timestamp
        )

    changed = True
    while changed:
        changed = False
        for foreign_key in schema.foreign_keys:
            child_mask = masks[foreign_key.child_table_id]
            assignments = database.table(foreign_key.child_table_id).column(
                foreign_key.child_column_id
            )
            valid = assignments >= 0
            keep = np.ones(len(assignments), dtype=bool)
            keep[valid] = masks[foreign_key.parent_table_id][assignments[valid]]
            updated = child_mask & keep
            if not np.array_equal(updated, child_mask):
                masks[foreign_key.child_table_id] = updated
                changed = True

    return TaskView(
        row_masks=masks,
        masked_columns=tuple(
            (plan.target_table_id, column_id)
            for column_id in plan.masked_column_ids
        ),
    )


__all__ = ["TaskView", "build_task_view"]
