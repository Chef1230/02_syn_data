"""In-memory anonymous relational database instance."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class TableData:
    table_id: str
    columns: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.table_id, str) or not self.table_id.strip():
            raise ValueError("table_id must be a non-empty string")
        if not isinstance(self.columns, Mapping) or not self.columns:
            raise ValueError("columns must be a non-empty mapping")
        values = dict(self.columns)
        lengths: set[int] = set()
        for column_id, array in values.items():
            if not isinstance(column_id, str) or not column_id.strip():
                raise ValueError("column IDs must be non-empty strings")
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise TypeError("table columns must be one-dimensional arrays")
            if array.dtype == object:
                raise TypeError("object arrays are not persistable")
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError("all table columns must have equal length")
        object.__setattr__(self, "columns", MappingProxyType(values))

    @property
    def row_count(self) -> int:
        return len(next(iter(self.columns.values())))

    def column(self, column_id: str) -> np.ndarray:
        try:
            return self.columns[column_id]
        except KeyError as error:
            raise KeyError(
                f"Table {self.table_id!r} has no column {column_id!r}"
            ) from error


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabaseInstance:
    instance_id: str
    schema_id: str
    plan_id: str
    tables: tuple[TableData, ...]

    def __post_init__(self) -> None:
        for name in ("instance_id", "schema_id", "plan_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.tables, tuple) or not self.tables:
            raise ValueError("tables must be a non-empty tuple")
        if not all(isinstance(table, TableData) for table in self.tables):
            raise TypeError("tables items must be TableData")
        ids = tuple(table.table_id for table in self.tables)
        if len(set(ids)) != len(ids):
            raise ValueError("table IDs must be unique")
        object.__setattr__(
            self,
            "tables",
            tuple(sorted(self.tables, key=lambda table: table.table_id)),
        )

    def table(self, table_id: str) -> TableData:
        for table in self.tables:
            if table.table_id == table_id:
                return table
        raise KeyError(f"DatabaseInstance has no table {table_id!r}")


__all__ = ["TableData", "DatabaseInstance"]
