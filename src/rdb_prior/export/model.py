"""In-memory representation of one RDBPFN-compatible DBB dataset."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNDataset:
    dataset_name: str
    task_name: str
    metadata: Mapping[str, Any]
    tables: Mapping[str, Mapping[str, np.ndarray]]
    splits: Mapping[str, Mapping[str, np.ndarray]]
    masked_columns: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("dataset_name", "task_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        tables = _array_tables(self.tables, field_name="tables")
        splits = _array_tables(self.splits, field_name="splits")
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("splits must contain train, validation, and test")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "tables", MappingProxyType(tables))
        object.__setattr__(self, "splits", MappingProxyType(splits))


def _array_tables(
    value: Mapping[str, Mapping[str, np.ndarray]],
    *,
    field_name: str,
) -> dict[str, Mapping[str, np.ndarray]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    result: dict[str, Mapping[str, np.ndarray]] = {}
    for table_name, columns in value.items():
        if not isinstance(table_name, str) or not table_name:
            raise ValueError(f"{field_name} names must be non-empty strings")
        if not isinstance(columns, Mapping) or not columns:
            raise ValueError(f"{field_name}.{table_name} must contain columns")
        encoded: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for column_name, array in columns.items():
            if not isinstance(column_name, str) or not column_name:
                raise ValueError("column names must be non-empty strings")
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise TypeError("exported columns must be one-dimensional arrays")
            encoded[column_name] = array
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError(f"{field_name}.{table_name} columns do not align")
        result[table_name] = MappingProxyType(encoded)
    return result


__all__ = ["RDBPFNDataset"]
