# src/rdb_prior/compilation/model.py
# -*- coding: utf-8 -*-
"""Immutable physical schema produced from a validated SchemaBlueprint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping

from rdb_prior.schema.spec import (
    Cardinality,
    IdentityDependency,
    Optionality,
    TableRole,
)


_SQL_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class PhysicalDataType(str, Enum):
    BIGINT = "bigint"
    INTEGER = "integer"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TEXT = "text"
    TIMESTAMP = "timestamp"


class ColumnKind(str, Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    FEATURE = "feature"
    TIME = "time"


def _require_identifier(name: str, value: Any) -> None:
    if isinstance(value, Enum) or not isinstance(value, str):
        raise TypeError(f"{name} must be a string logical identifier")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_sql_name(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _SQL_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must be a lowercase SQL-safe identifier: {value!r}"
        )


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalColumn:
    column_id: str
    name: str
    data_type: PhysicalDataType
    kind: ColumnKind
    ordinal: int
    nullable: bool
    unique: bool = False

    def __post_init__(self) -> None:
        _require_identifier("column_id", self.column_id)
        _require_sql_name("column name", self.name)
        if not isinstance(self.data_type, PhysicalDataType):
            raise TypeError("data_type must be PhysicalDataType")
        if not isinstance(self.kind, ColumnKind):
            raise TypeError("kind must be ColumnKind")
        _require_non_negative_int("ordinal", self.ordinal)
        _require_bool("nullable", self.nullable)
        _require_bool("unique", self.unique)

        if self.kind is ColumnKind.PRIMARY_KEY:
            if self.nullable:
                raise ValueError("Primary-key columns cannot be nullable")
            if not self.unique:
                raise ValueError("Primary-key columns must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalTable:
    table_id: str
    name: str
    role: TableRole
    rank: int
    columns: tuple[PhysicalColumn, ...]

    def __post_init__(self) -> None:
        _require_identifier("table_id", self.table_id)
        _require_sql_name("table name", self.name)
        if not isinstance(self.role, TableRole):
            raise TypeError("role must be TableRole")
        _require_non_negative_int("rank", self.rank)
        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple")
        if not self.columns:
            raise ValueError("PhysicalTable must contain columns")
        if not all(isinstance(column, PhysicalColumn) for column in self.columns):
            raise TypeError("columns items must be PhysicalColumn")

        column_ids = tuple(column.column_id for column in self.columns)
        column_names = tuple(column.name for column in self.columns)
        ordinals = tuple(column.ordinal for column in self.columns)
        _require_unique("column IDs", column_ids)
        _require_unique("column names", column_names)
        _require_unique("column ordinals", ordinals)

        primary_keys = tuple(
            column
            for column in self.columns
            if column.kind is ColumnKind.PRIMARY_KEY
        )
        if len(primary_keys) != 1:
            raise ValueError(
                "PhysicalTable must contain exactly one primary-key column"
            )

        canonical = tuple(sorted(self.columns, key=lambda column: column.ordinal))
        if tuple(column.ordinal for column in canonical) != tuple(
            range(len(canonical))
        ):
            raise ValueError("column ordinals must be contiguous from zero")
        object.__setattr__(self, "columns", canonical)

    @property
    def primary_key(self) -> PhysicalColumn:
        return next(
            column
            for column in self.columns
            if column.kind is ColumnKind.PRIMARY_KEY
        )

    def column(self, column_id: str) -> PhysicalColumn:
        for column in self.columns:
            if column.column_id == column_id:
                return column
        raise KeyError(f"Table {self.table_id!r} has no column {column_id!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalForeignKey:
    foreign_key_id: str
    name: str
    parent_table_id: str
    parent_column_id: str
    child_table_id: str
    child_column_id: str
    cardinality: Cardinality
    optionality: Optionality
    identity_dependency: IdentityDependency
    relation_strategy: str

    def __post_init__(self) -> None:
        _require_identifier("foreign_key_id", self.foreign_key_id)
        _require_sql_name("foreign-key name", self.name)
        _require_identifier("parent_table_id", self.parent_table_id)
        _require_identifier("parent_column_id", self.parent_column_id)
        _require_identifier("child_table_id", self.child_table_id)
        _require_identifier("child_column_id", self.child_column_id)
        if self.parent_table_id == self.child_table_id:
            raise ValueError("Physical foreign key cannot be a self-reference")
        if not isinstance(self.cardinality, Cardinality):
            raise TypeError("cardinality must be Cardinality")
        if self.cardinality is Cardinality.MANY_TO_MANY:
            raise ValueError("Physical FK cannot have many-to-many cardinality")
        if not isinstance(self.optionality, Optionality):
            raise TypeError("optionality must be Optionality")
        if not isinstance(self.identity_dependency, IdentityDependency):
            raise TypeError(
                "identity_dependency must be IdentityDependency"
            )
        _require_identifier("relation_strategy", self.relation_strategy)


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalSchema:
    schema_id: str
    blueprint_id: str
    tables: tuple[PhysicalTable, ...]
    foreign_keys: tuple[PhysicalForeignKey, ...]

    def __post_init__(self) -> None:
        _require_identifier("schema_id", self.schema_id)
        _require_identifier("blueprint_id", self.blueprint_id)
        if not isinstance(self.tables, tuple):
            raise TypeError("tables must be a tuple")
        if not self.tables:
            raise ValueError("PhysicalSchema must contain tables")
        if not all(isinstance(table, PhysicalTable) for table in self.tables):
            raise TypeError("tables items must be PhysicalTable")
        if not isinstance(self.foreign_keys, tuple):
            raise TypeError("foreign_keys must be a tuple")
        if not all(
            isinstance(foreign_key, PhysicalForeignKey)
            for foreign_key in self.foreign_keys
        ):
            raise TypeError(
                "foreign_keys items must be PhysicalForeignKey"
            )

        _require_unique(
            "table IDs",
            tuple(table.table_id for table in self.tables),
        )
        _require_unique(
            "table names",
            tuple(table.name for table in self.tables),
        )
        _require_unique(
            "foreign-key IDs",
            tuple(fk.foreign_key_id for fk in self.foreign_keys),
        )
        _require_unique(
            "foreign-key names",
            tuple(fk.name for fk in self.foreign_keys),
        )

        canonical_tables = tuple(
            sorted(self.tables, key=lambda table: table.table_id)
        )
        canonical_fks = tuple(
            sorted(
                self.foreign_keys,
                key=lambda foreign_key: foreign_key.foreign_key_id,
            )
        )
        object.__setattr__(self, "tables", canonical_tables)
        object.__setattr__(self, "foreign_keys", canonical_fks)

        tables = {
            table.table_id: table
            for table in canonical_tables
        }
        child_columns: set[tuple[str, str]] = set()
        for foreign_key in canonical_fks:
            parent = tables.get(foreign_key.parent_table_id)
            child = tables.get(foreign_key.child_table_id)
            if parent is None or child is None:
                raise ValueError(
                    f"Foreign key {foreign_key.foreign_key_id!r} "
                    "references an unknown table"
                )

            try:
                parent_column = parent.column(
                    foreign_key.parent_column_id
                )
                child_column = child.column(
                    foreign_key.child_column_id
                )
            except KeyError as error:
                raise ValueError(
                    f"Foreign key {foreign_key.foreign_key_id!r} "
                    "references an unknown column"
                ) from error

            if parent_column.kind is not ColumnKind.PRIMARY_KEY:
                raise ValueError("Foreign-key parent must be a primary key")
            if child_column.kind is not ColumnKind.FOREIGN_KEY:
                raise ValueError("Foreign-key child column has wrong kind")
            if parent_column.data_type is not child_column.data_type:
                raise ValueError("Foreign-key column types must match")

            child_key = (child.table_id, child_column.column_id)
            if child_key in child_columns:
                raise ValueError(
                    "A physical child column cannot implement multiple FKs"
                )
            child_columns.add(child_key)

            expected_nullable = (
                foreign_key.optionality is Optionality.OPTIONAL
            )
            if child_column.nullable is not expected_nullable:
                raise ValueError(
                    "FK column nullability must match relation optionality"
                )

    def table(self, table_id: str) -> PhysicalTable:
        for table in self.tables:
            if table.table_id == table_id:
                return table
        raise KeyError(f"PhysicalSchema has no table {table_id!r}")

    def table_map(self) -> Mapping[str, PhysicalTable]:
        return MappingProxyType(
            {table.table_id: table for table in self.tables}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "blueprint_id": self.blueprint_id,
            "tables": [
                {
                    "table_id": table.table_id,
                    "name": table.name,
                    "role": table.role.value,
                    "rank": table.rank,
                    "columns": [
                        {
                            "column_id": column.column_id,
                            "name": column.name,
                            "data_type": column.data_type.value,
                            "kind": column.kind.value,
                            "ordinal": column.ordinal,
                            "nullable": column.nullable,
                            "unique": column.unique,
                        }
                        for column in table.columns
                    ],
                }
                for table in self.tables
            ],
            "foreign_keys": [
                {
                    "foreign_key_id": foreign_key.foreign_key_id,
                    "name": foreign_key.name,
                    "parent_table_id": foreign_key.parent_table_id,
                    "parent_column_id": foreign_key.parent_column_id,
                    "child_table_id": foreign_key.child_table_id,
                    "child_column_id": foreign_key.child_column_id,
                    "cardinality": foreign_key.cardinality.value,
                    "optionality": foreign_key.optionality.value,
                    "identity_dependency": (
                        foreign_key.identity_dependency.value
                    ),
                    "relation_strategy": foreign_key.relation_strategy,
                }
                for foreign_key in self.foreign_keys
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhysicalSchema:
        if not isinstance(data, Mapping):
            raise TypeError("PhysicalSchema payload must be a mapping")

        table_payloads = data.get("tables")
        fk_payloads = data.get("foreign_keys")
        if not isinstance(table_payloads, list):
            raise ValueError("tables must be a list")
        if not isinstance(fk_payloads, list):
            raise ValueError("foreign_keys must be a list")

        tables: list[PhysicalTable] = []
        for table_data in table_payloads:
            if not isinstance(table_data, Mapping):
                raise ValueError("table payload must be a mapping")
            column_payloads = table_data.get("columns")
            if not isinstance(column_payloads, list):
                raise ValueError("columns must be a list")
            columns_list: list[PhysicalColumn] = []
            for column_data in column_payloads:
                if not isinstance(column_data, Mapping):
                    raise ValueError("column payload must be a mapping")
                columns_list.append(
                    PhysicalColumn(
                        column_id=column_data["column_id"],
                        name=column_data["name"],
                        data_type=PhysicalDataType(
                            column_data["data_type"]
                        ),
                        kind=ColumnKind(column_data["kind"]),
                        ordinal=column_data["ordinal"],
                        nullable=column_data["nullable"],
                        unique=column_data.get("unique", False),
                    )
                )
            columns = tuple(columns_list)
            tables.append(
                PhysicalTable(
                    table_id=table_data["table_id"],
                    name=table_data["name"],
                    role=TableRole(table_data["role"]),
                    rank=table_data["rank"],
                    columns=columns,
                )
            )

        foreign_key_values: list[PhysicalForeignKey] = []
        for fk_data in fk_payloads:
            if not isinstance(fk_data, Mapping):
                raise ValueError("foreign-key payload must be a mapping")
            foreign_key_values.append(
                PhysicalForeignKey(
                    foreign_key_id=fk_data["foreign_key_id"],
                    name=fk_data["name"],
                    parent_table_id=fk_data["parent_table_id"],
                    parent_column_id=fk_data["parent_column_id"],
                    child_table_id=fk_data["child_table_id"],
                    child_column_id=fk_data["child_column_id"],
                    cardinality=Cardinality(fk_data["cardinality"]),
                    optionality=Optionality(fk_data["optionality"]),
                    identity_dependency=IdentityDependency(
                        fk_data["identity_dependency"]
                    ),
                    relation_strategy=fk_data["relation_strategy"],
                )
            )
        foreign_keys = tuple(foreign_key_values)

        return cls(
            schema_id=data["schema_id"],
            blueprint_id=data["blueprint_id"],
            tables=tuple(tables),
            foreign_keys=foreign_keys,
        )


def _require_unique(name: str, values: tuple[Any, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


__all__ = [
    "PhysicalDataType",
    "ColumnKind",
    "PhysicalColumn",
    "PhysicalTable",
    "PhysicalForeignKey",
    "PhysicalSchema",
]
