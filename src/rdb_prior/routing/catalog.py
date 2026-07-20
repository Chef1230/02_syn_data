"""Bounded legal path templates over a physical schema graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalDataType,
    PhysicalForeignKey,
    PhysicalSchema,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PathHop:
    foreign_key_id: str
    source_table_id: str
    destination_table_id: str
    parent_to_child: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaPath:
    target_table_id: str
    hops: tuple[PathHop, ...]

    def __post_init__(self) -> None:
        if not self.hops:
            raise ValueError("a routed schema path must contain at least one hop")
        current = self.target_table_id
        for hop in self.hops:
            if hop.source_table_id != current:
                raise ValueError("path hops are not contiguous")
            current = hop.destination_table_id

    @property
    def source_table_id(self) -> str:
        return self.hops[-1].destination_table_id

    @property
    def foreign_key_ids(self) -> tuple[str, ...]:
        return tuple(hop.foreign_key_id for hop in self.hops)

    @property
    def signature(self) -> str:
        parts = [self.target_table_id]
        for hop in self.hops:
            direction = ">" if hop.parent_to_child else "<"
            parts.append(f"{direction}{hop.foreign_key_id}:{hop.destination_table_id}")
        return "|".join(parts)


def enumerate_schema_paths(
    schema: PhysicalSchema,
    *,
    target_table_id: str,
    max_depth: int = 2,
    max_candidates: int = 20,
    preferred_paths: Iterable[tuple[str, ...]] = (),
) -> tuple[SchemaPath, ...]:
    """Enumerate deterministic simple paths, prioritizing DSL-supervised ones."""
    if max_depth < 1 or max_candidates < 1:
        raise ValueError("max_depth and max_candidates must be positive")
    schema.table(target_table_id)
    adjacent: dict[str, list[tuple[PhysicalForeignKey, bool, str]]] = {
        table.table_id: [] for table in schema.tables
    }
    for foreign_key in schema.foreign_keys:
        adjacent[foreign_key.parent_table_id].append(
            (foreign_key, True, foreign_key.child_table_id)
        )
        adjacent[foreign_key.child_table_id].append(
            (foreign_key, False, foreign_key.parent_table_id)
        )
    for values in adjacent.values():
        values.sort(key=lambda item: (item[0].foreign_key_id, item[2]))

    paths: list[SchemaPath] = []
    frontier: list[tuple[str, tuple[PathHop, ...], frozenset[str]]] = [
        (target_table_id, (), frozenset({target_table_id}))
    ]
    for _depth in range(1, max_depth + 1):
        following: list[tuple[str, tuple[PathHop, ...], frozenset[str]]] = []
        for current, hops, visited in frontier:
            for foreign_key, parent_to_child, destination in adjacent[current]:
                if destination in visited:
                    continue
                new_hops = hops + (
                    PathHop(
                        foreign_key_id=foreign_key.foreign_key_id,
                        source_table_id=current,
                        destination_table_id=destination,
                        parent_to_child=parent_to_child,
                    ),
                )
                paths.append(
                    SchemaPath(
                        target_table_id=target_table_id,
                        hops=new_hops,
                    )
                )
                following.append(
                    (destination, new_hops, visited | {destination})
                )
        frontier = following

    preferred = {tuple(path): index for index, path in enumerate(preferred_paths)}
    paths.sort(
        key=lambda path: (
            0 if path.foreign_key_ids in preferred else 1,
            preferred.get(path.foreign_key_ids, len(preferred)),
            len(path.hops),
            path.signature,
        )
    )
    return tuple(paths[:max_candidates])


def path_feature_vector(
    schema: PhysicalSchema,
    path: SchemaPath,
    *,
    max_depth: int,
) -> np.ndarray:
    """Anonymous fixed-width descriptors used by the MLP router."""
    endpoint = schema.table(path.source_table_id)
    fks = {fk.foreign_key_id: fk for fk in schema.foreign_keys}
    optional = 0
    for hop in path.hops:
        foreign_key = fks[hop.foreign_key_id]
        optional += int(foreign_key.optionality.value == "optional")
    depth = len(path.hops)
    forward = sum(hop.parent_to_child for hop in path.hops)
    observable_columns = tuple(
        column
        for column in endpoint.columns
        if column.kind not in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}
    )
    numeric = sum(
        column.data_type
        in {
            PhysicalDataType.BIGINT,
            PhysicalDataType.INTEGER,
            PhysicalDataType.DOUBLE,
            PhysicalDataType.BOOLEAN,
        }
        for column in observable_columns
    )
    text = sum(
        column.data_type is PhysicalDataType.TEXT
        for column in observable_columns
    )
    has_time = any(
        column.kind is ColumnKind.TIME for column in observable_columns
    )
    column_count = max(1, len(observable_columns))
    return np.asarray(
        [
            depth / max_depth,
            forward / depth,
            (depth - forward) / depth,
            min(1.0, len(observable_columns) / 32.0),
            numeric / column_count,
            text / column_count,
            optional / depth,
            float(has_time),
        ],
        dtype=np.float32,
    )


def path_similarity_matrix(paths: tuple[SchemaPath, ...]) -> np.ndarray:
    count = len(paths)
    result = np.zeros((count, count), dtype=np.float32)
    for left_index, left in enumerate(paths):
        left_edges = set(left.foreign_key_ids)
        for right_index, right in enumerate(paths):
            if left_index == right_index:
                continue
            right_edges = set(right.foreign_key_ids)
            union = left_edges | right_edges
            edge_overlap = len(left_edges & right_edges) / max(1, len(union))
            same_endpoint = float(left.source_table_id == right.source_table_id)
            result[left_index, right_index] = 0.75 * edge_overlap + 0.25 * same_endpoint
    return result


__all__ = [
    "PathHop",
    "SchemaPath",
    "enumerate_schema_paths",
    "path_feature_vector",
    "path_similarity_matrix",
]
