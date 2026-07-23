"""Role-aware feature and temporal column generation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalSchema,
    PhysicalTable,
)
from rdb_prior.generation.feature_strategies import generate_feature_signal
from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.model import TableData
from rdb_prior.instance.plan import InstancePlan, TableMechanismPlan, TemporalFamily
from rdb_prior.schema.spec import TableRole


def generate_table_features(
    *,
    schema: PhysicalSchema,
    table: PhysicalTable,
    plan: InstancePlan,
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
) -> dict[str, np.ndarray]:
    table_plan = plan.table(table.table_id)
    rng = np.random.Generator(np.random.PCG64DXSM(table_plan.feature_seed))
    context = _causal_context(schema, table, latents, relations)
    values: dict[str, np.ndarray] = {}

    for column in table.columns:
        if column.kind in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}:
            continue
        if column.kind is ColumnKind.TIME:
            values[column.column_id] = _generate_time(
                schema=schema,
                table=table,
                column=column,
                table_plan=table_plan,
                relations=relations,
                generated_tables=generated_tables,
            )
            continue

        signal = generate_feature_signal(
            table_plan.feature_family,
            context,
            rng,
            noise_scale=table_plan.parameter_map["noise_scale"],
            signal_scale=table_plan.parameter_map["signal_scale"],
            activation_scale=table_plan.parameter_map["activation_scale"],
            output_scale=table_plan.parameter_map["output_scale"],
            long_tail_enabled=bool(
                table_plan.parameter_map["long_tail_enabled"]
            ),
            long_tail_alpha=table_plan.parameter_map["long_tail_alpha"],
            mlp_depth=int(table_plan.parameter_map.get("mlp_depth", 1)),
            mlp_hidden_factor=float(
                table_plan.parameter_map.get("mlp_hidden_factor", 2.0)
            ),
            mlp_dropout_rate=float(
                table_plan.parameter_map.get("mlp_dropout_rate", 0.0)
            ),
        )
        encoded = _encode_signal(
            signal,
            column,
            table.role,
            rng,
            cardinality=int(table_plan.parameter_map["categorical_cardinality"]),
        )
        values[column.column_id] = _apply_missing(
            encoded,
            column,
            rng,
            table_plan.parameter_map["missing_rate"],
        )
    return values


def _causal_context(
    schema: PhysicalSchema,
    table: PhysicalTable,
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
) -> np.ndarray:
    pieces = [latents.table(table.table_id).values]
    for foreign_key in schema.foreign_keys:
        if foreign_key.child_table_id != table.table_id:
            continue
        assignments = relations[foreign_key.foreign_key_id]
        parent = latents.table(foreign_key.parent_table_id).values
        selected = np.zeros((len(assignments), parent.shape[1]), dtype=np.float64)
        valid = assignments >= 0
        selected[valid] = parent[assignments[valid]]
        pieces.append(selected)
    return np.concatenate(pieces, axis=1)


def _generate_time(
    *,
    schema: PhysicalSchema,
    table: PhysicalTable,
    column: PhysicalColumn,
    table_plan: TableMechanismPlan,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
) -> np.ndarray:
    seed = table_plan.temporal_seed + column.ordinal * 104_729
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows = table_plan.population.row_count
    scale = table_plan.parameter_map["time_scale_seconds"]
    base_epoch = 1_577_836_800 + int(rng.integers(0, 5 * 365 * 86_400))

    incoming = tuple(
        foreign_key
        for foreign_key in schema.foreign_keys
        if foreign_key.child_table_id == table.table_id
    )
    if table_plan.temporal_family is TemporalFamily.TIME_LAGGED:
        for foreign_key in incoming:
            parent_table = schema.table(foreign_key.parent_table_id)
            parent_time = next(
                (
                    item
                    for item in parent_table.columns
                    if item.kind is ColumnKind.TIME
                ),
                None,
            )
            if parent_time is None or parent_table.table_id not in generated_tables:
                continue
            assignments = relations[foreign_key.foreign_key_id]
            parent_values = generated_tables[parent_table.table_id].column(
                parent_time.column_id
            )
            values = np.full(rows, base_epoch, dtype=np.int64)
            valid = assignments >= 0
            lag = np.maximum(1, rng.lognormal(np.log(scale), 0.8, size=rows))
            values[valid] = parent_values[assignments[valid]] + lag[valid].astype(
                np.int64
            )
            return values

    grouping = next(
        (
            relations[foreign_key.foreign_key_id]
            for foreign_key in incoming
            if foreign_key.relation_strategy != "lookup_assignment"
        ),
        np.zeros(rows, dtype=np.int64),
    )
    values = np.empty(rows, dtype=np.int64)
    for parent_index in np.unique(grouping):
        indices = np.flatnonzero(grouping == parent_index)
        group_base = base_epoch + int(rng.integers(0, 180 * 86_400))
        intervals = np.maximum(1, rng.exponential(scale, size=len(indices))).astype(
            np.int64
        )
        values[indices] = group_base + np.cumsum(intervals)
    return values


def _encode_signal(
    signal: np.ndarray,
    column: PhysicalColumn,
    role: TableRole,
    rng: np.random.Generator,
    *,
    cardinality: int,
) -> np.ndarray:
    if column.unique:
        order = np.argsort(signal, kind="stable")
        unique = np.empty(len(signal), dtype=np.int64)
        unique[order] = np.arange(len(signal), dtype=np.int64)
        if column.data_type is PhysicalDataType.TEXT:
            return np.char.add("v", unique.astype(str))
        return unique
    if column.data_type is PhysicalDataType.DOUBLE:
        return signal.astype(np.float64)
    if column.data_type is PhysicalDataType.INTEGER:
        if role is TableRole.LOOKUP:
            return _quantile_codes(signal, min(cardinality, len(signal)))
        return np.rint(signal * float(rng.uniform(2.0, 20.0))).astype(np.int64)
    if column.data_type is PhysicalDataType.BOOLEAN:
        threshold = float(np.quantile(signal, rng.uniform(0.3, 0.7)))
        return (signal > threshold).astype(np.int8)
    if column.data_type is PhysicalDataType.TEXT:
        codes = _quantile_codes(signal, min(cardinality, len(signal)))
        return np.char.add("v", codes.astype(str))
    if column.data_type is PhysicalDataType.TIMESTAMP:
        return (1_577_836_800 + signal * 86_400).astype(np.int64)
    raise ValueError(f"unsupported physical data type: {column.data_type}")


def _quantile_codes(signal: np.ndarray, cardinality: int) -> np.ndarray:
    cardinality = max(1, cardinality)
    if cardinality == 1:
        return np.zeros(len(signal), dtype=np.int64)
    boundaries = np.quantile(
        signal,
        np.linspace(0, 1, cardinality + 1)[1:-1],
    )
    return np.digitize(signal, boundaries).astype(np.int64)


def _apply_missing(
    values: np.ndarray,
    column: PhysicalColumn,
    rng: np.random.Generator,
    missing_rate: float,
) -> np.ndarray:
    if not column.nullable or missing_rate <= 0:
        return values
    missing = rng.random(len(values)) < missing_rate
    if values.dtype.kind in {"U", "S"}:
        width = max(1, values.dtype.itemsize // np.dtype("U1").itemsize)
        result = values.astype(f"<U{width}", copy=True)
        result[missing] = ""
        return result
    result = values.astype(np.float64, copy=True)
    result[missing] = np.nan
    return result


__all__ = ["generate_table_features"]
