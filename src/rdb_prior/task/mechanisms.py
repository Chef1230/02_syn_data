"""V1 task mechanisms derived only from a frozen database instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalForeignKey,
    PhysicalSchema,
)
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.schema.spec import TableRole
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


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationAttributeCandidate:
    table_id: str
    column_id: str
    prediction_type: PredictionType


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureEventCandidate:
    foreign_key_id: str
    entity_table_id: str
    event_table_id: str
    time_column_id: str


def relation_attribute_candidates(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    *,
    max_classification_categories: int,
) -> tuple[RelationAttributeCandidate, ...]:
    candidates: list[RelationAttributeCandidate] = []
    for table in schema.tables:
        if table.role not in {TableRole.EVENT, TableRole.BRIDGE}:
            continue
        data = database.table(table.table_id)
        for column in table.columns:
            if column.kind is not ColumnKind.FEATURE or column.unique:
                continue
            observed = _observed_values(data.column(column.column_id))
            if not len(observed):
                continue
            prediction_type = _prediction_type(
                column,
                observed,
                max_classification_categories=max_classification_categories,
            )
            if prediction_type is None:
                continue
            candidates.append(
                RelationAttributeCandidate(
                    table_id=table.table_id,
                    column_id=column.column_id,
                    prediction_type=prediction_type,
                )
            )
    return tuple(candidates)


def future_event_candidates(
    schema: PhysicalSchema,
) -> tuple[FutureEventCandidate, ...]:
    candidates: list[FutureEventCandidate] = []
    for foreign_key in schema.foreign_keys:
        parent = schema.table(foreign_key.parent_table_id)
        child = schema.table(foreign_key.child_table_id)
        if parent.role is not TableRole.ENTITY or child.role is not TableRole.EVENT:
            continue
        time_column = next(
            (
                column
                for column in child.columns
                if column.kind is ColumnKind.TIME
            ),
            None,
        )
        if time_column is not None:
            candidates.append(
                FutureEventCandidate(
                    foreign_key_id=foreign_key.foreign_key_id,
                    entity_table_id=parent.table_id,
                    event_table_id=child.table_id,
                    time_column_id=time_column.column_id,
                )
            )
    return tuple(candidates)


def build_relation_attribute_task(
    *,
    task_id: str,
    sample_id: str,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    candidate: RelationAttributeCandidate,
    seed: int,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count_per_split: int,
) -> PlannedTask | None:
    table = schema.table(candidate.table_id)
    column = table.column(candidate.column_id)
    data = database.table(table.table_id)
    labels = data.column(column.column_id)
    valid_rows = np.flatnonzero(_observed_mask(labels)).astype(np.int64)
    if len(valid_rows) < min_support_rows + min_query_rows:
        return None

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    time_column = next(
        (item for item in table.columns if item.kind is ColumnKind.TIME),
        None,
    )
    if time_column is not None:
        times = data.column(time_column.column_id)
        ordered = valid_rows[
            np.argsort(times[valid_rows], kind="stable")
        ]
        split_strategy = "temporal_rows"
    else:
        ordered = valid_rows.copy()
        rng.shuffle(ordered)
        split_strategy = "random_rows"

    support_count = round(len(ordered) * support_fraction)
    support_count = min(
        len(ordered) - min_query_rows,
        max(min_support_rows, support_count),
    )
    support_rows = ordered[:support_count]
    query_rows = ordered[support_count:]
    support_labels = labels[support_rows].copy()
    query_labels = labels[query_rows].copy()
    if candidate.prediction_type is PredictionType.CLASSIFICATION and not (
        _classification_split_is_valid(
            support_labels,
            query_labels,
            min_class_count=min_class_count_per_split,
        )
    ):
        return None

    plan = TaskPlan(
        task_id=task_id,
        sample_id=sample_id,
        instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.RELATION_ATTRIBUTE,
        prediction_type=candidate.prediction_type,
        target_table_id=table.table_id,
        source_table_id=table.table_id,
        target_column_id=column.column_id,
        split_strategy=split_strategy,
        seed=seed,
        masked_column_ids=(column.column_id,),
        route_supervision=_schema_route_labels(
            schema,
            target_table_id=table.table_id,
            optional_paths=tuple(
                (foreign_key.foreign_key_id,)
                for foreign_key in schema.foreign_keys
                if table.table_id
                in {
                    foreign_key.parent_table_id,
                    foreign_key.child_table_id,
                }
            ),
        ),
        parameters=(("support_fraction", support_fraction),),
    )
    return PlannedTask(
        plan=plan,
        data=TaskData(
            support_row_ids=support_rows,
            support_labels=support_labels,
            query_row_ids=query_rows,
            query_labels=query_labels,
        ),
    )


def build_future_event_existence_task(
    *,
    task_id: str,
    sample_id: str,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    candidate: FutureEventCandidate,
    seed: int,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count_per_split: int,
    cutoff_quantile_min: float,
    cutoff_quantile_max: float,
    horizon_fraction_min: float,
    horizon_fraction_max: float,
) -> PlannedTask | None:
    foreign_key = _foreign_key(schema, candidate.foreign_key_id)
    event_data = database.table(candidate.event_table_id)
    entity_data = database.table(candidate.entity_table_id)
    times = event_data.column(candidate.time_column_id)
    if len(times) < 2 or int(times.max()) <= int(times.min()):
        return None

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    cutoff_quantile = float(
        rng.uniform(cutoff_quantile_min, cutoff_quantile_max)
    )
    cutoff = int(np.quantile(times, cutoff_quantile))
    span = int(times.max()) - int(times.min())
    horizon_fraction = float(
        rng.uniform(horizon_fraction_min, horizon_fraction_max)
    )
    horizon_end = min(
        int(times.max()),
        cutoff + max(1, round(span * horizon_fraction)),
    )
    if horizon_end <= cutoff:
        return None

    assignments = event_data.column(foreign_key.child_column_id)
    future = (
        (times > cutoff)
        & (times <= horizon_end)
        & (assignments >= 0)
    )
    labels = np.zeros(entity_data.row_count, dtype=np.int8)
    labels[np.unique(assignments[future])] = 1
    split = _stratified_binary_split(
        labels,
        rng,
        support_fraction=support_fraction,
        min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support_rows, query_rows = split

    observation_rules = tuple(
        ObservationRule(
            table_id=table.table_id,
            time_column_id=time_column.column_id,
            max_timestamp=cutoff,
        )
        for table in schema.tables
        if table.role is TableRole.EVENT
        for time_column in table.columns
        if time_column.kind is ColumnKind.TIME
    )
    plan = TaskPlan(
        task_id=task_id,
        sample_id=sample_id,
        instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.FUTURE_EVENT_EXISTENCE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.entity_table_id,
        source_table_id=candidate.event_table_id,
        foreign_key_id=candidate.foreign_key_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff,
        horizon_end_time=horizon_end,
        split_strategy="stratified_entities",
        seed=seed,
        observation_rules=observation_rules,
        route_supervision=_schema_route_labels(
            schema,
            target_table_id=candidate.entity_table_id,
            required_paths=((candidate.foreign_key_id,),),
        ),
        parameters=(
            ("cutoff_quantile", cutoff_quantile),
            ("horizon_fraction", horizon_fraction),
            ("support_fraction", support_fraction),
        ),
    )
    return PlannedTask(
        plan=plan,
        data=TaskData(
            support_row_ids=support_rows,
            support_labels=labels[support_rows],
            query_row_ids=query_rows,
            query_labels=labels[query_rows],
        ),
    )


def future_event_labels(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    plan: TaskPlan,
) -> np.ndarray:
    if plan.mechanism is not TaskMechanism.FUTURE_EVENT_EXISTENCE:
        raise ValueError("plan is not a future event existence task")
    foreign_key = _foreign_key(schema, plan.foreign_key_id or "")
    event = database.table(plan.source_table_id)
    entity = database.table(plan.target_table_id)
    times = event.column(plan.time_column_id or "")
    assignments = event.column(foreign_key.child_column_id)
    mask = (
        (times > int(plan.cutoff_time))
        & (times <= int(plan.horizon_end_time))
        & (assignments >= 0)
    )
    labels = np.zeros(entity.row_count, dtype=np.int8)
    labels[np.unique(assignments[mask])] = 1
    return labels


def _prediction_type(
    column: PhysicalColumn,
    observed: np.ndarray,
    *,
    max_classification_categories: int,
) -> PredictionType | None:
    if column.data_type in {PhysicalDataType.BOOLEAN, PhysicalDataType.TEXT}:
        return PredictionType.CLASSIFICATION
    if column.data_type is PhysicalDataType.INTEGER:
        if len(np.unique(observed)) <= max_classification_categories:
            return PredictionType.CLASSIFICATION
        return PredictionType.REGRESSION
    if column.data_type is PhysicalDataType.DOUBLE:
        return PredictionType.REGRESSION
    return None


def _observed_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        return np.isfinite(values)
    if values.dtype.kind in {"U", "S"}:
        return values != ""
    return np.ones(len(values), dtype=bool)


def _observed_values(values: np.ndarray) -> np.ndarray:
    return values[_observed_mask(values)]


def _classification_split_is_valid(
    support: np.ndarray,
    query: np.ndarray,
    *,
    min_class_count: int,
) -> bool:
    support_values, support_counts = np.unique(support, return_counts=True)
    query_values, query_counts = np.unique(query, return_counts=True)
    if len(support_values) < 2 or len(query_values) < 2:
        return False
    if not set(query_values.tolist()) <= set(support_values.tolist()):
        return False
    support_count_map = dict(zip(support_values.tolist(), support_counts.tolist()))
    query_count_map = dict(zip(query_values.tolist(), query_counts.tolist()))
    return all(
        support_count_map[value] >= min_class_count
        and query_count_map.get(value, 0) >= min_class_count
        for value in query_count_map
    )


def _stratified_binary_split(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(group) < 2 * min_class_count for group in groups):
        return None
    support_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for group in groups:
        group = group.astype(np.int64)
        rng.shuffle(group)
        count = round(len(group) * support_fraction)
        count = min(
            len(group) - min_class_count,
            max(min_class_count, count),
        )
        support_parts.append(group[:count])
        query_parts.append(group[count:])
    support = np.concatenate(support_parts)
    query = np.concatenate(query_parts)
    rng.shuffle(support)
    rng.shuffle(query)
    if len(support) < min_support_rows or len(query) < min_query_rows:
        return None
    return support, query


def _foreign_key(
    schema: PhysicalSchema,
    foreign_key_id: str,
) -> PhysicalForeignKey:
    for foreign_key in schema.foreign_keys:
        if foreign_key.foreign_key_id == foreign_key_id:
            return foreign_key
    raise KeyError(f"PhysicalSchema has no FK {foreign_key_id!r}")


def _schema_route_labels(
    schema: PhysicalSchema,
    *,
    target_table_id: str,
    required_paths: tuple[tuple[str, ...], ...] = (),
    optional_paths: tuple[tuple[str, ...], ...] = (),
    max_depth: int = 2,
) -> tuple[RoutePathLabel, ...]:
    """Materialize required/optional/distractor labels in the Task DSL."""
    required = set(required_paths)
    optional = set(optional_paths)
    adjacent: dict[str, list[tuple[str, str]]] = {
        table.table_id: [] for table in schema.tables
    }
    for foreign_key in schema.foreign_keys:
        adjacent[foreign_key.parent_table_id].append(
            (foreign_key.foreign_key_id, foreign_key.child_table_id)
        )
        adjacent[foreign_key.child_table_id].append(
            (foreign_key.foreign_key_id, foreign_key.parent_table_id)
        )
    for values in adjacent.values():
        values.sort()
    frontier = [(target_table_id, (), frozenset({target_table_id}))]
    paths: list[tuple[str, ...]] = []
    for _depth in range(max_depth):
        following: list[tuple[str, tuple[str, ...], frozenset[str]]] = []
        for current, path, visited in frontier:
            for foreign_key_id, destination in adjacent[current]:
                if destination in visited:
                    continue
                candidate = path + (foreign_key_id,)
                paths.append(candidate)
                following.append(
                    (destination, candidate, visited | {destination})
                )
        frontier = following
    return tuple(
        RoutePathLabel(
            foreign_key_ids=path,
            role=(
                RouteRole.REQUIRED
                if path in required
                else RouteRole.OPTIONAL
                if path in optional
                else RouteRole.DISTRACTOR
            ),
        )
        for path in paths
    )


__all__ = [
    "RelationAttributeCandidate",
    "FutureEventCandidate",
    "relation_attribute_candidates",
    "future_event_candidates",
    "build_relation_attribute_task",
    "build_future_event_existence_task",
    "future_event_labels",
]
