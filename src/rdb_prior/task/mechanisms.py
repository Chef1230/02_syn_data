"""Classification task mechanisms with exact route provenance.

Every mechanism now explicitly samples three things from its target table
node: a primary key (row_id), a target column, and — when the target table
carries a TIME column — a time column.  Synthetic-label mechanisms use the
well-known sentinel ``__label__`` as their ``target_column_id`` so that the
export layer can always emit ``row_id | label | cutoff_time``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalForeignKey, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.model import (
    AggregateOperator,
    ClassificationKind,
    ObservationRule,
    PlannedTask,
    PredictionType,
    RoutePathLabel,
    RouteRole,
    TaskData,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.view import build_task_view

# Sentinel column id used by mechanisms whose label is computed rather than
# sourced from an existing feature column.
_SYNTHETIC_TARGET = "__label__"


# ---------------------------------------------------------------------------
# Unified target-table sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetSample:
    """The three columns every task samples from its target table node."""

    table_id: str
    primary_key_column_id: str
    target_column_id: str
    time_column_id: str | None


def _sample_target_columns(
    schema: PhysicalSchema,
    target_table_id: str,
    *,
    target_column_id: str,
    time_column_id: str | None = None,
) -> TargetSample:
    """Bundle primary key, target column, and optional time column."""
    target_table = schema.table(target_table_id)
    return TargetSample(
        table_id=target_table_id,
        primary_key_column_id=target_table.primary_key.column_id,
        target_column_id=target_column_id,
        time_column_id=time_column_id,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationAttributeCandidate:
    table_id: str
    column_id: str
    source_table_id: str
    source_column_id: str
    required_path: tuple[str, ...]
    classification_kind: ClassificationKind
    class_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureEventCandidate:
    foreign_key_id: str
    entity_table_id: str
    event_table_id: str
    time_column_id: str
    target_column_id: str = _SYNTHETIC_TARGET


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureEventAttributeCandidate:
    event_table_id: str
    target_column_id: str
    parent_table_id: str
    source_column_id: str
    foreign_key_id: str
    time_column_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalAggregateCandidate:
    target_table_id: str
    source_table_id: str
    required_path: tuple[str, ...]
    time_column_id: str
    operator: AggregateOperator
    source_column_id: str | None
    target_column_id: str = _SYNTHETIC_TARGET


def relation_attribute_candidates(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    *,
    max_classification_categories: int,
) -> tuple[RelationAttributeCandidate, ...]:
    """Use Event/Detail target columns and an actual parent source column."""
    result: list[RelationAttributeCandidate] = []
    for table in schema.tables:
        if table.role not in {TableRole.EVENT, TableRole.DETAIL}:
            continue
        target_columns = [
            column
            for column in table.columns
            if column.kind is ColumnKind.FEATURE and not column.unique
        ]
        if not target_columns:
            continue
        for fk in schema.foreign_keys:
            if fk.child_table_id != table.table_id:
                continue
            parent = schema.table(fk.parent_table_id)
            sources = _usable_feature_columns(schema, database, parent.table_id)
            for target in target_columns:
                observed = _observed_values(database.table(table.table_id).column(target.column_id))
                unique_count = len(np.unique(observed))
                kind = (
                    ClassificationKind.CATEGORICAL
                    if 2 < unique_count <= max_classification_categories
                    else ClassificationKind.BINARY
                )
                class_count = min(max(2, unique_count), max_classification_categories)
                for source in sources:
                    result.append(
                        RelationAttributeCandidate(
                            table_id=table.table_id,
                            column_id=target.column_id,
                            source_table_id=parent.table_id,
                            source_column_id=source,
                            required_path=(fk.foreign_key_id,),
                            classification_kind=kind,
                            class_count=class_count,
                        )
                    )
    return tuple(result)


def future_event_candidates(schema: PhysicalSchema) -> tuple[FutureEventCandidate, ...]:
    result: list[FutureEventCandidate] = []
    for fk in schema.foreign_keys:
        parent = schema.table(fk.parent_table_id)
        child = schema.table(fk.child_table_id)
        time_column = _time_column(child)
        if (
            parent.role is TableRole.ENTITY
            and child.role is TableRole.EVENT
            and time_column is not None
            and _has_incoming_fk(schema, parent.table_id)
        ):
            result.append(
                FutureEventCandidate(
                    foreign_key_id=fk.foreign_key_id,
                    entity_table_id=parent.table_id,
                    event_table_id=child.table_id,
                    time_column_id=time_column,
                )
            )
    return tuple(result)


def future_event_attribute_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[FutureEventAttributeCandidate, ...]:
    result: list[FutureEventAttributeCandidate] = []
    for event in schema.tables:
        if event.role is not TableRole.EVENT:
            continue
        time_column = _time_column(event)
        targets = [c.column_id for c in event.columns if c.kind is ColumnKind.FEATURE]
        if time_column is None or not targets:
            continue
        for fk in schema.foreign_keys:
            if fk.child_table_id != event.table_id:
                continue
            sources = _usable_feature_columns(schema, database, fk.parent_table_id)
            for target in targets:
                for source in sources:
                    result.append(
                        FutureEventAttributeCandidate(
                            event_table_id=event.table_id,
                            target_column_id=target,
                            parent_table_id=fk.parent_table_id,
                            source_column_id=source,
                            foreign_key_id=fk.foreign_key_id,
                            time_column_id=time_column,
                        )
                    )
    return tuple(result)


def temporal_aggregate_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[TemporalAggregateCandidate, ...]:
    result: list[TemporalAggregateCandidate] = []
    for target in schema.tables:
        if target.role not in {TableRole.ENTITY, TableRole.EVENT}:
            continue
        if not _has_incoming_fk(schema, target.table_id):
            continue
        for path, endpoint_id in _enumerate_paths(schema, target.table_id, max_depth=2):
            endpoint = schema.table(endpoint_id)
            if endpoint.role not in {TableRole.EVENT, TableRole.DETAIL}:
                continue
            if target.role is TableRole.EVENT and endpoint.role is not TableRole.DETAIL:
                continue
            time_column = _time_column(endpoint)
            if time_column is None:
                continue
            result.append(
                TemporalAggregateCandidate(
                    target_table_id=target.table_id,
                    source_table_id=endpoint.table_id,
                    required_path=path,
                    time_column_id=time_column,
                    operator=AggregateOperator.COUNT,
                    source_column_id=None,
                )
            )
            for source in _usable_feature_columns(schema, database, endpoint.table_id):
                for operator in (
                    AggregateOperator.SUM,
                    AggregateOperator.MAX,
                    AggregateOperator.MIN,
                ):
                    result.append(
                        TemporalAggregateCandidate(
                            target_table_id=target.table_id,
                            source_table_id=endpoint.table_id,
                            required_path=path,
                            time_column_id=time_column,
                            operator=operator,
                            source_column_id=source,
                        )
                    )
    return tuple(result)


def build_relation_attribute_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: RelationAttributeCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    rng = _rng(seed)
    scores = _path_source_scores(
        schema, database, candidate.table_id, candidate.required_path,
        candidate.source_column_id,
    )
    if candidate.classification_kind is ClassificationKind.CATEGORICAL:
        labels = _categorical_quantiles(scores, candidate.class_count)
        threshold = None
        requested_rate = None
    else:
        requested_rate = float(rng.uniform(positive_rate_min, positive_rate_max))
        labels, threshold = _threshold_labels(scores, requested_rate)
    # When the target table carries a TIME column, use temporal ordering for
    # the support/query split so that the export layer can emit cutoff_time.
    target_table = schema.table(candidate.table_id)
    time_col = _time_column(target_table)
    if time_col is not None:
        times = database.table(candidate.table_id).column(time_col)
        ordered = np.argsort(times, kind="stable").astype(np.int64)
        split = _temporal_split(
            labels, ordered, rng, support_fraction=support_fraction,
            min_support_rows=min_support_rows, min_query_rows=min_query_rows,
            min_class_count=min_class_count_per_split,
        )
        split_strategy = "temporal_rows"
        visibility_cutoff = int(times[ordered[-1]]) if len(ordered) else 0
        obs_rules = _observation_rules(schema, visibility_cutoff)
    else:
        split = _stratified_split(
            labels, rng, support_fraction=support_fraction,
            min_support_rows=min_support_rows, min_query_rows=min_query_rows,
            min_class_count=min_class_count_per_split,
        )
        split_strategy = "stratified_rows"
        visibility_cutoff = None
        obs_rules = ()
    if split is None:
        return None
    support, query = split
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id, mechanism=TaskMechanism.RELATION_ATTRIBUTE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.table_id,
        source_table_id=candidate.source_table_id,
        target_column_id=candidate.column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=time_col,
        cutoff_time=visibility_cutoff,
        split_strategy=split_strategy, seed=seed,
        masked_column_ids=(candidate.column_id,),
        observation_rules=obs_rules,
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.table_id,
            required_paths=(candidate.required_path,),
        ),
        classification_kind=candidate.classification_kind,
        threshold=threshold, requested_positive_rate=requested_rate,
        realized_positive_rate=(
            None
            if candidate.classification_kind is ClassificationKind.CATEGORICAL
            else float(np.mean(labels == 1))
        ),
        parameters=(("class_count", candidate.class_count), ("support_fraction", support_fraction)),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_future_event_existence_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.15, positive_rate_max: float = 0.65,
) -> PlannedTask | None:
    rng = _rng(seed)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    if len(times) < 2 or int(times.max()) <= int(times.min()):
        return None
    cutoff_q = float(rng.uniform(cutoff_quantile_min, cutoff_quantile_max))
    cutoff = int(np.quantile(times, cutoff_q))
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    low = cutoff + max(1, round((int(times.max()) - int(times.min())) * horizon_fraction_min))
    high = cutoff + max(1, round((int(times.max()) - int(times.min())) * horizon_fraction_max))
    horizon_candidates = np.unique(np.clip(times[(times > cutoff)], low, high))
    if not len(horizon_candidates):
        return None
    best: tuple[float, int, np.ndarray] | None = None
    for horizon in horizon_candidates:
        plan_stub = (cutoff, int(horizon))
        labels = _future_existence_values(schema, database, candidate, *plan_stub)
        item = (abs(float(np.mean(labels)) - desired), int(horizon), labels)
        if best is None or item[0] < best[0]:
            best = item
    assert best is not None
    _distance, horizon, labels = best
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.entity_table_id,
        source_table_id=candidate.event_table_id,
        target_column_id=candidate.target_column_id,
        foreign_key_id=candidate.foreign_key_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff, horizon_end_time=horizon,
        split_strategy="stratified_entities", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, cutoff),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.entity_table_id,
            required_paths=((candidate.foreign_key_id,),),
        ),
        classification_kind=ClassificationKind.BINARY,
        requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=(("cutoff_quantile", cutoff_q), ("support_fraction", support_fraction)),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_future_event_attribute_condition_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventAttributeCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    rng = _rng(seed)
    path = (candidate.foreign_key_id,)
    scores = _path_source_scores(
        schema, database, candidate.event_table_id, path, candidate.source_column_id
    )
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    labels, threshold = _threshold_labels(scores, desired)
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    event_times = database.table(candidate.event_table_id).column(candidate.time_column_id)
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.event_table_id,
        source_table_id=candidate.parent_table_id,
        target_column_id=candidate.target_column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=candidate.time_column_id,
        row_cutoff_time_column_id=candidate.time_column_id,
        split_strategy="stratified_event_rows", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, int(np.max(event_times))),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.event_table_id,
            required_paths=(path,),
        ),
        classification_kind=ClassificationKind.BINARY,
        threshold=threshold, requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=(("support_fraction", support_fraction),),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_temporal_relational_aggregate_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: TemporalAggregateCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    rng = _rng(seed)
    target = schema.table(candidate.target_table_id)
    source_times = database.table(candidate.source_table_id).column(candidate.time_column_id)
    if len(source_times) < 2 or int(source_times.max()) <= int(source_times.min()):
        return None
    row_cutoff_column = _time_column(target) if target.role is TableRole.EVENT else None
    span = int(source_times.max()) - int(source_times.min())
    window = max(1, round(span * float(rng.uniform(horizon_fraction_min, horizon_fraction_max))))
    if row_cutoff_column is None:
        cutoff = int(np.quantile(source_times, float(rng.uniform(cutoff_quantile_min, cutoff_quantile_max))))
        horizon = min(int(source_times.max()), cutoff + window)
        cutoffs = np.full(database.table(target.table_id).row_count, cutoff, dtype=np.int64)
    else:
        cutoff = None
        horizon = None
        cutoffs = database.table(target.table_id).column(row_cutoff_column).astype(np.int64)
    aggregates = _aggregate_values(schema, database, candidate, cutoffs, window)
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    labels, threshold = _threshold_labels(aggregates, desired)
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    visibility_cutoff = int(np.max(cutoffs)) if cutoff is None else int(cutoff)
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.target_table_id,
        source_table_id=candidate.source_table_id,
        target_column_id=candidate.target_column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff, horizon_end_time=horizon,
        row_cutoff_time_column_id=row_cutoff_column,
        split_strategy="stratified_rows", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, visibility_cutoff),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.target_table_id,
            required_paths=(candidate.required_path,),
        ),
        classification_kind=ClassificationKind.BINARY,
        aggregate_operator=candidate.operator,
        threshold=threshold, requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=(("window", window), ("support_fraction", support_fraction)),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def mechanism_labels(
    schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan
) -> np.ndarray:
    required = tuple(
        label.foreign_key_ids for label in plan.route_supervision
        if label.role is RouteRole.REQUIRED
    )
    if plan.mechanism is TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE:
        candidate = FutureEventCandidate(
            foreign_key_id=plan.foreign_key_id or "",
            entity_table_id=plan.target_table_id,
            event_table_id=plan.source_table_id,
            time_column_id=plan.time_column_id or "",
        )
        return _future_existence_values(
            schema, database, candidate, int(plan.cutoff_time), int(plan.horizon_end_time)
        )
    # Imported/legacy autocomplete tasks use the observed target column as the
    # benchmark label and therefore do not carry synthetic SCM source metadata.
    if (
        plan.mechanism is TaskMechanism.RELATION_ATTRIBUTE
        and plan.source_column_id is None
        and plan.target_column_id is not None
    ):
        return database.table(plan.target_table_id).column(plan.target_column_id)
    if not required:
        raise ValueError("task has no required path")
    if plan.mechanism in {
        TaskMechanism.RELATION_ATTRIBUTE,
        TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION,
    }:
        scores = _path_source_scores(
            schema, database, plan.target_table_id, required[0], plan.source_column_id or ""
        )
        if plan.classification_kind is ClassificationKind.CATEGORICAL:
            return _categorical_quantiles(scores, int(plan.parameter_map["class_count"]))
        return (scores > float(plan.threshold)).astype(np.int8)
    if plan.mechanism is TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE:
        target = schema.table(plan.target_table_id)
        row_time = plan.row_cutoff_time_column_id
        if row_time is None:
            cutoffs = np.full(database.table(target.table_id).row_count, int(plan.cutoff_time), dtype=np.int64)
        else:
            cutoffs = database.table(target.table_id).column(row_time).astype(np.int64)
        candidate = TemporalAggregateCandidate(
            target_table_id=plan.target_table_id,
            source_table_id=plan.source_table_id,
            required_path=required[0], time_column_id=plan.time_column_id or "",
            operator=plan.aggregate_operator or AggregateOperator.COUNT,
            source_column_id=plan.source_column_id,
        )
        values = _aggregate_values(
            schema, database, candidate, cutoffs, int(plan.parameter_map["window"])
        )
        return (values > float(plan.threshold)).astype(np.int8)
    raise ValueError(f"unsupported mechanism {plan.mechanism.value}")


def future_event_labels(schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan) -> np.ndarray:
    return mechanism_labels(schema, database, plan)


def _future_existence_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: FutureEventCandidate, cutoff: int, horizon: int,
) -> np.ndarray:
    fk = _foreign_key(schema, candidate.foreign_key_id)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    assignments = event.column(fk.child_column_id)
    selected = (times > cutoff) & (times <= horizon) & (assignments >= 0)
    labels = np.zeros(database.table(candidate.entity_table_id).row_count, dtype=np.int8)
    labels[np.unique(assignments[selected])] = 1
    return labels


def _path_source_scores(
    schema: PhysicalSchema, database: DatabaseInstance, target_table_id: str,
    path: tuple[str, ...], source_column_id: str,
) -> np.ndarray:
    row_sets, endpoint = _traverse_path(schema, database, target_table_id, path)
    values = _numeric(database.table(endpoint).column(source_column_id))
    scores = np.zeros(len(row_sets), dtype=np.float64)
    for index, rows in enumerate(row_sets):
        if len(rows):
            scores[index] = float(np.mean(values[rows]))
    # Stable tiny tie-breaker; it prevents discrete parent columns collapsing a quantile task.
    scores += np.arange(len(scores), dtype=np.float64) * 1e-9
    return scores


def _aggregate_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: TemporalAggregateCandidate, cutoffs: np.ndarray, window: int,
) -> np.ndarray:
    row_sets, endpoint = _traverse_path(
        schema, database, candidate.target_table_id, candidate.required_path
    )
    if endpoint != candidate.source_table_id:
        raise ValueError("required path endpoint does not match source table")
    source = database.table(endpoint)
    times = source.column(candidate.time_column_id)
    values = (
        None if candidate.source_column_id is None
        else _numeric(source.column(candidate.source_column_id))
    )
    output = np.zeros(len(row_sets), dtype=np.float64)
    for index, rows in enumerate(row_sets):
        selected = rows[(times[rows] > cutoffs[index]) & (times[rows] <= cutoffs[index] + window)]
        if candidate.operator is AggregateOperator.COUNT:
            output[index] = len(selected)
        elif not len(selected):
            output[index] = 0.0
        elif candidate.operator is AggregateOperator.SUM:
            output[index] = float(np.sum(values[selected]))
        elif candidate.operator is AggregateOperator.MAX:
            output[index] = float(np.max(values[selected]))
        else:
            output[index] = float(np.min(values[selected]))
    return output


def _traverse_path(
    schema: PhysicalSchema, database: DatabaseInstance,
    target_table_id: str, path: tuple[str, ...],
) -> tuple[list[np.ndarray], str]:
    groups = [np.asarray([row], dtype=np.int64) for row in range(database.table(target_table_id).row_count)]
    current = target_table_id
    for fk_id in path:
        fk = _foreign_key(schema, fk_id)
        if current == fk.child_table_id:
            assignments = database.table(current).column(fk.child_column_id)
            groups = [
                np.unique(assignments[rows][assignments[rows] >= 0]).astype(np.int64)
                for rows in groups
            ]
            current = fk.parent_table_id
        elif current == fk.parent_table_id:
            assignments = database.table(fk.child_table_id).column(fk.child_column_id)
            child_by_parent = [np.flatnonzero(assignments == row).astype(np.int64) for row in range(database.table(current).row_count)]
            groups = [
                np.unique(np.concatenate([child_by_parent[row] for row in rows])).astype(np.int64)
                if len(rows) else np.asarray([], dtype=np.int64)
                for rows in groups
            ]
            current = fk.child_table_id
        else:
            raise ValueError("required path is not contiguous from target table")
    return groups, current


def _threshold_labels(values: np.ndarray, positive_rate: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(values, 1.0 - positive_rate))
    return (values > threshold).astype(np.int8), threshold


def _categorical_quantiles(values: np.ndarray, class_count: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.int16)
    labels[order] = np.minimum(
        class_count - 1,
        (np.arange(len(values), dtype=np.int64) * class_count) // max(1, len(values)),
    )
    return labels


def _stratified_split(
    labels: np.ndarray, rng: np.random.Generator, *, support_fraction: float,
    min_support_rows: int, min_query_rows: int, min_class_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = [np.flatnonzero(labels == value).astype(np.int64) for value in np.unique(labels)]
    if len(groups) < 2 or any(len(group) < 2 * min_class_count for group in groups):
        return None
    support_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for group in groups:
        rng.shuffle(group)
        count = min(len(group) - min_class_count, max(min_class_count, round(len(group) * support_fraction)))
        support_parts.append(group[:count])
        query_parts.append(group[count:])
    support = np.concatenate(support_parts)
    query = np.concatenate(query_parts)
    rng.shuffle(support)
    rng.shuffle(query)
    if len(support) < min_support_rows or len(query) < min_query_rows:
        return None
    return support, query


def _planned(plan: TaskPlan, labels: np.ndarray, support: np.ndarray, query: np.ndarray) -> PlannedTask:
    return PlannedTask(
        plan=plan,
        data=TaskData(
            support_row_ids=support, support_labels=labels[support],
            query_row_ids=query, query_labels=labels[query],
        ),
    )


def _planned_if_visible(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    plan: TaskPlan,
    labels: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
) -> PlannedTask | None:
    task = _planned(plan, labels, support, query)
    target_mask = build_task_view(
        schema, database, plan
    ).row_masks[plan.target_table_id]
    supervised_rows = np.concatenate((support, query))
    if not np.all(target_mask[supervised_rows]):
        return None
    return task


def _usable_feature_columns(
    schema: PhysicalSchema, database: DatabaseInstance, table_id: str
) -> tuple[str, ...]:
    table = schema.table(table_id)
    return tuple(
        column.column_id for column in table.columns
        if column.kind is ColumnKind.FEATURE
        and len(_observed_values(database.table(table_id).column(column.column_id)))
        and len(np.unique(_observed_values(database.table(table_id).column(column.column_id)))) > 1
    )


def _time_column(table) -> str | None:
    return next((column.column_id for column in table.columns if column.kind is ColumnKind.TIME), None)


def _numeric(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"i", "u", "f", "b"}:
        result = values.astype(np.float64)
        if result.dtype.kind == "f":
            finite = np.isfinite(result)
            result[~finite] = float(np.mean(result[finite])) if np.any(finite) else 0.0
        return result
    uniques = sorted(value for value in np.unique(values).tolist() if value != "")
    mapping = {value: index for index, value in enumerate(uniques)}
    return np.asarray([mapping.get(value, -1) for value in values], dtype=np.float64)


def _observed_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        return np.isfinite(values)
    if values.dtype.kind in {"U", "S"}:
        return values != ""
    return np.ones(len(values), dtype=bool)


def _observed_values(values: np.ndarray) -> np.ndarray:
    return values[_observed_mask(values)]


def _observation_rules(schema: PhysicalSchema, cutoff: int) -> tuple[ObservationRule, ...]:
    return tuple(
        ObservationRule(table_id=table.table_id, time_column_id=column.column_id, max_timestamp=cutoff)
        for table in schema.tables
        for column in table.columns
        if table.role in {TableRole.EVENT, TableRole.DETAIL} and column.kind is ColumnKind.TIME
    )


def _has_incoming_fk(schema: PhysicalSchema, table_id: str) -> bool:
    """Return True when *table_id* has at least one incoming FK edge."""
    return any(
        fk.child_table_id == table_id for fk in schema.foreign_keys
    )


def _foreign_key(schema: PhysicalSchema, foreign_key_id: str) -> PhysicalForeignKey:
    return next(fk for fk in schema.foreign_keys if fk.foreign_key_id == foreign_key_id)


def _enumerate_paths(
    schema: PhysicalSchema, target_table_id: str, *, max_depth: int
) -> tuple[tuple[tuple[str, ...], str], ...]:
    adjacent: dict[str, list[tuple[str, str]]] = {table.table_id: [] for table in schema.tables}
    for fk in schema.foreign_keys:
        adjacent[fk.parent_table_id].append((fk.foreign_key_id, fk.child_table_id))
        adjacent[fk.child_table_id].append((fk.foreign_key_id, fk.parent_table_id))
    frontier = [(target_table_id, (), frozenset({target_table_id}))]
    result: list[tuple[tuple[str, ...], str]] = []
    for _ in range(max_depth):
        following = []
        for current, path, visited in frontier:
            for fk_id, destination in sorted(adjacent[current]):
                if destination in visited:
                    continue
                candidate = path + (fk_id,)
                result.append((candidate, destination))
                following.append((destination, candidate, visited | {destination}))
        frontier = following
    return tuple(result)


def _schema_route_labels(
    schema: PhysicalSchema, *, target_table_id: str,
    required_paths: tuple[tuple[str, ...], ...] = (),
    optional_paths: tuple[tuple[str, ...], ...] = (), max_depth: int = 2,
) -> tuple[RoutePathLabel, ...]:
    required = set(required_paths)
    optional = set(optional_paths)
    paths = [path for path, _endpoint in _enumerate_paths(schema, target_table_id, max_depth=max_depth)]
    missing = required - set(paths)
    if missing:
        raise ValueError(f"required paths are not legal schema paths: {sorted(missing)!r}")
    return tuple(
        RoutePathLabel(
            foreign_key_ids=path,
            role=(RouteRole.REQUIRED if path in required else RouteRole.OPTIONAL if path in optional else RouteRole.DISTRACTOR),
        )
        for path in paths
    )


def _temporal_split(
    labels: np.ndarray,
    ordered: np.ndarray,
    rng: np.random.Generator,
    *,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Support/query split respecting temporal order for cutoff_time export."""
    class_values = np.unique(labels)
    groups = [np.flatnonzero(labels == value).astype(np.int64) for value in class_values]
    if len(groups) < 2 or any(len(group) < 2 * min_class_count for group in groups):
        return None
    support_count = max(min_support_rows, round(len(ordered) * support_fraction))
    support_count = min(support_count, len(ordered) - min_query_rows)
    if support_count < min_support_rows:
        return None
    support = ordered[:support_count].copy()
    query = ordered[support_count:].copy()
    if len(query) < min_query_rows:
        return None
    for rows in (support, query):
        if any(
            np.count_nonzero(labels[rows] == value) < min_class_count
            for value in class_values
        ):
            return None
    rng.shuffle(support)
    rng.shuffle(query)
    return support, query


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(seed))


__all__ = [
    "TargetSample",
    "RelationAttributeCandidate", "FutureEventCandidate",
    "FutureEventAttributeCandidate", "TemporalAggregateCandidate",
    "relation_attribute_candidates", "future_event_candidates",
    "future_event_attribute_candidates", "temporal_aggregate_candidates",
    "build_relation_attribute_task", "build_future_event_existence_task",
    "build_future_event_attribute_condition_task",
    "build_temporal_relational_aggregate_task", "mechanism_labels",
    "future_event_labels",
]
