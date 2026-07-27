"""Deterministic eligibility matching and task sampling for stage 03."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.runtime import RuntimeContext
from rdb_prior.task.mechanisms import (
    build_future_event_attribute_condition_task,
    build_future_event_existence_task,
    build_relation_attribute_task,
    build_temporal_relational_aggregate_task,
    future_event_attribute_candidates,
    future_event_candidates,
    relation_attribute_candidates,
    temporal_aggregate_candidates,
)
from rdb_prior.task.model import PlannedTask, TaskMechanism


_DEFAULT_MECHANISM_WEIGHTS = (
    (TaskMechanism.RELATION_ATTRIBUTE, 0.35),
    (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 0.25),
    (TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION, 0.20),
    (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 0.20),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPlannerConfig:
    tasks_per_database: int = 2
    mechanism_weights: tuple[tuple[TaskMechanism, float], ...] = _DEFAULT_MECHANISM_WEIGHTS
    support_fraction: float = 0.7
    min_support_rows: int = 32
    min_query_rows: int = 16
    min_class_count_per_split: int = 2
    max_classification_categories: int = 12
    cutoff_quantile_min: float = 0.45
    cutoff_quantile_max: float = 0.7
    horizon_fraction_min: float = 0.12
    horizon_fraction_max: float = 0.3
    positive_rate_min: float = 0.2
    positive_rate_max: float = 0.8
    max_attempts_per_database: int = 128
    require_full_task_count: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tasks_per_database", "min_support_rows", "min_query_rows",
            "min_class_count_per_split", "max_classification_categories",
            "max_attempts_per_database",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.support_fraction < 1:
            raise ValueError("support_fraction must be in (0, 1)")
        for name, low, high in (
            ("cutoff quantile", self.cutoff_quantile_min, self.cutoff_quantile_max),
            ("horizon fraction", self.horizon_fraction_min, self.horizon_fraction_max),
            ("positive rate", self.positive_rate_min, self.positive_rate_max),
        ):
            _fraction_range(name, low, high)
        if not isinstance(self.require_full_task_count, bool):
            raise TypeError("require_full_task_count must be a boolean")
        if not isinstance(self.mechanism_weights, tuple) or not self.mechanism_weights:
            raise ValueError("mechanism_weights must be a non-empty tuple")
        mechanisms = tuple(item[0] for item in self.mechanism_weights)
        if len(set(mechanisms)) != len(mechanisms):
            raise ValueError("mechanism_weights contains duplicate mechanisms")
        for mechanism, weight in self.mechanism_weights:
            if not isinstance(mechanism, TaskMechanism):
                raise TypeError("mechanism_weights keys must be TaskMechanism")
            if weight <= 0:
                raise ValueError("mechanism weights must be positive")


class TaskPlanner:
    def __init__(self, config: TaskPlannerConfig | None = None) -> None:
        self.config = config or TaskPlannerConfig()

    def generate(
        self, *, sample_id: str, schema: PhysicalSchema,
        database: DatabaseInstance, runtime: RuntimeContext,
    ) -> tuple[PlannedTask, ...]:
        pools: dict[TaskMechanism, list[object]] = {
            TaskMechanism.RELATION_ATTRIBUTE: list(
                relation_attribute_candidates(
                    schema, database,
                    max_classification_categories=self.config.max_classification_categories,
                )
            ),
            TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE: list(future_event_candidates(schema)),
            TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION: list(
                future_event_attribute_candidates(schema, database)
            ),
            TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE: list(
                temporal_aggregate_candidates(schema, database)
            ),
        }
        rng = runtime.numpy_rng("task-selection")
        for pool in pools.values():
            rng.shuffle(pool)
        mechanisms, weights = zip(*self.config.mechanism_weights)
        generated: list[PlannedTask] = []
        signatures: set[tuple[object, ...]] = set()
        for attempt in range(self.config.max_attempts_per_database):
            if len(generated) >= self.config.tasks_per_database:
                break
            available = [mechanism for mechanism in mechanisms if pools.get(mechanism)]
            if not available:
                break
            available_weights = [weights[mechanisms.index(mechanism)] for mechanism in available]
            mechanism = available[int(rng.choice(len(available), p=_normalize(available_weights)))]
            candidate_pool = pools[mechanism]
            candidate = candidate_pool[int(rng.integers(0, len(candidate_pool)))]
            task_index = len(generated)
            common = dict(
                task_id=f"task_{sample_id}_{task_index:03d}", sample_id=sample_id,
                schema=schema, database=database, candidate=candidate,
                seed=runtime.seed("task", task_index, "attempt", attempt),
                support_fraction=self.config.support_fraction,
                min_support_rows=self.config.min_support_rows,
                min_query_rows=self.config.min_query_rows,
                min_class_count_per_split=self.config.min_class_count_per_split,
            )
            if mechanism is TaskMechanism.RELATION_ATTRIBUTE:
                task = build_relation_attribute_task(
                    **common, positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            elif mechanism is TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE:
                task = build_future_event_existence_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=min(self.config.positive_rate_max, 0.8),
                )
            elif mechanism is TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION:
                task = build_future_event_attribute_condition_task(
                    **common, positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            else:
                task = build_temporal_relational_aggregate_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            if task is None or task.plan.signature in signatures:
                continue
            signatures.add(task.plan.signature)
            generated.append(task)
        if self.config.require_full_task_count and len(generated) != self.config.tasks_per_database:
            raise ValueError(
                f"database {sample_id!r} yielded {len(generated)} valid tasks; "
                f"required {self.config.tasks_per_database}"
            )
        return tuple(generated)


def _fraction_range(name: str, low: float, high: float) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
        raise TypeError(f"{name} bounds must be numeric")
    if not 0 < low <= high < 1:
        raise ValueError(f"{name} bounds must satisfy 0 < min <= max < 1")


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


__all__ = ["TaskPlannerConfig", "TaskPlanner"]
