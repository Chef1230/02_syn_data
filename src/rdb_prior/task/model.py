"""Serializable task plans and support/query labels for stage 03."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class TaskMechanism(str, Enum):
    RELATION_ATTRIBUTE = "relation_attribute"
    FUTURE_EVENT_EXISTENCE = "future_event_existence"


class PredictionType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class RouteRole(str, Enum):
    """Synthetic Task DSL supervision for one exact FK path."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    DISTRACTOR = "distractor"


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutePathLabel:
    foreign_key_ids: tuple[str, ...]
    role: RouteRole

    def __post_init__(self) -> None:
        if not isinstance(self.foreign_key_ids, tuple) or not (
            self.foreign_key_ids
        ):
            raise ValueError("foreign_key_ids must be a non-empty tuple")
        for foreign_key_id in self.foreign_key_ids:
            _identifier("route foreign key", foreign_key_id)
        if not isinstance(self.role, RouteRole):
            raise TypeError("role must be RouteRole")

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreign_key_ids": list(self.foreign_key_ids),
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutePathLabel:
        return cls(
            foreign_key_ids=tuple(data["foreign_key_ids"]),
            role=RouteRole(data["role"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRule:
    table_id: str
    time_column_id: str
    max_timestamp: int

    def __post_init__(self) -> None:
        _identifier("table_id", self.table_id)
        _identifier("time_column_id", self.time_column_id)
        if isinstance(self.max_timestamp, bool) or not isinstance(
            self.max_timestamp, int
        ):
            raise TypeError("max_timestamp must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "time_column_id": self.time_column_id,
            "max_timestamp": self.max_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationRule:
        return cls(
            table_id=data["table_id"],
            time_column_id=data["time_column_id"],
            max_timestamp=data["max_timestamp"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPlan:
    task_id: str
    sample_id: str
    instance_id: str
    schema_id: str
    mechanism: TaskMechanism
    prediction_type: PredictionType
    target_table_id: str
    source_table_id: str
    split_strategy: str
    seed: int
    target_column_id: str | None = None
    foreign_key_id: str | None = None
    time_column_id: str | None = None
    cutoff_time: int | None = None
    horizon_end_time: int | None = None
    masked_column_ids: tuple[str, ...] = ()
    observation_rules: tuple[ObservationRule, ...] = ()
    route_supervision: tuple[RoutePathLabel, ...] = ()
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "sample_id",
            "instance_id",
            "schema_id",
            "target_table_id",
            "source_table_id",
            "split_strategy",
        ):
            _identifier(name, getattr(self, name))
        if not isinstance(self.mechanism, TaskMechanism):
            raise TypeError("mechanism must be TaskMechanism")
        if not isinstance(self.prediction_type, PredictionType):
            raise TypeError("prediction_type must be PredictionType")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("target_column_id", "foreign_key_id", "time_column_id"):
            value = getattr(self, name)
            if value is not None:
                _identifier(name, value)
        for name in ("cutoff_time", "horizon_end_time"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if (
            self.cutoff_time is not None
            and self.horizon_end_time is not None
            and self.horizon_end_time <= self.cutoff_time
        ):
            raise ValueError("horizon_end_time must be after cutoff_time")
        if not isinstance(self.masked_column_ids, tuple):
            raise TypeError("masked_column_ids must be a tuple")
        for column_id in self.masked_column_ids:
            _identifier("masked column", column_id)
        if len(set(self.masked_column_ids)) != len(self.masked_column_ids):
            raise ValueError("masked_column_ids must be unique")
        if not isinstance(self.observation_rules, tuple) or not all(
            isinstance(rule, ObservationRule) for rule in self.observation_rules
        ):
            raise TypeError("observation_rules must contain ObservationRule")
        if not isinstance(self.route_supervision, tuple) or not all(
            isinstance(label, RoutePathLabel)
            for label in self.route_supervision
        ):
            raise TypeError("route_supervision must contain RoutePathLabel")
        route_paths = tuple(
            label.foreign_key_ids for label in self.route_supervision
        )
        if len(set(route_paths)) != len(route_paths):
            raise ValueError("route_supervision paths must be unique")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        self._validate_mechanism_contract()

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.mechanism.value,
            self.target_table_id,
            self.source_table_id,
            self.target_column_id,
            self.foreign_key_id,
            self.cutoff_time,
            self.horizon_end_time,
        )

    def _validate_mechanism_contract(self) -> None:
        if self.mechanism is TaskMechanism.RELATION_ATTRIBUTE:
            if self.target_column_id is None:
                raise ValueError("relation attribute task requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("relation attribute target must be masked")
            if any(
                value is not None
                for value in (
                    self.foreign_key_id,
                    self.cutoff_time,
                    self.horizon_end_time,
                )
            ):
                raise ValueError("relation attribute task has temporal FK fields")
        elif self.mechanism is TaskMechanism.FUTURE_EVENT_EXISTENCE:
            if self.prediction_type is not PredictionType.CLASSIFICATION:
                raise ValueError("future event existence must be classification")
            if None in (
                self.foreign_key_id,
                self.time_column_id,
                self.cutoff_time,
                self.horizon_end_time,
            ):
                raise ValueError("future event existence requires FK and horizon")
            if self.target_column_id is not None or self.masked_column_ids:
                raise ValueError("future event existence uses a synthetic label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "instance_id": self.instance_id,
            "schema_id": self.schema_id,
            "mechanism": self.mechanism.value,
            "prediction_type": self.prediction_type.value,
            "target_table_id": self.target_table_id,
            "source_table_id": self.source_table_id,
            "target_column_id": self.target_column_id,
            "foreign_key_id": self.foreign_key_id,
            "time_column_id": self.time_column_id,
            "cutoff_time": self.cutoff_time,
            "horizon_end_time": self.horizon_end_time,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "masked_column_ids": list(self.masked_column_ids),
            "observation_rules": [
                rule.to_dict() for rule in self.observation_rules
            ],
            "route_supervision": [
                label.to_dict() for label in self.route_supervision
            ],
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskPlan:
        return cls(
            task_id=data["task_id"],
            sample_id=data["sample_id"],
            instance_id=data["instance_id"],
            schema_id=data["schema_id"],
            mechanism=TaskMechanism(data["mechanism"]),
            prediction_type=PredictionType(data["prediction_type"]),
            target_table_id=data["target_table_id"],
            source_table_id=data["source_table_id"],
            target_column_id=data.get("target_column_id"),
            foreign_key_id=data.get("foreign_key_id"),
            time_column_id=data.get("time_column_id"),
            cutoff_time=data.get("cutoff_time"),
            horizon_end_time=data.get("horizon_end_time"),
            split_strategy=data["split_strategy"],
            seed=data["seed"],
            masked_column_ids=tuple(data.get("masked_column_ids", ())),
            observation_rules=tuple(
                ObservationRule.from_dict(item)
                for item in data.get("observation_rules", ())
            ),
            route_supervision=tuple(
                RoutePathLabel.from_dict(item)
                for item in data.get("route_supervision", ())
            ),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskData:
    support_row_ids: np.ndarray
    support_labels: np.ndarray
    query_row_ids: np.ndarray
    query_labels: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "support_row_ids",
            "support_labels",
            "query_row_ids",
            "query_labels",
        ):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional ndarray")
            if value.dtype == object:
                raise TypeError(f"{name} cannot use object dtype")
        for name in ("support_row_ids", "query_row_ids"):
            value = getattr(self, name)
            if value.dtype.kind not in {"i", "u"}:
                raise TypeError(f"{name} must use an integer dtype")
            if np.any(value < 0) or len(np.unique(value)) != len(value):
                raise ValueError(f"{name} must contain unique non-negative rows")
        if len(self.support_row_ids) != len(self.support_labels):
            raise ValueError("support rows and labels must align")
        if len(self.query_row_ids) != len(self.query_labels):
            raise ValueError("query rows and labels must align")
        if not len(self.support_row_ids) or not len(self.query_row_ids):
            raise ValueError("support and query must both be non-empty")
        if np.intersect1d(self.support_row_ids, self.query_row_ids).size:
            raise ValueError("support and query rows must be disjoint")

    @property
    def total_rows(self) -> int:
        return len(self.support_row_ids) + len(self.query_row_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedTask:
    plan: TaskPlan
    data: TaskData


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _parameters(
    values: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple):
        raise TypeError("parameters must be a tuple")
    result: list[tuple[str, float]] = []
    for name, value in values:
        _identifier("parameter name", name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("parameter values must be numeric")
        result.append((name, float(value)))
    if len({name for name, _value in result}) != len(result):
        raise ValueError("parameter names must be unique")
    return tuple(sorted(result))


__all__ = [
    "TaskMechanism",
    "PredictionType",
    "RouteRole",
    "RoutePathLabel",
    "ObservationRule",
    "TaskPlan",
    "TaskData",
    "PlannedTask",
]
