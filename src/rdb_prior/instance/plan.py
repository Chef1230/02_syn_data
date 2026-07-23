"""Serializable execution plan for one relational database instance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from rdb_prior.schema.spec import TableRole


class FeatureSCMFamily(str, Enum):
    EXOGENOUS = "exogenous"
    LINEAR = "linear"
    CAM = "cam"
    MLP = "mlp"


class RootCauseFamily(str, Enum):
    """Distribution family for exogenous latent (root) variables in the SCM.

    Each table draws one family; its latent rows are then generated from
    that distribution and standardised to unit variance so that downstream
    feature SCMs receive inputs with diverse shape characteristics (skew,
    heavy tails, multi-modality) while maintaining consistent scale.
    """

    STANDARD_NORMAL = "standard_normal"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    LOGNORMAL = "lognormal"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


class TemporalFamily(str, Enum):
    NONE = "none"
    PARENT_BURST = "parent_burst"
    TIME_LAGGED = "time_lagged"


@dataclass(frozen=True, slots=True, kw_only=True)
class PopulationPlan:
    strategy: str
    row_count: int
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _identifier("strategy", self.strategy)
        _positive_int("row_count", self.row_count)
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters),
        )

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True, kw_only=True)
class TableMechanismPlan:
    table_id: str
    role: TableRole
    population: PopulationPlan
    latent_dimension: int
    feature_family: FeatureSCMFamily
    temporal_family: TemporalFamily
    latent_seed: int
    feature_seed: int
    temporal_seed: int
    root_cause_family: RootCauseFamily = RootCauseFamily.STANDARD_NORMAL
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _identifier("table_id", self.table_id)
        if not isinstance(self.role, TableRole):
            raise TypeError("role must be TableRole")
        if not isinstance(self.population, PopulationPlan):
            raise TypeError("population must be PopulationPlan")
        _positive_int("latent_dimension", self.latent_dimension)
        if not isinstance(self.feature_family, FeatureSCMFamily):
            raise TypeError("feature_family must be FeatureSCMFamily")
        if not isinstance(self.root_cause_family, RootCauseFamily):
            raise TypeError("root_cause_family must be RootCauseFamily")
        if not isinstance(self.temporal_family, TemporalFamily):
            raise TypeError("temporal_family must be TemporalFamily")
        for name in ("latent_seed", "feature_seed", "temporal_seed"):
            _seed(name, getattr(self, name))
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationMechanismPlan:
    relation_group_id: str
    foreign_key_ids: tuple[str, ...]
    parent_table_ids: tuple[str, ...]
    child_table_id: str
    family: str
    optional_rates: tuple[float, ...]
    seed: int
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _identifier("relation_group_id", self.relation_group_id)
        _identifier_tuple("foreign_key_ids", self.foreign_key_ids)
        _identifier_tuple("parent_table_ids", self.parent_table_ids)
        if len(self.foreign_key_ids) != len(self.parent_table_ids):
            raise ValueError("foreign_key_ids and parent_table_ids must align")
        _identifier("child_table_id", self.child_table_id)
        _identifier("family", self.family)
        if not isinstance(self.optional_rates, tuple):
            raise TypeError("optional_rates must be a tuple")
        if len(self.optional_rates) != len(self.foreign_key_ids):
            raise ValueError("optional_rates must align with foreign keys")
        for rate in self.optional_rates:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise TypeError("optional rates must be numeric")
            if not 0 <= rate < 1:
                raise ValueError("optional rates must be in [0, 1)")
        _seed("seed", self.seed)
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters),
        )

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True, kw_only=True)
class InstancePlan:
    plan_id: str
    sample_id: str
    schema_id: str
    blueprint_id: str
    global_seed: int
    generation_order: tuple[str, ...]
    tables: tuple[TableMechanismPlan, ...]
    relations: tuple[RelationMechanismPlan, ...]
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("plan_id", "sample_id", "schema_id", "blueprint_id"):
            _identifier(name, getattr(self, name))
        _seed("global_seed", self.global_seed)
        _identifier_tuple("generation_order", self.generation_order)
        if not isinstance(self.tables, tuple) or not all(
            isinstance(item, TableMechanismPlan) for item in self.tables
        ):
            raise TypeError("tables must contain TableMechanismPlan values")
        if not isinstance(self.relations, tuple) or not all(
            isinstance(item, RelationMechanismPlan) for item in self.relations
        ):
            raise TypeError(
                "relations must contain RelationMechanismPlan values"
            )
        table_ids = tuple(table.table_id for table in self.tables)
        if len(set(table_ids)) != len(table_ids):
            raise ValueError("table plan IDs must be unique")
        if set(self.generation_order) != set(table_ids):
            raise ValueError("generation_order must contain every table once")
        relation_ids = tuple(
            relation.relation_group_id for relation in self.relations
        )
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("relation group IDs must be unique")
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def table(self, table_id: str) -> TableMechanismPlan:
        return next(table for table in self.tables if table.table_id == table_id)

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "sample_id": self.sample_id,
            "schema_id": self.schema_id,
            "blueprint_id": self.blueprint_id,
            "global_seed": self.global_seed,
            "generation_order": list(self.generation_order),
            "tables": [
                {
                    "table_id": table.table_id,
                    "role": table.role.value,
                    "population": {
                        "strategy": table.population.strategy,
                        "row_count": table.population.row_count,
                        "parameters": dict(table.population.parameters),
                    },
                    "latent_dimension": table.latent_dimension,
                    "feature_family": table.feature_family.value,
                    "root_cause_family": table.root_cause_family.value,
                    "temporal_family": table.temporal_family.value,
                    "latent_seed": table.latent_seed,
                    "feature_seed": table.feature_seed,
                    "temporal_seed": table.temporal_seed,
                    "parameters": dict(table.parameters),
                }
                for table in self.tables
            ],
            "relations": [
                {
                    "relation_group_id": relation.relation_group_id,
                    "foreign_key_ids": list(relation.foreign_key_ids),
                    "parent_table_ids": list(relation.parent_table_ids),
                    "child_table_id": relation.child_table_id,
                    "family": relation.family,
                    "optional_rates": list(relation.optional_rates),
                    "seed": relation.seed,
                    "parameters": dict(relation.parameters),
                }
                for relation in self.relations
            ],
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InstancePlan:
        if not isinstance(data, Mapping):
            raise TypeError("InstancePlan payload must be a mapping")
        table_values: list[TableMechanismPlan] = []
        for item in data["tables"]:
            population = item["population"]
            table_values.append(
                TableMechanismPlan(
                    table_id=item["table_id"],
                    role=TableRole(item["role"]),
                    population=PopulationPlan(
                        strategy=population["strategy"],
                        row_count=population["row_count"],
                        parameters=tuple(population.get("parameters", {}).items()),
                    ),
                    latent_dimension=item["latent_dimension"],
                    feature_family=FeatureSCMFamily(item["feature_family"]),
                    root_cause_family=RootCauseFamily(
                        item.get("root_cause_family", "standard_normal")
                    ),
                    temporal_family=TemporalFamily(item["temporal_family"]),
                    latent_seed=item["latent_seed"],
                    feature_seed=item["feature_seed"],
                    temporal_seed=item["temporal_seed"],
                    parameters=tuple(item.get("parameters", {}).items()),
                )
            )
        relation_values = tuple(
            RelationMechanismPlan(
                relation_group_id=item["relation_group_id"],
                foreign_key_ids=tuple(item["foreign_key_ids"]),
                parent_table_ids=tuple(item["parent_table_ids"]),
                child_table_id=item["child_table_id"],
                family=item["family"],
                optional_rates=tuple(item["optional_rates"]),
                seed=item["seed"],
                parameters=tuple(item.get("parameters", {}).items()),
            )
            for item in data["relations"]
        )
        return cls(
            plan_id=data["plan_id"],
            sample_id=data["sample_id"],
            schema_id=data["schema_id"],
            blueprint_id=data["blueprint_id"],
            global_seed=data["global_seed"],
            generation_order=tuple(data["generation_order"]),
            tables=tuple(table_values),
            relations=relation_values,
            parameters=tuple(data.get("parameters", {}).items()),
        )


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _seed(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _identifier_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    for item in value:
        _identifier(name, item)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique values")


def _parameters(
    value: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, tuple):
        raise TypeError("parameters must be a tuple")
    result: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("parameters items must be pairs")
        name, parameter = item
        _identifier("parameter name", name)
        if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
            raise TypeError("parameter values must be numeric")
        result.append((name, float(parameter)))
    if len({name for name, _value in result}) != len(result):
        raise ValueError("parameter names must be unique")
    return tuple(sorted(result))


__all__ = [
    "FeatureSCMFamily",
    "RootCauseFamily",
    "TemporalFamily",
    "PopulationPlan",
    "TableMechanismPlan",
    "RelationMechanismPlan",
    "InstancePlan",
]
