# src/rdb_prior/schema/spec.py
# -*- coding: utf-8 -*-
"""Dependency-light schema enums and structural constraint contracts.

The V1 role catalog is intentionally closed and contains exactly five roles:
Entity, Event, Lookup, Bridge and Detail.  ``schema.roles`` owns the policy of
those roles; this module owns their stable identifiers and the immutable
constraint vocabulary shared by samplers, validators and compilers.

Constraints describe requirements only.  Evaluation belongs in
``schema.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import math
from typing import Any, Mapping, TypeAlias


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class TableRole(str, Enum):
    """Closed V1 structural-semantic role catalog."""

    ENTITY = "entity"
    EVENT = "event"
    LOOKUP = "lookup"
    BRIDGE = "bridge"
    DETAIL = "detail"


class EdgeKind(str, Enum):
    """Logical kind of a schema edge."""

    FOREIGN_KEY = "foreign_key"
    ASSOCIATION = "association"
    DERIVATION = "derivation"
    TEMPORAL_PRECEDENCE = "temporal_precedence"


class Cardinality(str, Enum):
    """Cardinality in referenced-parent -> referencing-child orientation."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"

    # Conceptual only.  A physical FK rule must realize many-to-many through
    # a Bridge table rather than attach this value to one FK edge.
    MANY_TO_MANY = "many_to_many"


class Optionality(str, Enum):
    """Whether the child-side reference is required."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class IdentityDependency(str, Enum):
    """Whether a child depends on its parent for row identity."""

    INDEPENDENT = "independent"
    IDENTIFYING = "identifying"
    WEAK = "weak"


class TemporalMode(str, Enum):
    """Temporal behavior expected from a table or relation."""

    NONE = "none"
    STATIC = "static"
    EVENT_TIME = "event_time"
    VALID_TIME = "valid_time"
    AS_OF_SUMMARY = "as_of_summary"


class ConstraintSeverity(str, Enum):
    """Whether a violation invalidates the artifact or emits a warning."""

    HARD = "hard"
    SOFT = "soft"


class ConstraintStage(str, Enum):
    """Earliest stage at which a constraint can be fully checked."""

    BLUEPRINT = "blueprint"
    COMPILATION = "compilation"
    INSTANCE_PLAN = "instance_plan"
    DATABASE = "database"


class DensityDefinition(str, Enum):
    """Exact denominator used by :class:`EdgeDensityConstraint`."""

    # m / n, where m is the selected edge count and n is table count.
    EDGES_PER_NODE = "edges_per_node"

    # m / (n * (n - 1)); self-loops are excluded and direction matters.
    SIMPLE_DIRECTED = "simple_directed"


class ConstraintKind(str, Enum):
    """Stable discriminator used for serialization and validator dispatch."""

    CONNECTED = "connected"
    ACYCLIC = "acyclic"
    NO_PARALLEL_EDGES = "no_parallel_edges"
    TABLE_COUNT = "table_count"
    EDGE_COUNT = "edge_count"
    EDGE_DENSITY = "edge_density"
    ROLE_COUNT = "role_count"
    ALLOWED_ROLE_EDGE = "allowed_role_edge"
    FORBIDDEN_ROLE_EDGE = "forbidden_role_edge"
    REQUIRED_ROLE_EDGE = "required_role_edge"
    PARENT_COUNT = "parent_count"
    CHILD_COUNT = "child_count"
    RANK_ORDER = "rank_order"
    REACHABILITY = "reachability"
    TEMPORAL_ORDER = "temporal_order"
    UNIQUE_LEAF = "unique_leaf"


# ---------------------------------------------------------------------------
# Stable logical identifiers
# ---------------------------------------------------------------------------


NodeId: TypeAlias = str
EdgeId: TypeAlias = str
ConstraintId: TypeAlias = str


# ---------------------------------------------------------------------------
# Constraint base
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstraintBase:
    """Fields shared by all concrete schema constraints."""

    constraint_id: ConstraintId
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    stage: ConstraintStage = ConstraintStage.BLUEPRINT
    description: str | None = None
    kind: ConstraintKind = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is ConstraintBase:
            raise TypeError("ConstraintBase is abstract; use a concrete constraint")

        _validate_identifier("constraint_id", self.constraint_id)
        _validate_enum("severity", self.severity, ConstraintSeverity)
        _validate_enum("stage", self.stage, ConstraintStage)

        if self.description is not None:
            _validate_identifier("description", self.description)


# ---------------------------------------------------------------------------
# Global graph constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectedConstraint(ConstraintBase):
    """Require the selected schema graph to be weakly connected."""

    kind: ConstraintKind = field(init=False, default=ConstraintKind.CONNECTED)


@dataclass(frozen=True, slots=True, kw_only=True)
class AcyclicConstraint(ConstraintBase):
    """Require the selected edge-kind projection to be acyclic."""

    edge_kinds: tuple[EdgeKind, ...] = (EdgeKind.FOREIGN_KEY,)
    kind: ConstraintKind = field(init=False, default=ConstraintKind.ACYCLIC)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_enum_tuple("edge_kinds", self.edge_kinds, EdgeKind)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoParallelEdgesConstraint(ConstraintBase):
    """Forbid duplicate parent/child/kind logical edges."""

    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(
        init=False,
        default=ConstraintKind.NO_PARALLEL_EDGES,
    )

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)


# ---------------------------------------------------------------------------
# Size and distribution constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TableCountConstraint(ConstraintBase):
    minimum: int | None = None
    maximum: int | None = None
    kind: ConstraintKind = field(init=False, default=ConstraintKind.TABLE_COUNT)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_bounds(
            name="table count",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeCountConstraint(ConstraintBase):
    minimum: int | None = None
    maximum: int | None = None
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(init=False, default=ConstraintKind.EDGE_COUNT)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)
        _validate_bounds(
            name="edge count",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeDensityConstraint(ConstraintBase):
    """Constrain edge density under one explicitly selected formula."""

    minimum: float | None = None
    maximum: float | None = None
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    definition: DensityDefinition = DensityDefinition.EDGES_PER_NODE
    kind: ConstraintKind = field(init=False, default=ConstraintKind.EDGE_DENSITY)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)
        _validate_enum("definition", self.definition, DensityDefinition)
        _validate_bounds(
            name="edge density",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=False,
        )

        if self.definition is DensityDefinition.SIMPLE_DIRECTED:
            for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
                if value is not None and value > 1:
                    raise ValueError(
                        f"simple-directed density {name} must not exceed 1"
                    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleCountConstraint(ConstraintBase):
    role: TableRole
    minimum: int | None = None
    maximum: int | None = None
    kind: ConstraintKind = field(init=False, default=ConstraintKind.ROLE_COUNT)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_enum("role", self.role, TableRole)
        _validate_bounds(
            name=f"{self.role.value} role count",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=True,
        )


# ---------------------------------------------------------------------------
# Role-aware edge constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AllowedRoleEdgeConstraint(ConstraintBase):
    """Permit a role pair within an already globally legal role policy.

    ``schema.roles.ROLE_EDGE_RULES`` remains the global hard upper bound.  This
    constraint may select a subset for one blueprint/domain, but it cannot make
    a globally forbidden pair legal and it does not require an edge to exist.
    """

    parent_role: TableRole
    child_role: TableRole
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(
        init=False,
        default=ConstraintKind.ALLOWED_ROLE_EDGE,
    )

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_role_edge_fields(self.parent_role, self.child_role, self.edge_kind)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForbiddenRoleEdgeConstraint(ConstraintBase):
    """Further forbid a role pair for one blueprint or domain."""

    parent_role: TableRole
    child_role: TableRole
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(
        init=False,
        default=ConstraintKind.FORBIDDEN_ROLE_EDGE,
    )

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_role_edge_fields(self.parent_role, self.child_role, self.edge_kind)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequiredRoleEdgeConstraint(ConstraintBase):
    """Require at least ``minimum`` edges for one globally legal role pair."""

    parent_role: TableRole
    child_role: TableRole
    minimum: int = 1
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(
        init=False,
        default=ConstraintKind.REQUIRED_ROLE_EDGE,
    )

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_role_edge_fields(self.parent_role, self.child_role, self.edge_kind)
        _validate_bounds(
            name="required role-edge count",
            minimum=self.minimum,
            maximum=None,
            integral=True,
        )
        if self.minimum < 1:
            raise ValueError("required role-edge minimum must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParentCountConstraint(ConstraintBase):
    """Constrain distinct incoming parent nodes of one logical node."""

    node_id: NodeId
    minimum: int | None = None
    maximum: int | None = None
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    distinct_parent_nodes: bool = True
    kind: ConstraintKind = field(init=False, default=ConstraintKind.PARENT_COUNT)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_identifier("node_id", self.node_id)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)
        _validate_bool("distinct_parent_nodes", self.distinct_parent_nodes)
        _validate_bounds(
            name=f"parent count for {self.node_id}",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=True,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildCountConstraint(ConstraintBase):
    """Constrain distinct outgoing child nodes of one logical node."""

    node_id: NodeId
    minimum: int | None = None
    maximum: int | None = None
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    distinct_child_nodes: bool = True
    kind: ConstraintKind = field(init=False, default=ConstraintKind.CHILD_COUNT)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_identifier("node_id", self.node_id)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)
        _validate_bool("distinct_child_nodes", self.distinct_child_nodes)
        _validate_bounds(
            name=f"child count for {self.node_id}",
            minimum=self.minimum,
            maximum=self.maximum,
            integral=True,
        )


# ---------------------------------------------------------------------------
# Rank and reachability constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RankOrderConstraint(ConstraintBase):
    """Require ``before_node`` to be at least one rank before ``after_node``."""

    before_node: NodeId
    after_node: NodeId
    minimum_gap: int = 1
    maximum_gap: int | None = None
    kind: ConstraintKind = field(init=False, default=ConstraintKind.RANK_ORDER)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_distinct_nodes(self.before_node, self.after_node)
        _validate_non_negative_int("minimum_gap", self.minimum_gap)
        if self.minimum_gap < 1:
            raise ValueError("minimum_gap must be at least 1")
        if self.maximum_gap is not None:
            _validate_non_negative_int("maximum_gap", self.maximum_gap)
            if self.maximum_gap < self.minimum_gap:
                raise ValueError("maximum_gap must be at least minimum_gap")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReachabilityConstraint(ConstraintBase):
    """Require a directed path between two distinct nodes."""

    source_node: NodeId
    target_node: NodeId
    minimum_hops: int = 1
    maximum_hops: int | None = None
    edge_kinds: tuple[EdgeKind, ...] = (EdgeKind.FOREIGN_KEY,)
    kind: ConstraintKind = field(init=False, default=ConstraintKind.REACHABILITY)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_distinct_nodes(self.source_node, self.target_node)
        _validate_non_negative_int("minimum_hops", self.minimum_hops)
        if self.minimum_hops < 1:
            raise ValueError("minimum_hops must be at least 1")
        if self.maximum_hops is not None:
            _validate_non_negative_int("maximum_hops", self.maximum_hops)
            if self.maximum_hops < self.minimum_hops:
                raise ValueError("maximum_hops must be at least minimum_hops")
        _validate_enum_tuple("edge_kinds", self.edge_kinds, EdgeKind)


# ---------------------------------------------------------------------------
# Temporal and leaf constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalOrderConstraint(ConstraintBase):
    """Compare row times paired through one declared logical relation.

    The validator resolves ``relation_id`` and compares each related row pair.
    Cutoff/horizon rules without a row-pairing relation belong in TaskPlan.
    """

    stage: ConstraintStage = ConstraintStage.DATABASE
    relation_id: EdgeId
    before_node: NodeId
    after_node: NodeId
    allow_equal_time: bool = True
    maximum_delay_days: int | None = None
    kind: ConstraintKind = field(init=False, default=ConstraintKind.TEMPORAL_ORDER)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        _validate_identifier("relation_id", self.relation_id)
        _validate_distinct_nodes(self.before_node, self.after_node)
        _validate_bool("allow_equal_time", self.allow_equal_time)
        if self.maximum_delay_days is not None:
            _validate_non_negative_int(
                "maximum_delay_days",
                self.maximum_delay_days,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class UniqueLeafConstraint(ConstraintBase):
    """Require one leaf in the union of explicitly selected nodes and roles."""

    node_ids: tuple[NodeId, ...] = ()
    roles: tuple[TableRole, ...] = ()
    edge_kind: EdgeKind = EdgeKind.FOREIGN_KEY
    kind: ConstraintKind = field(init=False, default=ConstraintKind.UNIQUE_LEAF)

    def __post_init__(self) -> None:
        ConstraintBase.__post_init__(self)
        if not self.node_ids and not self.roles:
            raise ValueError("UniqueLeafConstraint requires node_ids or roles")
        if self.node_ids:
            _validate_identifier_tuple("node_ids", self.node_ids)
        if self.roles:
            _validate_enum_tuple("roles", self.roles, TableRole)
        _validate_enum("edge_kind", self.edge_kind, EdgeKind)


SchemaConstraint: TypeAlias = (
    ConnectedConstraint
    | AcyclicConstraint
    | NoParallelEdgesConstraint
    | TableCountConstraint
    | EdgeCountConstraint
    | EdgeDensityConstraint
    | RoleCountConstraint
    | AllowedRoleEdgeConstraint
    | ForbiddenRoleEdgeConstraint
    | RequiredRoleEdgeConstraint
    | ParentCountConstraint
    | ChildCountConstraint
    | RankOrderConstraint
    | ReachabilityConstraint
    | TemporalOrderConstraint
    | UniqueLeafConstraint
)


# ---------------------------------------------------------------------------
# Constraint serialization
# ---------------------------------------------------------------------------


_CONSTRAINT_TYPES: dict[ConstraintKind, type[ConstraintBase]] = {
    ConstraintKind.CONNECTED: ConnectedConstraint,
    ConstraintKind.ACYCLIC: AcyclicConstraint,
    ConstraintKind.NO_PARALLEL_EDGES: NoParallelEdgesConstraint,
    ConstraintKind.TABLE_COUNT: TableCountConstraint,
    ConstraintKind.EDGE_COUNT: EdgeCountConstraint,
    ConstraintKind.EDGE_DENSITY: EdgeDensityConstraint,
    ConstraintKind.ROLE_COUNT: RoleCountConstraint,
    ConstraintKind.ALLOWED_ROLE_EDGE: AllowedRoleEdgeConstraint,
    ConstraintKind.FORBIDDEN_ROLE_EDGE: ForbiddenRoleEdgeConstraint,
    ConstraintKind.REQUIRED_ROLE_EDGE: RequiredRoleEdgeConstraint,
    ConstraintKind.PARENT_COUNT: ParentCountConstraint,
    ConstraintKind.CHILD_COUNT: ChildCountConstraint,
    ConstraintKind.RANK_ORDER: RankOrderConstraint,
    ConstraintKind.REACHABILITY: ReachabilityConstraint,
    ConstraintKind.TEMPORAL_ORDER: TemporalOrderConstraint,
    ConstraintKind.UNIQUE_LEAF: UniqueLeafConstraint,
}

_ENUM_FIELDS: dict[str, type[Enum]] = {
    "severity": ConstraintSeverity,
    "stage": ConstraintStage,
    "edge_kind": EdgeKind,
    "role": TableRole,
    "parent_role": TableRole,
    "child_role": TableRole,
    "definition": DensityDefinition,
}

_ENUM_TUPLE_FIELDS: dict[str, type[Enum]] = {
    "edge_kinds": EdgeKind,
    "roles": TableRole,
}

_STRING_TUPLE_FIELDS = {"node_ids"}


def constraint_to_dict(constraint: SchemaConstraint) -> dict[str, Any]:
    """Serialize one concrete constraint to a JSON-compatible dictionary."""

    if type(constraint) not in set(_CONSTRAINT_TYPES.values()):
        raise TypeError(f"Unsupported constraint type: {type(constraint).__name__}")

    result: dict[str, Any] = {}
    for item in fields(constraint):
        value = getattr(constraint, item.name)
        if isinstance(value, Enum):
            result[item.name] = value.value
        elif isinstance(value, tuple):
            result[item.name] = [
                part.value if isinstance(part, Enum) else part
                for part in value
            ]
        else:
            result[item.name] = value
    return result


def constraint_from_dict(data: Mapping[str, Any]) -> SchemaConstraint:
    """Deserialize and fully validate one discriminator-tagged constraint."""

    if not isinstance(data, Mapping):
        raise TypeError("constraint payload must be a mapping")

    raw_kind = data.get("kind")
    try:
        kind = ConstraintKind(raw_kind)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown constraint kind: {raw_kind!r}") from error

    constraint_type = _CONSTRAINT_TYPES[kind]
    field_map = {item.name: item for item in fields(constraint_type)}
    init_fields = {name for name, item in field_map.items() if item.init}
    unknown = set(data) - init_fields - {"kind"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown fields for {kind.value}: {names}")

    kwargs: dict[str, Any] = {}
    for name in init_fields:
        if name not in data:
            continue
        value = data[name]

        enum_type = _ENUM_FIELDS.get(name)
        if enum_type is not None:
            try:
                value = enum_type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {name}: {value!r}") from error

        tuple_enum_type = _ENUM_TUPLE_FIELDS.get(name)
        if tuple_enum_type is not None:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            try:
                value = tuple(tuple_enum_type(part) for part in value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {name}: {value!r}") from error

        if name in _STRING_TUPLE_FIELDS:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            value = tuple(value)

        kwargs[name] = value

    try:
        return constraint_type(**kwargs)  # type: ignore[return-value]
    except TypeError as error:
        raise ValueError(f"Invalid {kind.value} constraint payload") from error


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_identifier(name: str, value: Any) -> None:
    if isinstance(value, Enum) or not isinstance(value, str):
        raise TypeError(f"{name} must be a string logical identifier")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_enum(name: str, value: Any, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be {enum_type.__name__}")


def _validate_enum_tuple(
    name: str,
    value: Any,
    enum_type: type[Enum],
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    for part in value:
        _validate_enum(name, part, enum_type)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_identifier_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    for part in value:
        _validate_identifier(name, part)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_bounds(
    *,
    name: str,
    minimum: int | float | None,
    maximum: int | float | None,
    integral: bool,
) -> None:
    if minimum is None and maximum is None:
        raise ValueError(f"{name} constraint requires minimum or maximum")

    for bound_name, value in (("minimum", minimum), ("maximum", maximum)):
        if value is None:
            continue
        if isinstance(value, bool):
            raise TypeError(f"{name} {bound_name} must not be boolean")
        if integral:
            if not isinstance(value, int):
                raise TypeError(f"{name} {bound_name} must be an integer")
        else:
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} {bound_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} {bound_name} must be finite")
        if value < 0:
            raise ValueError(f"{name} {bound_name} must be non-negative")

    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed maximum")


def _validate_distinct_nodes(first: NodeId, second: NodeId) -> None:
    _validate_identifier("first node", first)
    _validate_identifier("second node", second)
    if first == second:
        raise ValueError("constraint requires two different nodes")


def _validate_role_edge_fields(
    parent_role: Any,
    child_role: Any,
    edge_kind: Any,
) -> None:
    _validate_enum("parent_role", parent_role, TableRole)
    _validate_enum("child_role", child_role, TableRole)
    _validate_enum("edge_kind", edge_kind, EdgeKind)


__all__ = [
    "TableRole",
    "EdgeKind",
    "Cardinality",
    "Optionality",
    "IdentityDependency",
    "TemporalMode",
    "ConstraintSeverity",
    "ConstraintStage",
    "DensityDefinition",
    "ConstraintKind",
    "NodeId",
    "EdgeId",
    "ConstraintId",
    "ConstraintBase",
    "ConnectedConstraint",
    "AcyclicConstraint",
    "NoParallelEdgesConstraint",
    "TableCountConstraint",
    "EdgeCountConstraint",
    "EdgeDensityConstraint",
    "RoleCountConstraint",
    "AllowedRoleEdgeConstraint",
    "ForbiddenRoleEdgeConstraint",
    "RequiredRoleEdgeConstraint",
    "ParentCountConstraint",
    "ChildCountConstraint",
    "RankOrderConstraint",
    "ReachabilityConstraint",
    "TemporalOrderConstraint",
    "UniqueLeafConstraint",
    "SchemaConstraint",
    "constraint_to_dict",
    "constraint_from_dict",
]
