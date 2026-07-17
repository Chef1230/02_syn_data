# src/rdb_prior/schema/roles.py
# -*- coding: utf-8 -*-
"""
Role definitions and role-aware edge policies.

Edge orientation
----------------
All physical foreign-key edges use:

    parent table -> child table

The parent table owns the referenced key. The child table stores the foreign
key column.

Responsibilities
----------------
This module defines:

1. Structural properties of each table role.
2. Allowed parent-role -> child-role combinations.
3. Default relation and feature generation strategies.
4. Lightweight lookup helpers used by samplers, compilers and validators.

This module does not:

- sample roles;
- generate schema nodes or edges;
- generate rows or features;
- validate a complete blueprint;
- contain experiment-specific sampling probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Iterable, Mapping

from rdb_prior.schema.spec import (
    Cardinality,
    IdentityDependency,
    Optionality,
    TableRole,
    TemporalMode,
)


# ---------------------------------------------------------------------------
# Public policy enums
# ---------------------------------------------------------------------------


class RootPolicy(str, Enum):
    """Whether a role may appear as an FK-graph root."""

    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    PREFERRED = "preferred"


class LeafPolicy(str, Enum):
    """Whether a role may appear as an FK-graph leaf."""

    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    PREFERRED = "preferred"


# ---------------------------------------------------------------------------
# Role models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleSpec:
    """
    Structural policy of one table role.

    ``structural parents`` exclude auxiliary references such as Lookup foreign
    keys. For example, an Event with one Entity parent and one Lookup reference
    has one structural parent, not two.
    """

    role: TableRole
    description: str

    root_policy: RootPolicy
    leaf_policy: LeafPolicy

    min_structural_parents: int
    max_structural_parents: int | None

    can_own_foreign_keys: bool
    can_be_referenced: bool

    temporal_capable: bool
    default_temporal_mode: TemporalMode

    default_feature_strategy: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, TableRole):
            raise TypeError("role must be TableRole")
        if not isinstance(self.root_policy, RootPolicy):
            raise TypeError("root_policy must be RootPolicy")
        if not isinstance(self.leaf_policy, LeafPolicy):
            raise TypeError("leaf_policy must be LeafPolicy")

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"description must not be empty for role {self.role.value}"
            )

        if (
            isinstance(self.min_structural_parents, bool)
            or not isinstance(self.min_structural_parents, int)
        ):
            raise TypeError("min_structural_parents must be an integer")

        if self.min_structural_parents < 0:
            raise ValueError(
                "min_structural_parents must be non-negative"
            )

        if (
            self.max_structural_parents is not None
            and (
                isinstance(self.max_structural_parents, bool)
                or not isinstance(self.max_structural_parents, int)
            )
        ):
            raise TypeError("max_structural_parents must be an integer")

        if (
            self.max_structural_parents is not None
            and self.max_structural_parents
            < self.min_structural_parents
        ):
            raise ValueError(
                "max_structural_parents must be greater than or equal to "
                "min_structural_parents"
            )

        for name, value in (
            ("can_own_foreign_keys", self.can_own_foreign_keys),
            ("can_be_referenced", self.can_be_referenced),
            ("temporal_capable", self.temporal_capable),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        if not isinstance(self.default_temporal_mode, TemporalMode):
            raise TypeError("default_temporal_mode must be TemporalMode")

        if (
            not self.temporal_capable
            and self.default_temporal_mode
            not in {TemporalMode.NONE, TemporalMode.STATIC}
        ):
            raise ValueError(
                f"Non-temporal role {self.role.value} cannot default to "
                f"{self.default_temporal_mode.value}"
            )

        if (
            self.root_policy is RootPolicy.FORBIDDEN
            and self.min_structural_parents < 1
        ):
            raise ValueError(
                "A root-forbidden role must require a structural parent"
            )

        if (
            not self.can_own_foreign_keys
            and self.max_structural_parents != 0
        ):
            raise ValueError(
                "A role that cannot own foreign keys must have zero "
                "structural parents"
            )

        if (
            not isinstance(self.default_feature_strategy, str)
            or not self.default_feature_strategy.strip()
        ):
            raise ValueError("default_feature_strategy must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleEdgeRule:
    """
    Policy for one allowed parent-role -> child-role foreign-key edge.

    Only allowed role pairs are registered. Absence from ``ROLE_EDGE_RULES``
    means that the edge is forbidden in V1.
    """

    parent_role: TableRole
    child_role: TableRole

    relation_strategy: str
    cardinality: Cardinality
    optionality: Optionality
    identity_dependency: IdentityDependency

    # Lookup references normally do not satisfy Event/Bridge/Detail structural
    # parent requirements.
    counts_as_structural_parent: bool = True

    # Used by the compiler and validator to distinguish main process edges from
    # auxiliary classification/context edges.
    auxiliary: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parent_role, TableRole):
            raise TypeError("parent_role must be TableRole")
        if not isinstance(self.child_role, TableRole):
            raise TypeError("child_role must be TableRole")
        if not isinstance(self.cardinality, Cardinality):
            raise TypeError("cardinality must be Cardinality")
        if not isinstance(self.optionality, Optionality):
            raise TypeError("optionality must be Optionality")
        if not isinstance(self.identity_dependency, IdentityDependency):
            raise TypeError("identity_dependency must be IdentityDependency")

        if (
            not isinstance(self.relation_strategy, str)
            or not self.relation_strategy.strip()
        ):
            raise ValueError("relation_strategy must not be empty")

        if not isinstance(self.counts_as_structural_parent, bool):
            raise TypeError("counts_as_structural_parent must be a boolean")
        if not isinstance(self.auxiliary, bool):
            raise TypeError("auxiliary must be a boolean")

        if self.cardinality is Cardinality.MANY_TO_MANY:
            raise ValueError(
                "A physical FK rule cannot be many-to-many; use a Bridge role"
            )

        if (
            self.identity_dependency is IdentityDependency.IDENTIFYING
            and self.optionality is not Optionality.REQUIRED
        ):
            raise ValueError("An identifying relation must be required")

        if self.auxiliary and self.counts_as_structural_parent:
            raise ValueError(
                "An auxiliary edge cannot count as a structural parent"
            )


class RoleCompatibilityError(ValueError):
    """Raised when a forbidden role edge is requested."""


# ---------------------------------------------------------------------------
# Role catalog
# ---------------------------------------------------------------------------


_ROLE_SPECS: Final[dict[TableRole, RoleSpec]] = {
    TableRole.ENTITY: RoleSpec(
        role=TableRole.ENTITY,
        description=(
            "An independently identifiable object or actor that persists "
            "across events."
        ),
        root_policy=RootPolicy.PREFERRED,
        leaf_policy=LeafPolicy.ALLOWED,
        min_structural_parents=0,
        max_structural_parents=2,
        can_own_foreign_keys=True,
        can_be_referenced=True,
        temporal_capable=False,
        default_temporal_mode=TemporalMode.STATIC,
        default_feature_strategy="entity",
    ),
    TableRole.EVENT: RoleSpec(
        role=TableRole.EVENT,
        description=(
            "A time-indexed interaction, transaction, observation or state "
            "transition involving one or more upstream objects."
        ),
        root_policy=RootPolicy.FORBIDDEN,
        leaf_policy=LeafPolicy.ALLOWED,
        min_structural_parents=1,
        max_structural_parents=3,
        can_own_foreign_keys=True,
        can_be_referenced=True,
        temporal_capable=True,
        default_temporal_mode=TemporalMode.EVENT_TIME,
        default_feature_strategy="event",
    ),
    TableRole.BRIDGE: RoleSpec(
        role=TableRole.BRIDGE,
        description=(
            "An associative table representing a many-to-many or "
            "multi-participant relationship."
        ),
        root_policy=RootPolicy.FORBIDDEN,
        leaf_policy=LeafPolicy.ALLOWED,
        min_structural_parents=2,
        max_structural_parents=3,
        can_own_foreign_keys=True,
        can_be_referenced=True,
        temporal_capable=True,
        default_temporal_mode=TemporalMode.NONE,
        default_feature_strategy="bridge",
    ),
    TableRole.LOOKUP: RoleSpec(
        role=TableRole.LOOKUP,
        description=(
            "A low-cardinality reference, vocabulary, category or static "
            "context table."
        ),
        root_policy=RootPolicy.PREFERRED,
        leaf_policy=LeafPolicy.ALLOWED,
        min_structural_parents=0,
        max_structural_parents=0,
        can_own_foreign_keys=False,
        can_be_referenced=True,
        temporal_capable=False,
        default_temporal_mode=TemporalMode.STATIC,
        default_feature_strategy="lookup",
    ),
    TableRole.DETAIL: RoleSpec(
        role=TableRole.DETAIL,
        description=(
            "A dependent child table containing line items, measurements, "
            "attributes, records or other details of an upstream object, "
            "event or relationship."
        ),
        root_policy=RootPolicy.FORBIDDEN,
        leaf_policy=LeafPolicy.PREFERRED,
        min_structural_parents=1,
        max_structural_parents=2,
        can_own_foreign_keys=True,
        can_be_referenced=False,
        temporal_capable=True,
        default_temporal_mode=TemporalMode.NONE,
        default_feature_strategy="detail",
    ),
}

ROLE_SPECS: Final[Mapping[TableRole, RoleSpec]] = MappingProxyType(
    _ROLE_SPECS
)


# ---------------------------------------------------------------------------
# Allowed parent-role -> child-role edges
# ---------------------------------------------------------------------------


def _edge_key(
    parent_role: TableRole,
    child_role: TableRole,
) -> tuple[TableRole, TableRole]:
    return parent_role, child_role


_ROLE_EDGE_RULES: Final[
    dict[tuple[TableRole, TableRole], RoleEdgeRule]
] = {
    # ------------------------------------------------------------------
    # Lookup references
    #
    # These are auxiliary classification/context edges. They do not count
    # toward the structural-parent requirements of Event, Bridge or Detail.
    # ------------------------------------------------------------------
    _edge_key(
        TableRole.LOOKUP,
        TableRole.ENTITY,
    ): RoleEdgeRule(
        parent_role=TableRole.LOOKUP,
        child_role=TableRole.ENTITY,
        relation_strategy="lookup_assignment",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
        counts_as_structural_parent=False,
        auxiliary=True,
    ),
    _edge_key(
        TableRole.LOOKUP,
        TableRole.EVENT,
    ): RoleEdgeRule(
        parent_role=TableRole.LOOKUP,
        child_role=TableRole.EVENT,
        relation_strategy="lookup_assignment",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
        counts_as_structural_parent=False,
        auxiliary=True,
    ),
    _edge_key(
        TableRole.LOOKUP,
        TableRole.BRIDGE,
    ): RoleEdgeRule(
        parent_role=TableRole.LOOKUP,
        child_role=TableRole.BRIDGE,
        relation_strategy="lookup_assignment",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
        counts_as_structural_parent=False,
        auxiliary=True,
    ),
    _edge_key(
        TableRole.LOOKUP,
        TableRole.DETAIL,
    ): RoleEdgeRule(
        parent_role=TableRole.LOOKUP,
        child_role=TableRole.DETAIL,
        relation_strategy="lookup_assignment",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
        counts_as_structural_parent=False,
        auxiliary=True,
    ),

    # ------------------------------------------------------------------
    # Entity relationships
    # ------------------------------------------------------------------
    _edge_key(
        TableRole.ENTITY,
        TableRole.ENTITY,
    ): RoleEdgeRule(
        parent_role=TableRole.ENTITY,
        child_role=TableRole.ENTITY,
        relation_strategy="entity_hierarchy",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
    ),
    _edge_key(
        TableRole.ENTITY,
        TableRole.EVENT,
    ): RoleEdgeRule(
        parent_role=TableRole.ENTITY,
        child_role=TableRole.EVENT,
        relation_strategy="entity_event",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.INDEPENDENT,
    ),
    _edge_key(
        TableRole.ENTITY,
        TableRole.BRIDGE,
    ): RoleEdgeRule(
        parent_role=TableRole.ENTITY,
        child_role=TableRole.BRIDGE,
        relation_strategy="affinity_bridge",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.WEAK,
    ),
    _edge_key(
        TableRole.ENTITY,
        TableRole.DETAIL,
    ): RoleEdgeRule(
        parent_role=TableRole.ENTITY,
        child_role=TableRole.DETAIL,
        relation_strategy="entity_detail",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.WEAK,
    ),

    # ------------------------------------------------------------------
    # Event relationships
    # ------------------------------------------------------------------
    _edge_key(
        TableRole.EVENT,
        TableRole.EVENT,
    ): RoleEdgeRule(
        parent_role=TableRole.EVENT,
        child_role=TableRole.EVENT,
        relation_strategy="history_sequence",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.OPTIONAL,
        identity_dependency=IdentityDependency.INDEPENDENT,
    ),
    _edge_key(
        TableRole.EVENT,
        TableRole.BRIDGE,
    ): RoleEdgeRule(
        parent_role=TableRole.EVENT,
        child_role=TableRole.BRIDGE,
        relation_strategy="event_bridge",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.WEAK,
    ),
    _edge_key(
        TableRole.EVENT,
        TableRole.DETAIL,
    ): RoleEdgeRule(
        parent_role=TableRole.EVENT,
        child_role=TableRole.DETAIL,
        relation_strategy="event_detail",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.WEAK,
    ),

    # ------------------------------------------------------------------
    # Bridge downstream relationships
    # ------------------------------------------------------------------
    _edge_key(
        TableRole.BRIDGE,
        TableRole.EVENT,
    ): RoleEdgeRule(
        parent_role=TableRole.BRIDGE,
        child_role=TableRole.EVENT,
        relation_strategy="bridge_event",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.INDEPENDENT,
    ),
    _edge_key(
        TableRole.BRIDGE,
        TableRole.DETAIL,
    ): RoleEdgeRule(
        parent_role=TableRole.BRIDGE,
        child_role=TableRole.DETAIL,
        relation_strategy="bridge_detail",
        cardinality=Cardinality.ONE_TO_MANY,
        optionality=Optionality.REQUIRED,
        identity_dependency=IdentityDependency.WEAK,
    ),
}

ROLE_EDGE_RULES: Final[
    Mapping[tuple[TableRole, TableRole], RoleEdgeRule]
] = MappingProxyType(_ROLE_EDGE_RULES)


# ---------------------------------------------------------------------------
# Public lookup helpers
# ---------------------------------------------------------------------------


def get_role_spec(role: TableRole) -> RoleSpec:
    """Return the immutable specification of one role."""
    try:
        return ROLE_SPECS[role]
    except KeyError as error:
        raise ValueError(f"Unknown table role: {role!r}") from error


def is_role_edge_allowed(
    parent_role: TableRole,
    child_role: TableRole,
) -> bool:
    """Return whether V1 permits this parent-role -> child-role edge."""
    return (parent_role, child_role) in ROLE_EDGE_RULES


def get_role_edge_rule(
    parent_role: TableRole,
    child_role: TableRole,
) -> RoleEdgeRule:
    """
    Return the policy of an allowed role edge.

    Raises
    ------
    RoleCompatibilityError
        If the role pair is forbidden.
    """
    try:
        return ROLE_EDGE_RULES[(parent_role, child_role)]
    except KeyError as error:
        raise RoleCompatibilityError(
            "Forbidden role edge in parent-to-child orientation: "
            f"{parent_role.value} -> {child_role.value}"
        ) from error


def resolve_feature_strategy(
    role: TableRole,
) -> str:
    """Return the default table feature strategy for a role."""
    return get_role_spec(role).default_feature_strategy


def resolve_relation_strategy(
    parent_role: TableRole,
    child_role: TableRole,
) -> str:
    """Return the default FK generation strategy for a role pair."""
    return get_role_edge_rule(
        parent_role,
        child_role,
    ).relation_strategy


def allowed_parent_roles(
    child_role: TableRole,
    *,
    structural_only: bool = False,
) -> tuple[TableRole, ...]:
    """
    Return roles that may point to ``child_role``.

    Results are sorted by enum value to avoid dependence on registry insertion
    order.
    """
    roles = {
        rule.parent_role
        for rule in ROLE_EDGE_RULES.values()
        if rule.child_role == child_role
        and (
            not structural_only
            or rule.counts_as_structural_parent
        )
    }

    return tuple(sorted(roles, key=lambda role: role.value))


def allowed_child_roles(
    parent_role: TableRole,
    *,
    structural_only: bool = False,
) -> tuple[TableRole, ...]:
    """Return child roles that may hold an FK referencing ``parent_role``."""
    roles = {
        rule.child_role
        for rule in ROLE_EDGE_RULES.values()
        if rule.parent_role == parent_role
        and (
            not structural_only
            or rule.counts_as_structural_parent
        )
    }

    return tuple(sorted(roles, key=lambda role: role.value))


def count_structural_parent_roles(
    parent_roles: Iterable[TableRole],
    child_role: TableRole,
) -> int:
    """
    Count parent-role occurrences that satisfy structural-parent requirements.

    This helper operates on roles only. A validator should separately check
    whether the corresponding parent node IDs are distinct.
    """
    count = 0

    for parent_role in parent_roles:
        rule = get_role_edge_rule(
            parent_role,
            child_role,
        )

        if rule.counts_as_structural_parent:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Catalog integrity checks
# ---------------------------------------------------------------------------


def _validate_catalog() -> None:
    missing_roles = set(TableRole) - set(ROLE_SPECS)

    if missing_roles:
        missing = ", ".join(
            sorted(role.value for role in missing_roles)
        )
        raise RuntimeError(
            f"ROLE_SPECS is missing roles: {missing}"
        )

    unknown_roles = set(ROLE_SPECS) - set(TableRole)

    if unknown_roles:
        unknown = ", ".join(
            sorted(str(role) for role in unknown_roles)
        )
        raise RuntimeError(
            f"ROLE_SPECS contains unknown roles: {unknown}"
        )

    for key, rule in ROLE_EDGE_RULES.items():
        expected_key = (
            rule.parent_role,
            rule.child_role,
        )

        if key != expected_key:
            raise RuntimeError(
                "Role edge registry key does not match rule: "
                f"key={key}, rule={expected_key}"
            )

        parent_spec = get_role_spec(rule.parent_role)
        child_spec = get_role_spec(rule.child_role)

        if not parent_spec.can_be_referenced:
            raise RuntimeError(
                f"Role {rule.parent_role.value} cannot be referenced "
                "but appears as an edge parent"
            )

        if not child_spec.can_own_foreign_keys:
            raise RuntimeError(
                f"Role {rule.child_role.value} cannot own foreign keys "
                "but appears as an edge child"
            )


_validate_catalog()


__all__ = [
    "RootPolicy",
    "LeafPolicy",
    "RoleSpec",
    "RoleEdgeRule",
    "RoleCompatibilityError",
    "ROLE_SPECS",
    "ROLE_EDGE_RULES",
    "get_role_spec",
    "get_role_edge_rule",
    "is_role_edge_allowed",
    "resolve_feature_strategy",
    "resolve_relation_strategy",
    "allowed_parent_roles",
    "allowed_child_roles",
    "count_structural_parent_roles",
]
