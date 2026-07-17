# src/rdb_prior/schema/motifs.py
# -*- coding: utf-8 -*-
"""
Anonymous role-aware structural motif specifications and static validation.

A schema motif is a small anonymous FK-graph template:

    anonymous slots
        + allowed latent roles
        + parent -> child edges
        + local rank constraints

Schema motifs do not contain:

- table names or business semantics;
- sampling probabilities;
- target occurrence counts;
- concrete table IDs;
- row counts or fanout parameters.
- temporal precedence, task semantics or process semantics;
- persisted occurrence metadata.

Sampling probabilities and occurrence bounds belong in configuration.
Temporal/process relations belong in a separate Task/Process motif layer.
Concrete node IDs are assigned while a motif is instantiated into a
SchemaBlueprint, but the completed schema does not retain which motifs were
used to construct it.

Edge orientation
----------------
All V1 motif edges use:

    referenced parent table -> FK-owning child table
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod
from types import MappingProxyType
from typing import Callable, Final, Iterable, Iterator, Mapping

from rdb_prior.schema.roles import (
    get_role_edge_rule,
    get_role_spec,
    is_role_edge_allowed,
)
from rdb_prior.schema.spec import TableRole


# ---------------------------------------------------------------------------
# Motif specification
# ---------------------------------------------------------------------------


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive_int(name: str, value: object) -> None:
    _validate_non_negative_int(name, value)
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_role_tuple(name: str, value: object) -> None:
    _validate_instance_tuple(name, value, TableRole)

    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_instance_tuple(
    name: str,
    value: object,
    expected_type: type,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")

    for item in value:
        if not isinstance(item, expected_type):
            raise TypeError(
                f"{name} items must be {expected_type.__name__}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifNodeSpec:
    """
    Anonymous node slot inside a motif.

    ``slot`` is local to one motif definition. It is not a concrete table ID.

    ``roles`` contains all roles that may be assigned to the slot. Most V1
    motifs should use one exact role per slot. Multiple roles are supported
    when all edge constraints admit at least one globally valid assignment.

    ``rank_offset`` is a local non-negative rank. During attachment, the
    composer must use::

        global_rank = anchor_global_rank + slot_offset - anchor_offset

    The completed schema stores only the resolved global rank.
    """

    slot: str
    roles: tuple[TableRole, ...]
    rank_offset: int

    def __post_init__(self) -> None:
        _validate_identifier("slot", self.slot)
        _validate_role_tuple("roles", self.roles)
        _validate_non_negative_int("rank_offset", self.rank_offset)


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifEdgeSpec:
    """
    Physical FK edge between two anonymous motif slots.

    ``minimum_rank_gap=1`` means that the child must be downstream of the
    parent. ``maximum_rank_gap=None`` permits rank-skip edges.
    """

    edge: str
    parent_slot: str
    child_slot: str

    minimum_rank_gap: int = 1
    maximum_rank_gap: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier("edge", self.edge)
        _validate_identifier("parent_slot", self.parent_slot)
        _validate_identifier("child_slot", self.child_slot)

        if self.parent_slot == self.child_slot:
            raise ValueError("A motif FK cannot be a self-loop")

        _validate_positive_int(
            "minimum_rank_gap",
            self.minimum_rank_gap,
        )

        if self.maximum_rank_gap is not None:
            _validate_positive_int(
                "maximum_rank_gap",
                self.maximum_rank_gap,
            )
            if self.maximum_rank_gap < self.minimum_rank_gap:
                raise ValueError(
                    "maximum_rank_gap must be at least minimum_rank_gap"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifSpec:
    """
    Immutable anonymous motif template.

    ``motif_type`` is a stable machine identifier such as:

        entity_event
        entity_event_detail
        entity_bridge_collider
        entity_event_fork

    It is not a generated natural-language description.

    ``anchor_slot`` identifies the only slot that may bind an existing schema
    node. Every other slot is instantiated as a fresh, distinct node. The
    anchor need not be a graph root.

    A schema motif is always connected and acyclic. Task/process relations are
    outside this structural library.
    """

    motif_type: str

    nodes: tuple[MotifNodeSpec, ...]
    edges: tuple[MotifEdgeSpec, ...]

    anchor_slot: str

    def __post_init__(self) -> None:
        _validate_identifier("motif_type", self.motif_type)
        _validate_instance_tuple("nodes", self.nodes, MotifNodeSpec)
        _validate_instance_tuple("edges", self.edges, MotifEdgeSpec)
        _validate_identifier("anchor_slot", self.anchor_slot)

        if not self.nodes:
            raise ValueError("nodes must not be empty")


# ---------------------------------------------------------------------------
# Static validation result
# ---------------------------------------------------------------------------


class MotifIssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifValidationIssue:
    level: MotifIssueLevel
    code: str
    message: str


class MotifDefinitionError(ValueError):
    """Raised when a motif definition contains static errors."""

    def __init__(
        self,
        motif_type: str,
        issues: Iterable[MotifValidationIssue],
    ) -> None:
        self.motif_type = motif_type
        self.issues = tuple(issues)

        details = "\n".join(
            f"- [{issue.code}] {issue.message}"
            for issue in self.issues
            if issue.level == MotifIssueLevel.ERROR
        )

        super().__init__(
            f"Invalid motif definition {motif_type!r}:\n{details}"
        )


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------


def validate_motif_spec(
    motif: MotifSpec,
    *,
    raise_on_error: bool = False,
    max_role_assignments: int = 4096,
) -> tuple[MotifValidationIssue, ...]:
    """
    Perform validation that does not require a sampled SchemaBlueprint.

    Checks
    ------
    1. Node and edge identifier uniqueness.
    2. Anchor existence.
    3. Edge endpoint existence and duplicate logical edges.
    4. Local rank consistency.
    5. Weak connectivity and directed acyclicity.
    6. Existence of a globally valid role assignment.
    7. Whole-node RoleSpec constraints for every non-anchor node.
    8. Role alternatives that never occur in any valid assignment.

    This function does not check:

    - sampled motif counts;
    - final schema connectivity;
    - cross-motif constraints;
    - external structural-parent counts of the anchor node;
    - final table ranks;
    - row-level FK integrity.
    """
    if not isinstance(motif, MotifSpec):
        raise TypeError("motif must be MotifSpec")
    if not isinstance(raise_on_error, bool):
        raise TypeError("raise_on_error must be a boolean")
    _validate_positive_int(
        "max_role_assignments",
        max_role_assignments,
    )

    issues: list[MotifValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(
            MotifValidationIssue(
                level=MotifIssueLevel.ERROR,
                code=code,
                message=message,
            )
        )

    def warning(code: str, message: str) -> None:
        issues.append(
            MotifValidationIssue(
                level=MotifIssueLevel.WARNING,
                code=code,
                message=message,
            )
        )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    node_by_slot: dict[str, MotifNodeSpec] = {}

    for node in motif.nodes:
        if node.slot in node_by_slot:
            error(
                "duplicate_node_slot",
                f"Duplicate node slot {node.slot!r}.",
            )
            continue

        node_by_slot[node.slot] = node

    if motif.anchor_slot not in node_by_slot:
        error(
            "unknown_anchor_slot",
            f"anchor_slot {motif.anchor_slot!r} does not reference "
            "a motif node.",
        )

    # Further graph checks require valid node slots.
    if not node_by_slot:
        return _finish_validation(
            motif,
            issues,
            raise_on_error=raise_on_error,
        )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    edge_ids: set[str] = set()
    logical_edges: set[tuple[str, str]] = set()
    valid_edges: list[MotifEdgeSpec] = []

    for edge in motif.edges:
        if edge.edge in edge_ids:
            error(
                "duplicate_edge_id",
                f"Duplicate edge ID {edge.edge!r}.",
            )
        else:
            edge_ids.add(edge.edge)

        endpoints_exist = True

        if edge.parent_slot not in node_by_slot:
            error(
                "unknown_parent_slot",
                f"Edge {edge.edge!r} references unknown parent slot "
                f"{edge.parent_slot!r}.",
            )
            endpoints_exist = False

        if edge.child_slot not in node_by_slot:
            error(
                "unknown_child_slot",
                f"Edge {edge.edge!r} references unknown child slot "
                f"{edge.child_slot!r}.",
            )
            endpoints_exist = False

        logical_key = (
            edge.parent_slot,
            edge.child_slot,
        )

        if logical_key in logical_edges:
            error(
                "duplicate_logical_edge",
                "Duplicate logical edge "
                f"{edge.parent_slot!r} -> {edge.child_slot!r}.",
            )
        else:
            logical_edges.add(logical_key)

        if endpoints_exist:
            valid_edges.append(edge)

            parent = node_by_slot[edge.parent_slot]
            child = node_by_slot[edge.child_slot]

            actual_gap = child.rank_offset - parent.rank_offset

            if actual_gap < edge.minimum_rank_gap:
                error(
                    "rank_order_violation",
                    f"Edge {edge.edge!r} requires rank gap >= "
                    f"{edge.minimum_rank_gap}, but its node offsets "
                    f"produce {actual_gap}.",
                )

            if (
                edge.maximum_rank_gap is not None
                and actual_gap > edge.maximum_rank_gap
            ):
                error(
                    "rank_skip_violation",
                    f"Edge {edge.edge!r} allows rank gap <= "
                    f"{edge.maximum_rank_gap}, but its node offsets "
                    f"produce {actual_gap}.",
                )

    # ------------------------------------------------------------------
    # Connectivity and acyclicity
    # ------------------------------------------------------------------

    if len(node_by_slot) > 1:
        if not _is_weakly_connected(
            node_slots=tuple(node_by_slot),
            edges=valid_edges,
        ):
            error(
                "disconnected_motif",
                "Motif nodes are not weakly connected.",
            )

    if not _is_directed_acyclic(
        node_slots=tuple(node_by_slot),
        edges=valid_edges,
    ):
        error(
            "cyclic_motif",
            "Motif contains a directed cycle.",
        )

    # ------------------------------------------------------------------
    # Role compatibility
    # ------------------------------------------------------------------

    if all(node.roles for node in node_by_slot.values()):
        assignment_count = prod(
            len(node.roles)
            for node in node_by_slot.values()
        )

        if assignment_count > max_role_assignments:
            error(
                "role_assignment_space_too_large",
                "Motif role assignment space contains "
                f"{assignment_count} combinations, exceeding the static "
                f"validation limit {max_role_assignments}. Use narrower "
                "role domains.",
            )
        else:
            edge_compatible_assignments = _valid_role_assignments(
                nodes=tuple(node_by_slot.values()),
                edges=valid_edges,
            )

            if not edge_compatible_assignments:
                error(
                    "no_valid_role_assignment",
                    "No complete role assignment satisfies all motif edges.",
                )
            else:
                valid_assignments = tuple(
                    assignment
                    for assignment in edge_compatible_assignments
                    if not _node_constraint_failures(
                        nodes=tuple(node_by_slot.values()),
                        edges=valid_edges,
                        assignment=assignment,
                        anchor_slot=motif.anchor_slot,
                    )
                )

                if not valid_assignments:
                    if all(
                        len(node.roles) == 1
                        for node in node_by_slot.values()
                    ):
                        for code, message in _node_constraint_failures(
                            nodes=tuple(node_by_slot.values()),
                            edges=valid_edges,
                            assignment=edge_compatible_assignments[0],
                            anchor_slot=motif.anchor_slot,
                        ):
                            error(code, message)
                    else:
                        error(
                            "no_complete_role_assignment",
                            "No edge-compatible role assignment satisfies "
                            "the whole-node structural constraints.",
                        )
                else:
                    _check_unused_role_alternatives(
                        nodes=tuple(node_by_slot.values()),
                        valid_assignments=valid_assignments,
                        warning=warning,
                    )

    return _finish_validation(
        motif,
        issues,
        raise_on_error=raise_on_error,
    )


def _finish_validation(
    motif: MotifSpec,
    issues: list[MotifValidationIssue],
    *,
    raise_on_error: bool,
) -> tuple[MotifValidationIssue, ...]:
    result = tuple(issues)

    if raise_on_error:
        errors = tuple(
            issue
            for issue in result
            if issue.level == MotifIssueLevel.ERROR
        )

        if errors:
            raise MotifDefinitionError(
                motif_type=motif.motif_type,
                issues=errors,
            )

    return result


# ---------------------------------------------------------------------------
# Instantiation helpers
# ---------------------------------------------------------------------------


def resolve_motif_global_ranks(
    motif: MotifSpec,
    *,
    anchor_global_rank: int,
) -> Mapping[str, int]:
    """Resolve local offsets into final schema ranks.

    The formula is fixed for every slot::

        global_rank = anchor_global_rank + slot_offset - anchor_offset

    A motif that would place a slot below global rank zero cannot be attached
    at the requested anchor rank.
    """
    if not isinstance(motif, MotifSpec):
        raise TypeError("motif must be MotifSpec")
    _validate_non_negative_int("anchor_global_rank", anchor_global_rank)
    validate_motif_spec(motif, raise_on_error=True)

    node_by_slot = {
        node.slot: node
        for node in motif.nodes
    }
    anchor_offset = node_by_slot[motif.anchor_slot].rank_offset
    ranks = {
        slot: (
            anchor_global_rank
            + node.rank_offset
            - anchor_offset
        )
        for slot, node in node_by_slot.items()
    }

    negative_slots = tuple(
        sorted(
            slot
            for slot, rank in ranks.items()
            if rank < 0
        )
    )
    if negative_slots:
        joined = ", ".join(repr(slot) for slot in negative_slots)
        raise ValueError(
            "Motif attachment would produce a negative global rank for "
            f"slots: {joined}"
        )

    return MappingProxyType(ranks)


def validate_motif_node_bindings(
    motif: MotifSpec,
    node_bindings: Mapping[str, str],
    *,
    existing_node_ids: Iterable[str] = (),
) -> None:
    """Validate ephemeral slot-to-node bindings during composition.

    All slots must bind distinct node IDs. Only the anchor slot may reuse a
    node that already exists in the partial schema. The bindings are a
    construction-time value and are not persisted in the completed schema.
    """
    if not isinstance(motif, MotifSpec):
        raise TypeError("motif must be MotifSpec")
    if not isinstance(node_bindings, Mapping):
        raise TypeError("node_bindings must be a mapping")
    if isinstance(existing_node_ids, (str, bytes)):
        raise TypeError("existing_node_ids must be an iterable of node IDs")

    for slot, node_id in node_bindings.items():
        _validate_identifier("binding slot", slot)
        _validate_identifier(f"node binding for {slot!r}", node_id)

    expected_slots = {
        node.slot
        for node in motif.nodes
    }
    actual_slots = set(node_bindings)

    if actual_slots != expected_slots:
        missing = tuple(sorted(expected_slots - actual_slots))
        unexpected = tuple(sorted(actual_slots - expected_slots))
        raise ValueError(
            "node_bindings must contain exactly the motif slots; "
            f"missing={missing}, unexpected={unexpected}"
        )

    bound_node_ids = tuple(node_bindings.values())
    if len(set(bound_node_ids)) != len(bound_node_ids):
        raise ValueError(
            "Each motif slot must bind a distinct schema node"
        )

    existing_values = tuple(existing_node_ids)
    for node_id in existing_values:
        _validate_identifier("existing node ID", node_id)
    existing = set(existing_values)

    reused_non_anchor = tuple(
        sorted(
            slot
            for slot, node_id in node_bindings.items()
            if slot != motif.anchor_slot and node_id in existing
        )
    )
    if reused_non_anchor:
        joined = ", ".join(repr(slot) for slot in reused_non_anchor)
        raise ValueError(
            "Only anchor_slot may reuse existing schema nodes; reused "
            f"non-anchor slots: {joined}"
        )


# ---------------------------------------------------------------------------
# Role assignment validation
# ---------------------------------------------------------------------------


def _valid_role_assignments(
    *,
    nodes: tuple[MotifNodeSpec, ...],
    edges: list[MotifEdgeSpec],
) -> tuple[Mapping[str, TableRole], ...]:
    """
    Enumerate globally valid role assignments.

    Motifs are deliberately small, so exhaustive enumeration is acceptable.
    The caller limits the assignment-space size.
    """
    slots = tuple(node.slot for node in nodes)
    role_domains = tuple(node.roles for node in nodes)

    valid: list[Mapping[str, TableRole]] = []

    for role_values in product(*role_domains):
        assignment = dict(zip(slots, role_values, strict=True))

        if all(
            is_role_edge_allowed(
                assignment[edge.parent_slot],
                assignment[edge.child_slot],
            )
            for edge in edges
        ):
            valid.append(MappingProxyType(assignment))

    return tuple(valid)


def _node_constraint_failures(
    *,
    nodes: tuple[MotifNodeSpec, ...],
    edges: list[MotifEdgeSpec],
    assignment: Mapping[str, TableRole],
    anchor_slot: str,
) -> tuple[tuple[str, str], ...]:
    """Return whole-node structural failures for one role assignment."""
    incoming_fk_count: dict[str, int] = defaultdict(int)
    incoming_structural_count: dict[str, int] = defaultdict(int)
    outgoing_fk_count: dict[str, int] = defaultdict(int)

    for edge in edges:
        parent_role = assignment[edge.parent_slot]
        child_role = assignment[edge.child_slot]
        rule = get_role_edge_rule(parent_role, child_role)

        outgoing_fk_count[edge.parent_slot] += 1
        incoming_fk_count[edge.child_slot] += 1
        if rule.counts_as_structural_parent:
            incoming_structural_count[edge.child_slot] += 1

    failures: list[tuple[str, str]] = []

    for node in nodes:
        role = assignment[node.slot]
        role_spec = get_role_spec(role)
        incoming_fk = incoming_fk_count[node.slot]
        incoming_structural = incoming_structural_count[node.slot]
        outgoing_fk = outgoing_fk_count[node.slot]

        if incoming_fk and not role_spec.can_own_foreign_keys:
            failures.append(
                (
                    "role_cannot_own_foreign_keys",
                    f"Node slot {node.slot!r} with role {role.value!r} "
                    "cannot own foreign keys.",
                )
            )

        if outgoing_fk and not role_spec.can_be_referenced:
            failures.append(
                (
                    "role_cannot_be_referenced",
                    f"Node slot {node.slot!r} with role {role.value!r} "
                    "cannot be referenced.",
                )
            )

        if (
            role_spec.max_structural_parents is not None
            and incoming_structural
            > role_spec.max_structural_parents
        ):
            failures.append(
                (
                    "too_many_structural_parents",
                    f"Node slot {node.slot!r} with role {role.value!r} "
                    f"has {incoming_structural} structural parents; maximum "
                    f"is {role_spec.max_structural_parents}.",
                )
            )

        # The anchor may already have structural parents in the partial
        # schema. Fresh non-anchor nodes must be locally complete.
        if (
            node.slot != anchor_slot
            and incoming_structural
            < role_spec.min_structural_parents
        ):
            failures.append(
                (
                    "insufficient_structural_parents",
                    f"Non-anchor node slot {node.slot!r} with role "
                    f"{role.value!r} has {incoming_structural} structural "
                    f"parents; minimum is "
                    f"{role_spec.min_structural_parents}.",
                )
            )

    return tuple(failures)


def _check_unused_role_alternatives(
    *,
    nodes: tuple[MotifNodeSpec, ...],
    valid_assignments: tuple[Mapping[str, TableRole], ...],
    warning: Callable[[str, str], None],
) -> None:
    """
    Warn about declared roles that cannot participate in any valid assignment.

    Such alternatives are usually configuration mistakes. They do not make the
    motif invalid because at least one globally valid assignment still exists.
    """
    used_roles: dict[str, set[TableRole]] = defaultdict(set)

    for assignment in valid_assignments:
        for slot, role in assignment.items():
            used_roles[slot].add(role)

    for node in nodes:
        unused = set(node.roles) - used_roles[node.slot]

        if unused:
            values = ", ".join(
                sorted(role.value for role in unused)
            )

            warning(
                "unused_role_alternative",
                f"Node slot {node.slot!r} declares role alternatives "
                f"that occur in no valid assignment: {values}.",
            )


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _is_weakly_connected(
    *,
    node_slots: tuple[str, ...],
    edges: list[MotifEdgeSpec],
) -> bool:
    if not node_slots:
        return True

    adjacency: dict[str, set[str]] = {
        slot: set()
        for slot in node_slots
    }

    for edge in edges:
        adjacency[edge.parent_slot].add(edge.child_slot)
        adjacency[edge.child_slot].add(edge.parent_slot)

    start = node_slots[0]
    visited = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(node_slots)


def _is_directed_acyclic(
    *,
    node_slots: tuple[str, ...],
    edges: list[MotifEdgeSpec],
) -> bool:
    indegree = {
        slot: 0
        for slot in node_slots
    }
    children: dict[str, list[str]] = {
        slot: []
        for slot in node_slots
    }

    for edge in edges:
        children[edge.parent_slot].append(edge.child_slot)
        indegree[edge.child_slot] += 1

    queue = deque(
        sorted(
            slot
            for slot, degree in indegree.items()
            if degree == 0
        )
    )

    visited_count = 0

    while queue:
        current = queue.popleft()
        visited_count += 1

        for child in sorted(children[current]):
            indegree[child] -= 1

            if indegree[child] == 0:
                queue.append(child)

    return visited_count == len(node_slots)


# ---------------------------------------------------------------------------
# Read-only motif library
# ---------------------------------------------------------------------------


class MotifLibrary:
    """
    Validated read-only collection of motif specifications.

    Motifs are validated when the library is constructed. The library does not
    contain sampling weights; configuration maps motif_type to its weight.
    """

    __slots__ = ("_motifs",)

    def __init__(
        self,
        motifs: Iterable[MotifSpec],
    ) -> None:
        registry: dict[str, MotifSpec] = {}

        for motif in motifs:
            validate_motif_spec(
                motif,
                raise_on_error=True,
            )

            if motif.motif_type in registry:
                raise ValueError(
                    f"Duplicate motif type: {motif.motif_type!r}"
                )

            registry[motif.motif_type] = motif

        self._motifs: Mapping[str, MotifSpec] = (
            MappingProxyType(registry)
        )

    def get(self, motif_type: str) -> MotifSpec:
        try:
            return self._motifs[motif_type]
        except KeyError as error:
            available = ", ".join(self.names())

            raise KeyError(
                f"Unknown motif type {motif_type!r}. "
                f"Available motifs: {available}"
            ) from error

    def contains(self, motif_type: str) -> bool:
        return motif_type in self._motifs

    def names(self) -> tuple[str, ...]:
        """Return stable names independent of registration order."""
        return tuple(sorted(self._motifs))

    def items(self) -> tuple[tuple[str, MotifSpec], ...]:
        return tuple(
            (name, self._motifs[name])
            for name in self.names()
        )

    def __len__(self) -> int:
        return len(self._motifs)

    def __iter__(self) -> Iterator[MotifSpec]:
        for name in self.names():
            yield self._motifs[name]


# ---------------------------------------------------------------------------
# V1 default motif definitions
# ---------------------------------------------------------------------------


ENTITY_EVENT = MotifSpec(
    motif_type="entity_event",
    anchor_slot="entity",
    nodes=(
        MotifNodeSpec(
            slot="entity",
            roles=(TableRole.ENTITY,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="event",
            roles=(TableRole.EVENT,),
            rank_offset=1,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="entity_to_event",
            parent_slot="entity",
            child_slot="event",
        ),
    ),
)


ENTITY_EVENT_DETAIL = MotifSpec(
    motif_type="entity_event_detail",
    anchor_slot="entity",
    nodes=(
        MotifNodeSpec(
            slot="entity",
            roles=(TableRole.ENTITY,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="event",
            roles=(TableRole.EVENT,),
            rank_offset=1,
        ),
        MotifNodeSpec(
            slot="detail",
            roles=(TableRole.DETAIL,),
            rank_offset=2,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="entity_to_event",
            parent_slot="entity",
            child_slot="event",
        ),
        MotifEdgeSpec(
            edge="event_to_detail",
            parent_slot="event",
            child_slot="detail",
        ),
    ),
)


ENTITY_BRIDGE_COLLIDER = MotifSpec(
    motif_type="entity_bridge_collider",
    anchor_slot="left_entity",
    nodes=(
        MotifNodeSpec(
            slot="left_entity",
            roles=(TableRole.ENTITY,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="right_entity",
            roles=(TableRole.ENTITY,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="bridge",
            roles=(TableRole.BRIDGE,),
            rank_offset=1,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="left_entity_to_bridge",
            parent_slot="left_entity",
            child_slot="bridge",
        ),
        MotifEdgeSpec(
            edge="right_entity_to_bridge",
            parent_slot="right_entity",
            child_slot="bridge",
        ),
    ),
)


ENTITY_EVENT_FORK = MotifSpec(
    motif_type="entity_event_fork",
    anchor_slot="entity",
    nodes=(
        MotifNodeSpec(
            slot="entity",
            roles=(TableRole.ENTITY,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="left_event",
            roles=(TableRole.EVENT,),
            rank_offset=1,
        ),
        MotifNodeSpec(
            slot="right_event",
            roles=(TableRole.EVENT,),
            rank_offset=1,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="entity_to_left_event",
            parent_slot="entity",
            child_slot="left_event",
        ),
        MotifEdgeSpec(
            edge="entity_to_right_event",
            parent_slot="entity",
            child_slot="right_event",
        ),
    ),
)


EVENT_REFERENCE_CHAIN = MotifSpec(
    motif_type="event_reference_chain",
    anchor_slot="parent_event",
    nodes=(
        MotifNodeSpec(
            slot="parent_event",
            roles=(TableRole.EVENT,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="child_event",
            roles=(TableRole.EVENT,),
            rank_offset=1,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="parent_event_to_child_event",
            parent_slot="parent_event",
            child_slot="child_event",
        ),
    ),
)


LOOKUP_ASSIGNMENT = MotifSpec(
    motif_type="lookup_assignment",
    anchor_slot="target",
    nodes=(
        MotifNodeSpec(
            slot="lookup",
            roles=(TableRole.LOOKUP,),
            rank_offset=0,
        ),
        MotifNodeSpec(
            slot="target",
            roles=(
                TableRole.ENTITY,
                TableRole.EVENT,
                TableRole.BRIDGE,
                TableRole.DETAIL,
            ),
            rank_offset=1,
        ),
    ),
    edges=(
        MotifEdgeSpec(
            edge="lookup_to_target",
            parent_slot="lookup",
            child_slot="target",
        ),
    ),
)


DEFAULT_MOTIF_LIBRARY: Final[MotifLibrary] = MotifLibrary(
    (
        ENTITY_EVENT,
        ENTITY_EVENT_DETAIL,
        ENTITY_BRIDGE_COLLIDER,
        ENTITY_EVENT_FORK,
        EVENT_REFERENCE_CHAIN,
        LOOKUP_ASSIGNMENT,
    )
)


__all__ = [
    "MotifNodeSpec",
    "MotifEdgeSpec",
    "MotifSpec",
    "MotifIssueLevel",
    "MotifValidationIssue",
    "MotifDefinitionError",
    "validate_motif_spec",
    "resolve_motif_global_ranks",
    "validate_motif_node_bindings",
    "MotifLibrary",
    "ENTITY_EVENT",
    "ENTITY_EVENT_DETAIL",
    "ENTITY_BRIDGE_COLLIDER",
    "ENTITY_EVENT_FORK",
    "EVENT_REFERENCE_CHAIN",
    "LOOKUP_ASSIGNMENT",
    "DEFAULT_MOTIF_LIBRARY",
]
