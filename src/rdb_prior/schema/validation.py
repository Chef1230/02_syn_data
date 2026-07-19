# src/rdb_prior/schema/validation.py
# -*- coding: utf-8 -*-
"""Layered validation for logical schema blueprints and schema motifs.

Validation is deliberately separated into four layers:

``structure``
    Logical IDs, endpoints, FK orientation and rank consistency.
``role``
    Global role-edge policy and whole-node RoleSpec invariants.
``motif``
    Static motif catalog, construction attachment and persisted provenance
    checks.
``semantic``
    Evaluation of the constraint vocabulary declared in ``schema.spec`` and
    cross-reference checks for constraints evaluated at later stages.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from rdb_prior.schema.blueprint import (
    BlueprintEdge,
    BlueprintNode,
    SchemaBlueprint,
)
from rdb_prior.schema.motifs import (
    DEFAULT_MOTIF_LIBRARY,
    MotifIssueLevel,
    MotifLibrary,
    MotifSpec,
    resolve_motif_global_ranks,
    validate_motif_node_bindings,
    validate_motif_spec,
)
from rdb_prior.schema.roles import (
    LeafPolicy,
    get_role_edge_rule,
    get_role_spec,
    is_role_edge_allowed,
)
from rdb_prior.schema.spec import (
    AcyclicConstraint,
    AllowedRoleEdgeConstraint,
    ChildCountConstraint,
    ConnectedConstraint,
    ConstraintBase,
    ConstraintSeverity,
    ConstraintStage,
    DensityDefinition,
    EdgeCountConstraint,
    EdgeDensityConstraint,
    EdgeKind,
    ForbiddenRoleEdgeConstraint,
    NoParallelEdgesConstraint,
    ParentCountConstraint,
    RankOrderConstraint,
    ReachabilityConstraint,
    RequiredRoleEdgeConstraint,
    RoleCountConstraint,
    TableCountConstraint,
    TemporalOrderConstraint,
    UniqueLeafConstraint,
)


class ValidationLayer(str, Enum):
    STRUCTURE = "structure"
    ROLE = "role"
    MOTIF = "motif"
    SEMANTIC = "semantic"


class ValidationLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationIssue:
    layer: ValidationLayer
    level: ValidationLevel
    code: str
    message: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    constraint_id: str | None = None
    motif_type: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    blueprint_id: str | None
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.level is ValidationLevel.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.level is ValidationLevel.WARNING
        )

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def for_layer(
        self,
        layer: ValidationLayer,
    ) -> tuple[ValidationIssue, ...]:
        if not isinstance(layer, ValidationLayer):
            raise TypeError("layer must be ValidationLayer")
        return tuple(
            issue
            for issue in self.issues
            if issue.layer is layer
        )


class BlueprintValidationError(ValueError):
    """Raised when a validation report contains one or more errors."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "\n".join(
            f"- [{issue.layer.value}:{issue.code}] {issue.message}"
            for issue in report.errors
        )
        super().__init__(
            f"Invalid blueprint {report.blueprint_id!r}:\n{details}"
        )


def _error(
    layer: ValidationLayer,
    code: str,
    message: str,
    *,
    node_ids: tuple[str, ...] = (),
    edge_ids: tuple[str, ...] = (),
    constraint_id: str | None = None,
    motif_type: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        layer=layer,
        level=ValidationLevel.ERROR,
        code=code,
        message=message,
        node_ids=node_ids,
        edge_ids=edge_ids,
        constraint_id=constraint_id,
        motif_type=motif_type,
    )


def _constraint_violation(
    constraint: ConstraintBase,
    code: str,
    message: str,
    *,
    node_ids: tuple[str, ...] = (),
    edge_ids: tuple[str, ...] = (),
) -> ValidationIssue:
    level = (
        ValidationLevel.ERROR
        if constraint.severity is ConstraintSeverity.HARD
        else ValidationLevel.WARNING
    )
    return ValidationIssue(
        layer=ValidationLayer.SEMANTIC,
        level=level,
        code=code,
        message=message,
        node_ids=node_ids,
        edge_ids=edge_ids,
        constraint_id=constraint.constraint_id,
    )


def _require_blueprint(blueprint: SchemaBlueprint) -> None:
    if not isinstance(blueprint, SchemaBlueprint):
        raise TypeError("blueprint must be SchemaBlueprint")


def _node_index(blueprint: SchemaBlueprint) -> dict[str, BlueprintNode]:
    result: dict[str, BlueprintNode] = {}
    for node in blueprint.nodes:
        result.setdefault(node.node_id, node)
    return result


def _edge_index(blueprint: SchemaBlueprint) -> dict[str, BlueprintEdge]:
    result: dict[str, BlueprintEdge] = {}
    for edge in blueprint.edges:
        result.setdefault(edge.edge_id, edge)
    return result


def _valid_edges(
    blueprint: SchemaBlueprint,
    *,
    kinds: tuple[EdgeKind, ...] | None = None,
) -> tuple[BlueprintEdge, ...]:
    if kinds is not None and EdgeKind.FOREIGN_KEY not in kinds:
        return ()

    node_ids = {
        node.node_id
        for node in blueprint.nodes
    }
    return tuple(
        edge
        for edge in blueprint.edges
        if edge.parent_node_id in node_ids
        and edge.child_node_id in node_ids
    )


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_structure(
    blueprint: SchemaBlueprint,
) -> tuple[ValidationIssue, ...]:
    """Validate intrinsic logical graph structure."""
    _require_blueprint(blueprint)
    issues: list[ValidationIssue] = []

    if not blueprint.nodes:
        issues.append(
            _error(
                ValidationLayer.STRUCTURE,
                "empty_blueprint",
                "A schema blueprint must contain at least one node.",
            )
        )

    node_counts = Counter(node.node_id for node in blueprint.nodes)
    for node_id, count in sorted(node_counts.items()):
        if count > 1:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "duplicate_node_id",
                    f"Node ID {node_id!r} occurs {count} times.",
                    node_ids=(node_id,),
                )
            )

    edge_counts = Counter(edge.edge_id for edge in blueprint.edges)
    for edge_id, count in sorted(edge_counts.items()):
        if count > 1:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "duplicate_edge_id",
                    f"Edge ID {edge_id!r} occurs {count} times.",
                    edge_ids=(edge_id,),
                )
            )

    constraint_counts = Counter(
        constraint.constraint_id
        for constraint in blueprint.constraints
    )
    for constraint_id, count in sorted(constraint_counts.items()):
        if count > 1:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "duplicate_constraint_id",
                    f"Constraint ID {constraint_id!r} occurs {count} times.",
                    constraint_id=constraint_id,
                )
            )

    nodes = _node_index(blueprint)

    for edge in blueprint.edges:
        parent = nodes.get(edge.parent_node_id)
        child = nodes.get(edge.child_node_id)

        if edge.parent_node_id == edge.child_node_id:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "self_loop",
                    f"Edge {edge.edge_id!r} is a self-loop on node "
                    f"{edge.parent_node_id!r}.",
                    node_ids=(edge.parent_node_id,),
                    edge_ids=(edge.edge_id,),
                )
            )

        if parent is None:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "unknown_parent_node",
                    f"Edge {edge.edge_id!r} references unknown parent "
                    f"{edge.parent_node_id!r}.",
                    node_ids=(edge.parent_node_id,),
                    edge_ids=(edge.edge_id,),
                )
            )

        if child is None:
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "unknown_child_node",
                    f"Edge {edge.edge_id!r} references unknown child "
                    f"{edge.child_node_id!r}.",
                    node_ids=(edge.child_node_id,),
                    edge_ids=(edge.edge_id,),
                )
            )

        if (
            parent is not None
            and child is not None
            and parent.rank >= child.rank
        ):
            issues.append(
                _error(
                    ValidationLayer.STRUCTURE,
                    "fk_rank_order_violation",
                    f"FK edge {edge.edge_id!r} requires parent rank < "
                    f"child rank, got {parent.rank} >= {child.rank}.",
                    node_ids=(parent.node_id, child.node_id),
                    edge_ids=(edge.edge_id,),
                )
            )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Role validation
# ---------------------------------------------------------------------------


def validate_roles(
    blueprint: SchemaBlueprint,
) -> tuple[ValidationIssue, ...]:
    """Validate global RoleSpec and RoleEdgeRule invariants."""
    _require_blueprint(blueprint)
    issues: list[ValidationIssue] = []
    nodes = _node_index(blueprint)

    incoming_fk: dict[str, set[str]] = defaultdict(set)
    incoming_structural: dict[str, set[str]] = defaultdict(set)
    outgoing_fk: dict[str, set[str]] = defaultdict(set)

    for edge in _valid_edges(
        blueprint,
        kinds=(EdgeKind.FOREIGN_KEY,),
    ):
        parent = nodes[edge.parent_node_id]
        child = nodes[edge.child_node_id]
        incoming_fk[child.node_id].add(parent.node_id)
        outgoing_fk[parent.node_id].add(child.node_id)

        if not is_role_edge_allowed(parent.role, child.role):
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "forbidden_role_edge",
                    "Global role policy forbids FK edge "
                    f"{parent.role.value} -> {child.role.value}.",
                    node_ids=(parent.node_id, child.node_id),
                    edge_ids=(edge.edge_id,),
                )
            )
            continue

        rule = get_role_edge_rule(parent.role, child.role)
        if rule.counts_as_structural_parent:
            incoming_structural[child.node_id].add(parent.node_id)

    for node in blueprint.nodes:
        role_spec = get_role_spec(node.role)
        parent_count = len(incoming_structural[node.node_id])

        if incoming_fk[node.node_id] and not role_spec.can_own_foreign_keys:
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "role_cannot_own_foreign_keys",
                    f"Role {node.role.value!r} cannot own foreign keys.",
                    node_ids=(node.node_id,),
                )
            )

        if outgoing_fk[node.node_id] and not role_spec.can_be_referenced:
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "role_cannot_be_referenced",
                    f"Role {node.role.value!r} cannot be referenced.",
                    node_ids=(node.node_id,),
                )
            )

        if parent_count < role_spec.min_structural_parents:
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "insufficient_structural_parents",
                    f"Node {node.node_id!r} with role {node.role.value!r} "
                    f"has {parent_count} distinct structural parents; "
                    f"minimum is {role_spec.min_structural_parents}.",
                    node_ids=(node.node_id,),
                )
            )

        if (
            role_spec.max_structural_parents is not None
            and parent_count > role_spec.max_structural_parents
        ):
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "too_many_structural_parents",
                    f"Node {node.node_id!r} with role {node.role.value!r} "
                    f"has {parent_count} distinct structural parents; "
                    f"maximum is {role_spec.max_structural_parents}.",
                    node_ids=(node.node_id,),
                )
            )

        if (
            role_spec.leaf_policy is LeafPolicy.FORBIDDEN
            and not outgoing_fk[node.node_id]
        ):
            issues.append(
                _error(
                    ValidationLayer.ROLE,
                    "forbidden_role_leaf",
                    f"Node {node.node_id!r} with role {node.role.value!r} "
                    "cannot be an FK-graph leaf.",
                    node_ids=(node.node_id,),
                )
            )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Motif validation
# ---------------------------------------------------------------------------


def validate_motif_library(
    motifs: MotifLibrary | Iterable[MotifSpec],
) -> tuple[ValidationIssue, ...]:
    """Validate a motif catalog without constructing a schema."""
    if isinstance(motifs, MotifLibrary):
        motif_values = tuple(motifs)
    else:
        if isinstance(motifs, (str, bytes)):
            raise TypeError("motifs must be MotifLibrary or iterable")
        motif_values = tuple(motifs)

    issues: list[ValidationIssue] = []
    type_counts: Counter[str] = Counter()

    for motif in motif_values:
        if not isinstance(motif, MotifSpec):
            raise TypeError("motif catalog items must be MotifSpec")
        type_counts[motif.motif_type] += 1

        for motif_issue in validate_motif_spec(motif):
            level = (
                ValidationLevel.ERROR
                if motif_issue.level is MotifIssueLevel.ERROR
                else ValidationLevel.WARNING
            )
            issues.append(
                ValidationIssue(
                    layer=ValidationLayer.MOTIF,
                    level=level,
                    code=motif_issue.code,
                    message=motif_issue.message,
                    motif_type=motif.motif_type,
                )
            )

    for motif_type, count in sorted(type_counts.items()):
        if count > 1:
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "duplicate_motif_type",
                    f"Motif type {motif_type!r} occurs {count} times.",
                    motif_type=motif_type,
                )
            )

    return tuple(issues)


def validate_motif_attachment(
    blueprint: SchemaBlueprint,
    motif: MotifSpec,
    node_bindings: Mapping[str, str],
    *,
    existing_node_ids: Iterable[str] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate one motif attachment against a partial or completed graph."""
    _require_blueprint(blueprint)
    if not isinstance(motif, MotifSpec):
        raise TypeError("motif must be MotifSpec")

    issues: list[ValidationIssue] = []
    definition_issues = validate_motif_spec(motif)
    for motif_issue in definition_issues:
        level = (
            ValidationLevel.ERROR
            if motif_issue.level is MotifIssueLevel.ERROR
            else ValidationLevel.WARNING
        )
        issues.append(
            ValidationIssue(
                layer=ValidationLayer.MOTIF,
                level=level,
                code=motif_issue.code,
                message=motif_issue.message,
                motif_type=motif.motif_type,
            )
        )

    if any(issue.level is ValidationLevel.ERROR for issue in issues):
        return tuple(issues)

    try:
        validate_motif_node_bindings(
            motif,
            node_bindings,
            existing_node_ids=existing_node_ids,
        )
    except (TypeError, ValueError) as error:
        issues.append(
            _error(
                ValidationLayer.MOTIF,
                "invalid_motif_binding",
                str(error),
                motif_type=motif.motif_type,
            )
        )
        return tuple(issues)

    nodes = _node_index(blueprint)
    motif_nodes = {
        node.slot: node
        for node in motif.nodes
    }
    bound_nodes: dict[str, BlueprintNode] = {}

    for slot, node_id in node_bindings.items():
        node = nodes.get(node_id)
        if node is None:
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "missing_bound_node",
                    f"Motif slot {slot!r} binds unknown blueprint node "
                    f"{node_id!r}.",
                    node_ids=(node_id,),
                    motif_type=motif.motif_type,
                )
            )
            continue

        bound_nodes[slot] = node
        if node.role not in motif_nodes[slot].roles:
            allowed = ", ".join(
                role.value
                for role in motif_nodes[slot].roles
            )
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "bound_role_not_allowed",
                    f"Motif slot {slot!r} allows roles [{allowed}], but "
                    f"node {node_id!r} has role {node.role.value!r}.",
                    node_ids=(node_id,),
                    motif_type=motif.motif_type,
                )
            )

    if len(bound_nodes) != len(motif.nodes):
        return tuple(issues)

    actual_edges = {
        (edge.parent_node_id, edge.child_node_id)
        for edge in _valid_edges(
            blueprint,
            kinds=(EdgeKind.FOREIGN_KEY,),
        )
    }
    local_structural_parents: dict[str, set[str]] = defaultdict(set)

    for edge in motif.edges:
        parent_node = bound_nodes[edge.parent_slot]
        child_node = bound_nodes[edge.child_slot]
        actual_pair = (parent_node.node_id, child_node.node_id)

        if actual_pair not in actual_edges:
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "missing_motif_edge",
                    f"Motif edge {edge.edge!r} is not realized as FK "
                    f"{actual_pair[0]!r} -> {actual_pair[1]!r}.",
                    node_ids=actual_pair,
                    motif_type=motif.motif_type,
                )
            )

        if not is_role_edge_allowed(parent_node.role, child_node.role):
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "invalid_bound_role_edge",
                    f"Bound roles do not allow motif edge "
                    f"{parent_node.role.value} -> "
                    f"{child_node.role.value}.",
                    node_ids=actual_pair,
                    motif_type=motif.motif_type,
                )
            )
            continue

        rule = get_role_edge_rule(parent_node.role, child_node.role)
        if rule.counts_as_structural_parent:
            local_structural_parents[edge.child_slot].add(edge.parent_slot)

    for slot, node in bound_nodes.items():
        role_spec = get_role_spec(node.role)
        parent_count = len(local_structural_parents[slot])

        if (
            slot != motif.anchor_slot
            and parent_count < role_spec.min_structural_parents
        ):
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "incomplete_bound_motif_node",
                    f"Bound non-anchor slot {slot!r} with role "
                    f"{node.role.value!r} has {parent_count} local "
                    f"structural parents; minimum is "
                    f"{role_spec.min_structural_parents}.",
                    node_ids=(node.node_id,),
                    motif_type=motif.motif_type,
                )
            )

        if (
            role_spec.max_structural_parents is not None
            and parent_count > role_spec.max_structural_parents
        ):
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "too_many_bound_motif_parents",
                    f"Bound slot {slot!r} with role {node.role.value!r} "
                    f"has {parent_count} local structural parents; maximum "
                    f"is {role_spec.max_structural_parents}.",
                    node_ids=(node.node_id,),
                    motif_type=motif.motif_type,
                )
            )

    anchor_rank = bound_nodes[motif.anchor_slot].rank
    try:
        expected_ranks = resolve_motif_global_ranks(
            motif,
            anchor_global_rank=anchor_rank,
        )
    except ValueError as error:
        issues.append(
            _error(
                ValidationLayer.MOTIF,
                "invalid_motif_rank_attachment",
                str(error),
                motif_type=motif.motif_type,
            )
        )
    else:
        for slot, expected_rank in expected_ranks.items():
            node = bound_nodes[slot]
            if node.rank != expected_rank:
                issues.append(
                    _error(
                        ValidationLayer.MOTIF,
                        "motif_rank_mismatch",
                        f"Motif slot {slot!r} expects global rank "
                        f"{expected_rank}, but node {node.node_id!r} has "
                        f"rank {node.rank}.",
                        node_ids=(node.node_id,),
                        motif_type=motif.motif_type,
                    )
                )

    return tuple(issues)


def validate_motif_occurrences(
    blueprint: SchemaBlueprint,
    motifs: MotifLibrary = DEFAULT_MOTIF_LIBRARY,
) -> tuple[ValidationIssue, ...]:
    """Validate persisted motif provenance and exact edge bindings."""
    _require_blueprint(blueprint)
    if not isinstance(motifs, MotifLibrary):
        raise TypeError("motifs must be MotifLibrary")

    issues: list[ValidationIssue] = []
    edges = _edge_index(blueprint)
    for occurrence in blueprint.motif_occurrences:
        if not motifs.contains(occurrence.motif_type):
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "unknown_occurrence_motif",
                    f"Occurrence {occurrence.occurrence_id!r} references "
                    f"unknown motif {occurrence.motif_type!r}.",
                    motif_type=occurrence.motif_type,
                )
            )
            continue

        motif = motifs.get(occurrence.motif_type)
        issues.extend(
            validate_motif_attachment(
                blueprint,
                motif,
                occurrence.nodes,
            )
        )
        expected_edge_slots = {edge.edge for edge in motif.edges}
        actual_edge_slots = set(occurrence.edges)
        if expected_edge_slots != actual_edge_slots:
            issues.append(
                _error(
                    ValidationLayer.MOTIF,
                    "invalid_occurrence_edge_slots",
                    f"Occurrence {occurrence.occurrence_id!r} edge slots "
                    "do not match its motif definition.",
                    motif_type=occurrence.motif_type,
                )
            )
            continue

        motif_edges = {edge.edge: edge for edge in motif.edges}
        for edge_slot, edge_id in occurrence.edge_bindings:
            edge = edges.get(edge_id)
            motif_edge = motif_edges[edge_slot]
            expected_parent = occurrence.nodes[motif_edge.parent_slot]
            expected_child = occurrence.nodes[motif_edge.child_slot]
            if edge is None or (
                edge.parent_node_id != expected_parent
                or edge.child_node_id != expected_child
            ):
                issues.append(
                    _error(
                        ValidationLayer.MOTIF,
                        "invalid_occurrence_edge_binding",
                        f"Occurrence {occurrence.occurrence_id!r} binds "
                        f"edge slot {edge_slot!r} to an incompatible edge.",
                        node_ids=(expected_parent, expected_child),
                        edge_ids=(edge_id,),
                        motif_type=occurrence.motif_type,
                    )
                )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Semantic constraint validation
# ---------------------------------------------------------------------------


_SUPPORTED_CONSTRAINT_TYPES = (
    ConnectedConstraint,
    AcyclicConstraint,
    NoParallelEdgesConstraint,
    TableCountConstraint,
    EdgeCountConstraint,
    EdgeDensityConstraint,
    RoleCountConstraint,
    AllowedRoleEdgeConstraint,
    ForbiddenRoleEdgeConstraint,
    RequiredRoleEdgeConstraint,
    ParentCountConstraint,
    ChildCountConstraint,
    RankOrderConstraint,
    ReachabilityConstraint,
    TemporalOrderConstraint,
    UniqueLeafConstraint,
)


def _bounds_satisfied(
    value: int | float,
    minimum: int | float | None,
    maximum: int | float | None,
) -> bool:
    return not (
        (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    )


def _weakly_connected(
    node_ids: tuple[str, ...],
    edges: tuple[BlueprintEdge, ...],
) -> bool:
    if not node_ids:
        return False

    adjacency: dict[str, set[str]] = {
        node_id: set()
        for node_id in node_ids
    }
    for edge in edges:
        adjacency[edge.parent_node_id].add(edge.child_node_id)
        adjacency[edge.child_node_id].add(edge.parent_node_id)

    visited = {node_ids[0]}
    queue = deque((node_ids[0],))
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(node_ids)


def _acyclic(
    node_ids: tuple[str, ...],
    edges: tuple[BlueprintEdge, ...],
) -> bool:
    indegree = {
        node_id: 0
        for node_id in node_ids
    }
    children: dict[str, list[str]] = {
        node_id: []
        for node_id in node_ids
    }

    for edge in edges:
        children[edge.parent_node_id].append(edge.child_node_id)
        indegree[edge.child_node_id] += 1

    queue = deque(
        sorted(
            node_id
            for node_id, degree in indegree.items()
            if degree == 0
        )
    )
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return visited == len(node_ids)


def _path_in_hop_bounds(
    source: str,
    target: str,
    edges: tuple[BlueprintEdge, ...],
    *,
    minimum_hops: int,
    maximum_hops: int | None,
    node_count: int,
) -> bool:
    children: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        children[edge.parent_node_id].add(edge.child_node_id)

    effective_maximum = (
        maximum_hops
        if maximum_hops is not None
        else max(node_count - 1, minimum_hops)
    )
    queue = deque(((source, 0),))
    visited = {(source, 0)}

    while queue:
        current, hops = queue.popleft()
        if current == target and minimum_hops <= hops <= effective_maximum:
            return True
        if hops >= effective_maximum:
            continue

        for child in children[current]:
            state = (child, hops + 1)
            if state not in visited:
                visited.add(state)
                queue.append(state)

    return False


def _validate_constraint_references(
    blueprint: SchemaBlueprint,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    nodes = _node_index(blueprint)
    edges = _edge_index(blueprint)

    for constraint in blueprint.constraints:
        if not isinstance(constraint, _SUPPORTED_CONSTRAINT_TYPES):
            issues.append(
                _error(
                    ValidationLayer.SEMANTIC,
                    "unsupported_constraint_type",
                    f"Unsupported constraint type "
                    f"{type(constraint).__name__!r}.",
                    constraint_id=constraint.constraint_id,
                )
            )
            continue

        referenced_nodes: tuple[str, ...] = ()
        if isinstance(constraint, (ParentCountConstraint, ChildCountConstraint)):
            referenced_nodes = (constraint.node_id,)
        elif isinstance(constraint, RankOrderConstraint):
            referenced_nodes = (
                constraint.before_node,
                constraint.after_node,
            )
        elif isinstance(constraint, ReachabilityConstraint):
            referenced_nodes = (
                constraint.source_node,
                constraint.target_node,
            )
        elif isinstance(constraint, UniqueLeafConstraint):
            referenced_nodes = constraint.node_ids
        elif isinstance(constraint, TemporalOrderConstraint):
            referenced_nodes = (
                constraint.before_node,
                constraint.after_node,
            )

        for node_id in referenced_nodes:
            if node_id not in nodes:
                issues.append(
                    _error(
                        ValidationLayer.SEMANTIC,
                        "constraint_unknown_node",
                        f"Constraint {constraint.constraint_id!r} "
                        f"references unknown node {node_id!r}.",
                        node_ids=(node_id,),
                        constraint_id=constraint.constraint_id,
                    )
                )

        edge_kinds: tuple[EdgeKind, ...] = ()
        if isinstance(
            constraint,
            (
                NoParallelEdgesConstraint,
                EdgeCountConstraint,
                EdgeDensityConstraint,
                AllowedRoleEdgeConstraint,
                ForbiddenRoleEdgeConstraint,
                RequiredRoleEdgeConstraint,
                ParentCountConstraint,
                ChildCountConstraint,
                UniqueLeafConstraint,
            ),
        ):
            edge_kinds = (constraint.edge_kind,)
        elif isinstance(constraint, (AcyclicConstraint, ReachabilityConstraint)):
            edge_kinds = constraint.edge_kinds

        unsupported_kinds = tuple(
            kind
            for kind in edge_kinds
            if kind is not EdgeKind.FOREIGN_KEY
        )
        if unsupported_kinds:
            values = ", ".join(kind.value for kind in unsupported_kinds)
            issues.append(
                _error(
                    ValidationLayer.SEMANTIC,
                    "non_structural_constraint_edge_kind",
                    f"Constraint {constraint.constraint_id!r} selects "
                    f"non-schema edge kinds: {values}.",
                    constraint_id=constraint.constraint_id,
                )
            )

        if isinstance(
            constraint,
            (AllowedRoleEdgeConstraint, RequiredRoleEdgeConstraint),
        ) and (
            constraint.edge_kind is EdgeKind.FOREIGN_KEY
            and not is_role_edge_allowed(
                constraint.parent_role,
                constraint.child_role,
            )
        ):
            issues.append(
                _error(
                    ValidationLayer.SEMANTIC,
                    "constraint_exceeds_global_role_policy",
                    f"Constraint {constraint.constraint_id!r} requests "
                    f"globally forbidden role edge "
                    f"{constraint.parent_role.value} -> "
                    f"{constraint.child_role.value}.",
                    constraint_id=constraint.constraint_id,
                )
            )

        if isinstance(constraint, TemporalOrderConstraint):
            relation = edges.get(constraint.relation_id)
            if relation is None:
                issues.append(
                    _error(
                        ValidationLayer.SEMANTIC,
                        "constraint_unknown_relation",
                        f"Temporal constraint {constraint.constraint_id!r} "
                        f"references unknown relation "
                        f"{constraint.relation_id!r}.",
                        edge_ids=(constraint.relation_id,),
                        constraint_id=constraint.constraint_id,
                    )
                )
            elif (
                relation.parent_node_id != constraint.before_node
                or relation.child_node_id != constraint.after_node
            ):
                issues.append(
                    _error(
                        ValidationLayer.SEMANTIC,
                        "temporal_relation_endpoint_mismatch",
                        f"Temporal relation {relation.edge_id!r} does not "
                        "match before_node -> after_node.",
                        node_ids=(
                            constraint.before_node,
                            constraint.after_node,
                        ),
                        edge_ids=(relation.edge_id,),
                        constraint_id=constraint.constraint_id,
                    )
                )

            for node_id in referenced_nodes:
                node = nodes.get(node_id)
                if node is None:
                    continue
                role_spec = get_role_spec(node.role)
                if not role_spec.temporal_capable:
                    issues.append(
                        _error(
                            ValidationLayer.SEMANTIC,
                            "temporal_constraint_on_non_temporal_node",
                            f"Temporal constraint "
                            f"{constraint.constraint_id!r} requires node "
                            f"{node_id!r} to have temporal semantics.",
                            node_ids=(node_id,),
                            constraint_id=constraint.constraint_id,
                        )
                    )

    return tuple(issues)


def validate_semantics(
    blueprint: SchemaBlueprint,
    *,
    stage: ConstraintStage = ConstraintStage.BLUEPRINT,
) -> tuple[ValidationIssue, ...]:
    """Validate constraint references and evaluate constraints for ``stage``."""
    _require_blueprint(blueprint)
    if not isinstance(stage, ConstraintStage):
        raise TypeError("stage must be ConstraintStage")

    issues = list(_validate_constraint_references(blueprint))
    nodes = _node_index(blueprint)
    node_ids = tuple(nodes)
    valid_edges = _valid_edges(blueprint)

    allowed_constraints: dict[
        EdgeKind,
        list[AllowedRoleEdgeConstraint],
    ] = defaultdict(list)
    for constraint in blueprint.constraints:
        if (
            isinstance(constraint, AllowedRoleEdgeConstraint)
            and constraint.stage is stage
        ):
            allowed_constraints[constraint.edge_kind].append(constraint)

    for edge_kind, constraints in allowed_constraints.items():
        if edge_kind is not EdgeKind.FOREIGN_KEY:
            continue
        allowed_pairs = {
            (constraint.parent_role, constraint.child_role)
            for constraint in constraints
        }
        representative = next(
            (
                constraint
                for constraint in constraints
                if constraint.severity is ConstraintSeverity.HARD
            ),
            constraints[0],
        )
        for edge in valid_edges:
            parent = nodes[edge.parent_node_id]
            child = nodes[edge.child_node_id]
            if (parent.role, child.role) not in allowed_pairs:
                issues.append(
                    _constraint_violation(
                        representative,
                        "allowed_role_edge_violation",
                        f"FK edge {edge.edge_id!r} uses role pair "
                        f"{parent.role.value} -> {child.role.value}, which "
                        "is outside the blueprint allowlist.",
                        node_ids=(parent.node_id, child.node_id),
                        edge_ids=(edge.edge_id,),
                    )
                )

    for constraint in blueprint.constraints:
        if not isinstance(constraint, _SUPPORTED_CONSTRAINT_TYPES):
            continue
        if constraint.stage is not stage:
            continue
        if isinstance(constraint, AllowedRoleEdgeConstraint):
            continue

        if isinstance(constraint, ConnectedConstraint):
            if not _weakly_connected(node_ids, valid_edges):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "connected_constraint_violation",
                        "Blueprint graph is not weakly connected.",
                    )
                )

        elif isinstance(constraint, AcyclicConstraint):
            edges = (
                valid_edges
                if EdgeKind.FOREIGN_KEY in constraint.edge_kinds
                else ()
            )
            if not _acyclic(node_ids, edges):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "acyclic_constraint_violation",
                        "Selected blueprint edge projection is cyclic.",
                    )
                )

        elif isinstance(constraint, NoParallelEdgesConstraint):
            selected_edges = (
                valid_edges
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
                else ()
            )
            logical_counts = Counter(
                (
                    edge.parent_node_id,
                    edge.child_node_id,
                    EdgeKind.FOREIGN_KEY,
                )
                for edge in selected_edges
            )
            for (parent, child, _kind), count in logical_counts.items():
                if count > 1:
                    issues.append(
                        _constraint_violation(
                            constraint,
                            "parallel_edge_constraint_violation",
                            f"Logical edge {parent!r} -> {child!r} occurs "
                            f"{count} times.",
                            node_ids=(parent, child),
                        )
                    )

        elif isinstance(constraint, TableCountConstraint):
            count = len(blueprint.nodes)
            if not _bounds_satisfied(
                count,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "table_count_constraint_violation",
                        f"Table count {count} is outside requested bounds.",
                    )
                )

        elif isinstance(constraint, EdgeCountConstraint):
            count = (
                len(valid_edges)
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
                else 0
            )
            if not _bounds_satisfied(
                count,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "edge_count_constraint_violation",
                        f"Selected edge count {count} is outside requested "
                        "bounds.",
                    )
                )

        elif isinstance(constraint, EdgeDensityConstraint):
            edge_count = (
                len(valid_edges)
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
                else 0
            )
            node_count = len(blueprint.nodes)
            if constraint.definition is DensityDefinition.EDGES_PER_NODE:
                density = edge_count / node_count if node_count else 0.0
            else:
                denominator = node_count * (node_count - 1)
                density = edge_count / denominator if denominator else 0.0

            if not _bounds_satisfied(
                density,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "edge_density_constraint_violation",
                        f"Selected edge density {density:.6g} is outside "
                        "requested bounds.",
                    )
                )

        elif isinstance(constraint, RoleCountConstraint):
            count = sum(
                node.role is constraint.role
                for node in blueprint.nodes
            )
            if not _bounds_satisfied(
                count,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "role_count_constraint_violation",
                        f"Role {constraint.role.value!r} count {count} is "
                        "outside requested bounds.",
                    )
                )

        elif isinstance(constraint, ForbiddenRoleEdgeConstraint):
            if constraint.edge_kind is not EdgeKind.FOREIGN_KEY:
                continue
            for edge in valid_edges:
                parent = nodes[edge.parent_node_id]
                child = nodes[edge.child_node_id]
                if (
                    parent.role is constraint.parent_role
                    and child.role is constraint.child_role
                ):
                    issues.append(
                        _constraint_violation(
                            constraint,
                            "forbidden_role_edge_constraint_violation",
                            f"Edge {edge.edge_id!r} realizes forbidden role "
                            f"pair {parent.role.value} -> "
                            f"{child.role.value}.",
                            node_ids=(parent.node_id, child.node_id),
                            edge_ids=(edge.edge_id,),
                        )
                    )

        elif isinstance(constraint, RequiredRoleEdgeConstraint):
            if constraint.edge_kind is not EdgeKind.FOREIGN_KEY:
                continue
            count = 0
            for edge in valid_edges:
                parent = nodes[edge.parent_node_id]
                child = nodes[edge.child_node_id]
                if (
                    parent.role is constraint.parent_role
                    and child.role is constraint.child_role
                ):
                    count += 1
            if count < constraint.minimum:
                issues.append(
                    _constraint_violation(
                        constraint,
                        "required_role_edge_constraint_violation",
                        f"Role edge {constraint.parent_role.value} -> "
                        f"{constraint.child_role.value} occurs {count} "
                        f"times; minimum is {constraint.minimum}.",
                    )
                )

        elif isinstance(constraint, ParentCountConstraint):
            if constraint.node_id not in nodes:
                continue
            selected = tuple(
                edge
                for edge in valid_edges
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
                and edge.child_node_id == constraint.node_id
            )
            count = (
                len({edge.parent_node_id for edge in selected})
                if constraint.distinct_parent_nodes
                else len(selected)
            )
            if not _bounds_satisfied(
                count,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "parent_count_constraint_violation",
                        f"Parent count for {constraint.node_id!r} is {count}, "
                        "outside requested bounds.",
                        node_ids=(constraint.node_id,),
                    )
                )

        elif isinstance(constraint, ChildCountConstraint):
            if constraint.node_id not in nodes:
                continue
            selected = tuple(
                edge
                for edge in valid_edges
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
                and edge.parent_node_id == constraint.node_id
            )
            count = (
                len({edge.child_node_id for edge in selected})
                if constraint.distinct_child_nodes
                else len(selected)
            )
            if not _bounds_satisfied(
                count,
                constraint.minimum,
                constraint.maximum,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "child_count_constraint_violation",
                        f"Child count for {constraint.node_id!r} is {count}, "
                        "outside requested bounds.",
                        node_ids=(constraint.node_id,),
                    )
                )

        elif isinstance(constraint, RankOrderConstraint):
            before = nodes.get(constraint.before_node)
            after = nodes.get(constraint.after_node)
            if before is None or after is None:
                continue
            gap = after.rank - before.rank
            if not _bounds_satisfied(
                gap,
                constraint.minimum_gap,
                constraint.maximum_gap,
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "rank_order_constraint_violation",
                        f"Rank gap {gap} for {before.node_id!r} -> "
                        f"{after.node_id!r} is outside requested bounds.",
                        node_ids=(before.node_id, after.node_id),
                    )
                )

        elif isinstance(constraint, ReachabilityConstraint):
            if (
                constraint.source_node not in nodes
                or constraint.target_node not in nodes
            ):
                continue
            edges = (
                valid_edges
                if EdgeKind.FOREIGN_KEY in constraint.edge_kinds
                else ()
            )
            if not _path_in_hop_bounds(
                constraint.source_node,
                constraint.target_node,
                edges,
                minimum_hops=constraint.minimum_hops,
                maximum_hops=constraint.maximum_hops,
                node_count=len(nodes),
            ):
                issues.append(
                    _constraint_violation(
                        constraint,
                        "reachability_constraint_violation",
                        f"No path from {constraint.source_node!r} to "
                        f"{constraint.target_node!r} satisfies requested "
                        "hop bounds.",
                        node_ids=(
                            constraint.source_node,
                            constraint.target_node,
                        ),
                    )
                )

        elif isinstance(constraint, UniqueLeafConstraint):
            candidates = {
                node.node_id
                for node in blueprint.nodes
                if node.role in constraint.roles
            }
            candidates.update(
                node_id
                for node_id in constraint.node_ids
                if node_id in nodes
            )
            parent_ids = {
                edge.parent_node_id
                for edge in valid_edges
                if constraint.edge_kind is EdgeKind.FOREIGN_KEY
            }
            leaves = tuple(
                sorted(candidates - parent_ids)
            )
            if len(leaves) != 1:
                issues.append(
                    _constraint_violation(
                        constraint,
                        "unique_leaf_constraint_violation",
                        f"Selected node set contains {len(leaves)} leaves; "
                        "exactly one is required.",
                        node_ids=leaves,
                    )
                )

        # TemporalOrderConstraint is reference-checked above. Row-time
        # comparison requires a DatabaseInstance and is intentionally not
        # evaluated against SchemaBlueprint.

    return tuple(issues)


# ---------------------------------------------------------------------------
# Aggregate API
# ---------------------------------------------------------------------------


def validate_blueprint(
    blueprint: SchemaBlueprint,
    *,
    raise_on_error: bool = False,
    motifs: MotifLibrary = DEFAULT_MOTIF_LIBRARY,
) -> ValidationReport:
    """Run structure, role and semantic validation for one blueprint."""
    _require_blueprint(blueprint)
    if not isinstance(raise_on_error, bool):
        raise TypeError("raise_on_error must be a boolean")
    if not isinstance(motifs, MotifLibrary):
        raise TypeError("motifs must be MotifLibrary")

    report = ValidationReport(
        blueprint_id=blueprint.blueprint_id,
        issues=(
            *validate_structure(blueprint),
            *validate_roles(blueprint),
            *validate_motif_occurrences(blueprint, motifs),
            *validate_semantics(blueprint),
        ),
    )

    if raise_on_error and not report.is_valid:
        raise BlueprintValidationError(report)

    return report


__all__ = [
    "ValidationLayer",
    "ValidationLevel",
    "ValidationIssue",
    "ValidationReport",
    "BlueprintValidationError",
    "validate_structure",
    "validate_roles",
    "validate_motif_library",
    "validate_motif_attachment",
    "validate_motif_occurrences",
    "validate_semantics",
    "validate_blueprint",
]
