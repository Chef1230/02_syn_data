# src/rdb_prior/schema/blueprint.py
# -*- coding: utf-8 -*-
"""
Immutable logical schema blueprint.

A SchemaBlueprint is the final anonymous logical schema produced by the
constructive motif composer:

    roles.py + spec.py + motifs.py
                  ↓
           Blueprint Builder
                  ↓
           SchemaBlueprint
                  ↓
              Compiler
                  ↓
            PhysicalSchema

The Blueprint contains only:

- stable anonymous logical node IDs;
- latent structural table roles;
- final logical ranks;
- stable anonymous logical FK edge IDs;
- parent -> child logical FK topology;
- schema constraints that the compiled schema must satisfy.

It intentionally does not contain:

- motif types, versions, occurrences, or slot bindings;
- motif sampling weights or occurrence counts;
- domain or natural-language semantics;
- task, process, or label-generation definitions;
- physical table or column names;
- SQL types, PK/FK column implementations;
- row counts, FK values, or generated data.

Edge orientation
----------------
Every logical edge follows:

    referenced parent node -> FK-owning child node
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from rdb_prior.schema.spec import (
    ConstraintBase,
    EdgeId,
    NodeId,
    SchemaConstraint,
    TableRole,
)


# ---------------------------------------------------------------------------
# Logical graph objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlueprintNode:
    """
    One anonymous logical table node.

    Parameters
    ----------
    node_id:
        Stable logical identifier, such as ``N000``.

        It must remain stable across traversal order, worker count, and
        parallel execution changes.

    role:
        Latent structural role used by the anonymous generator.

    rank:
        Final logical rank after all motifs have been composed and merged.

        Rank is part of the completed logical schema, not a temporary motif
        offset or compiler hint. The compiler may verify it but should not
        silently reinterpret it.
    """

    node_id: NodeId
    role: TableRole
    rank: int

    def __post_init__(self) -> None:
        _require_identifier(
            self.node_id,
            field_name="node_id",
        )

        if not isinstance(self.role, TableRole):
            raise TypeError("role must be TableRole")

        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(
                f"rank must be an integer, got {type(self.rank).__name__}"
            )

        if self.rank < 0:
            raise ValueError("rank must be non-negative")


@dataclass(frozen=True, slots=True)
class BlueprintEdge:
    """
    One anonymous logical foreign-key edge.

    The edge represents only logical parent-child topology. Physical FK
    columns, cardinality defaults, nullability, relation strategies, and SQL
    types are resolved later by the compiler.
    """

    edge_id: EdgeId
    parent_node_id: NodeId
    child_node_id: NodeId

    def __post_init__(self) -> None:
        _require_identifier(
            self.edge_id,
            field_name="edge_id",
        )
        _require_identifier(
            self.parent_node_id,
            field_name="parent_node_id",
        )
        _require_identifier(
            self.child_node_id,
            field_name="child_node_id",
        )


# ---------------------------------------------------------------------------
# Completed logical schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchemaBlueprint:
    """
    Completed immutable logical schema.

    Construction order is not semantically meaningful. Nodes, edges, and
    constraints are canonicalized by their stable IDs so serialized output is
    independent of motif composition order or internal dictionary iteration.

    Only local representation invariants are checked here:

    - IDs are non-empty;
    - node IDs are unique;
    - edge IDs are unique;
    - constraint IDs are unique;
    - every edge endpoint exists.

    Complete graph validation belongs in ``schema.validation``:

    - connectivity;
    - DAG property;
    - duplicate logical parent-child edges;
    - self-loops;
    - rank consistency;
    - allowed role edges;
    - structural parent counts;
    - root and leaf constraints;
    - all declared SchemaConstraint objects.
    """

    blueprint_id: str
    nodes: tuple[BlueprintNode, ...]
    edges: tuple[BlueprintEdge, ...]
    constraints: tuple[SchemaConstraint, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(
            self.blueprint_id,
            field_name="blueprint_id",
        )

        if not isinstance(self.nodes, tuple):
            raise TypeError("nodes must be a tuple")

        if not isinstance(self.edges, tuple):
            raise TypeError("edges must be a tuple")

        if not isinstance(self.constraints, tuple):
            raise TypeError("constraints must be a tuple")

        if not self.nodes:
            raise ValueError(
                "SchemaBlueprint must contain at least one node"
            )

        for node in self.nodes:
            if not isinstance(node, BlueprintNode):
                raise TypeError("nodes items must be BlueprintNode")

        for edge in self.edges:
            if not isinstance(edge, BlueprintEdge):
                raise TypeError("edges items must be BlueprintEdge")

        for constraint in self.constraints:
            if not isinstance(constraint, ConstraintBase):
                raise TypeError(
                    "constraints items must be concrete SchemaConstraint "
                    "instances"
                )

        # Canonical ordering ensures deterministic persistence and comparison.
        canonical_nodes = tuple(
            sorted(
                self.nodes,
                key=lambda node: node.node_id,
            )
        )
        canonical_edges = tuple(
            sorted(
                self.edges,
                key=lambda edge: edge.edge_id,
            )
        )
        canonical_constraints = tuple(
            sorted(
                self.constraints,
                key=lambda constraint: constraint.constraint_id,
            )
        )

        object.__setattr__(
            self,
            "nodes",
            canonical_nodes,
        )
        object.__setattr__(
            self,
            "edges",
            canonical_edges,
        )
        object.__setattr__(
            self,
            "constraints",
            canonical_constraints,
        )

        node_ids = tuple(
            node.node_id
            for node in canonical_nodes
        )
        edge_ids = tuple(
            edge.edge_id
            for edge in canonical_edges
        )
        constraint_ids = tuple(
            constraint.constraint_id
            for constraint in canonical_constraints
        )

        _require_unique(
            node_ids,
            field_name="Blueprint node IDs",
        )
        _require_unique(
            edge_ids,
            field_name="Blueprint edge IDs",
        )
        _require_unique(
            constraint_ids,
            field_name="Blueprint constraint IDs",
        )

        known_node_ids = frozenset(node_ids)

        # Referential integrity is a representation invariant rather than a
        # graph policy, so it is checked immediately.
        for edge in canonical_edges:
            if edge.parent_node_id not in known_node_ids:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown parent node "
                    f"{edge.parent_node_id!r}"
                )

            if edge.child_node_id not in known_node_ids:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown child node "
                    f"{edge.child_node_id!r}"
                )

    # ------------------------------------------------------------------
    # Basic lookup
    # ------------------------------------------------------------------

    def node(
        self,
        node_id: NodeId,
    ) -> BlueprintNode:
        """Return one logical node by stable ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node

        raise KeyError(
            f"Blueprint {self.blueprint_id!r} has no node {node_id!r}"
        )

    def edge(
        self,
        edge_id: EdgeId,
    ) -> BlueprintEdge:
        """Return one logical edge by stable ID."""
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge

        raise KeyError(
            f"Blueprint {self.blueprint_id!r} has no edge {edge_id!r}"
        )

    def node_map(
        self,
    ) -> Mapping[NodeId, BlueprintNode]:
        """Return a read-only node lookup mapping."""
        return MappingProxyType(
            {
                node.node_id: node
                for node in self.nodes
            }
        )

    def edge_map(
        self,
    ) -> Mapping[EdgeId, BlueprintEdge]:
        """Return a read-only edge lookup mapping."""
        return MappingProxyType(
            {
                edge.edge_id: edge
                for edge in self.edges
            }
        )

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    def incoming_edges(
        self,
        node_id: NodeId,
    ) -> tuple[BlueprintEdge, ...]:
        """
        Return logical FK edges whose child is ``node_id``.

        These correspond to FK columns owned by the node after compilation.
        """
        self.node(node_id)

        return tuple(
            edge
            for edge in self.edges
            if edge.child_node_id == node_id
        )

    def outgoing_edges(
        self,
        node_id: NodeId,
    ) -> tuple[BlueprintEdge, ...]:
        """
        Return logical FK edges whose parent is ``node_id``.

        These represent downstream tables referencing this node.
        """
        self.node(node_id)

        return tuple(
            edge
            for edge in self.edges
            if edge.parent_node_id == node_id
        )

    def parents(
        self,
        node_id: NodeId,
    ) -> tuple[BlueprintNode, ...]:
        """Return parent nodes in stable node-ID order."""
        parent_ids = {
            edge.parent_node_id
            for edge in self.incoming_edges(node_id)
        }

        return tuple(
            node
            for node in self.nodes
            if node.node_id in parent_ids
        )

    def children(
        self,
        node_id: NodeId,
    ) -> tuple[BlueprintNode, ...]:
        """Return child nodes in stable node-ID order."""
        child_ids = {
            edge.child_node_id
            for edge in self.outgoing_edges(node_id)
        }

        return tuple(
            node
            for node in self.nodes
            if node.node_id in child_ids
        )

    # ------------------------------------------------------------------
    # Role and rank access
    # ------------------------------------------------------------------

    def nodes_by_role(
        self,
        role: TableRole,
    ) -> tuple[BlueprintNode, ...]:
        """Return all logical nodes assigned to one role."""
        return tuple(
            node
            for node in self.nodes
            if node.role == role
        )

    def nodes_at_rank(
        self,
        rank: int,
    ) -> tuple[BlueprintNode, ...]:
        """Return all nodes at one final logical rank."""
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")

        if rank < 0:
            raise ValueError("rank must be non-negative")

        return tuple(
            node
            for node in self.nodes
            if node.rank == rank
        )

    @property
    def max_rank(self) -> int:
        """Return the maximum final logical rank."""
        return max(
            node.rank
            for node in self.nodes
        )

    @property
    def root_nodes(self) -> tuple[BlueprintNode, ...]:
        """
        Return nodes with no incoming logical FK edge.

        This is purely topology-based. Whether a role is allowed to be a root
        is checked in ``schema.validation``.
        """
        child_ids = {
            edge.child_node_id
            for edge in self.edges
        }

        return tuple(
            node
            for node in self.nodes
            if node.node_id not in child_ids
        )

    @property
    def leaf_nodes(self) -> tuple[BlueprintNode, ...]:
        """
        Return nodes with no outgoing logical FK edge.

        Whether a role should or may be a leaf is checked separately.
        """
        parent_ids = {
            edge.parent_node_id
            for edge in self.edges
        }

        return tuple(
            node
            for node in self.nodes
            if node.node_id not in parent_ids
        )


# ---------------------------------------------------------------------------
# Internal invariant helpers
# ---------------------------------------------------------------------------


def _require_identifier(
    value: str,
    *,
    field_name: str,
) -> None:
    if isinstance(value, Enum) or not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string, "
            f"got {type(value).__name__}"
        )

    if not value:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain leading or trailing whitespace"
        )


def _require_unique(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    if duplicates:
        duplicate_text = ", ".join(
            sorted(duplicates)
        )

        raise ValueError(
            f"{field_name} must be unique; duplicates: "
            f"{duplicate_text}"
        )


__all__ = [
    "BlueprintNode",
    "BlueprintEdge",
    "SchemaBlueprint",
]
