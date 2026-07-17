# src/rdb_prior/schema/sampler.py
# -*- coding: utf-8 -*-
"""Deterministic role/motif-aware SchemaBlueprint sampling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Final, Mapping

from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import (
    BlueprintEdge,
    BlueprintNode,
    SchemaBlueprint,
)
from rdb_prior.schema.motifs import (
    DEFAULT_MOTIF_LIBRARY,
    MotifLibrary,
    MotifSpec,
    resolve_motif_global_ranks,
)
from rdb_prior.schema.roles import (
    get_role_edge_rule,
    get_role_spec,
    is_role_edge_allowed,
)
from rdb_prior.schema.spec import (
    AcyclicConstraint,
    ConnectedConstraint,
    EdgeCountConstraint,
    NoParallelEdgesConstraint,
    RequiredRoleEdgeConstraint,
    RoleCountConstraint,
    TableCountConstraint,
    TableRole,
)
from rdb_prior.schema.validation import (
    BlueprintValidationError,
    validate_blueprint,
    validate_motif_attachment,
)


_DEFAULT_MOTIF_WEIGHTS: Final[tuple[tuple[str, float], ...]] = (
    ("entity_event", 1.0),
    ("entity_event_detail", 1.4),
    ("entity_bridge_collider", 1.0),
    ("entity_event_fork", 1.0),
    ("event_reference_chain", 0.8),
    ("lookup_assignment", 0.8),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlueprintSamplerConfig:
    """Configuration containing only logical schema sampling choices."""

    min_tables: int = 3
    max_tables: int = 12
    table_count_values: tuple[int, ...] = ()
    table_count_weights: tuple[float, ...] = ()
    max_rank: int = 3
    max_extra_edges: int = 2
    extra_edge_probability: float = 0.35
    blueprint_id_prefix: str = "blueprint"
    motif_weights: tuple[
        tuple[str, float], ...
    ] = _DEFAULT_MOTIF_WEIGHTS

    def __post_init__(self) -> None:
        for name, value in (
            ("min_tables", self.min_tables),
            ("max_tables", self.max_tables),
            ("max_rank", self.max_rank),
            ("max_extra_edges", self.max_extra_edges),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")

        if self.min_tables < 2:
            raise ValueError("min_tables must be at least 2")
        if self.max_tables < self.min_tables:
            raise ValueError("max_tables must be at least min_tables")

        if not isinstance(self.table_count_values, tuple):
            raise TypeError("table_count_values must be a tuple")
        if not isinstance(self.table_count_weights, tuple):
            raise TypeError("table_count_weights must be a tuple")
        if bool(self.table_count_values) != bool(self.table_count_weights):
            raise ValueError(
                "table_count_values and table_count_weights must either "
                "both be empty or both be populated"
            )
        if len(self.table_count_values) != len(self.table_count_weights):
            raise ValueError(
                "table_count_values and table_count_weights must have "
                "the same length"
            )
        if len(set(self.table_count_values)) != len(
            self.table_count_values
        ):
            raise ValueError("table_count_values must be unique")
        for value in self.table_count_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("table_count_values items must be integers")
            if not self.min_tables <= value <= self.max_tables:
                raise ValueError(
                    "table_count_values items must be within min_tables "
                    "and max_tables"
                )
        for weight in self.table_count_weights:
            if isinstance(weight, bool) or not isinstance(
                weight,
                (int, float),
            ):
                raise TypeError("table_count_weights items must be numeric")
            if weight <= 0:
                raise ValueError(
                    "table_count_weights items must be positive"
                )
        if self.max_rank < 1:
            raise ValueError("max_rank must be at least 1")
        if self.max_extra_edges < 0:
            raise ValueError("max_extra_edges must be non-negative")

        if (
            isinstance(self.extra_edge_probability, bool)
            or not isinstance(self.extra_edge_probability, (int, float))
        ):
            raise TypeError("extra_edge_probability must be numeric")
        if not 0 <= self.extra_edge_probability <= 1:
            raise ValueError(
                "extra_edge_probability must be between zero and one"
            )

        if (
            not isinstance(self.blueprint_id_prefix, str)
            or not self.blueprint_id_prefix.strip()
        ):
            raise ValueError("blueprint_id_prefix must not be empty")

        if not isinstance(self.motif_weights, tuple):
            raise TypeError("motif_weights must be a tuple")
        if not self.motif_weights:
            raise ValueError("motif_weights must not be empty")

        seen: set[str] = set()
        for entry in self.motif_weights:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "motif_weights entries must be (motif_type, weight)"
                )
            motif_type, weight = entry
            if not isinstance(motif_type, str) or not motif_type.strip():
                raise ValueError("motif type must not be empty")
            if motif_type in seen:
                raise ValueError("motif_weights contains duplicate types")
            seen.add(motif_type)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
            ):
                raise TypeError("motif weight must be numeric")
            if weight <= 0:
                raise ValueError("motif weight must be positive")


@dataclass(frozen=True, slots=True)
class _AttachmentOption:
    motif: MotifSpec
    anchor: BlueprintNode
    roles: Mapping[str, TableRole]
    ranks: Mapping[str, int]


class BlueprintSampler:
    """Sample complete valid blueprints through ephemeral motif attachment."""

    __slots__ = ("config", "motifs", "_weights", "_base_motif")

    def __init__(
        self,
        config: BlueprintSamplerConfig | None = None,
        motifs: MotifLibrary = DEFAULT_MOTIF_LIBRARY,
    ) -> None:
        self.config = config or BlueprintSamplerConfig()
        if not isinstance(motifs, MotifLibrary):
            raise TypeError("motifs must be MotifLibrary")
        self.motifs = motifs

        weights = dict(self.config.motif_weights)
        unknown = set(weights) - set(motifs.names())
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configured motif types: {names}")
        self._weights = weights
        if not motifs.contains("entity_event"):
            raise ValueError(
                "Motif library must contain the entity_event base motif"
            )
        self._base_motif = motifs.get("entity_event")

    def sample(
        self,
        sample_id: str | int,
        runtime: RuntimeContext,
    ) -> SchemaBlueprint:
        """Sample one deterministic complete logical schema."""
        if isinstance(sample_id, bool) or not isinstance(
            sample_id,
            (str, int),
        ):
            raise TypeError("sample_id must be a string or integer")
        if isinstance(sample_id, str) and not sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext")

        count_rng = runtime.python_rng(
            "schema",
            "blueprint",
            "table-count",
        )
        if self.config.table_count_values:
            target_tables = count_rng.choices(
                self.config.table_count_values,
                weights=self.config.table_count_weights,
                k=1,
            )[0]
        else:
            target_tables = count_rng.randint(
                self.config.min_tables,
                self.config.max_tables,
            )

        nodes: list[BlueprintNode] = [
            BlueprintNode(
                node_id="N000",
                role=TableRole.ENTITY,
                rank=0,
            )
        ]
        edges: list[BlueprintEdge] = []

        self._attach(
            motif=self._base_motif,
            anchor=nodes[0],
            roles={
                "entity": TableRole.ENTITY,
                "event": TableRole.EVENT,
            },
            ranks={"entity": 0, "event": 1},
            nodes=nodes,
            edges=edges,
            existing_node_ids=(nodes[0].node_id,),
        )

        step = 0
        while len(nodes) < target_tables:
            remaining = target_tables - len(nodes)
            options_by_type = self._feasible_options(
                nodes=tuple(nodes),
                remaining=remaining,
            )
            if not options_by_type:
                raise RuntimeError(
                    "No feasible motif attachment can reach requested "
                    f"table count {target_tables} from {len(nodes)} nodes"
                )

            step_rng = runtime.python_rng(
                "schema",
                "blueprint",
                "motif-step",
                step,
            )
            motif_types = tuple(sorted(options_by_type))
            weights = tuple(
                self._weights[motif_type]
                for motif_type in motif_types
            )
            selected_type = step_rng.choices(
                motif_types,
                weights=weights,
                k=1,
            )[0]
            options = options_by_type[selected_type]
            option = options[step_rng.randrange(len(options))]
            existing_node_ids = tuple(node.node_id for node in nodes)

            self._attach(
                motif=option.motif,
                anchor=option.anchor,
                roles=option.roles,
                ranks=option.ranks,
                nodes=nodes,
                edges=edges,
                existing_node_ids=existing_node_ids,
            )
            step += 1

        self._sample_extra_edges(
            nodes=nodes,
            edges=edges,
            runtime=runtime,
        )

        blueprint = SchemaBlueprint(
            blueprint_id=(
                f"{self.config.blueprint_id_prefix}_{sample_id}"
            ),
            nodes=tuple(nodes),
            edges=tuple(edges),
            constraints=(
                ConnectedConstraint(constraint_id="connected"),
                AcyclicConstraint(constraint_id="acyclic_fk"),
                NoParallelEdgesConstraint(
                    constraint_id="no_parallel_fk"
                ),
                TableCountConstraint(
                    constraint_id="exact_table_count",
                    minimum=target_tables,
                    maximum=target_tables,
                ),
                EdgeCountConstraint(
                    constraint_id="exact_edge_count",
                    minimum=len(edges),
                    maximum=len(edges),
                ),
                RoleCountConstraint(
                    constraint_id="require_entity",
                    role=TableRole.ENTITY,
                    minimum=1,
                ),
                RoleCountConstraint(
                    constraint_id="require_event",
                    role=TableRole.EVENT,
                    minimum=1,
                ),
                RequiredRoleEdgeConstraint(
                    constraint_id="require_entity_event",
                    parent_role=TableRole.ENTITY,
                    child_role=TableRole.EVENT,
                    minimum=1,
                ),
            ),
        )

        report = validate_blueprint(blueprint)
        if not report.is_valid:
            raise BlueprintValidationError(report)
        return blueprint

    def _feasible_options(
        self,
        *,
        nodes: tuple[BlueprintNode, ...],
        remaining: int,
    ) -> dict[str, tuple[_AttachmentOption, ...]]:
        result: dict[str, tuple[_AttachmentOption, ...]] = {}

        for motif in self.motifs:
            if motif.motif_type not in self._weights:
                continue
            new_node_count = len(motif.nodes) - 1
            if new_node_count > remaining:
                continue
            options: list[_AttachmentOption] = []
            anchor_spec = next(
                node
                for node in motif.nodes
                if node.slot == motif.anchor_slot
            )

            for anchor in nodes:
                if anchor.role not in anchor_spec.roles:
                    continue
                try:
                    ranks = resolve_motif_global_ranks(
                        motif,
                        anchor_global_rank=anchor.rank,
                    )
                except ValueError:
                    continue
                if max(ranks.values()) > self.config.max_rank:
                    continue

                assignments = self._role_assignments(
                    motif,
                    anchor.role,
                )
                for assignment in assignments:
                    options.append(
                        _AttachmentOption(
                            motif=motif,
                            anchor=anchor,
                            roles=assignment,
                            ranks=ranks,
                        )
                    )

            if options:
                result[motif.motif_type] = tuple(options)

        return result

    @staticmethod
    def _role_assignments(
        motif: MotifSpec,
        anchor_role: TableRole,
    ) -> tuple[Mapping[str, TableRole], ...]:
        slots = tuple(node.slot for node in motif.nodes)
        domains = tuple(
            (
                (anchor_role,)
                if node.slot == motif.anchor_slot
                else node.roles
            )
            for node in motif.nodes
        )
        valid: list[Mapping[str, TableRole]] = []

        for values in product(*domains):
            assignment = dict(zip(slots, values, strict=True))
            if not all(
                is_role_edge_allowed(
                    assignment[edge.parent_slot],
                    assignment[edge.child_slot],
                )
                for edge in motif.edges
            ):
                continue

            structural_counts: dict[str, int] = defaultdict(int)
            for edge in motif.edges:
                rule = get_role_edge_rule(
                    assignment[edge.parent_slot],
                    assignment[edge.child_slot],
                )
                if rule.counts_as_structural_parent:
                    structural_counts[edge.child_slot] += 1

            complete = True
            for node in motif.nodes:
                role_spec = get_role_spec(assignment[node.slot])
                count = structural_counts[node.slot]
                if (
                    node.slot != motif.anchor_slot
                    and count < role_spec.min_structural_parents
                ):
                    complete = False
                    break
                if (
                    role_spec.max_structural_parents is not None
                    and count > role_spec.max_structural_parents
                ):
                    complete = False
                    break

            if complete:
                valid.append(assignment)

        return tuple(valid)

    @staticmethod
    def _attach(
        *,
        motif: MotifSpec,
        anchor: BlueprintNode,
        roles: Mapping[str, TableRole],
        ranks: Mapping[str, int],
        nodes: list[BlueprintNode],
        edges: list[BlueprintEdge],
        existing_node_ids: tuple[str, ...],
    ) -> None:
        bindings: dict[str, str] = {
            motif.anchor_slot: anchor.node_id,
        }

        for node_spec in motif.nodes:
            if node_spec.slot == motif.anchor_slot:
                continue
            node_id = f"N{len(nodes):03d}"
            nodes.append(
                BlueprintNode(
                    node_id=node_id,
                    role=roles[node_spec.slot],
                    rank=ranks[node_spec.slot],
                )
            )
            bindings[node_spec.slot] = node_id

        for edge_spec in motif.edges:
            edges.append(
                BlueprintEdge(
                    edge_id=f"E{len(edges):03d}",
                    parent_node_id=bindings[edge_spec.parent_slot],
                    child_node_id=bindings[edge_spec.child_slot],
                )
            )

        partial = SchemaBlueprint(
            blueprint_id="partial",
            nodes=tuple(nodes),
            edges=tuple(edges),
        )
        issues = validate_motif_attachment(
            partial,
            motif,
            bindings,
            existing_node_ids=existing_node_ids,
        )
        errors = tuple(
            issue
            for issue in issues
            if issue.level.value == "error"
        )
        if errors:
            details = "; ".join(
                f"{issue.code}: {issue.message}"
                for issue in errors
            )
            raise RuntimeError(
                f"Motif attachment {motif.motif_type!r} failed: {details}"
            )

    def _sample_extra_edges(
        self,
        *,
        nodes: list[BlueprintNode],
        edges: list[BlueprintEdge],
        runtime: RuntimeContext,
    ) -> None:
        if self.config.max_extra_edges == 0:
            return

        existing_pairs = {
            (edge.parent_node_id, edge.child_node_id)
            for edge in edges
        }
        structural_parents: dict[str, set[str]] = defaultdict(set)
        node_by_id = {node.node_id: node for node in nodes}
        for edge in edges:
            parent = node_by_id[edge.parent_node_id]
            child = node_by_id[edge.child_node_id]
            rule = get_role_edge_rule(parent.role, child.role)
            if rule.counts_as_structural_parent:
                structural_parents[child.node_id].add(parent.node_id)

        candidates: list[tuple[BlueprintNode, BlueprintNode]] = []
        for parent in nodes:
            for child in nodes:
                pair = (parent.node_id, child.node_id)
                if parent.rank >= child.rank or pair in existing_pairs:
                    continue
                if not is_role_edge_allowed(parent.role, child.role):
                    continue
                rule = get_role_edge_rule(parent.role, child.role)
                role_spec = get_role_spec(child.role)
                projected_count = len(structural_parents[child.node_id])
                if (
                    rule.counts_as_structural_parent
                    and parent.node_id
                    not in structural_parents[child.node_id]
                ):
                    projected_count += 1
                if (
                    role_spec.max_structural_parents is not None
                    and projected_count > role_spec.max_structural_parents
                ):
                    continue
                candidates.append((parent, child))

        rng = runtime.python_rng(
            "schema",
            "blueprint",
            "extra-edges",
        )
        rng.shuffle(candidates)
        added = 0
        for parent, child in candidates:
            if added >= self.config.max_extra_edges:
                break
            if rng.random() > self.config.extra_edge_probability:
                continue

            rule = get_role_edge_rule(parent.role, child.role)
            role_spec = get_role_spec(child.role)
            projected_count = len(structural_parents[child.node_id])
            if (
                rule.counts_as_structural_parent
                and parent.node_id
                not in structural_parents[child.node_id]
            ):
                projected_count += 1
            if (
                role_spec.max_structural_parents is not None
                and projected_count > role_spec.max_structural_parents
            ):
                continue

            edges.append(
                BlueprintEdge(
                    edge_id=f"E{len(edges):03d}",
                    parent_node_id=parent.node_id,
                    child_node_id=child.node_id,
                )
            )
            if rule.counts_as_structural_parent:
                structural_parents[child.node_id].add(parent.node_id)
            added += 1


__all__ = [
    "BlueprintSamplerConfig",
    "BlueprintSampler",
]
