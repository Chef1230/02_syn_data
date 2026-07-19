from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.schema.blueprint import (
    BlueprintEdge,
    BlueprintNode,
    MotifOccurrence,
    SchemaBlueprint,
)
from rdb_prior.schema.motifs import (
    DEFAULT_MOTIF_LIBRARY,
    ENTITY_BRIDGE_COLLIDER,
)
from rdb_prior.schema.spec import (
    AcyclicConstraint,
    AllowedRoleEdgeConstraint,
    ChildCountConstraint,
    ConnectedConstraint,
    ConstraintSeverity,
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
    TableRole,
    TemporalOrderConstraint,
    UniqueLeafConstraint,
)
from rdb_prior.schema.validation import (
    BlueprintValidationError,
    ValidationLayer,
    ValidationLevel,
    validate_blueprint,
    validate_motif_attachment,
    validate_motif_library,
    validate_motif_occurrences,
    validate_roles,
    validate_semantics,
    validate_structure,
)


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def _valid_constraints():
    return (
        ConnectedConstraint(constraint_id="connected"),
        AcyclicConstraint(constraint_id="acyclic"),
        NoParallelEdgesConstraint(constraint_id="no_parallel"),
        TableCountConstraint(
            constraint_id="table_count",
            minimum=3,
            maximum=3,
        ),
        EdgeCountConstraint(
            constraint_id="edge_count",
            minimum=2,
            maximum=2,
        ),
        EdgeDensityConstraint(
            constraint_id="density",
            minimum=0.6,
            maximum=0.7,
        ),
        RoleCountConstraint(
            constraint_id="entity_count",
            role=TableRole.ENTITY,
            minimum=1,
            maximum=1,
        ),
        AllowedRoleEdgeConstraint(
            constraint_id="allow_entity_event",
            parent_role=TableRole.ENTITY,
            child_role=TableRole.EVENT,
        ),
        AllowedRoleEdgeConstraint(
            constraint_id="allow_event_detail",
            parent_role=TableRole.EVENT,
            child_role=TableRole.DETAIL,
        ),
        ForbiddenRoleEdgeConstraint(
            constraint_id="forbid_detail_entity",
            parent_role=TableRole.DETAIL,
            child_role=TableRole.ENTITY,
        ),
        RequiredRoleEdgeConstraint(
            constraint_id="require_entity_event",
            parent_role=TableRole.ENTITY,
            child_role=TableRole.EVENT,
        ),
        ParentCountConstraint(
            constraint_id="event_parent_count",
            node_id="event",
            minimum=1,
            maximum=1,
        ),
        ChildCountConstraint(
            constraint_id="entity_child_count",
            node_id="entity",
            minimum=1,
            maximum=1,
        ),
        RankOrderConstraint(
            constraint_id="event_before_detail",
            before_node="event",
            after_node="detail",
            minimum_gap=1,
            maximum_gap=1,
        ),
        ReachabilityConstraint(
            constraint_id="entity_reaches_detail",
            source_node="entity",
            target_node="detail",
            minimum_hops=2,
            maximum_hops=2,
        ),
        TemporalOrderConstraint(
            constraint_id="event_time_before_detail",
            relation_id="event_to_detail",
            before_node="event",
            after_node="detail",
        ),
        UniqueLeafConstraint(
            constraint_id="detail_is_leaf",
            node_ids=("detail",),
        ),
    )


def _valid_blueprint(*, constraints=None) -> SchemaBlueprint:
    return SchemaBlueprint(
        blueprint_id="valid_blueprint",
        nodes=(
            BlueprintNode(
                node_id="entity",
                role=TableRole.ENTITY,
                rank=0,
            ),
            BlueprintNode(
                node_id="event",
                role=TableRole.EVENT,
                rank=1,
            ),
            BlueprintNode(
                node_id="detail",
                role=TableRole.DETAIL,
                rank=2,
            ),
        ),
        edges=(
            BlueprintEdge(
                edge_id="entity_to_event",
                parent_node_id="entity",
                child_node_id="event",
            ),
            BlueprintEdge(
                edge_id="event_to_detail",
                parent_node_id="event",
                child_node_id="detail",
            ),
        ),
        constraints=(
            _valid_constraints()
            if constraints is None
            else constraints
        ),
    )


class BlueprintModelTests(unittest.TestCase):
    def test_local_types_are_strict(self) -> None:
        with self.assertRaisesRegex(TypeError, "TableRole"):
            BlueprintNode(node_id="node", role="entity", rank=0)

        with self.assertRaisesRegex(TypeError, "integer"):
            BlueprintNode(
                node_id="node",
                role=TableRole.ENTITY,
                rank=True,
            )

        with self.assertRaisesRegex(TypeError, "tuple"):
            SchemaBlueprint(
                blueprint_id="bad",
                nodes=[],
                edges=(),
            )

    def test_edge_kind_is_implicitly_foreign_key_only(self) -> None:
        with self.assertRaisesRegex(TypeError, "kind"):
            BlueprintEdge(
                edge_id="derived",
                parent_node_id="parent",
                child_node_id="child",
                kind=EdgeKind.DERIVATION,
            )


class StructuralValidationTests(unittest.TestCase):
    def test_valid_blueprint_has_no_structural_issues(self) -> None:
        self.assertEqual((), validate_structure(_valid_blueprint()))

    def test_duplicate_ids_unknown_endpoint_and_rank_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            SchemaBlueprint(
                blueprint_id="duplicate_nodes",
                nodes=(
                    BlueprintNode(
                        node_id="entity",
                        role=TableRole.ENTITY,
                        rank=0,
                    ),
                    BlueprintNode(
                        node_id="entity",
                        role=TableRole.ENTITY,
                        rank=0,
                    ),
                ),
                edges=(),
            )

        with self.assertRaisesRegex(ValueError, "unknown child"):
            SchemaBlueprint(
                blueprint_id="unknown_endpoint",
                nodes=(
                    BlueprintNode(
                        node_id="entity",
                        role=TableRole.ENTITY,
                        rank=0,
                    ),
                ),
                edges=(
                    BlueprintEdge(
                        edge_id="missing",
                        parent_node_id="entity",
                        child_node_id="missing_node",
                    ),
                ),
            )

        blueprint = SchemaBlueprint(
            blueprint_id="bad_rank",
            nodes=(
                BlueprintNode(
                    node_id="entity",
                    role=TableRole.ENTITY,
                    rank=1,
                ),
                BlueprintNode(
                    node_id="event",
                    role=TableRole.EVENT,
                    rank=0,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="bad_rank",
                    parent_node_id="entity",
                    child_node_id="event",
                ),
            ),
        )

        codes = _codes(validate_structure(blueprint))
        self.assertIn("fk_rank_order_violation", codes)

    def test_self_loop_is_reported(self) -> None:
        blueprint = SchemaBlueprint(
            blueprint_id="self_loop",
            nodes=(
                BlueprintNode(
                    node_id="entity",
                    role=TableRole.ENTITY,
                    rank=0,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="loop",
                    parent_node_id="entity",
                    child_node_id="entity",
                ),
            ),
        )

        self.assertIn(
            "self_loop",
            _codes(validate_structure(blueprint)),
        )


class RoleValidationTests(unittest.TestCase):
    def test_valid_blueprint_satisfies_role_catalog(self) -> None:
        self.assertEqual((), validate_roles(_valid_blueprint()))

    def test_bridge_requires_two_distinct_structural_parents(self) -> None:
        blueprint = SchemaBlueprint(
            blueprint_id="one_parent_bridge",
            nodes=(
                BlueprintNode(
                    node_id="entity",
                    role=TableRole.ENTITY,
                    rank=0,
                ),
                BlueprintNode(
                    node_id="bridge",
                    role=TableRole.BRIDGE,
                    rank=1,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="entity_to_bridge",
                    parent_node_id="entity",
                    child_node_id="bridge",
                ),
            ),
        )

        self.assertIn(
            "insufficient_structural_parents",
            _codes(validate_roles(blueprint)),
        )

    def test_lookup_parent_is_auxiliary_not_structural(self) -> None:
        blueprint = SchemaBlueprint(
            blueprint_id="lookup_only_event",
            nodes=(
                BlueprintNode(
                    node_id="lookup",
                    role=TableRole.LOOKUP,
                    rank=0,
                ),
                BlueprintNode(
                    node_id="event",
                    role=TableRole.EVENT,
                    rank=1,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="lookup_to_event",
                    parent_node_id="lookup",
                    child_node_id="event",
                ),
            ),
        )

        self.assertIn(
            "insufficient_structural_parents",
            _codes(validate_roles(blueprint)),
        )

    def test_detail_role_cannot_be_referenced(self) -> None:
        blueprint = SchemaBlueprint(
            blueprint_id="detail_parent",
            nodes=(
                BlueprintNode(
                    node_id="detail",
                    role=TableRole.DETAIL,
                    rank=0,
                ),
                BlueprintNode(
                    node_id="entity",
                    role=TableRole.ENTITY,
                    rank=1,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="detail_to_entity",
                    parent_node_id="detail",
                    child_node_id="entity",
                ),
            ),
        )

        self.assertIn(
            "role_cannot_be_referenced",
            _codes(validate_roles(blueprint)),
        )


class MotifValidationTests(unittest.TestCase):
    def _collider_blueprint(self, *, bridge_rank: int = 1) -> SchemaBlueprint:
        return SchemaBlueprint(
            blueprint_id="collider",
            nodes=(
                BlueprintNode(
                    node_id="left",
                    role=TableRole.ENTITY,
                    rank=0,
                ),
                BlueprintNode(
                    node_id="right",
                    role=TableRole.ENTITY,
                    rank=0,
                ),
                BlueprintNode(
                    node_id="bridge",
                    role=TableRole.BRIDGE,
                    rank=bridge_rank,
                ),
            ),
            edges=(
                BlueprintEdge(
                    edge_id="left_to_bridge",
                    parent_node_id="left",
                    child_node_id="bridge",
                ),
                BlueprintEdge(
                    edge_id="right_to_bridge",
                    parent_node_id="right",
                    child_node_id="bridge",
                ),
            ),
        )

    def test_default_motif_library_is_valid(self) -> None:
        self.assertEqual(
            (),
            validate_motif_library(DEFAULT_MOTIF_LIBRARY),
        )

    def test_valid_attachment_can_be_recorded_as_provenance(self) -> None:
        issues = validate_motif_attachment(
            self._collider_blueprint(),
            ENTITY_BRIDGE_COLLIDER,
            {
                "left_entity": "left",
                "right_entity": "right",
                "bridge": "bridge",
            },
            existing_node_ids=("left",),
        )

        self.assertEqual((), issues)

    def test_occurrence_rejects_incompatible_edge_binding(self) -> None:
        blueprint = self._collider_blueprint()
        blueprint = SchemaBlueprint(
            blueprint_id=blueprint.blueprint_id,
            nodes=blueprint.nodes,
            edges=blueprint.edges,
            motif_occurrences=(
                MotifOccurrence(
                    occurrence_id="M000",
                    motif_type="entity_bridge_collider",
                    node_bindings=(
                        ("left_entity", "left"),
                        ("right_entity", "right"),
                        ("bridge", "bridge"),
                    ),
                    edge_bindings=(
                        ("left_entity_to_bridge", "right_to_bridge"),
                        ("right_entity_to_bridge", "left_to_bridge"),
                    ),
                ),
            ),
        )

        self.assertIn(
            "invalid_occurrence_edge_binding",
            _codes(validate_motif_occurrences(blueprint)),
        )

    def test_attachment_rejects_slot_merge(self) -> None:
        issues = validate_motif_attachment(
            self._collider_blueprint(),
            ENTITY_BRIDGE_COLLIDER,
            {
                "left_entity": "left",
                "right_entity": "left",
                "bridge": "bridge",
            },
        )

        self.assertIn("invalid_motif_binding", _codes(issues))

    def test_attachment_detects_rank_mismatch(self) -> None:
        issues = validate_motif_attachment(
            self._collider_blueprint(bridge_rank=2),
            ENTITY_BRIDGE_COLLIDER,
            {
                "left_entity": "left",
                "right_entity": "right",
                "bridge": "bridge",
            },
        )

        self.assertIn("motif_rank_mismatch", _codes(issues))

    def test_attachment_detects_missing_motif_edge(self) -> None:
        blueprint = self._collider_blueprint()
        blueprint = SchemaBlueprint(
            blueprint_id=blueprint.blueprint_id,
            nodes=blueprint.nodes,
            edges=blueprint.edges[:1],
        )
        issues = validate_motif_attachment(
            blueprint,
            ENTITY_BRIDGE_COLLIDER,
            {
                "left_entity": "left",
                "right_entity": "right",
                "bridge": "bridge",
            },
        )

        self.assertIn("missing_motif_edge", _codes(issues))


class SemanticValidationTests(unittest.TestCase):
    def test_all_blueprint_constraint_types_are_evaluated(self) -> None:
        self.assertEqual((), validate_semantics(_valid_blueprint()))

    def test_allowed_role_edges_form_a_blueprint_allowlist(self) -> None:
        constraints = (
            AllowedRoleEdgeConstraint(
                constraint_id="only_entity_event",
                parent_role=TableRole.ENTITY,
                child_role=TableRole.EVENT,
            ),
        )

        self.assertIn(
            "allowed_role_edge_violation",
            _codes(
                validate_semantics(
                    _valid_blueprint(constraints=constraints)
                )
            ),
        )

    def test_unknown_constraint_reference_is_an_error(self) -> None:
        constraints = (
            ParentCountConstraint(
                constraint_id="unknown_parent_target",
                node_id="missing",
                minimum=1,
            ),
        )

        self.assertIn(
            "constraint_unknown_node",
            _codes(
                validate_semantics(
                    _valid_blueprint(constraints=constraints)
                )
            ),
        )

    def test_non_schema_edge_kind_constraint_is_rejected(self) -> None:
        constraints = (
            ReachabilityConstraint(
                constraint_id="process_reachability",
                source_node="entity",
                target_node="detail",
                edge_kinds=(EdgeKind.DERIVATION,),
            ),
        )

        self.assertIn(
            "non_structural_constraint_edge_kind",
            _codes(
                validate_semantics(
                    _valid_blueprint(constraints=constraints)
                )
            ),
        )

    def test_soft_constraint_violation_is_only_a_warning(self) -> None:
        constraints = (
            TableCountConstraint(
                constraint_id="prefer_four_tables",
                severity=ConstraintSeverity.SOFT,
                minimum=4,
            ),
        )
        report = validate_blueprint(
            _valid_blueprint(constraints=constraints)
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(1, len(report.warnings))
        self.assertIs(ValidationLevel.WARNING, report.warnings[0].level)

    def test_temporal_constraint_requires_temporal_endpoints(self) -> None:
        constraints = (
            TemporalOrderConstraint(
                constraint_id="entity_before_event",
                relation_id="entity_to_event",
                before_node="entity",
                after_node="event",
            ),
        )

        self.assertIn(
            "temporal_constraint_on_non_temporal_node",
            _codes(
                validate_semantics(
                    _valid_blueprint(constraints=constraints)
                )
            ),
        )


class AggregateValidationTests(unittest.TestCase):
    def test_valid_blueprint_passes_all_layers(self) -> None:
        report = validate_blueprint(_valid_blueprint())

        self.assertTrue(report.is_valid)
        self.assertEqual((), report.issues)
        self.assertEqual(
            (),
            report.for_layer(ValidationLayer.STRUCTURE),
        )

    def test_raise_on_error_contains_full_report(self) -> None:
        blueprint = SchemaBlueprint(
            blueprint_id="invalid",
            nodes=(
                BlueprintNode(
                    node_id="event",
                    role=TableRole.EVENT,
                    rank=0,
                ),
            ),
            edges=(),
        )

        with self.assertRaises(BlueprintValidationError) as captured:
            validate_blueprint(blueprint, raise_on_error=True)

        self.assertFalse(captured.exception.report.is_valid)
        self.assertIn(
            "insufficient_structural_parents",
            _codes(captured.exception.report.errors),
        )


if __name__ == "__main__":
    unittest.main()
