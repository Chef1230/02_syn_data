from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.schema.spec import (
    AcyclicConstraint,
    AllowedRoleEdgeConstraint,
    ChildCountConstraint,
    ConnectedConstraint,
    ConstraintBase,
    ConstraintKind,
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
    TableRole,
    TemporalOrderConstraint,
    UniqueLeafConstraint,
    constraint_from_dict,
    constraint_to_dict,
)


class SchemaSpecTests(unittest.TestCase):
    def _samples(self):
        return (
            ConnectedConstraint(constraint_id="connected"),
            AcyclicConstraint(constraint_id="acyclic"),
            NoParallelEdgesConstraint(constraint_id="parallel"),
            TableCountConstraint(
                constraint_id="tables",
                minimum=3,
                maximum=12,
            ),
            EdgeCountConstraint(constraint_id="edges", minimum=2),
            EdgeDensityConstraint(
                constraint_id="density",
                minimum=0.5,
                maximum=1.5,
            ),
            RoleCountConstraint(
                constraint_id="entities",
                role=TableRole.ENTITY,
                minimum=1,
            ),
            AllowedRoleEdgeConstraint(
                constraint_id="allow_entity_event",
                parent_role=TableRole.ENTITY,
                child_role=TableRole.EVENT,
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
                minimum=1,
            ),
            ParentCountConstraint(
                constraint_id="bridge_parents",
                node_id="bridge_0",
                minimum=2,
            ),
            ChildCountConstraint(
                constraint_id="entity_children",
                node_id="entity_0",
                maximum=4,
            ),
            RankOrderConstraint(
                constraint_id="rank",
                before_node="entity_0",
                after_node="event_0",
            ),
            ReachabilityConstraint(
                constraint_id="reach",
                source_node="entity_0",
                target_node="detail_0",
                edge_kinds=(EdgeKind.FOREIGN_KEY, EdgeKind.DERIVATION),
            ),
            TemporalOrderConstraint(
                constraint_id="temporal",
                relation_id="event_detail_fk",
                before_node="event_0",
                after_node="detail_0",
            ),
            UniqueLeafConstraint(
                constraint_id="unique_leaf",
                node_ids=("detail_0",),
                roles=(TableRole.DETAIL,),
            ),
        )

    def test_v1_role_catalog_is_exactly_five_roles(self) -> None:
        self.assertEqual(
            {"entity", "event", "lookup", "bridge", "detail"},
            {role.value for role in TableRole},
        )

    def test_all_concrete_constraints_construct(self) -> None:
        samples = self._samples()

        self.assertEqual(set(ConstraintKind), {item.kind for item in samples})

    def test_constraint_base_is_abstract(self) -> None:
        with self.assertRaisesRegex(TypeError, "abstract"):
            ConstraintBase(constraint_id="base")

    def test_constraint_round_trip(self) -> None:
        for constraint in self._samples():
            with self.subTest(kind=constraint.kind):
                encoded = constraint_to_dict(constraint)
                json_payload = json.loads(json.dumps(encoded))
                decoded = constraint_from_dict(json_payload)
                self.assertEqual(constraint, decoded)

    def test_unknown_constraint_kind_and_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown constraint kind"):
            constraint_from_dict({"kind": "future_kind"})

        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            constraint_from_dict(
                {
                    "kind": "connected",
                    "constraint_id": "connected",
                    "unexpected": 1,
                }
            )

    def test_schema_constraints_do_not_persist_motif_metadata(self) -> None:
        self.assertNotIn(
            "motif_count",
            {kind.value for kind in ConstraintKind},
        )
        self.assertNotIn(
            "motif_membership",
            {kind.value for kind in ConstraintKind},
        )

        with self.assertRaisesRegex(ValueError, "Unknown constraint kind"):
            constraint_from_dict(
                {
                    "kind": "motif_membership",
                    "constraint_id": "legacy_motif_record",
                    "motif_type": "entity_event",
                    "node_ids": ["entity_0", "event_0"],
                }
            )

    def test_raw_string_enums_are_rejected_by_core_models(self) -> None:
        with self.assertRaisesRegex(TypeError, "parent_role"):
            AllowedRoleEdgeConstraint(
                constraint_id="bad",
                parent_role="entity",  # type: ignore[arg-type]
                child_role=TableRole.EVENT,
            )

        with self.assertRaisesRegex(TypeError, "edge_kind"):
            NoParallelEdgesConstraint(
                constraint_id="bad",
                edge_kind="foreign_key",  # type: ignore[arg-type]
            )

    def test_count_bounds_require_non_boolean_integers(self) -> None:
        for value in (True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    TableCountConstraint(
                        constraint_id="bad_count",
                        minimum=value,  # type: ignore[arg-type]
                    )

    def test_density_is_finite_and_definition_is_explicit(self) -> None:
        for value in (math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EdgeDensityConstraint(
                        constraint_id="bad_density",
                        minimum=value,
                    )

        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            EdgeDensityConstraint(
                constraint_id="bad_simple_density",
                minimum=1.1,
                definition=DensityDefinition.SIMPLE_DIRECTED,
            )

    def test_edge_kind_sets_must_be_nonempty_and_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            AcyclicConstraint(constraint_id="empty", edge_kinds=())

        with self.assertRaisesRegex(ValueError, "duplicates"):
            ReachabilityConstraint(
                constraint_id="duplicate",
                source_node="a",
                target_node="b",
                edge_kinds=(EdgeKind.FOREIGN_KEY, EdgeKind.FOREIGN_KEY),
            )

    def test_ids_must_be_nonempty_and_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ParentCountConstraint(
                constraint_id="bad_node",
                node_id="   ",
                minimum=1,
            )

        with self.assertRaisesRegex(ValueError, "duplicates"):
            UniqueLeafConstraint(
                constraint_id="duplicates",
                node_ids=("a", "a"),
            )

    def test_rank_order_requires_positive_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            RankOrderConstraint(
                constraint_id="bad_rank",
                before_node="a",
                after_node="b",
                minimum_gap=0,
            )

    def test_temporal_order_is_database_stage_and_relation_bound(self) -> None:
        constraint = TemporalOrderConstraint(
            constraint_id="temporal",
            relation_id="event_detail_fk",
            before_node="event",
            after_node="detail",
        )

        self.assertIs(ConstraintStage.DATABASE, constraint.stage)

        with self.assertRaisesRegex(ValueError, "relation_id"):
            TemporalOrderConstraint(
                constraint_id="bad_temporal",
                relation_id=" ",
                before_node="event",
                after_node="detail",
            )


if __name__ == "__main__":
    unittest.main()
