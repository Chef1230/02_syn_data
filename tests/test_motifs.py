from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.schema.motifs import (
    DEFAULT_MOTIF_LIBRARY,
    ENTITY_BRIDGE_COLLIDER,
    ENTITY_EVENT,
    EVENT_REFERENCE_CHAIN,
    LOOKUP_ASSIGNMENT,
    MotifEdgeSpec,
    MotifIssueLevel,
    MotifLibrary,
    MotifNodeSpec,
    MotifSpec,
    resolve_motif_global_ranks,
    validate_motif_node_bindings,
    validate_motif_spec,
)
from rdb_prior.schema.spec import EdgeKind, TableRole


def _issue_codes(motif: MotifSpec) -> set[str]:
    return {
        issue.code
        for issue in validate_motif_spec(motif)
        if issue.level is MotifIssueLevel.ERROR
    }


class MotifDefinitionTests(unittest.TestCase):
    def test_default_library_is_structurally_valid(self) -> None:
        self.assertEqual(6, len(DEFAULT_MOTIF_LIBRARY))

        for motif in DEFAULT_MOTIF_LIBRARY:
            with self.subTest(motif=motif.motif_type):
                self.assertEqual((), validate_motif_spec(motif))

    def test_motif_identity_has_no_version_field(self) -> None:
        self.assertNotIn("version", {field.name for field in fields(MotifSpec)})

        with self.assertRaisesRegex(TypeError, "version"):
            MotifSpec(
                motif_type="entity_event",
                version=1,
                anchor_slot="entity",
                nodes=ENTITY_EVENT.nodes,
                edges=ENTITY_EVENT.edges,
            )

    def test_library_rejects_duplicate_motif_type(self) -> None:
        duplicate = MotifSpec(
            motif_type=ENTITY_EVENT.motif_type,
            anchor_slot=ENTITY_EVENT.anchor_slot,
            nodes=ENTITY_EVENT.nodes,
            edges=ENTITY_EVENT.edges,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate motif type"):
            MotifLibrary((ENTITY_EVENT, duplicate))

    def test_temporal_sequence_is_not_a_schema_motif(self) -> None:
        self.assertFalse(DEFAULT_MOTIF_LIBRARY.contains("event_sequence"))
        self.assertTrue(
            DEFAULT_MOTIF_LIBRARY.contains("event_reference_chain")
        )
        self.assertEqual(
            "event_reference_chain",
            EVENT_REFERENCE_CHAIN.motif_type,
        )


class MotifLocalInvariantTests(unittest.TestCase):
    def test_raw_string_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "TableRole"):
            MotifNodeSpec(
                slot="entity",
                roles=("entity",),
                rank_offset=0,
            )

    def test_roles_must_be_an_immutable_tuple(self) -> None:
        with self.assertRaisesRegex(TypeError, "tuple"):
            MotifNodeSpec(
                slot="entity",
                roles=[TableRole.ENTITY],
                rank_offset=0,
            )

    def test_rank_offset_rejects_bool_float_and_negative(self) -> None:
        for invalid in (True, 0.5, -1):
            with self.subTest(value=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    MotifNodeSpec(
                        slot="entity",
                        roles=(TableRole.ENTITY,),
                        rank_offset=invalid,
                    )

    def test_fk_rank_gap_must_be_a_positive_integer(self) -> None:
        for invalid in (False, 0, 1.5, -1):
            with self.subTest(value=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    MotifEdgeSpec(
                        edge="entity_to_event",
                        parent_slot="entity",
                        child_slot="event",
                        minimum_rank_gap=invalid,
                    )

    def test_schema_motif_edge_cannot_declare_process_kind(self) -> None:
        with self.assertRaisesRegex(TypeError, "kind"):
            MotifEdgeSpec(
                edge="derived",
                parent_slot="entity",
                child_slot="event",
                kind=EdgeKind.DERIVATION,
            )

    def test_assignment_limit_rejects_bool_and_non_positive_values(self) -> None:
        for invalid in (True, 0, -1):
            with self.subTest(value=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    validate_motif_spec(
                        ENTITY_EVENT,
                        max_role_assignments=invalid,
                    )


class MotifWholeNodeConstraintTests(unittest.TestCase):
    def test_non_anchor_bridge_requires_two_structural_parents(self) -> None:
        motif = MotifSpec(
            motif_type="one_parent_bridge",
            anchor_slot="entity",
            nodes=(
                MotifNodeSpec(
                    slot="entity",
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
                    edge="entity_to_bridge",
                    parent_slot="entity",
                    child_slot="bridge",
                ),
            ),
        )

        self.assertIn(
            "insufficient_structural_parents",
            _issue_codes(motif),
        )

    def test_auxiliary_lookup_parent_does_not_complete_event(self) -> None:
        motif = MotifSpec(
            motif_type="lookup_only_event",
            anchor_slot="lookup",
            nodes=(
                MotifNodeSpec(
                    slot="lookup",
                    roles=(TableRole.LOOKUP,),
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
                    edge="lookup_to_event",
                    parent_slot="lookup",
                    child_slot="event",
                ),
            ),
        )

        self.assertIn(
            "insufficient_structural_parents",
            _issue_codes(motif),
        )

    def test_anchor_may_receive_missing_parents_from_existing_schema(self) -> None:
        motif = MotifSpec(
            motif_type="attach_lookup_to_existing_bridge",
            anchor_slot="bridge",
            nodes=(
                MotifNodeSpec(
                    slot="lookup",
                    roles=(TableRole.LOOKUP,),
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
                    edge="lookup_to_bridge",
                    parent_slot="lookup",
                    child_slot="bridge",
                ),
            ),
        )

        self.assertEqual((), validate_motif_spec(motif))

    def test_bridge_collider_satisfies_whole_node_constraint(self) -> None:
        self.assertEqual((), validate_motif_spec(ENTITY_BRIDGE_COLLIDER))


class MotifInstantiationContractTests(unittest.TestCase):
    def test_global_rank_formula_uses_anchor_offset(self) -> None:
        self.assertEqual(
            {"lookup": 4, "target": 5},
            dict(
                resolve_motif_global_ranks(
                    LOOKUP_ASSIGNMENT,
                    anchor_global_rank=5,
                )
            ),
        )

    def test_attachment_rejects_negative_global_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative global rank"):
            resolve_motif_global_ranks(
                LOOKUP_ASSIGNMENT,
                anchor_global_rank=0,
            )

    def test_all_slots_must_bind_distinct_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_motif_node_bindings(
                ENTITY_BRIDGE_COLLIDER,
                {
                    "left_entity": "entity_0",
                    "right_entity": "entity_0",
                    "bridge": "bridge_0",
                },
            )

    def test_only_anchor_may_reuse_an_existing_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only anchor_slot"):
            validate_motif_node_bindings(
                ENTITY_BRIDGE_COLLIDER,
                {
                    "left_entity": "entity_0",
                    "right_entity": "entity_1",
                    "bridge": "bridge_0",
                },
                existing_node_ids=("entity_0", "entity_1"),
            )

    def test_valid_binding_may_reuse_anchor_only(self) -> None:
        validate_motif_node_bindings(
            ENTITY_BRIDGE_COLLIDER,
            {
                "left_entity": "entity_0",
                "right_entity": "entity_1",
                "bridge": "bridge_0",
            },
            existing_node_ids=("entity_0",),
        )


if __name__ == "__main__":
    unittest.main()
