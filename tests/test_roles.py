from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.schema.roles import (
    ROLE_EDGE_RULES,
    ROLE_SPECS,
    LeafPolicy,
    RoleCompatibilityError,
    RoleEdgeRule,
    RoleSpec,
    RootPolicy,
    allowed_child_roles,
    count_structural_parent_roles,
    get_role_edge_rule,
)
from rdb_prior.schema.spec import (
    Cardinality,
    IdentityDependency,
    Optionality,
    TableRole,
    TemporalMode,
)


class RoleCatalogTests(unittest.TestCase):
    def test_every_v1_role_has_exactly_one_spec(self) -> None:
        self.assertEqual(set(TableRole), set(ROLE_SPECS))
        self.assertEqual(5, len(ROLE_SPECS))

    def test_all_registered_edges_respect_endpoint_capabilities(self) -> None:
        for (parent_role, child_role), rule in ROLE_EDGE_RULES.items():
            with self.subTest(parent=parent_role, child=child_role):
                self.assertEqual(parent_role, rule.parent_role)
                self.assertEqual(child_role, rule.child_role)
                self.assertTrue(ROLE_SPECS[parent_role].can_be_referenced)
                self.assertTrue(ROLE_SPECS[child_role].can_own_foreign_keys)

    def test_lookup_edges_are_auxiliary_not_structural(self) -> None:
        rule = get_role_edge_rule(TableRole.LOOKUP, TableRole.EVENT)

        self.assertTrue(rule.auxiliary)
        self.assertFalse(rule.counts_as_structural_parent)
        self.assertEqual(
            1,
            count_structural_parent_roles(
                (TableRole.LOOKUP, TableRole.ENTITY),
                TableRole.EVENT,
            ),
        )

    def test_forbidden_role_edge_raises(self) -> None:
        with self.assertRaises(RoleCompatibilityError):
            get_role_edge_rule(TableRole.DETAIL, TableRole.ENTITY)

    def test_allowed_children_are_sorted_and_use_parent_orientation(self) -> None:
        children = allowed_child_roles(TableRole.ENTITY)

        self.assertEqual(tuple(sorted(children, key=lambda item: item.value)), children)
        self.assertIn(TableRole.EVENT, children)
        self.assertIn(TableRole.BRIDGE, children)

    def test_root_forbidden_role_requires_parent(self) -> None:
        with self.assertRaisesRegex(ValueError, "root-forbidden"):
            RoleSpec(
                role=TableRole.EVENT,
                description="invalid",
                root_policy=RootPolicy.FORBIDDEN,
                leaf_policy=LeafPolicy.ALLOWED,
                min_structural_parents=0,
                max_structural_parents=1,
                can_own_foreign_keys=True,
                can_be_referenced=True,
                temporal_capable=True,
                default_temporal_mode=TemporalMode.EVENT_TIME,
                default_feature_strategy="event",
            )

    def test_physical_fk_rule_cannot_be_many_to_many(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bridge"):
            RoleEdgeRule(
                parent_role=TableRole.ENTITY,
                child_role=TableRole.EVENT,
                relation_strategy="invalid",
                cardinality=Cardinality.MANY_TO_MANY,
                optionality=Optionality.REQUIRED,
                identity_dependency=IdentityDependency.INDEPENDENT,
            )


if __name__ == "__main__":
    unittest.main()
