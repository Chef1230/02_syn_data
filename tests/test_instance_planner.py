from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.instance.plan import (
    FeatureSCMFamily,
    InstancePlan,
    TemporalFamily,
)
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.validation.checks import validate_instance_plan


class InstancePlannerTests(unittest.TestCase):
    def _plan(self, sample_id: str = "instance_plan"):
        runtime = RuntimeContext(91).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=6, max_tables=6)
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
        planner = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=24,
                entity_rows_max=32,
                lookup_rows_min=4,
                lookup_rows_max=8,
                max_rows_per_table=96,
            )
        )
        return schema, planner.plan(
            sample_id=sample_id,
            schema=schema,
            runtime=runtime.child("database-instance"),
        )

    def test_plan_is_deterministic_valid_and_round_trips(self) -> None:
        schema, first = self._plan()
        _schema, second = self._plan()

        self.assertEqual(first, second)
        self.assertEqual(first, InstancePlan.from_dict(first.to_dict()))
        self.assertTrue(validate_instance_plan(schema, first).is_valid)
        self.assertEqual(
            {foreign_key.foreign_key_id for foreign_key in schema.foreign_keys},
            {
                fk_id
                for relation in first.relations
                for fk_id in relation.foreign_key_ids
            },
        )

    def test_role_mechanisms_and_root_constraints(self) -> None:
        schema, plan = self._plan("role_mechanisms")
        for table_plan in plan.tables:
            physical = schema.table(table_plan.table_id)
            incoming = [
                foreign_key
                for foreign_key in schema.foreign_keys
                if foreign_key.child_table_id == physical.table_id
                and foreign_key.relation_strategy != "lookup_assignment"
            ]
            if not incoming:
                self.assertIn(physical.role, {TableRole.ENTITY, TableRole.LOOKUP})
            if physical.role is TableRole.LOOKUP:
                self.assertIs(FeatureSCMFamily.EXOGENOUS, table_plan.feature_family)
            else:
                self.assertIn(
                    table_plan.feature_family,
                    {
                        FeatureSCMFamily.LINEAR,
                        FeatureSCMFamily.CAM,
                        FeatureSCMFamily.MLP,
                    },
                )
            expected_time = (
                TemporalFamily.NONE
                if physical.role is not TableRole.EVENT
                else table_plan.temporal_family
            )
            self.assertIs(expected_time, table_plan.temporal_family)
            self.assertIn("missing_rate", table_plan.parameter_map)

    def test_bridge_structural_fks_share_one_joint_plan(self) -> None:
        for suffix in range(30):
            schema, plan = self._plan(f"bridge_{suffix}")
            bridge_groups = [
                relation
                for relation in plan.relations
                if relation.family == "affinity_bridge"
            ]
            if not bridge_groups:
                continue
            for relation in bridge_groups:
                self.assertGreaterEqual(len(relation.foreign_key_ids), 2)
                self.assertIs(
                    TableRole.BRIDGE,
                    schema.table(relation.child_table_id).role,
                )
            return
        self.fail("sampler did not produce a bridge schema")


if __name__ == "__main__":
    unittest.main()
