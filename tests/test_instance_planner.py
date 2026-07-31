from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.instance.plan import (
    FeatureSCMFamily,
    InstancePlan,
    RootCauseFamily,
    TemporalFamily,
)
from rdb_prior.instance.planner import (
    InstancePlanner,
    InstancePlannerConfig,
    RoleSCMPrior,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.validation.checks import validate_instance_plan


class InstancePlannerTests(unittest.TestCase):
    def test_default_scm_prior_is_signal_sparse(self) -> None:
        weights = dict(InstancePlannerConfig().scm_weights)
        self.assertEqual(0.30, weights[FeatureSCMFamily.EXOGENOUS])
        self.assertEqual(0.40, weights[FeatureSCMFamily.LINEAR])
        self.assertEqual(0.20, weights[FeatureSCMFamily.CAM])
        self.assertEqual(0.10, weights[FeatureSCMFamily.MLP])
        self.assertAlmostEqual(1.0, sum(weights.values()))

    def test_role_scm_override_controls_families_and_scales(self) -> None:
        runtime = RuntimeContext(92).for_sample("role_scm_override")
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=6, max_tables=6)
        ).sample("role_scm_override", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            "role_scm_override",
            runtime,
        )
        common = {
            "entity_rows_min": 24,
            "entity_rows_max": 32,
            "lookup_rows_min": 4,
            "lookup_rows_max": 8,
            "max_rows_per_table": 96,
        }
        baseline = InstancePlanner(InstancePlannerConfig(**common)).plan(
            sample_id="role_scm_override",
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        event_prior = RoleSCMPrior(
            scm_weights=((FeatureSCMFamily.MLP, 1.0),),
            root_cause_weights=((RootCauseFamily.NONLINEAR, 1.0),),
            signal_scale_multiplier=2.0,
            noise_scale_multiplier=3.0,
        )
        conditioned = InstancePlanner(
            InstancePlannerConfig(
                **common,
                role_scm=((TableRole.EVENT, event_prior),),
            )
        ).plan(
            sample_id="role_scm_override",
            schema=schema,
            runtime=runtime.child("database-instance"),
        )

        baseline_tables = {table.table_id: table for table in baseline.tables}
        events = [
            table
            for table in conditioned.tables
            if table.role is TableRole.EVENT
        ]
        self.assertTrue(events)
        for table in events:
            original = baseline_tables[table.table_id]
            self.assertIs(FeatureSCMFamily.MLP, table.feature_family)
            self.assertIs(
                RootCauseFamily.NONLINEAR,
                table.root_cause_family,
            )
            self.assertAlmostEqual(
                2.0 * original.parameter_map["signal_scale"],
                table.parameter_map["signal_scale"],
            )
            self.assertAlmostEqual(
                3.0 * original.parameter_map["noise_scale"],
                table.parameter_map["noise_scale"],
            )

        lookup_prior = InstancePlannerConfig().scm_prior_for_role(
            TableRole.LOOKUP
        )
        self.assertEqual(
            ((FeatureSCMFamily.EXOGENOUS, 1.0),),
            lookup_prior.scm_weights,
        )
        self.assertEqual(
            ((RootCauseFamily.STANDARD_NORMAL, 1.0),),
            lookup_prior.root_cause_weights,
        )

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
        self.assertIn("scm_signal_mean", first.parameter_map)
        self.assertIn("scm_noise_mean", first.parameter_map)
        self.assertIn("scm_long_tail_enabled", first.parameter_map)
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
                        FeatureSCMFamily.EXOGENOUS,
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
            self.assertGreater(table_plan.parameter_map["signal_scale"], 0)
            self.assertGreater(table_plan.parameter_map["noise_scale"], 0)
            self.assertGreater(table_plan.parameter_map["activation_scale"], 0)
            self.assertGreater(table_plan.parameter_map["output_scale"], 0)
            self.assertGreater(table_plan.parameter_map["long_tail_alpha"], 1)

    def test_meta_prior_varies_across_databases_and_prefers_low_noise(self) -> None:
        noise_means: list[float] = []
        signal_means: list[float] = []
        long_tail_values: set[float] = set()
        for suffix in range(80):
            _schema, plan = self._plan(f"meta_prior_{suffix}")
            noise_means.append(plan.parameter_map["scm_noise_mean"])
            signal_means.append(plan.parameter_map["scm_signal_mean"])
            long_tail_values.add(
                plan.parameter_map["scm_long_tail_enabled"]
            )

        self.assertLess(float(np.median(noise_means)), 0.01)
        self.assertGreater(max(noise_means) / min(noise_means), 100.0)
        self.assertGreater(max(signal_means) / min(signal_means), 100.0)
        self.assertEqual({0.0, 1.0}, long_tail_values)

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

    def test_realistic_prior_is_bounded_and_zero_inflated(self) -> None:
        runtime = RuntimeContext(121).for_sample("realistic_prior")
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=8, max_tables=8)
        ).sample("realistic_prior", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            "realistic_prior",
            runtime,
        )
        config = InstancePlannerConfig(
            entity_rows_min=100,
            entity_rows_max=20000,
            entity_rows_distribution="lognormal",
            entity_rows_median=1500,
            entity_rows_log_sigma=1.15,
            lookup_rows_min=3,
            lookup_rows_max=200,
            lookup_rows_distribution="loguniform",
            population_scale_distribution="lognormal",
            population_scale_log_sigma=1.0,
            max_rows_per_table=50000,
            feature_missing_rate_min=0.01,
            feature_missing_rate_max=0.80,
            feature_missing_zero_probability=1.0,
            feature_missing_beta_alpha=1.2,
            feature_missing_beta_beta=4.0,
            feature_noise_scale_min=0.001,
            feature_noise_scale_max=0.5,
            categorical_cardinality_min=2,
            categorical_cardinality_max=30,
            categorical_high_cardinality_probability=1.0,
            categorical_high_cardinality_min=31,
            categorical_high_cardinality_max=2000,
        )
        plan = InstancePlanner(config).plan(
            sample_id="realistic_prior",
            schema=schema,
            runtime=runtime.child("database-instance"),
        )

        self.assertTrue(validate_instance_plan(schema, plan).is_valid)
        self.assertTrue(
            all(
                table.population.row_count <= config.max_rows_per_table
                for table in plan.tables
            )
        )
        for table in plan.tables:
            self.assertEqual(0.0, table.parameter_map["missing_rate"])
            self.assertGreaterEqual(
                table.parameter_map["categorical_cardinality"],
                config.categorical_high_cardinality_min,
            )
            self.assertLessEqual(
                table.parameter_map["categorical_cardinality"],
                config.categorical_high_cardinality_max,
            )


if __name__ == "__main__":
    unittest.main()
