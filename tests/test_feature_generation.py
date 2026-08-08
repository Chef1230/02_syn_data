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
from rdb_prior.compilation.model import ColumnKind, PhysicalColumn, PhysicalDataType
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.validation.checks import validate_database_instance


class FeatureGenerationTests(unittest.TestCase):
    def _generate(self, sample_id: str):
        runtime = RuntimeContext(121).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(
                min_tables=5,
                max_tables=5,
                motif_weights=(("event_reference_chain", 1.0),),
            )
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=24,
                entity_rows_max=28,
                lookup_rows_min=4,
                lookup_rows_max=6,
                max_rows_per_table=80,
            )
        ).plan(
            sample_id=sample_id,
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        database = DatabaseGenerator().generate(schema=schema, plan=plan)
        return schema, plan, database

    def test_database_generation_is_deterministic_and_valid(self) -> None:
        schema, plan, first = self._generate("features")
        _schema, _plan, second = self._generate("features")

        self.assertTrue(validate_database_instance(schema, plan, first).is_valid)
        for first_table in first.tables:
            second_table = second.table(first_table.table_id)
            for column_id, values in first_table.columns.items():
                np.testing.assert_equal(values, second_table.column(column_id))

    def test_columns_use_persistable_role_appropriate_dtypes(self) -> None:
        schema, _plan, database = self._generate("dtypes")
        for table in schema.tables:
            data = database.table(table.table_id)
            for column in table.columns:
                values = data.column(column.column_id)
                self.assertNotEqual("O", values.dtype.kind)
                if column.data_type is PhysicalDataType.TEXT:
                    self.assertIn(values.dtype.kind, {"U", "S"})
                if column.kind in {
                    ColumnKind.PRIMARY_KEY,
                    ColumnKind.FOREIGN_KEY,
                    ColumnKind.TIME,
                }:
                    self.assertIn(values.dtype.kind, {"i", "u"})

    def test_missingness_never_removes_all_observed_values(self) -> None:
        from rdb_prior.generation.features import _apply_missing

        rng = np.random.default_rng(3)
        for data_type, values in (
            (PhysicalDataType.DOUBLE, np.array([1.0, 2.0, 3.0])),
            (PhysicalDataType.TEXT, np.array(["a", "b", "c"])),
        ):
            column = PhysicalColumn(
                column_id="c",
                name="c",
                data_type=data_type,
                kind=ColumnKind.FEATURE,
                ordinal=0,
                nullable=True,
            )
            masked = _apply_missing(values, column, rng, 1.0)
            if data_type is PhysicalDataType.DOUBLE:
                observed = masked[~np.isnan(masked)]
            else:
                observed = masked[masked != ""]
            self.assertEqual(
                1,
                len(observed),
                "a fully-masked column must keep at least one observed value",
            )

    def test_event_to_event_time_is_strictly_lagged(self) -> None:
        schema, _plan, database = self._generate("time_lag")
        for foreign_key in schema.foreign_keys:
            parent = schema.table(foreign_key.parent_table_id)
            child = schema.table(foreign_key.child_table_id)
            if parent.role is not TableRole.EVENT or child.role is not TableRole.EVENT:
                continue
            parent_time = next(
                column for column in parent.columns if column.kind is ColumnKind.TIME
            )
            child_time = next(
                column for column in child.columns if column.kind is ColumnKind.TIME
            )
            assignments = database.table(child.table_id).column(
                foreign_key.child_column_id
            )
            valid = assignments >= 0
            self.assertTrue(
                np.all(
                    database.table(child.table_id).column(child_time.column_id)[valid]
                    > database.table(parent.table_id).column(parent_time.column_id)[
                        assignments[valid]
                    ]
                )
            )
            return
        self.fail("event_reference_chain did not produce an Event -> Event FK")

    def test_all_time_columns_stay_within_calendar_interval(self) -> None:
        schema, plan, database = self._generate("bounded_time")
        for table in schema.tables:
            data = database.table(table.table_id)
            for column in table.columns:
                if column.kind is not ColumnKind.TIME:
                    continue
                values = data.column(column.column_id)
                self.assertTrue(
                    np.all(values >= plan.calendar_start_seconds),
                    f"{table.table_id}.{column.column_id} below calendar start",
                )
                self.assertTrue(
                    np.all(values <= plan.calendar_end_seconds),
                    f"{table.table_id}.{column.column_id} above calendar end",
                )
        self.assertTrue(
            validate_database_instance(schema, plan, database).is_valid
        )

    def test_event_span_is_bounded_by_calendar(self) -> None:
        schema, plan, database = self._generate("bounded_span")
        calendar_span = plan.calendar_end_seconds - plan.calendar_start_seconds
        for table in schema.tables:
            data = database.table(table.table_id)
            for column in table.columns:
                if column.kind is not ColumnKind.TIME:
                    continue
                values = data.column(column.column_id)
                self.assertLessEqual(
                    int(values.max()) - int(values.min()),
                    calendar_span,
                )

    def test_time_out_of_calendar_is_reported(self) -> None:
        schema, plan, database = self._generate("time_out_calendar")
        corrupted = False
        for table in schema.tables:
            data = database.table(table.table_id)
            for column in table.columns:
                if column.kind is not ColumnKind.TIME:
                    continue
                values = data.column(column.column_id)
                values[0] = plan.calendar_end_seconds + 1
                corrupted = True
                break
            if corrupted:
                break
        self.assertTrue(corrupted)
        report = validate_database_instance(schema, plan, database)
        codes = {issue.code for issue in report.issues}
        self.assertIn("time_out_of_calendar", codes)

    def test_burst_and_churn_mechanisms_produce_varied_shapes(self) -> None:
        from rdb_prior.generation.features import _mechanism_ticks
        from rdb_prior.instance.plan import (
            EventTemporalMechanism,
            FeatureSCMFamily,
            PopulationPlan,
            TableMechanismPlan,
            TemporalFamily,
        )
        from rdb_prior.schema.spec import TableRole

        params = (
            ("time_scale_seconds", 3600.0),
            ("burst_max_clusters", 3.0),
            ("burst_cluster_width_min", 0.01),
            ("burst_cluster_width_max", 0.12),
            ("churn_exponent_min", 1.2),
            ("churn_exponent_max", 3.0),
            ("seasonal_period_days_min", 30.0),
            ("seasonal_period_days_max", 365.0),
        )

        def plan_for(mechanism: EventTemporalMechanism) -> TableMechanismPlan:
            return TableMechanismPlan(
                table_id="t",
                role=TableRole.EVENT,
                population=PopulationPlan(strategy="s", row_count=1000),
                latent_dimension=4,
                feature_family=FeatureSCMFamily.EXOGENOUS,
                temporal_family=TemporalFamily.PARENT_BURST,
                event_mechanism=mechanism,
                latent_seed=1,
                feature_seed=2,
                temporal_seed=3,
                parameters=params,
            )

        def ticks(mechanism: EventTemporalMechanism) -> np.ndarray:
            rng = np.random.Generator(np.random.PCG64DXSM(11))
            return _mechanism_ticks(
                rng, mechanism, plan_for(mechanism), 2000, 1_000_000
            )

        stationary = ticks(EventTemporalMechanism.STATIONARY)
        burst = ticks(EventTemporalMechanism.BURST)
        churn = ticks(EventTemporalMechanism.CHURN)
        seasonal = ticks(EventTemporalMechanism.SEASONAL)
        for label, values in (
            ("stationary", stationary),
            ("burst", burst),
            ("churn", churn),
            ("seasonal", seasonal),
        ):
            self.assertTrue(
                np.all(values >= 0) and np.all(values <= 1_000_000),
                f"{label} not bounded in [0, span]",
            )
        # Churn concentrates mass at the window start: median and 90th
        # percentile sit below the stationary values for any exponent > 1.
        self.assertLess(np.median(churn), np.median(stationary))
        self.assertLess(np.percentile(churn, 90), np.percentile(stationary, 90))
        # Bursts create long silences: the largest inter-arrival gap clearly
        # exceeds the stationary max gap.
        burst_gaps = np.diff(burst)
        stationary_gaps = np.diff(stationary)
        self.assertGreater(
            float(burst_gaps.max()),
            2.0 * float(stationary_gaps.max()),
        )


if __name__ == "__main__":
    unittest.main()
