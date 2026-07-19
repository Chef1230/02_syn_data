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
from rdb_prior.compilation.model import ColumnKind, PhysicalDataType
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


if __name__ == "__main__":
    unittest.main()
