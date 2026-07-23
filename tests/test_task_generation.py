from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.artifacts import load_instance_artifact
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.task.artifacts import load_task_artifact
from rdb_prior.task.mechanisms import future_event_labels, mechanism_labels
from rdb_prior.task.model import RouteRole, TaskMechanism, TaskPlan
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.task.planner import TaskPlanner, TaskPlannerConfig
from rdb_prior.task.validation import validate_task
from rdb_prior.task.view import build_task_view


class TaskGenerationTests(unittest.TestCase):
    def _database(self, sample_id: str, *, min_tables: int = 4, max_tables: int = 4):
        runtime = RuntimeContext(303).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=min_tables, max_tables=max_tables)
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=40,
                lookup_rows_min=3,
                lookup_rows_max=5,
                max_rows_per_table=128,
            )
        ).plan(
            sample_id=sample_id,
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        database = DatabaseGenerator().generate(schema=schema, plan=plan)
        return runtime, schema, database

    def test_relation_attribute_task_masks_target_and_round_trips(self) -> None:
        runtime, schema, database = self._database("attribute_task")
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=2,
                mechanism_weights=((TaskMechanism.RELATION_ATTRIBUTE, 1.0),),
                min_support_rows=12,
                min_query_rows=6,
            )
        )

        tasks = planner.generate(
            sample_id="attribute_task",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )

        self.assertEqual(2, len(tasks))
        for task in tasks:
            self.assertIs(TaskMechanism.RELATION_ATTRIBUTE, task.plan.mechanism)
            self.assertIn(
                task.plan.target_column_id,
                task.plan.masked_column_ids,
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            view = build_task_view(schema, database, task.plan)
            self.assertTrue(
                view.is_column_masked(
                    task.plan.target_table_id,
                    task.plan.target_column_id or "",
                )
            )
            self.assertEqual(task.plan, TaskPlan.from_dict(task.plan.to_dict()))

    def test_future_event_task_recomputes_labels_and_cuts_visibility(self) -> None:
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"future_task_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            task = tasks[0]
            expected = future_event_labels(schema, database, task.plan)

            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            self.assertTrue(task.plan.observation_rules)
            self.assertTrue(
                all(
                    rule.max_timestamp == task.plan.cutoff_time
                    for rule in task.plan.observation_rules
                )
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            view = build_task_view(schema, database, task.plan)
            for rule in task.plan.observation_rules:
                visible = view.visible_rows(rule.table_id)
                times = database.table(rule.table_id).column(rule.time_column_id)
                self.assertTrue(np.all(times[visible] <= rule.max_timestamp))
            for foreign_key in schema.foreign_keys:
                child_rows = view.visible_rows(foreign_key.child_table_id)
                assignments = database.table(foreign_key.child_table_id).column(
                    foreign_key.child_column_id
                )[child_rows]
                valid = assignments >= 0
                parent_mask = view.row_masks[foreign_key.parent_table_id]
                self.assertTrue(np.all(parent_mask[assignments[valid]]))
            return
        self.fail("no balanced future-event task found in 20 databases")

    def test_full_task_pipeline_honors_tasks_per_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=2,
                    base_seed=71,
                    sampler=BlueprintSamplerConfig(min_tables=3, max_tables=4),
                )
            )
            instance_result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance",
                    planner=InstancePlannerConfig(
                        entity_rows_min=24,
                        entity_rows_max=32,
                        lookup_rows_min=3,
                        lookup_rows_max=5,
                        max_rows_per_table=96,
                    ),
                )
            )
            result = generate_tasks(
                TaskPipelineConfig(
                    instance_manifest=instance_result.manifest_path,
                    output_root=root / "task",
                    planner=TaskPlannerConfig(
                        tasks_per_database=2,
                        mechanism_weights=(
                            (TaskMechanism.RELATION_ATTRIBUTE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                    ),
                )
            )

            self.assertEqual(2, result.database_count)
            self.assertEqual(4, result.task_count)
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(2, manifest["database_count"])
            self.assertEqual(4, manifest["task_count"])
            for artifact_path in result.artifact_paths:
                artifact = load_task_artifact(artifact_path)
                self.assertTrue(artifact.validation.is_valid)
                instance = load_instance_artifact(artifact.instance_artifact)
                self.assertEqual(
                    instance.database.instance_id,
                    artifact.task.plan.instance_id,
                )

    def test_all_mechanisms_emit_recomputable_exact_required_paths(self) -> None:
        sample_id = "mechanism_route_audit"
        runtime = RuntimeContext(991).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=6, max_tables=8)
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint, sample_id, runtime
        )
        instance_plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=48,
                max_rows_per_table=160,
            )
        ).plan(
            sample_id=sample_id,
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        database = DatabaseGenerator().generate(
            schema=schema,
            plan=instance_plan,
        )

        for mechanism in TaskMechanism:
            tasks = TaskPlanner(
                TaskPlannerConfig(
                    tasks_per_database=1,
                    mechanism_weights=((mechanism, 1.0),),
                    min_support_rows=8,
                    min_query_rows=4,
                    min_class_count_per_split=1,
                    max_attempts_per_database=512,
                )
            ).generate(
                sample_id=sample_id,
                schema=schema,
                database=database,
                runtime=runtime.child("task", mechanism.value),
            )
            task = tasks[0]
            required = [
                label
                for label in task.plan.route_supervision
                if label.role is RouteRole.REQUIRED
            ]
            self.assertTrue(required, mechanism.value)
            expected = mechanism_labels(schema, database, task.plan)
            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)


if __name__ == "__main__":
    unittest.main()
