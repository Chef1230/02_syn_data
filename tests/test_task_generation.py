from __future__ import annotations

from dataclasses import replace
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
from rdb_prior.task.mechanisms import (
    _temporal_split,
    future_event_labels,
    mechanism_labels,
)
from rdb_prior.task.model import RouteRole, TaskMechanism, TaskPlan
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.task.planner import (
    TaskPlanner,
    TaskPlannerConfig,
    _bind_instance_calendar,
)
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

    def test_temporal_split_rejects_single_class_query(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
        ordered = np.arange(len(labels), dtype=np.int64)

        split = _temporal_split(
            labels,
            ordered,
            np.random.default_rng(17),
            support_fraction=0.5,
            min_support_rows=4,
            min_query_rows=4,
            min_class_count=2,
        )

        self.assertIsNone(split)

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
            target_mask = view.row_masks[task.plan.target_table_id]
            self.assertTrue(
                np.all(target_mask[task.data.support_row_ids])
            )
            self.assertTrue(
                np.all(target_mask[task.data.query_row_ids])
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

    def test_history_gated_future_activity_gates_on_prior_events(self) -> None:
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"history_gated_task_{index}"
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
            plan = task.plan

            # Recompute the label from raw event rows: 1 iff the entity has
            # at least one event at or before the cutoff and none within
            # (cutoff, horizon].
            fk = next(
                foreign_key
                for foreign_key in schema.foreign_keys
                if foreign_key.foreign_key_id == plan.foreign_key_id
            )
            event = database.table(plan.source_table_id)
            times = event.column(plan.time_column_id or "")
            assignments = event.column(fk.child_column_id)
            entity_count = database.table(plan.target_table_id).row_count
            valid = assignments >= 0
            has_history = np.zeros(entity_count, dtype=bool)
            has_future = np.zeros(entity_count, dtype=bool)
            has_history[
                np.unique(assignments[valid & (times <= plan.cutoff_time)])
            ] = True
            has_future[np.unique(assignments[
                valid
                & (times > plan.cutoff_time)
                & (times <= plan.horizon_end_time)
            ])] = True
            expected = (has_history & ~has_future).astype(np.int8)

            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            np.testing.assert_array_equal(
                expected, mechanism_labels(schema, database, plan)
            )
            # Both classes must occur: labels are gated on having history.
            self.assertEqual({0, 1}, set(expected.tolist()))
            self.assertTrue(np.any(has_history))
            self.assertTrue(validate_task(schema, database, task).is_valid)
            return
        self.fail("no balanced history-gated task found in 40 databases")

    def _expected_submode_labels(
        self,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        plan: TaskPlan,
        mechanism: TaskMechanism,
    ) -> np.ndarray:
        """Independently derive the sub-mode label array from raw event rows.

        Entities without any history get ``-1`` (excluded from the sample);
        the two sub-modes only differ in which group is the positive class:
        - ``HISTORY_GATED_FUTURE_INACTIVE``: positive = no future events
        - ``HISTORY_GATED_FUTURE_ACTIVE``:  positive = has future events
        """
        fk = next(
            foreign_key
            for foreign_key in schema.foreign_keys
            if foreign_key.foreign_key_id == plan.foreign_key_id
        )
        event = database.table(plan.source_table_id)
        times = event.column(plan.time_column_id or "")
        assignments = event.column(fk.child_column_id)
        entity_count = database.table(plan.target_table_id).row_count
        valid = assignments >= 0
        has_history = np.zeros(entity_count, dtype=bool)
        has_future = np.zeros(entity_count, dtype=bool)
        has_history[
            np.unique(assignments[valid & (times <= plan.cutoff_time)])
        ] = True
        has_future[np.unique(assignments[
            valid
            & (times > plan.cutoff_time)
            & (times <= plan.horizon_end_time)
        ])] = True

        labels = np.full(entity_count, -1, dtype=np.int8)
        if mechanism is TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE:
            labels[has_history & ~has_future] = 1
            labels[has_history & has_future] = 0
        else:
            labels[has_history & has_future] = 1
            labels[has_history & ~has_future] = 0
        return labels

    def _test_history_gated_submode(self, mechanism: TaskMechanism) -> None:
        """Shared test body for the two history-gated sub-modes.

        Both sub-modes restrict samples to entities with prior history and
        only differ in which group is the positive class.
        """
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=((mechanism, 1.0),),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"hg_submode_{mechanism.value}_{index}"
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
            plan = task.plan

            # Independently recompute expected labels from raw event rows —
            # not via mechanism_labels, so the task and the recompute path
            # are both cross-checked against the sub-mode semantics.
            expected = self._expected_submode_labels(
                schema, database, plan, mechanism
            )
            # Entities without history are excluded from the sample.
            supervised = np.concatenate(
                (task.data.support_row_ids, task.data.query_row_ids)
            )
            self.assertTrue(
                np.all(expected[supervised] >= 0),
                f"Sub-mode {mechanism.value}: all supervised entities must have history",
            )
            # Both classes must be present.
            supervised_labels = expected[supervised]
            self.assertEqual(
                {0, 1},
                set(np.unique(supervised_labels).tolist()),
                f"Sub-mode {mechanism.value}: both classes must appear",
            )

            # The task's stored labels and the recompute path must both agree
            # with the independently derived expectation.
            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            np.testing.assert_array_equal(
                expected, mechanism_labels(schema, database, plan)
            )
            self.assertTrue(
                np.any(expected >= 0),
                f"Sub-mode {mechanism.value}: some entities must have history",
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            return
        self.fail(f"no balanced {mechanism.value} task found in 40 databases")

    def test_history_gated_future_inactive_submode(self) -> None:
        """Sub-mode 1: entities with history, positive = no future events."""
        self._test_history_gated_submode(
            TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE
        )

    def test_history_gated_future_active_submode(self) -> None:
        """Sub-mode 2: entities with history, positive = has future events."""
        self._test_history_gated_submode(
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE
        )

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
        # A small 6-8 table schema does not always admit every mechanism,
        # and the eligible set shifts whenever the motif library changes.
        # Audit the first deterministic sample that supports all mechanisms.
        chosen = None
        for suffix in range(16):
            sample_id = (
                "mechanism_route_audit"
                if suffix == 0
                else f"mechanism_route_audit_{suffix}"
            )
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
            tasks_by_mechanism = {}
            for mechanism in TaskMechanism:
                try:
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
                except ValueError:
                    break
                tasks_by_mechanism[mechanism] = tasks[0]
            if len(tasks_by_mechanism) == len(TaskMechanism):
                chosen = (schema, database, tasks_by_mechanism)
                break

        self.assertIsNotNone(
            chosen, "no sampled database supported all mechanisms"
        )
        schema, database, tasks_by_mechanism = chosen

        for mechanism in TaskMechanism:
            task = tasks_by_mechanism[mechanism]
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
            view = build_task_view(schema, database, task.plan)
            target_mask = view.row_masks[task.plan.target_table_id]
            self.assertTrue(
                np.all(target_mask[task.data.support_row_ids])
            )
            self.assertTrue(
                np.all(target_mask[task.data.query_row_ids])
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)

    def test_task_cutoff_and_horizon_within_calendar_interval(self) -> None:
        generated: list[TaskPlan] = []
        for suffix in range(8):
            sample_id = (
                "task_calendar_bounds"
                if suffix == 0
                else f"task_calendar_bounds_{suffix}"
            )
            runtime = RuntimeContext(404).for_sample(sample_id)
            blueprint = BlueprintSampler(
                BlueprintSamplerConfig(min_tables=4, max_tables=4)
            ).sample(sample_id, runtime)
            schema = PhysicalSchemaCompiler().compile(
                blueprint, sample_id, runtime
            )
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
            try:
                tasks = TaskPlanner(
                    TaskPlannerConfig(
                        tasks_per_database=2,
                        mechanism_weights=(
                            (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                        min_class_count_per_split=1,
                        max_attempts_per_database=512,
                    )
                ).generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                    instance_plan=plan,
                )
            except ValueError:
                continue
            for task in tasks:
                task_plan = task.plan
                self.assertEqual(
                    task_plan.db_start_seconds,
                    plan.calendar_start_seconds,
                )
                self.assertEqual(
                    task_plan.db_end_seconds,
                    plan.calendar_end_seconds,
                )
                for name in ("cutoff_time", "horizon_end_time"):
                    value = getattr(task_plan, name)
                    if value is not None:
                        self.assertGreaterEqual(
                            value, task_plan.db_start_seconds
                        )
                        self.assertLessEqual(value, task_plan.db_end_seconds)
                for rule in task_plan.observation_rules:
                    self.assertGreaterEqual(
                        rule.max_timestamp, task_plan.db_start_seconds
                    )
                    self.assertLessEqual(
                        rule.max_timestamp, task_plan.db_end_seconds
                    )
                generated.append(task_plan)
            if len(generated) >= 3:
                break
        self.assertTrue(generated)

    def test_calendar_binding_rejects_invalid_candidate_without_clipping(self) -> None:
        runtime, schema, database = self._database("calendar_binding")
        instance_plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=40,
                lookup_rows_min=3,
                lookup_rows_max=5,
                max_rows_per_table=128,
            )
        ).plan(
            sample_id="calendar_binding",
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        task = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=((TaskMechanism.RELATION_ATTRIBUTE, 1.0),),
                min_support_rows=8,
                min_query_rows=4,
            )
        ).generate(
            sample_id="calendar_binding",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )[0]

        invalid = replace(
            task,
            plan=replace(task.plan, cutoff_time=0),
        )
        self.assertIsNone(_bind_instance_calendar(invalid, instance_plan))


if __name__ == "__main__":
    unittest.main()
