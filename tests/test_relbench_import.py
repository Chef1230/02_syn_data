from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency environment.
    pd = None


from rdb_prior.importers.relbench import (  # noqa: E402
    RelBenchImportConfig,
    convert_relbench_objects,
)
from rdb_prior.evaluation.relbench import (  # noqa: E402
    RelBenchScoreConfig,
    score_relbench_predictions,
)
from rdb_prior.routing.catalog import enumerate_schema_paths  # noqa: E402
from rdb_prior.routing.data import (  # noqa: E402
    RoutingTaskStore,
    RoutingTaskTensorizer,
    _traverse_path,
)
from rdb_prior.routing.config import RouterModelConfig  # noqa: E402


class _FakeRelBenchTable:
    def __init__(
        self,
        frame,
        *,
        pkey_col=None,
        time_col=None,
        foreign_keys=None,
    ) -> None:
        self.df = frame
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.fkey_col_to_pkey_table = dict(foreign_keys or {})


class _FakeDatabase:
    def __init__(self, tables) -> None:
        self.table_dict = tables


class _FakeDataset:
    def __init__(self, database) -> None:
        self.database = database

    def get_db(self, upto_test_timestamp=True):
        del upto_test_timestamp
        return self.database


class _FakeEntityTask:
    task_type = SimpleNamespace(value="binary_classification")
    entity_table = "users"
    entity_col = "user_id"
    time_col = "timestamp"
    target_col = "label"

    def __init__(self, split_frames) -> None:
        self.split_frames = split_frames

    def get_table(self, split, mask_input_cols=False):
        del mask_input_cols
        return _FakeRelBenchTable(self.split_frames[split].copy())


class _FakeOfficialTask:
    def __init__(self) -> None:
        self.prediction = None

    def evaluate(self, prediction):
        self.prediction = prediction
        return {"mean_score": float(np.mean(prediction))}


@unittest.skipIf(pd is None, "pandas is not installed")
class RelBenchImportTests(unittest.TestCase):
    def _objects(self):
        users = _FakeRelBenchTable(
            pd.DataFrame(
                {
                    "user_id": ["user-a", "user-b"],
                    "country": ["cn", "us"],
                }
            ),
            pkey_col="user_id",
        )
        events = _FakeRelBenchTable(
            pd.DataFrame(
                {
                    "user_id": ["user-a", "user-a", "user-a", "user-b"],
                    "event_time": pd.to_datetime(
                        ["2020-01-05", "2020-01-20", "2020-01-30", "2020-01-05"]
                    ),
                    "amount": [1.0, 2.0, 99.0, 3.0],
                }
            ),
            time_col="event_time",
            foreign_keys={"user_id": "users"},
        )
        database = _FakeDatabase({"users": users, "events": events})
        split_frames = {
            "train": pd.DataFrame(
                {
                    "user_id": ["user-a", "user-b"],
                    "timestamp": pd.to_datetime(["2020-01-10", "2020-01-10"]),
                    "label": [0, 1],
                }
            ),
            "val": pd.DataFrame(
                {
                    "user_id": ["user-a"],
                    "timestamp": pd.to_datetime(["2020-01-15"]),
                    "label": [0],
                }
            ),
            "test": pd.DataFrame(
                {
                    "user_id": ["user-a", "user-b", "user-a"],
                    "timestamp": pd.to_datetime(
                        ["2020-01-25", "2020-01-25", "2020-01-25"]
                    ),
                    "label": [1, 0, 1],
                }
            ),
        }
        return _FakeDataset(database), _FakeEntityTask(split_frames)

    def test_conversion_chunks_all_test_rows_and_is_temporally_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "relbench"
            dataset, task = self._objects()
            progress: list[tuple[int, int, str]] = []
            result = convert_relbench_objects(
                RelBenchImportConfig(
                    dataset_name="rel-fake",
                    task_name="user-churn",
                    output_root=root,
                    max_rows_per_task=5,
                    query_rows_per_task=2,
                    support_rows=3,
                ),
                dataset=dataset,
                task=task,
                progress=lambda completed, total, item: progress.append(
                    (completed, total, item)
                ),
            )
            self.assertEqual(2, result.task_count)
            self.assertEqual(3, result.query_row_count)
            manifest = json.loads(
                result.task_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(2, manifest["task_count"])
            self.assertEqual((0, 7, "prepare"), progress[0])
            self.assertEqual((7, 7, "metadata"), progress[-1])
            items = [item for _completed, _total, item in progress]
            for expected in ("table:events", "table:users", "schema", "instance"):
                self.assertIn(expected, items)

            store = RoutingTaskStore(result.task_manifest, cache_size=1)
            raw_tasks = [store.load(reference) for reference in store.references]
            query_rows = np.concatenate(
                [raw.task_artifact.task.data.query_row_ids for raw in raw_tasks]
            )
            np.testing.assert_array_equal(query_rows, np.asarray([3, 4, 5]))
            for raw in raw_tasks:
                tensors = RoutingTaskTensorizer(
                    RouterModelConfig(
                        max_path_depth=2,
                        max_candidates=8,
                        top_k_paths=2,
                        max_source_columns=3,
                        min_rows_per_hop=1,
                        rows_per_hop=8,
                        max_rows_per_task=5,
                        token_dim=16,
                        type_embedding_dim=4,
                        router_hidden_dim=24,
                        transformer_heads=4,
                        transformer_layers=1,
                    )
                ).tensorize(raw)
                self.assertEqual(
                    len(raw.task_artifact.task.data.support_row_ids)
                    + len(raw.task_artifact.task.data.query_row_ids),
                    len(tensors.row_ids),
                )

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            event_table_id = next(
                item["table_id"]
                for item in metadata["tables"]
                if item["original_name"] == "events"
            )
            raw = raw_tasks[0]
            event_path = next(
                path
                for path in enumerate_schema_paths(
                    raw.schema,
                    target_table_id=raw.task_artifact.task.plan.target_table_id,
                    max_depth=2,
                    max_candidates=8,
                )
                if path.source_table_id == event_table_id
            )
            train_related, _reads, _expanded = _traverse_path(
                raw,
                event_path,
                0,
                min_rows=8,
                max_rows=8,
                seed_key="train",
            )
            test_related, _reads, _expanded = _traverse_path(
                raw,
                event_path,
                3,
                min_rows=8,
                max_rows=8,
                seed_key="test",
            )
            np.testing.assert_array_equal(train_related, np.asarray([0]))
            np.testing.assert_array_equal(test_related, np.asarray([0, 1]))

            predictions_path = root / "predictions.jsonl"
            prediction_rows = (
                {"row_id": 5, "probabilities": [0.3, 0.7]},
                {"row_id": 3, "probabilities": [0.2, 0.8]},
                {"row_id": 4, "probabilities": [0.9, 0.1]},
            )
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in prediction_rows),
                encoding="utf-8",
            )
            official_task = _FakeOfficialTask()
            official = score_relbench_predictions(
                RelBenchScoreConfig(
                    metadata_path=result.metadata_path,
                    predictions_path=predictions_path,
                    output_path=root / "relbench_metrics.json",
                ),
                task=official_task,
            )
            np.testing.assert_allclose(
                official_task.prediction, np.asarray([0.8, 0.1, 0.7])
            )
            self.assertAlmostEqual(official.metrics["mean_score"], 1.6 / 3)

    def test_pre_unix_epoch_timestamps_are_shifted_without_losing_order(self) -> None:
        dataset, task = self._objects()
        for frame in task.split_frames.values():
            frame["timestamp"] = frame["timestamp"] - pd.DateOffset(years=70)
        with tempfile.TemporaryDirectory() as directory:
            result = convert_relbench_objects(
                RelBenchImportConfig(
                    dataset_name="rel-f1",
                    task_name="driver-dnf",
                    output_root=Path(directory),
                    max_rows_per_task=5,
                    query_rows_per_task=2,
                    support_rows=3,
                ),
                dataset=dataset,
                task=task,
            )

            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertLess(metadata["timestamp_origin_ns"], 0)
            store = RoutingTaskStore(result.task_manifest, cache_size=1)
            raw = store.load(store.references[0])
            times = raw.database.table(
                raw.task_artifact.task.plan.target_table_id
            ).column(raw.task_artifact.task.plan.row_cutoff_time_column_id)
            self.assertTrue(np.all(times >= 0))

    def test_missing_task_timestamp_is_still_rejected(self) -> None:
        dataset, task = self._objects()
        task.split_frames["train"].loc[0, "timestamp"] = pd.NaT
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "timestamps cannot be missing"):
                convert_relbench_objects(
                    RelBenchImportConfig(
                        dataset_name="rel-fake",
                        task_name="missing-time",
                        output_root=Path(directory),
                    ),
                    dataset=dataset,
                    task=task,
                )

    def test_rejects_task_types_that_need_non_scalar_targets(self) -> None:
        dataset, task = self._objects()
        task.task_type = SimpleNamespace(value="link_prediction")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported RelBench task type"):
                convert_relbench_objects(
                    RelBenchImportConfig(
                        dataset_name="rel-fake",
                        task_name="links",
                        output_root=Path(directory),
                    ),
                    dataset=dataset,
                    task=task,
                )


if __name__ == "__main__":
    unittest.main()
