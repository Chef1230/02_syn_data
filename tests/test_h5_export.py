from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.export.h5 import H5ExportConfig, export_processed_dbb_to_h5


class H5ExportTests(unittest.TestCase):
    def test_streams_processed_classification_tasks_in_training_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed = root / "rdbpfn-processed"
            self._write_dataset(
                processed / "classification-dfs-1",
                task_type="classification",
            )
            self._write_dataset(
                processed / "regression-dfs-1",
                task_type="regression",
            )
            output = root / "prior.h5"
            progress: list[tuple[int, int, str]] = []

            result = export_processed_dbb_to_h5(
                H5ExportConfig(
                    processed_root=processed,
                    output_path=output,
                    total_rows=12,
                    max_columns=4,
                    seed=17,
                    dataset_names=(
                        "classification-dfs-1",
                        "regression-dfs-1",
                    ),
                ),
                progress=lambda completed, total, name: progress.append(
                    (completed, total, name)
                ),
            )

            self.assertEqual(2, result.dataset_count)
            self.assertEqual(1, result.task_count)
            self.assertEqual(1, result.skipped_task_count)
            self.assertEqual((2, 2), progress[-1][:2])
            with h5py.File(output, "r") as handle:
                self.assertEqual(
                    {
                        "X",
                        "y",
                        "num_features",
                        "num_available_features",
                        "num_datapoints",
                        "single_eval_pos",
                        "feature_is_categorical",
                        "max_num_classes",
                    },
                    set(handle.keys()),
                )
                self.assertEqual((1, 12, 4), handle["X"].shape)
                self.assertEqual((1, 12), handle["y"].shape)
                self.assertEqual(2, int(handle["num_features"][0]))
                self.assertEqual(2, int(handle["num_available_features"][0]))
                self.assertEqual(12, int(handle["num_datapoints"][0]))
                split = int(handle["single_eval_pos"][0])
                self.assertEqual(7, split)
                np.testing.assert_array_equal(
                    np.asarray([0, 1, 0, 0], dtype=np.uint8),
                    handle["feature_is_categorical"][0],
                )
                self.assertEqual({0, 1}, set(handle["y"][0, :split].tolist()))
                self.assertEqual({0, 1}, set(handle["y"][0, split:].tolist()))
                np.testing.assert_array_equal(
                    np.asarray([1], dtype=np.int32),
                    handle["max_num_classes"][:],
                )

            with self.assertRaises(FileExistsError):
                export_processed_dbb_to_h5(
                    H5ExportConfig(
                        processed_root=processed,
                        output_path=output,
                        total_rows=12,
                        max_columns=4,
                        dataset_names=("classification-dfs-1",),
                    )
                )

    @staticmethod
    def _write_dataset(path: Path, *, task_type: str) -> None:
        task_name = path.name
        task_directory = path / task_name
        task_directory.mkdir(parents=True)
        columns = [
            {"name": "feature_float", "dtype": "float", "in_size": 1},
            {
                "name": "feature_category",
                "dtype": "category",
                "num_categories": 3,
            },
            {
                "name": "label",
                "dtype": "category" if task_type == "classification" else "float",
                "num_categories": 2,
            },
        ]
        metadata = {
            "dataset_name": task_name,
            "tables": [],
            "tasks": [
                {
                    "name": task_name,
                    "source": f"{task_name}/{{split}}.npz",
                    "format": "numpy",
                    "columns": columns,
                    "time_column": None,
                    "evaluation_metric": "auroc",
                    "target_column": "label",
                    "target_table": "target",
                    "task_type": task_type,
                    "num_classes": 2,
                }
            ],
        }
        (path / "metadata.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False),
            encoding="utf-8",
        )
        splits = {
            "train": (
                np.asarray([0.0, 1.0, np.nan, 3.0], dtype=np.float32),
                np.asarray(["a", "b", "a", "c"], dtype=object),
                np.asarray([0, 1, 0, 1]),
            ),
            "validation": (
                np.asarray([4.0, 5.0], dtype=np.float32),
                np.asarray(["b", "c"], dtype=object),
                np.asarray([0, 1]),
            ),
            "test": (
                np.asarray([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
                np.asarray(["a", "unseen", "b", "c"], dtype=object),
                np.asarray([0, 1, 0, 1]),
            ),
        }
        for split_name, (numeric, category, labels) in splits.items():
            np.savez(
                task_directory / f"{split_name}.npz",
                feature_float=numeric,
                feature_category=category,
                label=labels.astype(
                    np.int64 if task_type == "classification" else np.float32
                ),
            )


if __name__ == "__main__":
    unittest.main()
