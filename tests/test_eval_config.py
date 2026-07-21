from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdb_prior.eval_config import load_eval_environment  # noqa: E402


class EvalConfigTests(unittest.TestCase):
    def test_loads_values_and_resolves_project_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "eval.yaml"
            config.write_text(
                "\n".join(
                    [
                        "relbench:",
                        "  dataset: rel-f1",
                        "  tasks: [driver-dnf, driver-top3]",
                        "  output: outputs/f1",
                        "  download: false",
                        "  reuse_converted: true",
                        "router:",
                        "  checkpoint: checkpoints/router.pt",
                        "runtime:",
                        "  progress_every: 1",
                    ]
                ),
                encoding="utf-8",
            )

            values = load_eval_environment(config, project_root=root)

            self.assertEqual(values["RELBENCH_DATASET"], "rel-f1")
            self.assertEqual(values["RELBENCH_TASKS"], "driver-dnf,driver-top3")
            self.assertEqual(values["DOWNLOAD"], "0")
            self.assertEqual(values["REUSE_CONVERTED"], "1")
            self.assertEqual(values["PROGRESS_EVERY"], "1")
            self.assertEqual(values["RELBENCH_OUTPUT"], str(root / "outputs/f1"))
            self.assertEqual(
                values["ROUTER_CHECKPOINT"], str(root / "checkpoints/router.pt")
            )

    def test_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "eval.yaml"
            config.write_text("runtime:\n  progres_every: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "runtime.progres_every"):
                load_eval_environment(config, project_root=root)


if __name__ == "__main__":
    unittest.main()
