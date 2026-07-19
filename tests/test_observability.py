from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.observability import (
    ProgressReporter,
    close_logging,
    configure_logging,
)
from rdb_prior.pipeline import SchemaPipelineConfig, generate_physical_schemas
from rdb_prior.schema.sampler import BlueprintSamplerConfig


class ObservabilityTests(unittest.TestCase):
    def test_logging_writes_console_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stream = StringIO()
            log_path = Path(temporary_directory) / "logs" / "run.log"
            logger = configure_logging(
                level="DEBUG",
                log_file=log_path,
                stream=stream,
            ).getChild("test")

            logger.info("pipeline ready")
            for handler in logger.parent.handlers:
                handler.flush()

            self.assertIn("pipeline ready", stream.getvalue())
            self.assertIn(
                "pipeline ready",
                log_path.read_text(encoding="utf-8"),
            )
            close_logging()

    def test_forced_progress_bar_contains_rate_and_eta(self) -> None:
        stream = StringIO()
        reporter = ProgressReporter(
            stage="task",
            total=2,
            enabled=True,
            width=12,
            stream=stream,
        )

        reporter.update(1, 2, "sample_000000", detail="tasks=2")
        reporter.update(2, 2, "sample_000001", detail="tasks=2")
        reporter.close()

        output = stream.getvalue()
        self.assertIn("[task]", output)
        self.assertIn("2/2", output)
        self.assertIn("100.0%", output)
        self.assertIn("ETA", output)
        self.assertIn("tasks=2", output)

    def test_pipeline_reports_every_completed_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            events: list[tuple[int, int, str]] = []
            generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=Path(temporary_directory),
                    num_schemas=3,
                    progress_every=100,
                    sampler=BlueprintSamplerConfig(min_tables=3, max_tables=3),
                ),
                progress=lambda completed, total, sample_id: events.append(
                    (completed, total, sample_id)
                ),
            )

            self.assertEqual([1, 2, 3], [event[0] for event in events])
            self.assertTrue(all(event[1] == 3 for event in events))


if __name__ == "__main__":
    unittest.main()
