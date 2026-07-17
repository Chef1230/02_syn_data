from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.pipeline import (
    SchemaPipelineConfig,
    generate_physical_schemas,
)
from rdb_prior.schema.sampler import BlueprintSamplerConfig


class SchemaPipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "schemas"
            config = SchemaPipelineConfig(
                output_root=output_root,
                num_schemas=3,
                base_seed=17,
                progress_every=1,
                sampler=BlueprintSamplerConfig(
                    min_tables=3,
                    max_tables=5,
                ),
            )

            result = generate_physical_schemas(config)

            self.assertEqual(3, result.generated_count)
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(3, len(manifest["entries"]))

            for artifact_path in result.artifact_paths:
                payload = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
                self.assertEqual("physical_schema", payload["artifact_type"])
                self.assertTrue(payload["validation"]["is_valid"])
                self.assertNotIn("motifs", payload["blueprint"])
                self.assertNotIn(
                    "motif_occurrences",
                    payload["blueprint"],
                )
                PhysicalSchema.from_dict(payload["physical_schema"])

    def test_existing_artifact_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = SchemaPipelineConfig(
                output_root=Path(temporary_directory),
                num_schemas=1,
            )
            generate_physical_schemas(config)

            with self.assertRaises(FileExistsError):
                generate_physical_schemas(config)


if __name__ == "__main__":
    unittest.main()
