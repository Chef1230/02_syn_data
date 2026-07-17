from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.cli import main
from rdb_prior.config import (
    SchemaConfigError,
    SchemaConfigOverrides,
    load_schema_pipeline_config,
)


class SchemaConfigTests(unittest.TestCase):
    def test_reference_yaml_loads_all_schema_sections(self) -> None:
        config = load_schema_pipeline_config(
            PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        )

        self.assertEqual(20, config.num_schemas)
        self.assertEqual(42, config.base_seed)
        self.assertEqual(tuple(range(3, 16)), config.sampler.table_count_values)
        self.assertEqual(6, len(config.sampler.motif_weights))
        self.assertEqual(
            4,
            len(config.compiler.feature_columns_by_table_count),
        )
        self.assertEqual((), config.compiler.feature_columns_by_role)
        self.assertEqual(
            (PROJECT_ROOT / "outputs" / "schema_v1_sample").resolve(),
            config.output_root,
        )

    def test_unknown_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.yaml"
            path.write_text(
                "config_version: 1\nschema:\n  unknown_knob: 3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SchemaConfigError,
                "unknown option",
            ):
                load_schema_pipeline_config(path)

    def test_unknown_motif_is_rejected_during_config_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_motif.yaml"
            path.write_text(
                "config_version: 1\nmotifs:\n  weights:\n"
                "    imaginary_motif: 1.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SchemaConfigError,
                "Unknown configured motif",
            ):
                load_schema_pipeline_config(path)

    def test_cli_bounds_override_disables_configured_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "schemas"
            config = load_schema_pipeline_config(
                PROJECT_ROOT / "configs" / "refactor_v1.yaml",
                overrides=SchemaConfigOverrides(
                    output_root=output,
                    num_schemas=2,
                    min_tables=3,
                    max_tables=4,
                    min_feature_columns=1,
                    max_feature_columns=2,
                ),
            )

            self.assertEqual(2, config.num_schemas)
            self.assertEqual((), config.sampler.table_count_values)
            self.assertEqual(
                (),
                config.compiler.feature_columns_by_table_count,
            )
            self.assertEqual(1, config.compiler.min_feature_columns)
            self.assertEqual(2, config.compiler.max_feature_columns)

    def test_schema_cli_generates_from_yaml_with_run_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            config_path = root / "schema.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "config_version: 1",
                        "seed: 7",
                        "generation:",
                        "  num_schemas: 9",
                        "  progress_every: 0",
                        "schema:",
                        "  min_tables: 3",
                        "  max_tables: 3",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "schema",
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output),
                        "--count",
                        "2",
                    )
                )

            self.assertEqual(0, exit_code)
            summary = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(2, summary["generated_count"])
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(manifest["entries"]))
            self.assertTrue(
                all(entry["table_count"] == 3 for entry in manifest["entries"])
            )


if __name__ == "__main__":
    unittest.main()
