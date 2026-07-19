from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.graph import (
    SchemaGraphArtifactWriter,
    SchemaGraphConfig,
    physical_schema_to_dot,
)
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig


class SchemaGraphTests(unittest.TestCase):
    @staticmethod
    def _schema(sample_id: str = "graph_sample"):
        runtime = RuntimeContext(73).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=4, max_tables=4)
        ).sample(sample_id, runtime)
        return PhysicalSchemaCompiler().compile(
            blueprint,
            sample_id,
            runtime,
        )

    def test_dot_contains_tables_columns_roles_and_directed_fks(self) -> None:
        schema = self._schema()

        first = physical_schema_to_dot(schema)
        second = physical_schema_to_dot(schema)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("digraph schema {\n"))
        self.assertIn("PK", first)
        self.assertIn("FK", first)
        for table in schema.tables:
            self.assertIn(table.name, first)
            self.assertIn(f"role={table.role.value}", first)
        for foreign_key in schema.foreign_keys:
            edge = (
                f'"{foreign_key.parent_table_id}" -> '
                f'"{foreign_key.child_table_id}"'
            )
            self.assertIn(edge, first)
            child_column = schema.table(
                foreign_key.child_table_id
            ).column(foreign_key.child_column_id)
            self.assertIn(child_column.name, first)

    def test_writer_creates_dot_and_honors_overwrite(self) -> None:
        schema = self._schema("dot_writer")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = SchemaGraphArtifactWriter(output_root=root)

            artifacts = writer.commit(
                sample_id="dot_writer",
                schema=schema,
            )

            self.assertEqual(
                root / "schema_graphs" / "dot_writer.dot",
                artifacts.dot_path,
            )
            self.assertIsNone(artifacts.image_path)
            self.assertTrue(artifacts.dot_path.is_file())
            with self.assertRaises(FileExistsError):
                writer.commit(sample_id="dot_writer", schema=schema)

    @patch("rdb_prior.schema.graph.subprocess.run")
    @patch("rdb_prior.schema.graph.shutil.which", return_value="dot")
    def test_writer_can_render_png_with_graphviz(
        self,
        _which,
        run,
    ) -> None:
        schema = self._schema("png_writer")

        def render(command, **kwargs):
            self.assertEqual("dot", command[0])
            self.assertEqual("-Tpng", command[1])
            self.assertEqual("-o", command[3])
            Path(command[4]).write_bytes(b"fake-png")
            return subprocess.CompletedProcess(command, 0)

        run.side_effect = render
        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = SchemaGraphArtifactWriter(
                output_root=Path(temporary_directory),
                config=SchemaGraphConfig(render_format="png"),
            )

            artifacts = writer.commit(
                sample_id="png_writer",
                schema=schema,
            )

            self.assertTrue(artifacts.dot_path.is_file())
            self.assertEqual(b"fake-png", artifacts.image_path.read_bytes())
            run.assert_called_once()
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertTrue(run.call_args.kwargs["capture_output"])

    @patch("rdb_prior.schema.graph.shutil.which", return_value=None)
    def test_requested_rendering_requires_graphviz(self, _which) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "Graphviz"):
                SchemaGraphArtifactWriter(
                    output_root=Path(temporary_directory),
                    config=SchemaGraphConfig(render_format="svg"),
                )

    def test_rendering_cannot_be_enabled_without_dot(self) -> None:
        with self.assertRaisesRegex(ValueError, "write_dot"):
            SchemaGraphConfig(write_dot=False, render_format="png")


if __name__ == "__main__":
    unittest.main()
