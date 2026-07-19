from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.extensions.interfaces import ExtensionBundle
from rdb_prior.pipeline import SchemaPipelineConfig, generate_physical_schemas
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig


class ExtensionPipelineTests(unittest.TestCase):
    def test_pipeline_invokes_all_reserved_extension_boundaries(self) -> None:
        calls: list[str] = []
        sampler = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=3, max_tables=3)
        )
        compiler = PhysicalSchemaCompiler()

        class Domain:
            def sample(self, runtime):
                calls.append("domain")
                return {"anonymous": True}

        class Blueprint:
            def sample(self, sample_id, runtime, domain):
                calls.append("blueprint")
                return sampler.sample(sample_id, runtime)

        class Process:
            def instantiate(self, domain, blueprint, runtime):
                calls.append("process")
                return ()

        class Task:
            def plan(self, domain, blueprint, processes, runtime):
                calls.append("task")
                return None

        class Design:
            def sample(self, blueprint, task_plan, runtime):
                calls.append("design")
                return None

        class Compiler:
            def compile(self, blueprint, design, sample_id, runtime):
                calls.append("compiler")
                return compiler.compile_result(blueprint, sample_id, runtime)

        extensions = ExtensionBundle(
            domain=Domain(),
            blueprint=Blueprint(),
            process=Process(),
            task=Task(),
            design=Design(),
            compiler=Compiler(),
        )
        with tempfile.TemporaryDirectory() as directory:
            generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=Path(directory),
                    num_schemas=1,
                    sampler=BlueprintSamplerConfig(
                        min_tables=3,
                        max_tables=3,
                    ),
                ),
                extensions=extensions,
            )

        self.assertEqual(
            ["domain", "blueprint", "process", "task", "design", "compiler"],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
