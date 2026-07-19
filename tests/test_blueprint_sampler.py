from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.sampler import (
    BlueprintSampler,
    BlueprintSamplerConfig,
)
from rdb_prior.schema.spec import TableRole
from rdb_prior.schema.validation import validate_blueprint


class BlueprintSamplerTests(unittest.TestCase):
    def test_sample_is_deterministic_for_logical_sample_id(self) -> None:
        sampler = BlueprintSampler()
        root = RuntimeContext(123)

        first = sampler.sample("sample_7", root.for_sample("sample_7"))
        second = sampler.sample("sample_7", root.for_sample("sample_7"))

        self.assertEqual(first, second)

    def test_sample_respects_size_and_required_roles(self) -> None:
        config = BlueprintSamplerConfig(
            min_tables=4,
            max_tables=7,
            max_rank=3,
        )
        sampler = BlueprintSampler(config)
        root = RuntimeContext(42)

        for sample_index in range(200):
            with self.subTest(sample_index=sample_index):
                blueprint = sampler.sample(
                    sample_index,
                    root.for_sample(sample_index),
                )
                self.assertGreaterEqual(len(blueprint.nodes), 4)
                self.assertLessEqual(len(blueprint.nodes), 7)
                self.assertLessEqual(blueprint.max_rank, 3)
                self.assertTrue(
                    blueprint.nodes_by_role(TableRole.ENTITY)
                )
                self.assertTrue(
                    blueprint.nodes_by_role(TableRole.EVENT)
                )
                self.assertTrue(validate_blueprint(blueprint).is_valid)

    def test_completed_blueprint_preserves_anonymous_motif_provenance(self) -> None:
        sampler = BlueprintSampler()
        blueprint = sampler.sample(
            "no_trace",
            RuntimeContext(42).for_sample("no_trace"),
        )

        self.assertFalse(hasattr(blueprint, "motifs"))
        self.assertGreaterEqual(len(blueprint.motif_occurrences), 1)
        for occurrence in blueprint.motif_occurrences:
            self.assertTrue(occurrence.node_bindings)
            self.assertTrue(
                set(occurrence.edges.values())
                <= {edge.edge_id for edge in blueprint.edges}
            )
        self.assertEqual(
            blueprint,
            SchemaBlueprint.from_dict(blueprint.to_dict()),
        )

    def test_background_attachments_bound_motif_count(self) -> None:
        sampler = BlueprintSampler(
            BlueprintSamplerConfig(
                min_tables=7,
                max_tables=7,
                min_motif_occurrences=1,
                max_motif_occurrences=1,
                background_attachment_probability=1.0,
            )
        )
        blueprint = sampler.sample(
            "background",
            RuntimeContext(42).for_sample("background"),
        )

        self.assertEqual(1, len(blueprint.motif_occurrences))
        motif_nodes = {
            node_id
            for occurrence in blueprint.motif_occurrences
            for node_id in occurrence.nodes.values()
        }
        self.assertLess(len(motif_nodes), len(blueprint.nodes))

    def test_configuration_rejects_unknown_motif(self) -> None:
        config = BlueprintSamplerConfig(
            motif_weights=(("unknown_motif", 1.0),)
        )

        with self.assertRaisesRegex(ValueError, "Unknown configured"):
            BlueprintSampler(config)

    def test_weighted_table_count_distribution_is_used(self) -> None:
        sampler = BlueprintSampler(
            BlueprintSamplerConfig(
                min_tables=3,
                max_tables=7,
                table_count_values=(7,),
                table_count_weights=(1.0,),
            )
        )
        root = RuntimeContext(19)

        for sample_index in range(20):
            blueprint = sampler.sample(
                sample_index,
                root.for_sample(sample_index),
            )
            self.assertEqual(7, len(blueprint.nodes))


if __name__ == "__main__":
    unittest.main()
