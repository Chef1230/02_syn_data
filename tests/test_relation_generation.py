from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.generation.latent import LatentRegistry, TableLatent
from rdb_prior.generation.relation_strategies import (
    generate_affinity_bridge,
    generate_single_relation,
)
from rdb_prior.instance.plan import RelationMechanismPlan


def _table(values: np.ndarray) -> TableLatent:
    return TableLatent(
        values=values.astype(np.float64),
        activity=np.ones(len(values), dtype=np.float64),
    )


class RelationGenerationTests(unittest.TestCase):
    def test_softmax_fk_responds_to_shared_child_and_parent_latent(self) -> None:
        parent = np.linspace(-2.5, 2.5, 21)[:, None]
        child = np.concatenate(
            [np.full((200, 1), -2.0), np.full((200, 1), 2.0)]
        )
        latents = LatentRegistry(
            {"parent": _table(parent), "child": _table(child)}
        )
        plan = RelationMechanismPlan(
            relation_group_id="relation",
            foreign_key_ids=("fk",),
            parent_table_ids=("parent",),
            child_table_id="child",
            family="lookup_softmax",
            optional_rates=(0.0,),
            seed=7,
            parameters=(("affinity_strength", 5.0), ("degree_strength", 0.0)),
        )

        first = generate_single_relation(plan, child_rows=400, latents=latents)["fk"]
        second = generate_single_relation(plan, child_rows=400, latents=latents)["fk"]

        np.testing.assert_array_equal(first, second)
        self.assertGreater(abs(first[:200].mean() - first[200:].mean()), 5.0)
        self.assertTrue(np.all((first >= 0) & (first < len(parent))))

    def test_affinity_bridge_generates_unique_joint_parent_tuples(self) -> None:
        rng = np.random.default_rng(4)
        latents = LatentRegistry(
            {
                "left": _table(rng.normal(size=(12, 3))),
                "right": _table(rng.normal(size=(13, 3))),
                "bridge": _table(rng.normal(size=(80, 3))),
            }
        )
        plan = RelationMechanismPlan(
            relation_group_id="bridge_relation",
            foreign_key_ids=("left_fk", "right_fk"),
            parent_table_ids=("left", "right"),
            child_table_id="bridge",
            family="affinity_bridge",
            optional_rates=(0.0, 0.0),
            seed=31,
            parameters=(("affinity_strength", 1.0), ("degree_strength", 0.8)),
        )

        result = generate_affinity_bridge(plan, child_rows=80, latents=latents)
        pairs = np.column_stack([result["left_fk"], result["right_fk"]])

        self.assertEqual(80, len(np.unique(pairs, axis=0)))
        self.assertTrue(np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < 12)))
        self.assertTrue(np.all((pairs[:, 1] >= 0) & (pairs[:, 1] < 13)))

    def test_lookup_transition_is_persistent_in_latent_event_order(self) -> None:
        rng = np.random.default_rng(18)
        child = rng.normal(size=(500, 2))
        latents = LatentRegistry(
            {
                "lookup": _table(rng.normal(size=(6, 2))),
                "event": _table(child),
            }
        )
        plan = RelationMechanismPlan(
            relation_group_id="transition",
            foreign_key_ids=("status_fk",),
            parent_table_ids=("lookup",),
            child_table_id="event",
            family="lookup_transition",
            optional_rates=(0.0,),
            seed=83,
        )

        values = generate_single_relation(plan, child_rows=500, latents=latents)[
            "status_fk"
        ]
        ordered = values[np.argsort(child[:, 0], kind="stable")]

        self.assertGreater(np.mean(ordered[1:] == ordered[:-1]), 0.5)
        self.assertTrue(np.all((values >= 0) & (values < 6)))


if __name__ == "__main__":
    unittest.main()
