"""Relation-plan dispatcher."""

from __future__ import annotations

import numpy as np

from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.relation_strategies import (
    generate_affinity_bridge,
    generate_single_relation,
)
from rdb_prior.instance.plan import InstancePlan


def generate_relations(
    plan: InstancePlan,
    latents: LatentRegistry,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for relation in plan.relations:
        child_rows = plan.table(
            relation.child_table_id
        ).population.row_count
        if relation.family == "affinity_bridge":
            generated = generate_affinity_bridge(
                relation,
                child_rows=child_rows,
                latents=latents,
            )
        else:
            generated = generate_single_relation(
                relation,
                child_rows=child_rows,
                latents=latents,
            )
        overlap = set(values) & set(generated)
        if overlap:
            raise RuntimeError(f"foreign keys generated twice: {sorted(overlap)}")
        values.update(generated)
    return values


__all__ = ["generate_relations"]
