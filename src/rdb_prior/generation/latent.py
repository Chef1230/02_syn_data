"""Shared hierarchical latent variables used by relations and features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.instance.plan import InstancePlan


@dataclass(frozen=True, slots=True, kw_only=True)
class TableLatent:
    values: np.ndarray
    activity: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("latent values must be a matrix")
        if self.activity.shape != (self.values.shape[0],):
            raise ValueError("activity must align with latent rows")


class LatentRegistry:
    def __init__(self, values: dict[str, TableLatent]) -> None:
        self._values = dict(values)

    def table(self, table_id: str) -> TableLatent:
        try:
            return self._values[table_id]
        except KeyError as error:
            raise KeyError(f"No latent values for table {table_id!r}") from error


def generate_latent_registry(plan: InstancePlan) -> LatentRegistry:
    if not isinstance(plan, InstancePlan):
        raise TypeError("plan must be InstancePlan")
    dimension = max(table.latent_dimension for table in plan.tables)
    global_rng = np.random.Generator(np.random.PCG64DXSM(plan.global_seed))
    global_latent = global_rng.normal(size=dimension)
    results: dict[str, TableLatent] = {}

    for table in plan.tables:
        rng = np.random.Generator(np.random.PCG64DXSM(table.latent_seed))
        rows = table.population.row_count
        dims = table.latent_dimension
        projection = rng.normal(scale=0.5, size=(dims, dimension))
        table_effect = projection @ global_latent / np.sqrt(dimension)
        values = rng.normal(size=(rows, dims)) + table_effect

        block_count = min(4, max(2, round(np.sqrt(rows) / 4)))
        block_ids = rng.integers(0, block_count, size=rows)
        block_effects = rng.normal(scale=0.65, size=(block_count, dims))
        values += block_effects[block_ids]
        values = _standardize(values)

        raw_activity = 0.8 * values[:, 0]
        if dims > 1:
            raw_activity += 0.25 * values[:, 1]
        raw_activity += rng.normal(scale=0.2, size=rows)
        activity = np.exp(np.clip(raw_activity, -3.0, 3.0))
        results[table.table_id] = TableLatent(
            values=values.astype(np.float64),
            activity=activity.astype(np.float64),
        )
    return LatentRegistry(results)


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale


__all__ = ["TableLatent", "LatentRegistry", "generate_latent_registry"]
