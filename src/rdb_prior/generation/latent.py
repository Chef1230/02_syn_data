"""Shared hierarchical latent variables used by relations and features.

Each table draws a ``RootCauseFamily`` that controls the shape of its
exogenous latent (root) variables.  The raw samples are then standardised
to unit variance so that downstream feature SCMs receive inputs with
diverse distributional characteristics — skew, heavy tails, multi-modality —
while maintaining a consistent scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.instance.plan import InstancePlan, RootCauseFamily


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

        # ------------------------------------------------------------------
        # Root-cause family dispatch — generates the raw exogenous values
        # before hierarchical effects are added.
        # ------------------------------------------------------------------
        raw_latent = _generate_root_cause(
            family=table.root_cause_family,
            rng=rng,
            rows=rows,
            dims=dims,
        )

        values = raw_latent + table_effect

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


# ---------------------------------------------------------------------------
# Root-cause family generators
# ---------------------------------------------------------------------------


def _generate_root_cause(
    family: RootCauseFamily,
    rng: np.random.Generator,
    rows: int,
    dims: int,
) -> np.ndarray:
    """Generate raw exogenous latent rows from a distribution family.

    The returned values are NOT standardised here — the caller adds
    hierarchical effects and standardises afterwards so that downstream
    feature SCMs see diverse shapes at consistent scale.
    """
    if family is RootCauseFamily.STANDARD_NORMAL:
        return rng.normal(size=(rows, dims))

    if family is RootCauseFamily.LINEAR:
        return _linear_root_cause(rng, rows, dims)

    if family is RootCauseFamily.NONLINEAR:
        return _nonlinear_root_cause(rng, rows, dims)

    if family is RootCauseFamily.LOGNORMAL:
        return _lognormal_root_cause(rng, rows, dims)

    if family is RootCauseFamily.GAUSSIAN_MIXTURE:
        return _gaussian_mixture_root_cause(rng, rows, dims)

    raise ValueError(f"unsupported root-cause family: {family}")


# ---------------------------------------------------------------------------
# STANDARD_NORMAL — identity (handled inline above; kept here for symmetry)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LINEAR — random rotation + anisotropic scaling of standard normals
# ---------------------------------------------------------------------------


def _linear_root_cause(
    rng: np.random.Generator,
    rows: int,
    dims: int,
) -> np.ndarray:
    """Apply a random orthogonal-ish transformation to change the correlation
    structure between latent dimensions without introducing non-Gaussianity."""
    base = rng.normal(size=(rows, dims))
    if dims < 2:
        return base
    # Build a random matrix with controlled singular values.
    raw = rng.normal(scale=1.0 / np.sqrt(dims), size=(dims, dims))
    # Mix in a diagonal scaling so some dimensions dominate.
    scales = np.exp(rng.uniform(-1.5, 1.5, size=dims))
    transform = raw * scales[np.newaxis, :]
    return base @ transform


# ---------------------------------------------------------------------------
# NONLINEAR — element-wise nonlinearities applied to standard normals
# ---------------------------------------------------------------------------


def _nonlinear_root_cause(
    rng: np.random.Generator,
    rows: int,
    dims: int,
) -> np.ndarray:
    """Apply a random nonlinearity to each latent dimension independently.

    Each dimension draws one of several canonical SCM-style transformations:
    cubic (heavy-tailed symmetric), signed-square, tanh saturation, or
    exponential (positive-skewed).
    """
    base = rng.normal(size=(rows, dims))
    result = np.empty((rows, dims), dtype=np.float64)

    for d in range(dims):
        choice = int(rng.integers(0, 4))
        col = base[:, d]
        if choice == 0:
            # Cubic: preserves symmetry, fattens tails.
            result[:, d] = col**3
        elif choice == 1:
            # Signed square: asymmetric, spreads extremes.
            result[:, d] = np.sign(col) * col**2
        elif choice == 2:
            # Tanh saturation: bounded, compresses extremes.
            scale = float(np.exp(rng.uniform(-1.0, 1.5)))
            result[:, d] = np.tanh(scale * col)
        else:
            # Exponential-like: strong positive skew.
            result[:, d] = np.exp(np.clip(col, -4.0, 4.0))
    return result


# ---------------------------------------------------------------------------
# LOGNORMAL — heavy-tailed, positive-skewed
# ---------------------------------------------------------------------------


def _lognormal_root_cause(
    rng: np.random.Generator,
    rows: int,
    dims: int,
) -> np.ndarray:
    """Log-normal exogenous variables — common in SCMs for variables that
    represent concentrations, counts, or durations."""
    sigma = float(np.exp(rng.uniform(-1.2, 0.5)))
    mu = -0.5 * sigma**2  # centre so median ≈ 1
    raw = rng.normal(loc=mu, scale=sigma, size=(rows, dims))
    return np.exp(raw)


# ---------------------------------------------------------------------------
# GAUSSIAN_MIXTURE — multi-modal latent structure
# ---------------------------------------------------------------------------


def _gaussian_mixture_root_cause(
    rng: np.random.Generator,
    rows: int,
    dims: int,
) -> np.ndarray:
    """Mixture of 2–5 Gaussian components — introduces clustering structure
    into the latent space, common in SCMs with hidden confounders or
    sub-populations."""
    component_count = int(rng.integers(2, min(6, rows // 4 + 1)))
    # Component means spread apart.
    means = rng.normal(scale=float(np.exp(rng.uniform(0.0, 1.5))),
                       size=(component_count, dims))
    # Each component may have different variance.
    stds = np.exp(rng.uniform(-1.0, 0.8, size=(component_count, dims)))
    # Mixing proportions — Dirichlet-like.
    raw_weights = np.exp(rng.uniform(-0.5, 1.0, size=component_count))
    weights = raw_weights / raw_weights.sum()

    assignments = rng.choice(component_count, size=rows, p=weights)
    result = np.empty((rows, dims), dtype=np.float64)
    for k in range(component_count):
        mask = assignments == k
        count = int(np.sum(mask))
        if count == 0:
            continue
        result[mask] = rng.normal(
            loc=means[k],
            scale=stds[k],
            size=(count, dims),
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale


__all__ = ["TableLatent", "LatentRegistry", "generate_latent_registry"]
