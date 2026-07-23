"""RDB-PFN-style hierarchical hyper-prior for existing feature SCMs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class SCMMetaParameters:
    """Database-level hyperparameters shared by all table SCM draws."""

    signal_mean: float
    signal_std: float
    noise_mean: float
    noise_std: float
    activation_log_center: float
    output_log_std: float
    long_tail_enabled: bool
    long_tail_alpha: float
    # MLP structural prior — sampled once per database, each table draws
    # conditional realizations from these ranges.
    mlp_depth_min: int
    mlp_depth_max: int
    mlp_hidden_factor_min: float
    mlp_hidden_factor_max: float
    mlp_dropout_probability: float
    mlp_dropout_rate_min: float
    mlp_dropout_rate_max: float

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        return (
            ("scm_activation_log_center", self.activation_log_center),
            ("scm_long_tail_alpha", self.long_tail_alpha),
            ("scm_long_tail_enabled", float(self.long_tail_enabled)),
            ("scm_mlp_depth_max", float(self.mlp_depth_max)),
            ("scm_mlp_depth_min", float(self.mlp_depth_min)),
            ("scm_mlp_dropout_probability", self.mlp_dropout_probability),
            ("scm_mlp_dropout_rate_max", self.mlp_dropout_rate_max),
            ("scm_mlp_dropout_rate_min", self.mlp_dropout_rate_min),
            ("scm_mlp_hidden_factor_max", self.mlp_hidden_factor_max),
            ("scm_mlp_hidden_factor_min", self.mlp_hidden_factor_min),
            ("scm_noise_mean", self.noise_mean),
            ("scm_noise_std", self.noise_std),
            ("scm_output_log_std", self.output_log_std),
            ("scm_signal_mean", self.signal_mean),
            ("scm_signal_std", self.signal_std),
        )


def sample_scm_meta_parameters(
    rng: np.random.Generator,
    *,
    signal_mean_min: float,
    signal_mean_max: float,
    noise_mean_min: float,
    noise_mean_max: float,
    relative_std_min: float,
    relative_std_max: float,
    activation_scale_min: float,
    activation_scale_max: float,
    output_log_std: float,
    long_tail_probability: float,
    long_tail_alpha_min: float,
    long_tail_alpha_max: float,
    mlp_depth_min: int,
    mlp_depth_max: int,
    mlp_hidden_factor_min: float,
    mlp_hidden_factor_max: float,
    mlp_dropout_probability: float,
    mlp_dropout_rate_min: float,
    mlp_dropout_rate_max: float,
) -> SCMMetaParameters:
    """Sample one database-level hyper-prior realization.

    The mean and relative standard deviation are themselves sampled
    log-uniformly. This mirrors the two-level, log-scaled hyperparameter
    sampling used by RDB-PFN instead of fixing one narrow scale globally.

    MLP structural priors (depth, hidden factor, dropout) are sampled once
    per database so that every table drawing the MLP family within the same
    database shares a consistent architecture distribution.
    """

    signal_mean = _log_uniform(rng, signal_mean_min, signal_mean_max)
    signal_relative_std = _log_uniform(
        rng,
        relative_std_min,
        relative_std_max,
    )
    noise_mean = _log_uniform(rng, noise_mean_min, noise_mean_max)
    noise_relative_std = _log_uniform(
        rng,
        relative_std_min,
        relative_std_max,
    )
    return SCMMetaParameters(
        signal_mean=signal_mean,
        signal_std=signal_mean * signal_relative_std,
        noise_mean=noise_mean,
        noise_std=noise_mean * noise_relative_std,
        activation_log_center=float(
            rng.uniform(log(activation_scale_min), log(activation_scale_max))
        ),
        output_log_std=output_log_std,
        long_tail_enabled=bool(rng.random() < long_tail_probability),
        long_tail_alpha=float(
            rng.uniform(long_tail_alpha_min, long_tail_alpha_max)
        ),
        mlp_depth_min=mlp_depth_min,
        mlp_depth_max=mlp_depth_max,
        mlp_hidden_factor_min=mlp_hidden_factor_min,
        mlp_hidden_factor_max=mlp_hidden_factor_max,
        mlp_dropout_probability=mlp_dropout_probability,
        mlp_dropout_rate_min=mlp_dropout_rate_min,
        mlp_dropout_rate_max=mlp_dropout_rate_max,
    )


def sample_table_scm_parameters(
    rng: np.random.Generator,
    meta: SCMMetaParameters,
    *,
    activation_scale_min: float,
    activation_scale_max: float,
) -> tuple[tuple[str, float], ...]:
    """Sample table-level parameters conditional on one database meta draw.

    For the MLP family this also samples structural realizations (depth,
    hidden factor, dropout) so that every MLP table draws its own
    architecture within the database-level prior.
    """

    signal_scale = _positive_truncated_normal(
        rng,
        mean=meta.signal_mean,
        std=meta.signal_std,
        floor=1e-6,
    )
    noise_scale = _positive_truncated_normal(
        rng,
        mean=meta.noise_mean,
        std=meta.noise_std,
        floor=1e-8,
    )
    activation_scale = float(
        np.clip(
            exp(rng.normal(meta.activation_log_center, 0.5)),
            activation_scale_min,
            activation_scale_max,
        )
    )
    output_scale = float(
        np.clip(exp(rng.normal(0.0, meta.output_log_std)), 1e-3, 1e3)
    )
    # MLP structural realizations — each MLP table draws its own.
    mlp_depth = int(
        rng.integers(meta.mlp_depth_min, meta.mlp_depth_max + 1)
    )
    mlp_hidden_factor = float(
        rng.uniform(meta.mlp_hidden_factor_min, meta.mlp_hidden_factor_max)
    )
    mlp_dropout_enabled = bool(
        rng.random() < meta.mlp_dropout_probability
    )
    mlp_dropout_rate = (
        float(
            rng.uniform(
                meta.mlp_dropout_rate_min,
                meta.mlp_dropout_rate_max,
            )
        )
        if mlp_dropout_enabled
        else 0.0
    )
    return (
        ("activation_scale", activation_scale),
        ("long_tail_alpha", meta.long_tail_alpha),
        ("long_tail_enabled", float(meta.long_tail_enabled)),
        ("mlp_depth", float(mlp_depth)),
        ("mlp_dropout_rate", mlp_dropout_rate),
        ("mlp_hidden_factor", mlp_hidden_factor),
        ("noise_scale", noise_scale),
        ("output_scale", output_scale),
        ("signal_scale", signal_scale),
    )


def _log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    if low <= 0 or high < low:
        raise ValueError("log-uniform bounds must satisfy 0 < low <= high")
    return float(exp(rng.uniform(log(low), log(high))))


def _positive_truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    floor: float,
) -> float:
    if mean <= 0 or std <= 0 or floor <= 0:
        raise ValueError("positive truncated-normal parameters must be positive")
    for _attempt in range(32):
        value = float(rng.normal(mean, std))
        if value > floor:
            return value
    return max(floor, abs(float(rng.normal(mean, std))))


__all__ = [
    "SCMMetaParameters",
    "sample_scm_meta_parameters",
    "sample_table_scm_parameters",
]
