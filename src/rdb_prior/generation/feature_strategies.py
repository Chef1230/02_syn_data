"""Numerical structural causal mechanisms for anonymous columns."""

from __future__ import annotations

import numpy as np

from rdb_prior.instance.plan import FeatureSCMFamily


def generate_feature_signal(
    family: FeatureSCMFamily,
    context: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_scale: float,
) -> np.ndarray:
    """Generate one standardized column from a row-aligned causal context."""
    if context.ndim != 2 or context.shape[0] < 1:
        raise ValueError("feature context must be a non-empty matrix")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")

    if family is FeatureSCMFamily.EXOGENOUS:
        signal = rng.normal(size=context.shape[0])
    elif family is FeatureSCMFamily.LINEAR:
        weights = rng.normal(
            scale=1.0 / np.sqrt(context.shape[1]),
            size=context.shape[1],
        )
        signal = context @ weights
    elif family is FeatureSCMFamily.CAM:
        signal = _cam_signal(context, rng)
    elif family is FeatureSCMFamily.MLP:
        signal = _mlp_signal(context, rng)
    else:
        raise ValueError(f"unsupported feature SCM family: {family}")

    signal = signal + rng.normal(scale=noise_scale, size=len(signal))
    return _standardize(signal)


def _cam_signal(
    context: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    term_count = min(context.shape[1], int(rng.integers(2, 6)))
    indices = rng.choice(context.shape[1], size=term_count, replace=False)
    result = np.zeros(context.shape[0], dtype=np.float64)
    functions = (
        lambda value, scale: np.sin(scale * value),
        lambda value, scale: np.tanh(scale * value),
        lambda value, scale: np.sign(value) * np.sqrt(np.abs(value) + 1e-8),
        lambda value, scale: np.clip(value, -2.5, 2.5) ** 2,
    )
    for index in indices:
        coefficient = float(rng.normal())
        scale = float(rng.uniform(0.5, 2.0))
        function = functions[int(rng.integers(0, len(functions)))]
        result += coefficient * function(context[:, index], scale)
    return result


def _mlp_signal(
    context: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    hidden = min(16, max(4, context.shape[1] * 2))
    first = rng.normal(
        scale=1.0 / np.sqrt(context.shape[1]),
        size=(context.shape[1], hidden),
    )
    bias = rng.normal(scale=0.35, size=hidden)
    second = rng.normal(scale=1.0 / np.sqrt(hidden), size=hidden)
    return np.tanh(context @ first + bias) @ second


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale < 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / scale


__all__ = ["generate_feature_signal"]
