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
    signal_scale: float = 1.0,
    activation_scale: float = 1.0,
    output_scale: float = 1.0,
    long_tail_enabled: bool = False,
    long_tail_alpha: float = 1.5,
    mlp_depth: int = 1,
    mlp_hidden_factor: float = 2.0,
    mlp_dropout_rate: float = 0.0,
) -> np.ndarray:
    """Generate one standardized column from a row-aligned causal context."""
    if context.ndim != 2 or context.shape[0] < 1:
        raise ValueError("feature context must be a non-empty matrix")
    if min(noise_scale, signal_scale, activation_scale, output_scale) <= 0:
        raise ValueError("SCM scales must be positive")
    if long_tail_alpha <= 1:
        raise ValueError("long_tail_alpha must be greater than 1")

    if family is FeatureSCMFamily.EXOGENOUS:
        signal = rng.normal(size=context.shape[0])
    elif family is FeatureSCMFamily.LINEAR:
        weights = _coefficient_vector(
            rng,
            context.shape[1],
            scale=signal_scale / np.sqrt(context.shape[1]),
            long_tail_enabled=long_tail_enabled,
            long_tail_alpha=long_tail_alpha,
        )
        signal = context @ weights
    elif family is FeatureSCMFamily.CAM:
        signal = _cam_signal(
            context,
            rng,
            signal_scale=signal_scale,
            activation_scale=activation_scale,
            long_tail_enabled=long_tail_enabled,
            long_tail_alpha=long_tail_alpha,
        )
    elif family is FeatureSCMFamily.MLP:
        signal = _mlp_signal(
            context,
            rng,
            signal_scale=signal_scale,
            activation_scale=activation_scale,
            long_tail_enabled=long_tail_enabled,
            long_tail_alpha=long_tail_alpha,
            depth=mlp_depth,
            hidden_factor=mlp_hidden_factor,
            dropout_rate=mlp_dropout_rate,
        )
    else:
        raise ValueError(f"unsupported feature SCM family: {family}")

    signal = output_scale * signal
    signal = signal + rng.normal(scale=noise_scale, size=len(signal))
    return _standardize(signal)


def _cam_signal(
    context: np.ndarray,
    rng: np.random.Generator,
    *,
    signal_scale: float,
    activation_scale: float,
    long_tail_enabled: bool,
    long_tail_alpha: float,
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
    coefficients = _coefficient_vector(
        rng,
        term_count,
        scale=signal_scale / np.sqrt(term_count),
        long_tail_enabled=long_tail_enabled,
        long_tail_alpha=long_tail_alpha,
    )
    for index, coefficient in zip(indices, coefficients, strict=True):
        scale = float(activation_scale * rng.uniform(0.5, 2.0))
        function = functions[int(rng.integers(0, len(functions)))]
        result += float(coefficient) * function(context[:, index], scale)
    return result


def _mlp_signal(
    context: np.ndarray,
    rng: np.random.Generator,
    *,
    signal_scale: float,
    activation_scale: float,
    long_tail_enabled: bool,
    long_tail_alpha: float,
    depth: int,
    hidden_factor: float,
    dropout_rate: float,
) -> np.ndarray:
    """Multi-layer perceptron with sampled depth, hidden width, and dropout.

    Each hidden layer maps its input through a random weight matrix followed
    by an optional dropout mask and a tanh non-linearity. The final layer
    projects to a scalar output.

    When *depth* is 1 and *dropout_rate* is 0 this reproduces the original
    single-hidden-layer behaviour.
    """
    input_dim = context.shape[1]
    hidden_dim = max(4, min(64, round(input_dim * hidden_factor)))
    x = context

    for layer_index in range(depth):
        fan_in = x.shape[1]
        fan_out = hidden_dim if layer_index < depth - 1 else 1
        weight = _coefficient_vector(
            rng,
            fan_in * fan_out,
            scale=signal_scale / np.sqrt(fan_in),
            long_tail_enabled=long_tail_enabled,
            long_tail_alpha=long_tail_alpha,
        ).reshape(fan_in, fan_out)
        bias = rng.normal(scale=0.35, size=fan_out)
        preactivation = activation_scale * (x @ weight + bias)
        x = np.tanh(np.clip(preactivation, -20.0, 20.0))
        # Apply dropout after every hidden layer except the output layer.
        if dropout_rate > 0 and layer_index < depth - 1:
            mask = rng.random(x.shape[1]) > dropout_rate
            scale = 1.0 / (1.0 - dropout_rate)
            x = x * mask.astype(np.float64) * scale

    return x.ravel()


def _coefficient_vector(
    rng: np.random.Generator,
    size: int,
    *,
    scale: float,
    long_tail_enabled: bool,
    long_tail_alpha: float,
) -> np.ndarray:
    if not long_tail_enabled:
        return rng.normal(scale=scale, size=size)
    magnitudes = 1.0 + rng.pareto(long_tail_alpha, size=size)
    root_mean_square = float(np.sqrt(np.mean(magnitudes**2)))
    magnitudes = magnitudes / max(root_mean_square, 1e-12)
    signs = rng.choice(np.array([-1.0, 1.0]), size=size)
    permutation = rng.permutation(size)
    return scale * signs * magnitudes[permutation]


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale < 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / scale


__all__ = ["generate_feature_signal"]
