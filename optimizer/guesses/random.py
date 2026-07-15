"""Random and smooth-random guess generators.

Why this file exists
--------------------
Multiple starts are essential when a difficult optimization landscape has many local
basins.  Raw random controls are useful for stress testing, but smooth random and
random Fourier guesses are usually better starting points because they avoid excessive
high-frequency content while still exploring different pulse shapes.

How it fits the architecture
----------------------------
- generated random matrices are passed through the same amplitude/envelope/scaling
  machinery as deterministic guesses.
- seeds make experiments reproducible.
- random Fourier guesses reuse the deterministic Fourier synthesis pathway.

Reviewer invariants
-------------------
- all random draws come from ``numpy.random.default_rng``.
- a two-number amplitude tuple means a random per-channel amplitude range.
- smooth random guesses use vectorized convolution per channel and then normalize.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.guesses.base import channel_indices, finalize_guess, resolve_spec, validate_positive
from optimizer.guesses.harmonic import fourier_guess


def _is_numeric_pair(value: Any) -> bool:
    """Return whether ``value`` should be treated as an amplitude range."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 2:
        return False
    try:
        float(value[0])
        float(value[1])
        return True
    except (TypeError, ValueError):
        return False


def _random_amplitudes(
    spec: Any,
    amplitude: Any,
    *,
    channels: str | Iterable[str] | None,
    rng: np.random.Generator,
) -> Any:
    """Return amplitude argument, sampling ranges for random guess families."""

    if not _is_numeric_pair(amplitude):
        return amplitude
    low = float(amplitude[0])
    high = float(amplitude[1])
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("amplitude range must be finite and ordered as (low, high).")
    values = np.zeros(spec.n_controls, dtype=float)
    for row in channel_indices(spec, channels):
        values[row] = rng.uniform(low, high)
    return values


def random_guess(
    target: Any,
    *,
    amplitude: float | tuple[float, float] | Mapping[str, float] | Sequence[float] = 1.0,
    distribution: str = "uniform",
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    seed: int | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a raw random control guess."""

    spec = resolve_spec(target)
    rng = np.random.default_rng(seed)
    kind = str(distribution).lower()
    if kind == "uniform":
        raw = rng.uniform(-1.0, 1.0, size=spec.shape)
    elif kind == "normal":
        raw = rng.normal(size=spec.shape)
    elif kind == "rademacher":
        raw = rng.choice(np.array([-1.0, 1.0]), size=spec.shape)
    else:
        raise ValueError("distribution must be one of: uniform, normal, rademacher.")
    amplitudes = _random_amplitudes(spec, amplitude, channels=channels, rng=rng)
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitudes,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "random_guess",
        meta={"guess": "random", "distribution": kind, "seed": seed, **dict(meta or {})},
    )


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a normalized 1D Gaussian smoothing kernel."""

    sigma = validate_positive("correlation", sigma)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


def _smooth_rows(matrix: np.ndarray, *, sigma: float) -> np.ndarray:
    """Smooth each channel row using edge padding and 1D convolution."""

    kernel = _gaussian_kernel(sigma)
    pad = kernel.size // 2
    out = np.zeros_like(matrix, dtype=float)
    for row in range(matrix.shape[0]):
        padded = np.pad(matrix[row], pad_width=pad, mode="edge")
        out[row] = np.convolve(padded, kernel, mode="valid")
    return out


def random_smooth_guess(
    target: Any,
    *,
    amplitude: float | tuple[float, float] | Mapping[str, float] | Sequence[float] = 1.0,
    correlation: float = 0.15,
    distribution: str = "normal",
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    seed: int | None = None,
    envelope: str | None = "hann",
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a random guess smoothed by a Gaussian correlation kernel."""

    spec = resolve_spec(target)
    rng = np.random.default_rng(seed)
    kind = str(distribution).lower()
    if kind == "uniform":
        raw = rng.uniform(-1.0, 1.0, size=spec.shape)
    elif kind == "normal":
        raw = rng.normal(size=spec.shape)
    else:
        raise ValueError("distribution must be one of: uniform, normal.")
    sigma = float(correlation)
    if 0.0 < sigma <= 1.0:
        sigma = max(1.0, sigma * spec.control_dim)
    raw = _smooth_rows(raw, sigma=sigma)
    amplitudes = _random_amplitudes(spec, amplitude, channels=channels, rng=rng)
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitudes,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "random_smooth_guess",
        meta={
            "guess": "random_smooth",
            "distribution": kind,
            "correlation": float(correlation),
            "seed": seed,
            **dict(meta or {}),
        },
    )


def random_fourier_guess(
    target: Any,
    *,
    amplitude: float | tuple[float, float] | Mapping[str, float] | Sequence[float] = 1.0,
    modes: int = 5,
    frequency_base: float = 1.0,
    coefficient_scale: float = 1.0,
    decay: str = "1/k",
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    seed: int | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a random low-frequency Fourier-series guess."""

    spec = resolve_spec(target)
    if int(modes) < 1:
        raise ValueError("modes must be >= 1.")
    validate_positive("coefficient_scale", coefficient_scale)
    rng = np.random.default_rng(seed)
    k = np.arange(1, int(modes) + 1, dtype=float)
    if decay == "flat":
        decay_weights = np.ones_like(k)
    elif decay == "1/k":
        decay_weights = 1.0 / k
    elif decay == "1/k2":
        decay_weights = 1.0 / (k * k)
    else:
        raise ValueError("decay must be one of: flat, 1/k, 1/k2.")
    coefficients = float(coefficient_scale) * rng.normal(size=(spec.n_controls, int(modes))) * decay_weights
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(spec.n_controls, int(modes)))
    amplitudes = _random_amplitudes(spec, amplitude, channels=channels, rng=rng)
    guess = fourier_guess(
        spec,
        amplitude=amplitudes,
        modes=int(modes),
        frequency_base=frequency_base,
        coefficients=coefficients,
        phases=phases,
        decay="flat",
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "random_fourier_guess",
        meta={
            "guess": "random_fourier",
            "modes": int(modes),
            "frequency_base": float(frequency_base),
            "coefficient_scale": float(coefficient_scale),
            "decay": decay,
            "seed": seed,
            **dict(meta or {}),
        },
    )
    return guess
