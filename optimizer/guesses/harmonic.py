"""Smooth harmonic and localized waveform guesses.

Why this file exists
--------------------
Smooth structured guesses are often more useful than raw random controls.  Sine,
cosine, sinc, Gaussian, and Fourier templates provide controllable amplitude,
frequency content, phase, width, and endpoint behavior.  They give optimizers a
reasonable starting shape without committing to a physical objective.

How it fits the architecture
----------------------------
- public functions accept a system or ``ControlSpec``.
- common amplitude, scale, envelope, endpoint, and channel handling comes from
  ``guesses.base``.
- random Fourier starts live in ``guesses.random``; this module is deterministic
  unless coefficients/phases supplied by the caller are random.

Reviewer invariants
-------------------
- generated raw templates are dimensionless; ``finalize_guess`` applies amplitude.
- frequency is measured in cycles over the normalized control interval.
- Fourier mode arrays are validated before synthesis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.guesses.base import (
    channel_values,
    finalize_guess,
    resolve_spec,
    time_grid,
    validate_positive,
)


def _wave_parameter(
    target: Any,
    value: float | Mapping[str, float] | Sequence[float],
    *,
    channels: str | Iterable[str] | None,
) -> np.ndarray:
    """Return one waveform parameter per channel."""

    return channel_values(resolve_spec(target), value, channels=channels, default=0.0)


def sine_guess(
    target: Any,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    frequency: float | Mapping[str, float] | Sequence[float] = 1.0,
    phase: float | Mapping[str, float] | Sequence[float] = 0.0,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a sine-wave guess with channel-wise frequency and phase."""

    spec = resolve_spec(target)
    t = time_grid(spec)
    frequencies = channel_values(spec, frequency, channels=channels, default=0.0)
    phases = channel_values(spec, phase, channels=channels, default=0.0)
    raw = np.zeros(spec.shape, dtype=float)
    for row in range(spec.n_controls):
        raw[row] = np.sin(2.0 * np.pi * frequencies[row] * t + phases[row])
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "sine_guess",
        meta={"guess": "sine", **dict(meta or {})},
    )


def cosine_guess(
    target: Any,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    frequency: float | Mapping[str, float] | Sequence[float] = 1.0,
    phase: float | Mapping[str, float] | Sequence[float] = 0.0,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a cosine-wave guess with channel-wise frequency and phase."""

    spec = resolve_spec(target)
    t = time_grid(spec)
    frequencies = channel_values(spec, frequency, channels=channels, default=0.0)
    phases = channel_values(spec, phase, channels=channels, default=0.0)
    raw = np.zeros(spec.shape, dtype=float)
    for row in range(spec.n_controls):
        raw[row] = np.cos(2.0 * np.pi * frequencies[row] * t + phases[row])
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "cosine_guess",
        meta={"guess": "cosine", **dict(meta or {})},
    )


def gaussian_guess(
    target: Any,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    center: float | Mapping[str, float] | Sequence[float] = 0.5,
    width: float | Mapping[str, float] | Sequence[float] = 0.15,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a localized Gaussian pulse guess."""

    spec = resolve_spec(target)
    t = time_grid(spec)
    centers = channel_values(spec, center, channels=channels, default=0.5)
    widths = channel_values(spec, width, channels=channels, default=0.15)
    raw = np.zeros(spec.shape, dtype=float)
    for row in range(spec.n_controls):
        validate_positive("width", widths[row])
        raw[row] = np.exp(-0.5 * ((t - centers[row]) / widths[row]) ** 2)
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "gaussian_guess",
        meta={"guess": "gaussian", **dict(meta or {})},
    )


def sinc_guess(
    target: Any,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    center: float | Mapping[str, float] | Sequence[float] = 0.5,
    width: float | Mapping[str, float] | Sequence[float] = 4.0,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a centered sinc pulse with controllable side-lobe width."""

    spec = resolve_spec(target)
    t = time_grid(spec)
    centers = channel_values(spec, center, channels=channels, default=0.5)
    widths = channel_values(spec, width, channels=channels, default=4.0)
    raw = np.zeros(spec.shape, dtype=float)
    for row in range(spec.n_controls):
        validate_positive("width", widths[row])
        raw[row] = np.sinc(widths[row] * (t - centers[row]))
    return finalize_guess(
        spec,
        raw,
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "sinc_guess",
        meta={"guess": "sinc", **dict(meta or {})},
    )


def _coefficient_matrix(
    spec: Any,
    coefficients: Any,
    *,
    modes: int,
    decay: str,
) -> np.ndarray:
    """Return coefficient matrix with shape ``(n_controls, modes)``."""

    if coefficients is None:
        k = np.arange(1, int(modes) + 1, dtype=float)
        if decay == "flat":
            base = np.ones_like(k)
        elif decay == "1/k":
            base = 1.0 / k
        elif decay == "1/k2":
            base = 1.0 / (k * k)
        else:
            raise ValueError("decay must be one of: flat, 1/k, 1/k2.")
        return np.tile(base.reshape(1, -1), (spec.n_controls, 1))

    if isinstance(coefficients, Mapping):
        out = np.zeros((spec.n_controls, int(modes)), dtype=float)
        for key, values in coefficients.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape != (int(modes),):
                raise ValueError(f"coefficients for {key!r} must have length {modes}.")
            out[spec.channel_index(str(key))] = arr
        return out

    arr = np.asarray(coefficients, dtype=float)
    if arr.shape == (int(modes),):
        return np.tile(arr.reshape(1, -1), (spec.n_controls, 1))
    if arr.shape != (spec.n_controls, int(modes)):
        raise ValueError(f"coefficients must have shape ({modes},) or {(spec.n_controls, int(modes))}.")
    return arr.copy()


def fourier_guess(
    target: Any,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    modes: int = 5,
    frequency_base: float = 1.0,
    coefficients: Any = None,
    phases: Any = 0.0,
    decay: str = "1/k",
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a low-frequency Fourier-series guess."""

    spec = resolve_spec(target)
    if int(modes) < 1:
        raise ValueError("modes must be >= 1.")
    validate_positive("frequency_base", frequency_base)
    t = time_grid(spec)
    k = np.arange(1, int(modes) + 1, dtype=float)
    coeffs = _coefficient_matrix(spec, coefficients, modes=int(modes), decay=str(decay).lower())
    if isinstance(phases, Mapping):
        phase_matrix = np.zeros((spec.n_controls, int(modes)), dtype=float)
        for key, values in phases.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape == ():
                phase_matrix[spec.channel_index(str(key)), :] = float(arr)
            elif arr.shape == (int(modes),):
                phase_matrix[spec.channel_index(str(key)), :] = arr
            else:
                raise ValueError(f"phases for {key!r} must be scalar or length {modes}.")
    else:
        arr = np.asarray(phases, dtype=float)
        if arr.shape == ():
            phase_matrix = np.full((spec.n_controls, int(modes)), float(arr), dtype=float)
        elif arr.shape == (int(modes),):
            phase_matrix = np.tile(arr.reshape(1, -1), (spec.n_controls, 1))
        elif arr.shape == (spec.n_controls, int(modes)):
            phase_matrix = arr.copy()
        else:
            raise ValueError("phases must be scalar, length modes, or shape (n_controls, modes).")

    raw = np.zeros(spec.shape, dtype=float)
    for row in range(spec.n_controls):
        angles = 2.0 * np.pi * float(frequency_base) * k[:, None] * t[None, :] + phase_matrix[row, :, None]
        raw[row] = np.sum(coeffs[row, :, None] * np.sin(angles), axis=0)

    return finalize_guess(
        spec,
        raw,
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        envelope=envelope,
        endpoint=endpoint,
        scale=scale,
        name=name or "fourier_guess",
        meta={
            "guess": "fourier",
            "modes": int(modes),
            "frequency_base": float(frequency_base),
            "decay": str(decay).lower(),
            **dict(meta or {}),
        },
    )
