"""Shared machinery for guess generators.

Why this file exists
--------------------
Initial guesses are not cosmetic.  For difficult control problems, the starting pulse
can decide whether an optimizer reaches a useful basin or wastes iterations.  Every
guess family therefore needs the same careful controls: choose channels, set
amplitude, apply optional envelopes, handle endpoints, and normalize the generated
waveform in a predictable way.

How it fits the architecture
----------------------------
- public guess functions accept either a full system or a ``ControlSpec``.
- this module resolves the target layout and builds ``Controls``.
- simple, harmonic, random, and composite guess modules reuse the same amplitude and
  endpoint behavior.

What this file deliberately does not do
---------------------------------------
It does not know physics or objectives, and it does not run optimizers.  It only
turns mathematical waveform templates into validated ``Controls``.

Reviewer invariants
-------------------
- all generated matrices have shape ``(n_controls, control_dim)``.
- non-selected channels remain zero unless a caller explicitly mixes existing
  controls.
- amplitude mapping can be scalar, per-channel mapping, or per-channel sequence.
- scale modes have stable meanings across all guess families.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls


GuessTarget = Any


def resolve_spec(target: GuessTarget) -> ControlSpec:
    """Return a ``ControlSpec`` from either a system or a spec object."""

    if isinstance(target, ControlSpec):
        return target
    hook = getattr(target, "control_spec", None)
    if callable(hook):
        spec = hook()
        if not isinstance(spec, ControlSpec):
            raise TypeError("target.control_spec() must return ControlSpec.")
        return spec
    raise TypeError("guess target must be a ControlSpec or provide control_spec().")


def time_grid(spec: ControlSpec) -> np.ndarray:
    """Return normalized endpoint-inclusive time samples in ``[0, 1]``."""

    if spec.control_dim == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, 1.0, spec.control_dim, dtype=float)


def channel_indices(spec: ControlSpec, channels: str | Iterable[str] | None) -> tuple[int, ...]:
    """Return row indices selected by user-facing channel names."""

    if channels is None:
        return tuple(range(spec.n_controls))
    if isinstance(channels, str):
        names = (channels,)
    else:
        names = tuple(str(channel) for channel in channels)
    return tuple(spec.channel_index(name) for name in names)


def channel_values(
    spec: ControlSpec,
    value: float | Mapping[str, float] | Sequence[float],
    *,
    channels: str | Iterable[str] | None = None,
    default: float = 0.0,
) -> np.ndarray:
    """Return one scalar per control channel.

    Scalars apply to selected channels.  Mappings use channel names.  Sequences can be
    full per-channel arrays or match only the selected channels.
    """

    selected = channel_indices(spec, channels)
    out = np.full(spec.n_controls, float(default), dtype=float)
    if isinstance(value, Mapping):
        for key, scalar in value.items():
            out[spec.channel_index(str(key))] = float(scalar)
    elif isinstance(value, np.ndarray) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        seq = [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
        if len(seq) == spec.n_controls:
            out[:] = seq
        elif len(seq) == len(selected):
            for row, scalar in zip(selected, seq, strict=True):
                out[row] = scalar
        else:
            raise ValueError(
                f"Expected {spec.n_controls} values or {len(selected)} selected-channel values, got {len(seq)}."
            )
    else:
        for row in selected:
            out[row] = float(value)
    if not np.all(np.isfinite(out)):
        raise ValueError("channel values must be finite.")
    return out


def envelope_values(spec: ControlSpec, envelope: str | None) -> np.ndarray:
    """Return a length-``control_dim`` envelope."""

    name = "none" if envelope is None else str(envelope).lower()
    t = time_grid(spec)
    if name in {"none", "free", "flat"}:
        env = np.ones(spec.control_dim, dtype=float)
    elif name == "hann":
        env = np.hanning(spec.control_dim)
    elif name == "sin2":
        env = np.sin(np.pi * t) ** 2
    elif name == "gaussian":
        env = np.exp(-0.5 * ((t - 0.5) / 0.2) ** 2)
    elif name == "smoothstep":
        edge = (3.0 * t * t - 2.0 * t * t * t) * (3.0 * (1.0 - t) ** 2 - 2.0 * (1.0 - t) ** 3)
        max_edge = float(np.max(edge))
        env = edge / max_edge if max_edge > 0.0 else edge
    else:
        raise ValueError("envelope must be one of: none, hann, sin2, gaussian, smoothstep.")
    if env.size and np.max(np.abs(env)) > 0.0:
        env = env / np.max(np.abs(env))
    return env.astype(float, copy=False)


def apply_endpoint(matrix: np.ndarray, *, endpoint: str | None, rows: tuple[int, ...]) -> np.ndarray:
    """Apply endpoint policy to selected channels."""

    policy = "free" if endpoint is None else str(endpoint).lower()
    out = np.asarray(matrix, dtype=float).copy()
    if policy in {"free", "hold"}:
        return out
    if policy == "zero":
        if out.shape[1] >= 1:
            out[list(rows), 0] = 0.0
            out[list(rows), -1] = 0.0
        return out
    raise ValueError("endpoint must be one of: free, hold, zero.")


def _scaled_channel(
    raw: np.ndarray,
    *,
    amplitude: float,
    scale: str,
    dt: float,
) -> np.ndarray:
    """Scale one raw channel according to the requested amplitude convention."""

    raw = np.asarray(raw, dtype=float)
    mode = str(scale).lower()
    if mode == "none":
        return amplitude * raw
    if mode == "max_abs":
        denom = float(np.max(np.abs(raw))) if raw.size else 0.0
    elif mode == "l2":
        denom = float(np.linalg.norm(raw))
    elif mode == "energy":
        denom = float(np.sqrt(dt * np.sum(raw * raw)))
    else:
        raise ValueError("scale must be one of: none, max_abs, l2, energy.")
    if denom <= np.finfo(float).tiny:
        return np.zeros_like(raw)
    return (float(amplitude) / denom) * raw


def finalize_guess(
    spec: ControlSpec,
    raw: np.ndarray,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    envelope: str | None = None,
    endpoint: str | None = "free",
    scale: str = "max_abs",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Convert a raw unit waveform into a validated ``Controls`` guess."""

    raw_matrix = np.asarray(raw, dtype=float)
    if raw_matrix.shape != spec.shape:
        raise ValueError(f"raw guess must have shape {spec.shape}, got {raw_matrix.shape}.")
    if not np.all(np.isfinite(raw_matrix)):
        raise ValueError("raw guess contains non-finite values.")

    selected = channel_indices(spec, channels)
    amplitudes = channel_values(spec, amplitude, channels=channels, default=0.0)
    offsets = channel_values(spec, offset, channels=channels, default=0.0)
    env = envelope_values(spec, envelope)
    dt = 1.0 if spec.dt is None else float(spec.dt)

    matrix = np.zeros(spec.shape, dtype=float)
    for row in selected:
        shaped = raw_matrix[row] * env
        matrix[row] = offsets[row] + _scaled_channel(
            shaped,
            amplitude=amplitudes[row],
            scale=scale,
            dt=dt,
        )
    matrix = apply_endpoint(matrix, endpoint=endpoint, rows=selected)
    return Controls.from_matrix(spec, matrix, name=name or "guess", meta=dict(meta or {}), copy=False)


def unit_matrix(spec: ControlSpec, value: float = 0.0) -> np.ndarray:
    """Return a raw matrix filled with a scalar."""

    return np.full(spec.shape, float(value), dtype=float)


def validate_positive(name: str, value: float) -> float:
    """Validate positive finite scalar parameters."""

    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return out
