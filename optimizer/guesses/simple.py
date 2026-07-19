"""Simple deterministic guess generators.

Why this file exists
--------------------
The first guesses users reach for should be explicit and predictable: constant
channels and ramps.  These are not sophisticated, but they are essential for checking
objectives and establishing baseline optimizer behavior before trying random or
Fourier starts.

How it fits the architecture
----------------------------
- public functions accept a system or ``ControlSpec``.
- generated values are returned as the standard ``Controls`` container.
- amplitude/channel/endpoint behavior is delegated to ``guesses.base``.

Reviewer invariants
-------------------
- constant and ramp guesses still support selected-channel generation.
- no random state is used in this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.guesses.base import finalize_guess, resolve_spec, time_grid, unit_matrix


def constant_guess(
    target: Any,
    *,
    value: float | Mapping[str, float] | Sequence[float] = 0.0,
    channels: str | Iterable[str] | None = None,
    endpoint: str | None = "free",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a selected-channel constant guess."""

    spec = resolve_spec(target)
    raw = unit_matrix(spec, 1.0)
    return finalize_guess(
        spec,
        raw,
        amplitude=value,
        offset=0.0,
        channels=channels,
        endpoint=endpoint,
        scale="max_abs",
        name=name or "constant_guess",
        meta={"guess": "constant", **dict(meta or {})},
    )


def ramp_guess(
    target: Any,
    *,
    start: float | Mapping[str, float] | Sequence[float] = 0.0,
    stop: float | Mapping[str, float] | Sequence[float] = 1.0,
    channels: str | Iterable[str] | None = None,
    kind: str = "linear",
    endpoint: str | None = "free",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a smooth monotone ramp between per-channel start/stop values."""

    spec = resolve_spec(target)
    t = time_grid(spec)
    ramp_kind = str(kind).lower()
    if ramp_kind == "linear":
        profile = t
    elif ramp_kind == "quadratic":
        profile = t * t
    elif ramp_kind == "smoothstep":
        profile = 3.0 * t * t - 2.0 * t * t * t
    else:
        raise ValueError("kind must be one of: linear, quadratic, smoothstep.")

    # Build the ramp explicitly rather than using ``finalize_guess`` amplitude
    # scaling, because start/stop are physical channel values, not a normalized
    # waveform amplitude.
    from optimizer.guesses.base import apply_endpoint, channel_indices, channel_values

    selected = channel_indices(spec, channels)
    starts = channel_values(spec, start, channels=channels, default=0.0)
    stops = channel_values(spec, stop, channels=channels, default=0.0)
    matrix = np.zeros(spec.shape, dtype=float)
    for row in selected:
        matrix[row] = starts[row] + (stops[row] - starts[row]) * profile
    matrix = apply_endpoint(matrix, endpoint=endpoint, rows=selected)
    return Controls.from_matrix(
        spec,
        matrix,
        name=name or "ramp_guess",
        meta={"guess": "ramp", "kind": ramp_kind, **dict(meta or {})},
        copy=False,
    )
