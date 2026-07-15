"""Composite and restart guess utilities.

Why this file exists
--------------------
Useful guesses are often built from other guesses: a structured Gaussian plus a small
random Fourier perturbation, a previous best pulse rescaled to a new amplitude, or a
weighted mix of several candidates.  These operations are not optimizers; they are
ways to prepare starting controls for optimizer comparisons and restarts.

How it fits the architecture
----------------------------
- functions operate directly on ``Controls``.
- perturbation uses the same random guess families as fresh starts.
- scaling reuses clear amplitude semantics for existing controls.

Reviewer invariants
-------------------
- mixed controls must share the same ``ControlSpec``.
- perturbations preserve the base control layout.
- scaling never mutates the original controls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.guesses.base import finalize_guess
from optimizer.guesses.harmonic import fourier_guess
from optimizer.guesses.random import random_fourier_guess, random_guess, random_smooth_guess


def scale_guess(
    controls: Controls,
    *,
    amplitude: float | Mapping[str, float] | Sequence[float] = 1.0,
    offset: float | Mapping[str, float] | Sequence[float] = 0.0,
    scale: str = "max_abs",
    channels: str | Sequence[str] | None = None,
    endpoint: str | None = "free",
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a rescaled copy of existing controls."""

    return finalize_guess(
        controls.spec,
        controls.as_matrix(copy=False),
        amplitude=amplitude,
        offset=offset,
        channels=channels,
        endpoint=endpoint,
        scale=scale,
        name=name or f"scaled_{controls.name or 'controls'}",
        meta={"guess": "scale", "source": controls.name, **dict(meta or {})},
    )


def mix_guess(
    guesses: Sequence[Controls] | Controls,
    *more_guesses: Controls,
    weights: Sequence[float] | None = None,
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Controls:
    """Return a weighted mix of compatible control guesses."""

    if isinstance(guesses, Controls):
        items = (guesses, *more_guesses)
    else:
        items = tuple(guesses) + tuple(more_guesses)
    if not items:
        raise ValueError("mix_guess requires at least one Controls object.")
    spec = items[0].spec
    for item in items:
        if not isinstance(item, Controls):
            raise TypeError("mix_guess inputs must be Controls objects.")
        if item.spec.keys != spec.keys or item.spec.control_dim != spec.control_dim:
            raise ValueError("all guesses must have matching control specs.")
    if weights is None:
        weight_array = np.full(len(items), 1.0 / len(items), dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if weight_array.shape != (len(items),):
            raise ValueError(f"weights must have length {len(items)}.")
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be finite.")
    matrix = np.zeros(spec.shape, dtype=float)
    for weight, item in zip(weight_array, items, strict=True):
        matrix += float(weight) * item.as_matrix(copy=False)
    return Controls.from_matrix(
        spec,
        matrix,
        name=name or "mixed_guess",
        meta={"guess": "mix", "sources": [item.name for item in items], **dict(meta or {})},
        copy=False,
    )


def perturb_guess(
    controls: Controls,
    *,
    amplitude: float | tuple[float, float] | Mapping[str, float] | Sequence[float] = 0.01,
    kind: str = "random_fourier",
    seed: int | None = None,
    channels: str | Sequence[str] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> Controls:
    """Return existing controls plus a generated perturbation."""

    perturb_kind = str(kind).lower()
    if perturb_kind == "random_fourier":
        noise = random_fourier_guess(
            controls.spec,
            amplitude=amplitude,
            seed=seed,
            channels=channels,
            name="perturbation_random_fourier",
            **kwargs,
        )
    elif perturb_kind == "random_smooth":
        noise = random_smooth_guess(
            controls.spec,
            amplitude=amplitude,
            seed=seed,
            channels=channels,
            name="perturbation_random_smooth",
            **kwargs,
        )
    elif perturb_kind == "random":
        noise = random_guess(
            controls.spec,
            amplitude=amplitude,
            seed=seed,
            channels=channels,
            name="perturbation_random",
            **kwargs,
        )
    elif perturb_kind == "fourier":
        noise = fourier_guess(
            controls.spec,
            amplitude=amplitude,
            channels=channels,
            name="perturbation_fourier",
            **kwargs,
        )
    else:
        raise ValueError("kind must be one of: random_fourier, random_smooth, random, fourier.")

    return Controls.from_matrix(
        controls.spec,
        controls.as_matrix(copy=False) + noise.as_matrix(copy=False),
        name=name or f"perturbed_{controls.name or 'controls'}",
        meta={
            "guess": "perturb",
            "source": controls.name,
            "perturbation_kind": perturb_kind,
            "seed": seed,
        },
        copy=False,
    )
