"""Shared helpers for first-order optimizer modules.

Why this file exists
--------------------
The first optimizer implementations all need the same small pieces of engineering:
validate numeric hyperparameters, flatten controls into vector updates, clip a step
without changing its direction, and coerce warmstart inputs into ``WarmStartState``.
Keeping those details here prevents Adam, momentum, and line search from copying the
same defensive code.

How it fits the architecture
----------------------------
- optimizer modules use these helpers to build ``StepProposal`` objects.
- the shared engine remains method-agnostic.
- user-facing functions stay direct and compact.

What this file deliberately does not do
---------------------------------------
It does not choose optimizer directions, accept/reject trials, inspect physical
metrics, or implement schedules.  It only handles reusable data plumbing.

Reviewer invariants
-------------------
- returned flat vectors are independent copies unless explicitly created fresh.
- clipping preserves direction and only changes vector length.
- warmstart conversion always names the target optimizer explicitly.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from optimizer.controls import Controls
from optimizer.state import RunState, WarmStartState


def require_variant(name: str, valid: Iterable[str], *, family: str) -> str:
    """Normalize and validate an optimizer variant name."""

    variant = str(name).lower()
    valid_set = set(valid)
    if variant not in valid_set:
        choices = ", ".join(sorted(valid_set))
        raise ValueError(f"{family} variant must be one of: {choices}.")
    return variant


def require_finite(
    name: str,
    value: float | None,
    *,
    default: float | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    """Validate a scalar hyperparameter and return it as ``float``."""

    if value is None:
        value = default
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def require_probability_like(name: str, value: float) -> float:
    """Validate coefficients such as beta/momentum that must lie in ``[0, 1)``."""

    out = require_finite(name, value)
    if out is None or out < 0.0 or out >= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")
    return out


def flat_state_array(
    state: dict[str, Any],
    key: str,
    *,
    size: int,
    default: float = 0.0,
) -> np.ndarray:
    """Return an optimizer-state vector with the expected flat control size."""

    if key not in state or state[key] is None:
        return np.full(int(size), float(default), dtype=float)
    arr = np.asarray(state[key], dtype=float).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"optimizer_state[{key!r}] has shape {arr.shape}, expected ({size},).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"optimizer_state[{key!r}] contains non-finite values.")
    return arr.copy()


def controls_from_flat_like(reference: Controls, values: np.ndarray, *, name: str) -> Controls:
    """Create controls from a flat vector using ``reference`` layout."""

    flat = np.asarray(values, dtype=reference.spec.dtype).reshape(-1)
    if flat.shape != (reference.spec.size,):
        raise ValueError(f"flat values must have shape ({reference.spec.size},), got {flat.shape}.")
    return Controls.from_flat(reference.spec, flat, name=name)


def clip_vector(vector: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, float, bool]:
    """Clip a vector by L2 norm and report ``(vector, original_norm, clipped)``."""

    raw = np.asarray(vector, dtype=float).reshape(-1)
    original_norm = float(np.linalg.norm(raw))
    if max_norm is None:
        return raw.copy(), original_norm, False
    limit = require_finite("max_step_norm", max_norm, positive=True)
    assert limit is not None
    if original_norm <= limit or original_norm == 0.0:
        return raw.copy(), original_norm, False
    return raw * (limit / original_norm), original_norm, True


def coerce_warmstart(value: Any, *, target_optimizer: str) -> WarmStartState | None:
    """Convert result/state-like warmstart input into ``WarmStartState``."""

    if value is None:
        return None
    if isinstance(value, WarmStartState):
        return value
    if isinstance(value, RunState):
        return WarmStartState.from_run_state(value, target_optimizer=target_optimizer)
    return WarmStartState.from_result(value, target_optimizer=target_optimizer)
