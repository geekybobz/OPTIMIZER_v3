"""Reusable step-size schedule objects.

Why this file exists
--------------------
Line search, momentum, Adam, and repair loops all need simple step-size policies.
Hard-coding those policies inside each algorithm makes behavior harder to review and
harder to keep consistent.  These schedule objects are small, deterministic training
aids that can be used by optimizers now and by curriculum/mode code later.

How it fits the architecture
----------------------------
- schedules own scalar step-size policy only.
- optimizers still own update directions and acceptance.
- run state stores the actual current step size between chunks.

What this file deliberately does not do
---------------------------------------
It does not evaluate systems, inspect gradients, or enforce metric guards.

Reviewer invariants
-------------------
- every returned step size is finite and inside optional min/max limits.
- adaptive schedules update only from accepted/rejected outcome booleans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite(name: str, value: float, *, positive: bool = True) -> float:
    """Validate a scalar schedule parameter."""

    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return out


@dataclass(frozen=True)
class ConstantSchedule:
    """Schedule that always returns the same step size."""

    value: float

    def __post_init__(self) -> None:
        _finite("value", self.value)

    def initial(self) -> float:
        """Return the initial step size."""

        return float(self.value)

    def update(
        self,
        current: float | None = None,
        *,
        accepted: bool | None = None,
        iteration: int | None = None,
    ) -> float:
        """Return the unchanged step size."""

        del current, accepted, iteration
        return float(self.value)


@dataclass(frozen=True)
class AdaptiveStepSchedule:
    """Shrink/grow step-size policy for training loops."""

    initial_step: float = 1.0e-3
    shrink: float = 0.5
    grow: float = 1.0
    min_step: float = 1.0e-12
    max_step: float | None = None

    def __post_init__(self) -> None:
        _finite("initial_step", self.initial_step)
        _finite("shrink", self.shrink)
        _finite("grow", self.grow)
        _finite("min_step", self.min_step)
        if not 0.0 < float(self.shrink) < 1.0:
            raise ValueError("shrink must satisfy 0 < shrink < 1.")
        if self.max_step is not None:
            _finite("max_step", self.max_step)
            if float(self.max_step) < float(self.min_step):
                raise ValueError("max_step must be >= min_step.")

    def _clamp(self, value: float) -> float:
        """Apply configured min/max bounds to a scalar step size."""

        out = max(float(self.min_step), float(value))
        if self.max_step is not None:
            out = min(out, float(self.max_step))
        return out

    def initial(self) -> float:
        """Return clamped initial step size."""

        return self._clamp(float(self.initial_step))

    def update(
        self,
        current: float | None = None,
        *,
        accepted: bool | None = None,
        iteration: int | None = None,
    ) -> float:
        """Return next step size after accepted/rejected outcome."""

        del iteration
        base = self.initial() if current is None else self._clamp(float(current))
        if accepted is None:
            return base
        if bool(accepted):
            return self._clamp(base * float(self.grow))
        return self._clamp(base * float(self.shrink))


@dataclass(frozen=True)
class ExponentialDecaySchedule:
    """Multiply the initial step by ``decay`` after each accepted iteration."""

    initial_step: float = 1.0e-3
    decay: float = 0.99
    min_step: float = 1.0e-12

    def __post_init__(self) -> None:
        _finite("initial_step", self.initial_step)
        _finite("decay", self.decay)
        _finite("min_step", self.min_step)
        if not 0.0 < float(self.decay) <= 1.0:
            raise ValueError("decay must satisfy 0 < decay <= 1.")

    def initial(self) -> float:
        return max(float(self.min_step), float(self.initial_step))

    def update(
        self,
        current: float | None = None,
        *,
        accepted: bool | None = None,
        iteration: int | None = None,
    ) -> float:
        del current
        index = max(0, int(iteration or 0))
        if accepted is False:
            index = max(0, index - 1)
        return max(float(self.min_step), float(self.initial_step) * float(self.decay) ** index)


@dataclass(frozen=True)
class LinearWarmupSchedule:
    """Increase a step linearly from ``initial_step`` to ``target_step``."""

    initial_step: float
    target_step: float
    warmup_steps: int

    def __post_init__(self) -> None:
        _finite("initial_step", self.initial_step)
        _finite("target_step", self.target_step)
        if int(self.warmup_steps) < 1:
            raise ValueError("warmup_steps must be >= 1.")

    def initial(self) -> float:
        return float(self.initial_step)

    def update(
        self,
        current: float | None = None,
        *,
        accepted: bool | None = None,
        iteration: int | None = None,
    ) -> float:
        if accepted is False:
            return self.initial() if current is None else float(current)
        fraction = min(1.0, max(0.0, float(iteration or 0) / int(self.warmup_steps)))
        return float(self.initial_step) + fraction * (float(self.target_step) - float(self.initial_step))


@dataclass(frozen=True)
class CosineDecaySchedule:
    """Cosine decay from ``initial_step`` to ``min_step`` over a fixed horizon."""

    initial_step: float
    total_steps: int
    min_step: float = 0.0

    def __post_init__(self) -> None:
        _finite("initial_step", self.initial_step)
        _finite("min_step", self.min_step, positive=False)
        if float(self.min_step) < 0.0 or float(self.min_step) > float(self.initial_step):
            raise ValueError("min_step must satisfy 0 <= min_step <= initial_step.")
        if int(self.total_steps) < 1:
            raise ValueError("total_steps must be >= 1.")

    def initial(self) -> float:
        return float(self.initial_step)

    def update(
        self,
        current: float | None = None,
        *,
        accepted: bool | None = None,
        iteration: int | None = None,
    ) -> float:
        if accepted is False:
            return self.initial() if current is None else float(current)
        progress = min(1.0, max(0.0, float(iteration or 0) / int(self.total_steps)))
        weight = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(self.min_step) + weight * (float(self.initial_step) - float(self.min_step))
