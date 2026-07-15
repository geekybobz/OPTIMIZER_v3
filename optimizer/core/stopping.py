"""Reusable stopping rules for optimizer chunks.

Why this file exists
--------------------
All first-order optimizers need the same basic stopping behavior.  A chunk should
stop after a maximum number of iterations, when a target metric is reached, when
metrics become non-finite, or when progress stalls.  If every optimizer implements
those checks by itself, the library will drift into several subtly different loops.

This module keeps those decisions in one place.  The engine owns a ``StopTracker``
for each chunk and asks it before and after iteration attempts.

How it fits the architecture
----------------------------
- ``engine.py`` uses ``StoppingConfig`` to build a ``StopTracker``.
- future public optimizers expose user-friendly arguments that map into this config.
- logs can record ``StopDecision.reason`` and ``details`` without knowing rule logic.

What this file deliberately does not do
---------------------------------------
It does not evaluate systems, inspect gradients, accept trials, or change controls.
It only answers whether a chunk should stop based on iteration counts and metrics.

Reviewer invariants
-------------------
- ``maxiter`` is interpreted as attempted engine iterations.
- target checks are independent from stall checks.
- stall checks compare against the best seen value of one metric.
- non-finite numeric payloads stop the chunk before target/stall logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StopDecision:
    """Decision returned by stopping checks."""

    stop: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def continue_(cls) -> "StopDecision":
        return cls(stop=False, reason=None, details={})

    @classmethod
    def stop_now(cls, reason: str, **details: Any) -> "StopDecision":
        return cls(stop=True, reason=str(reason), details=dict(details))


@dataclass(frozen=True)
class StoppingConfig:
    """Configuration for engine-level stopping checks."""

    maxiter: int
    target_value: float | None = None
    target_metric: str = "J"
    target_mode: str = "le"
    stall_patience: int | None = None
    stall_tolerance: float = 0.0
    stall_metric: str = "J"
    stall_mode: str = "min"
    check_finite: bool = True

    def __post_init__(self) -> None:
        maxiter = int(self.maxiter)
        if maxiter < 0:
            raise ValueError("maxiter must be >= 0.")
        if self.target_mode not in {"le", "ge"}:
            raise ValueError("target_mode must be 'le' or 'ge'.")
        if self.stall_mode not in {"min", "max"}:
            raise ValueError("stall_mode must be 'min' or 'max'.")
        if self.stall_patience is not None and int(self.stall_patience) < 1:
            raise ValueError("stall_patience must be >= 1 when provided.")
        stall_tolerance = float(self.stall_tolerance)
        if stall_tolerance < 0.0 or not np.isfinite(stall_tolerance):
            raise ValueError("stall_tolerance must be finite and >= 0.")
        object.__setattr__(self, "maxiter", maxiter)
        object.__setattr__(self, "stall_tolerance", stall_tolerance)
        if self.stall_patience is not None:
            object.__setattr__(self, "stall_patience", int(self.stall_patience))
        if self.target_value is not None:
            target = float(self.target_value)
            if not np.isfinite(target):
                raise ValueError("target_value must be finite when provided.")
            object.__setattr__(self, "target_value", target)


def _metric_scalar(metrics: Mapping[str, Any], key: str) -> float:
    """Return one metric as a finite-or-nonfinite scalar float."""

    if key not in metrics:
        raise KeyError(f"metrics do not include {key!r}.")
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f"metric {key!r} must be scalar for stopping checks.")
    return float(value)


def _numeric_payload_is_finite(value: Any) -> bool:
    """Return whether numeric metric payloads are finite.

    Non-numeric metadata is ignored because systems may include labels or mode names
    in their metric dictionaries.
    """

    if isinstance(value, Number):
        return bool(np.isfinite(float(value)))
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return True
        return bool(np.all(np.isfinite(arr)))
    return True


def metrics_are_finite(metrics: Mapping[str, Any]) -> bool:
    """Return whether all numeric metric payloads are finite."""

    return all(_numeric_payload_is_finite(value) for value in metrics.values())


class StopTracker:
    """Stateful stopping helper for one optimizer chunk."""

    # ------------------------------------------------------------------
    # Construction and initial metric state
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: StoppingConfig,
        *,
        initial_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.best_value: float | None = None
        self.stall_count = 0
        if initial_metrics is not None and config.stall_patience is not None:
            self.best_value = _metric_scalar(initial_metrics, config.stall_metric)

    # ------------------------------------------------------------------
    # Public checks used by the engine
    # ------------------------------------------------------------------

    def check_before_iteration(self, iteration: int) -> StopDecision:
        """Stop when the attempted iteration budget has already been consumed."""

        if int(iteration) >= self.config.maxiter:
            return StopDecision.stop_now("maxiter", iteration=int(iteration), maxiter=self.config.maxiter)
        return StopDecision.continue_()

    def check_initial_metrics(self, metrics: Mapping[str, Any]) -> StopDecision:
        """Check initial metrics without charging the stall counter."""

        finite_decision = self._check_finite(metrics)
        if finite_decision.stop:
            return finite_decision
        return self._check_target(metrics)

    def check_metrics(self, metrics: Mapping[str, Any]) -> StopDecision:
        """Check finite, target, and stall rules against current metrics."""

        finite_decision = self._check_finite(metrics)
        if finite_decision.stop:
            return finite_decision

        target_decision = self._check_target(metrics)
        if target_decision.stop:
            return target_decision

        stall_decision = self._check_stall(metrics)
        if stall_decision.stop:
            return stall_decision

        return StopDecision.continue_()

    # ------------------------------------------------------------------
    # Individual stopping rules
    # ------------------------------------------------------------------

    def _check_finite(self, metrics: Mapping[str, Any]) -> StopDecision:
        if self.config.check_finite and not metrics_are_finite(metrics):
            return StopDecision.stop_now("nonfinite", metrics=dict(metrics))
        return StopDecision.continue_()

    def _check_target(self, metrics: Mapping[str, Any]) -> StopDecision:
        if self.config.target_value is None:
            return StopDecision.continue_()

        value = _metric_scalar(metrics, self.config.target_metric)
        target = self.config.target_value
        reached = value <= target if self.config.target_mode == "le" else value >= target
        if reached:
            return StopDecision.stop_now(
                "target",
                metric=self.config.target_metric,
                value=value,
                target=target,
                mode=self.config.target_mode,
            )
        return StopDecision.continue_()

    def _check_stall(self, metrics: Mapping[str, Any]) -> StopDecision:
        if self.config.stall_patience is None:
            return StopDecision.continue_()

        value = _metric_scalar(metrics, self.config.stall_metric)
        if self.best_value is None:
            self.best_value = value
            self.stall_count = 0
            return StopDecision.continue_()

        tolerance = self.config.stall_tolerance
        if self.config.stall_mode == "min":
            improved = value < self.best_value - tolerance
        else:
            improved = value > self.best_value + tolerance

        if improved:
            self.best_value = value
            self.stall_count = 0
            return StopDecision.continue_()

        self.stall_count += 1
        if self.stall_count >= int(self.config.stall_patience):
            return StopDecision.stop_now(
                "stall",
                metric=self.config.stall_metric,
                value=value,
                best=self.best_value,
                patience=self.config.stall_patience,
                tolerance=tolerance,
            )
        return StopDecision.continue_()
