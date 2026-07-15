"""Validated system evaluation helpers for the shared optimizer engine.

Why this file exists
--------------------
The physical systems in OPTIMIZER v3 own their analytical objective and gradient.
Every optimizer still needs the same surrounding safety work: validate that controls
match the system, normalize metric dictionaries, reject non-finite results, and avoid
repeating an expensive propagation when the exact same controls are evaluated twice.

This module centralizes that boundary in ``SystemEvaluator``.  It is intentionally
small: optimizers should call the engine, the engine should call this evaluator, and
the system should remain the only place where physics is computed.

How it fits the architecture
----------------------------
- ``system.py`` defines the required hooks and validation helpers.
- ``result.py`` defines ``Evaluation`` snapshots.
- ``engine.py`` uses this evaluator for current/trial metrics and gradients.
- future diagnostics can reuse the same cache behavior for batch probes.

What this file deliberately does not do
---------------------------------------
It does not choose steps, accept or reject trials, update run state, or log anything.
Those decisions belong to the engine.  This file only wraps expensive calls with
consistent validation and lightweight caching.

Reviewer invariants
-------------------
- a cached evaluation is keyed by control layout and numeric content.
- ``evaluate`` returns a validated ``Evaluation`` or raises clearly.
- ``try_evaluate`` and ``try_gradient`` convert failures into structured outcomes
  when the engine needs to stop gracefully.
- gradients are never finite-difference fallbacks here; they come from the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.result import Evaluation
from optimizer.system import OptimizerSystem, evaluate_system, gradient_system, require_system


ControlCacheKey = tuple[tuple[str, ...], int, str, tuple[int, int], bytes]


@dataclass(frozen=True)
class EvaluationOutcome:
    """Structured result for an evaluation attempt that may fail."""

    evaluation: Evaluation | None
    ok: bool
    reason: str
    error: str | None = None


@dataclass(frozen=True)
class GradientOutcome:
    """Structured result for a gradient attempt that may fail."""

    gradient: Controls | None
    ok: bool
    reason: str
    error: str | None = None


def controls_cache_key(controls: Controls) -> ControlCacheKey:
    """Return a deterministic key for the exact numeric control content."""

    matrix = np.ascontiguousarray(controls.as_matrix(copy=False))
    return (
        tuple(controls.spec.keys),
        int(controls.spec.control_dim),
        str(controls.spec.dtype),
        controls.spec.shape,
        matrix.tobytes(),
    )


class SystemEvaluator:
    """Validation and cache wrapper around an optimizer-facing system."""

    # ------------------------------------------------------------------
    # Construction and cache ownership
    # ------------------------------------------------------------------

    def __init__(self, system: Any, *, use_cache: bool = True) -> None:
        self.system: OptimizerSystem = require_system(system)
        self.use_cache = bool(use_cache)
        self._evaluations: dict[ControlCacheKey, Evaluation] = {}
        self.evaluation_count = 0
        self.gradient_count = 0

    def clear_cache(self) -> None:
        """Drop cached evaluations without changing system or counters."""

        self._evaluations.clear()

    # ------------------------------------------------------------------
    # Validated direct calls
    # ------------------------------------------------------------------

    def evaluate(self, controls: Controls) -> Evaluation:
        """Evaluate controls, using the cache when numeric content is unchanged."""

        key = controls_cache_key(controls)
        if self.use_cache and key in self._evaluations:
            return self._evaluations[key]

        metrics = evaluate_system(self.system, controls)
        evaluation = Evaluation.from_metrics(controls.copy(name=controls.name), metrics)
        self.evaluation_count += 1
        if self.use_cache:
            self._evaluations[key] = evaluation
        return evaluation

    def gradient(self, controls: Controls) -> Controls:
        """Return a validated analytical gradient from the system."""

        gradient = gradient_system(self.system, controls)
        self.gradient_count += 1
        return gradient

    # ------------------------------------------------------------------
    # Graceful engine-facing attempts
    # ------------------------------------------------------------------

    def try_evaluate(self, controls: Controls) -> EvaluationOutcome:
        """Evaluate controls and capture validation/evaluation failures."""

        try:
            return EvaluationOutcome(
                evaluation=self.evaluate(controls),
                ok=True,
                reason="ok",
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - engine needs a stop reason, not a traceback.
            return EvaluationOutcome(
                evaluation=None,
                ok=False,
                reason="evaluation_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def try_gradient(self, controls: Controls) -> GradientOutcome:
        """Compute gradient and capture validation/gradient failures."""

        try:
            return GradientOutcome(
                gradient=self.gradient(controls),
                ok=True,
                reason="ok",
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - engine records the failure and stops cleanly.
            return GradientOutcome(
                gradient=None,
                ok=False,
                reason="gradient_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
