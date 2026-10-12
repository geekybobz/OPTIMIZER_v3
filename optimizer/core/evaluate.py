"""Validated system evaluation helpers for the shared optimizer engine.

Why this file exists
--------------------
The physical systems in OPTIMIZER v4 own their analytical objective and gradient.
Every optimizer still needs the same surrounding safety work: validate that controls
match the system, normalize metric dictionaries, reject non-finite results, and avoid
repeating an expensive propagation when the exact same controls are evaluated twice.

This module centralizes that boundary in ``SystemEvaluator``.  It is intentionally
small: optimizers should call the engine, the engine should call this evaluator, and
the system should remain the only place where physics is computed.

How it fits the architecture
----------------------------
- ``system_olgs`` defines the required hooks and validation helpers.
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
- successful-call and attempted-call counters remain distinct, so a failed trial
  is visible as consumed work without changing the legacy successful-call count.
- gradients are never finite-difference fallbacks here; they come from the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core._cache import DEFAULT_CACHE_SIZE
from optimizer.core.runtime import EvaluatorBase, candidate_key
from optimizer.result import Evaluation
from optimizer.system_olgs import (
    ControlBounds,
    OptimizerSystem,
    ValueGradient,
    evaluate_system,
    get_system_spec,
    gradient_system,
    optional_control_bounds,
    optional_hvp,
    optional_jacobian,
    optional_plot_data,
    optional_prox,
    optional_residuals,
    optional_smooth_gradient,
    probe_system,
    require_system,
    validate_control_spec,
    value_and_gradient_system,
)


ControlCacheKey = tuple[str, bytes, str]


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


@dataclass(frozen=True)
class ValueGradientOutcome:
    """Structured result for a combined value/gradient attempt."""

    value_gradient: ValueGradient | None
    ok: bool
    reason: str
    error: str | None = None


def controls_cache_key(controls: Controls) -> ControlCacheKey:
    """Return a deterministic key for the exact numeric control content."""

    return candidate_key(controls)


class SystemEvaluator(EvaluatorBase):
    """Validation and cache wrapper around an optimizer-facing system."""

    # ------------------------------------------------------------------
    # Construction and cache ownership
    # ------------------------------------------------------------------

    def __init__(
        self,
        system: Any,
        *,
        use_cache: bool = True,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.system: OptimizerSystem = require_system(system)
        self.system_spec = get_system_spec(self.system)
        control_spec = validate_control_spec(self.system)
        declared_controls = self.system_spec["controls"]
        if tuple(declared_controls["keys"]) != control_spec.keys or int(
            declared_controls["control_dim"]
        ) != control_spec.control_dim:
            raise ValueError("system_spec controls do not match system.control_spec().")
        self._initialize_runtime(use_cache=use_cache, cache_size=cache_size)
        self._evaluations = self._new_cache()
        self._value_gradients = self._new_cache()
        self.evaluation_count = 0
        self.gradient_count = 0
        self.evaluation_attempt_count = 0
        self.gradient_attempt_count = 0
        self.value_gradient_count = 0
        self.value_gradient_attempt_count = 0
        self.residual_count = 0
        self.residual_attempt_count = 0
        self.jacobian_count = 0
        self.jacobian_attempt_count = 0
        self.hvp_count = 0
        self.hvp_attempt_count = 0
        self.smooth_gradient_count = 0
        self.smooth_gradient_attempt_count = 0
        self.prox_count = 0
        self.prox_attempt_count = 0
        self.bounds_count = 0
        self.bounds_attempt_count = 0
        self.plot_data_count = 0
        self.plot_data_attempt_count = 0

    def clear_cache(self) -> None:
        """Drop cached evaluations without changing system or counters."""

        super().clear_cache()

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return hook availability without invoking numerical work."""

        probe = probe_system(self.system)
        return probe.to_dict()

    def counters(self, *, compact: bool = False) -> dict[str, int]:
        """Return a stable snapshot of all successful and attempted operations."""

        counters = {
            "evaluation": self.evaluation_count,
            "gradient": self.gradient_count,
            "value_gradient": self.value_gradient_count,
            "residual": self.residual_count,
            "jacobian": self.jacobian_count,
            "hvp": self.hvp_count,
            "smooth_gradient": self.smooth_gradient_count,
            "prox": self.prox_count,
            "bounds": self.bounds_count,
            "plot_data": self.plot_data_count,
            "evaluation_attempt": self.evaluation_attempt_count,
            "gradient_attempt": self.gradient_attempt_count,
            "value_gradient_attempt": self.value_gradient_attempt_count,
            "residual_attempt": self.residual_attempt_count,
            "jacobian_attempt": self.jacobian_attempt_count,
            "hvp_attempt": self.hvp_attempt_count,
            "smooth_gradient_attempt": self.smooth_gradient_attempt_count,
            "prox_attempt": self.prox_attempt_count,
            "bounds_attempt": self.bounds_attempt_count,
            "plot_data_attempt": self.plot_data_attempt_count,
        }
        if not compact:
            return counters
        base = {"evaluation", "gradient", "evaluation_attempt", "gradient_attempt"}
        return {name: value for name, value in counters.items() if name in base or value != 0}

    # ------------------------------------------------------------------
    # Validated direct calls
    # ------------------------------------------------------------------

    def evaluate(self, controls: Controls) -> Evaluation:
        """Evaluate controls, using the cache when numeric content is unchanged."""

        key = controls_cache_key(controls)
        if self.use_cache and key in self._evaluations:
            return self._evaluations[key].copy(name=controls.name)

        self.evaluation_attempt_count += 1
        metrics = evaluate_system(self.system, controls, system_spec=self.system_spec)
        evaluation = Evaluation.from_metrics(
            controls.copy(name=controls.name),
            metrics,
            objective_metric=str(self.system_spec["objective"]["metric_id"]),
        )
        self.evaluation_count += 1
        if self.use_cache:
            self._evaluations[key] = evaluation.copy(name=evaluation.controls.name)
        return evaluation

    def gradient(self, controls: Controls) -> Controls:
        """Return a validated analytical gradient from the system."""

        gradient = self._counted("gradient", gradient_system, self.system, controls)
        return gradient

    def value_and_gradient(self, controls: Controls) -> tuple[Evaluation, Controls]:
        """Return a validated combined value/gradient result, cached by controls."""

        key = controls_cache_key(controls)
        if self.use_cache and key in self._value_gradients and key in self._evaluations:
            pair = self._value_gradients[key]
            evaluation = self._evaluations[key]
            return evaluation.copy(name=controls.name), pair.gradient.copy(name=pair.gradient.name)
        self.value_gradient_attempt_count += 1
        pair = value_and_gradient_system(self.system, controls, system_spec=self.system_spec)
        gradient = pair.gradient.copy(name=pair.gradient.name)
        objective_metric = str(self.system_spec["objective"]["metric_id"])
        evaluation = Evaluation.from_metrics(
            controls.copy(name=controls.name),
            pair.metrics,
            objective_metric=objective_metric,
        )
        self.value_gradient_count += 1
        self.evaluation_count += 1
        self.gradient_count += 1
        if self.use_cache:
            self._evaluations[key] = evaluation.copy(name=evaluation.controls.name)
            self._value_gradients[key] = ValueGradient(metrics=dict(pair.metrics), gradient=gradient.copy(name=gradient.name))
        return evaluation, gradient

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        value = self._counted("residual", optional_residuals, self.system, controls, name=name)
        return value

    def jacobian(self, controls: Controls, *, name: str = "hard") -> np.ndarray:
        value = self._counted("jacobian", optional_jacobian, self.system, controls, name=name)
        return value

    def hvp(self, controls: Controls, vector: Controls | np.ndarray) -> Controls:
        value = self._counted("hvp", optional_hvp, self.system, controls, vector)
        return value

    def smooth_gradient(self, controls: Controls) -> Controls:
        value = self._counted("smooth_gradient", optional_smooth_gradient, self.system, controls)
        return value

    def prox(self, controls: Controls, step_size: float) -> Controls:
        value = self._counted("prox", optional_prox, self.system, controls, step_size)
        return value

    def control_bounds(self) -> ControlBounds:
        value = self._counted("bounds", optional_control_bounds, self.system)
        return value

    def plot_data(self, controls: Controls, *, kind: str, **options: Any) -> Any:
        """Return one explicit, validated system-owned visualization data product."""

        value = self._counted("plot_data", optional_plot_data, self.system, controls, kind=kind, **options)
        return value

    # ------------------------------------------------------------------
    # Graceful engine-facing attempts
    # ------------------------------------------------------------------

    def try_evaluate(self, controls: Controls) -> EvaluationOutcome:
        outcome = self.attempt(self.evaluate, controls, reason="evaluation_failed")
        return EvaluationOutcome(outcome.value, outcome.ok, outcome.reason, outcome.error)

    def try_gradient(self, controls: Controls) -> GradientOutcome:
        outcome = self.attempt(self.gradient, controls, reason="gradient_failed")
        return GradientOutcome(outcome.value, outcome.ok, outcome.reason, outcome.error)

    def try_value_and_gradient(self, controls: Controls) -> ValueGradientOutcome:
        outcome = self.attempt(self.value_and_gradient, controls, reason="value_gradient_failed")
        pair = None if not outcome.ok else ValueGradient(metrics=outcome.value[0].metrics, gradient=outcome.value[1])
        return ValueGradientOutcome(pair, outcome.ok, outcome.reason, outcome.error)

    def try_smooth_gradient(self, controls: Controls) -> GradientOutcome:
        outcome = self.attempt(self.smooth_gradient, controls, reason="smooth_gradient_failed")
        return GradientOutcome(outcome.value, outcome.ok, outcome.reason, outcome.error)
