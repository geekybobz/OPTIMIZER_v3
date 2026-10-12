"""Validated, counted, accuracy-aware OLSS evaluation runtime."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from optimizer.core._cache import DEFAULT_CACHE_SIZE, BoundedCache
from optimizer.system_olss import (
    ShootingEvaluation,
    ShootingVariables,
    ShootingVerification,
    cost_gradient_system,
    evaluate_shooting_system,
    get_shooting_system_spec,
    jacobian_system,
    residuals_and_jacobian_system,
    residuals_system,
    shooting_bounds_system,
    verify_shooting_system,
    require_shooting_system,
)


def resolve_verification_request(
    system_spec: Mapping[str, Any],
    *,
    working_options: Mapping[str, Any] | None,
    verification_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool, str]:
    """Resolve verification options and whether they are independent evidence."""

    working = dict(working_options or {})
    if verification_options is not None:
        if not isinstance(verification_options, Mapping):
            raise TypeError("verification_options must be a mapping or None.")
        requested = dict(verification_options)
        independent = bool(requested) and requested != working
        source = "explicit_options" if independent else "explicit_options_not_distinct"
        return requested, independent, source
    declared = dict(system_spec.get("verification") or {}).get("default_options")
    if declared is not None:
        requested = dict(declared)
        independent = bool(requested) and requested != working
        source = "system_default" if independent else "system_default_not_distinct"
        return requested, independent, source
    return working, False, "working_options_reused"


def resolve_verification_tolerance(
    system_spec: Mapping[str, Any],
    *,
    working_tolerance: float,
    verification_tolerance: float | None,
) -> tuple[float, str]:
    """Resolve the qualification tolerance and record where it came from.

    Precedence: the caller's explicit value, then the system's declared
    ``verification.default_tolerance``, then the working solver tolerance.
    Verification re-evaluates at distinct accuracy, so its residual includes
    the working discretization error; a system states that scale once here.
    """

    if verification_tolerance is not None:
        value, source = float(verification_tolerance), "explicit"
    else:
        declared = dict(system_spec.get("verification") or {}).get("default_tolerance")
        if declared is not None:
            value, source = float(declared), "system_default"
        else:
            value, source = float(working_tolerance), "working_tolerance"
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("verification_tolerance must be finite and nonnegative.")
    return value, source


def _json_default(value: Any) -> str:
    if isinstance(value, np.generic):
        return repr(value.item())
    return repr(value)


def _options_key(options: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(options or {}), sort_keys=True, default=_json_default, separators=(",", ":"))


def _candidate_key(variables: ShootingVariables, options: Mapping[str, Any] | None) -> tuple[Any, ...]:
    values = np.ascontiguousarray(variables.flatten(copy=False))
    return (
        variables.spec.names,
        variables.spec.size,
        variables.spec.dtype,
        values.tobytes(),
        _options_key(options),
    )


class ShootingEvaluator:
    """Own OLSS validation, caches, and numerical-call counters."""

    def __init__(
        self,
        system: Any,
        *,
        use_cache: bool = True,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.system = require_shooting_system(system)
        self.system_spec = get_shooting_system_spec(system)
        self.use_cache = bool(use_cache)
        self._residuals = BoundedCache(cache_size)
        self._jacobians = BoundedCache(cache_size)
        self._metrics = BoundedCache(cache_size)
        self._verification = BoundedCache(cache_size)
        self._counts = {
            "residual": 0,
            "residual_attempt": 0,
            "jacobian": 0,
            "jacobian_attempt": 0,
            "combined": 0,
            "combined_attempt": 0,
            "evaluation": 0,
            "evaluation_attempt": 0,
            "verification": 0,
            "verification_attempt": 0,
            "cost_gradient": 0,
            "cost_gradient_attempt": 0,
            "bounds": 0,
            "bounds_attempt": 0,
        }

    def clear_cache(self) -> None:
        self._residuals.clear()
        self._jacobians.clear()
        self._metrics.clear()
        self._verification.clear()

    def counters(self, *, compact: bool = False) -> dict[str, int]:
        if not compact:
            return dict(self._counts)
        return {key: value for key, value in self._counts.items() if value}

    @staticmethod
    def _base_metrics(residuals: np.ndarray) -> dict[str, float]:
        return {
            "residual_norm": float(np.linalg.norm(residuals)),
            "residual_max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        }

    def residuals(
        self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        key = _candidate_key(variables, options)
        if self.use_cache and key in self._residuals:
            return self._residuals[key].copy()
        self._counts["residual_attempt"] += 1
        value = residuals_system(self.system, variables, options=options)
        self._counts["residual"] += 1
        if self.use_cache:
            self._residuals[key] = value.copy()
        return value

    def jacobian(
        self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        key = _candidate_key(variables, options)
        if self.use_cache and key in self._jacobians:
            return self._jacobians[key].copy()
        self._counts["jacobian_attempt"] += 1
        value = jacobian_system(self.system, variables, options=options)
        self._counts["jacobian"] += 1
        if self.use_cache:
            self._jacobians[key] = value.copy()
        return value

    def residuals_and_jacobian(
        self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        key = _candidate_key(variables, options)
        if self.use_cache and key in self._residuals and key in self._jacobians:
            return self._residuals[key].copy(), self._jacobians[key].copy()
        if callable(getattr(self.system, "residuals_and_jacobian", None)):
            self._counts["combined_attempt"] += 1
            residuals, jacobian = residuals_and_jacobian_system(
                self.system, variables, options=options
            )
            self._counts["combined"] += 1
            self._counts["residual"] += 1
            self._counts["jacobian"] += 1
            if self.use_cache:
                self._residuals[key] = residuals.copy()
                self._jacobians[key] = jacobian.copy()
            return residuals, jacobian
        return self.residuals(variables, options=options), self.jacobian(variables, options=options)

    def metrics(
        self,
        variables: ShootingVariables,
        *,
        residuals: np.ndarray | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = _candidate_key(variables, options)
        if self.use_cache and key in self._metrics:
            return dict(self._metrics[key])
        held = self.residuals(variables, options=options) if residuals is None else np.asarray(residuals)
        base = self._base_metrics(held)
        extra_declared = set(self.system_spec["metrics"]) - set(base)
        if extra_declared:
            self._counts["evaluation_attempt"] += 1
            metrics = evaluate_shooting_system(self.system, variables, options=options)
            self._counts["evaluation"] += 1
            metrics.update(base)
        else:
            metrics = base
        if self.use_cache:
            self._metrics[key] = dict(metrics)
        return dict(metrics)

    def evaluate(
        self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None
    ) -> ShootingEvaluation:
        residuals = self.residuals(variables, options=options)
        metrics = self.metrics(variables, residuals=residuals, options=options)
        return ShootingEvaluation(
            variables=variables.copy(),
            residuals=residuals,
            metrics=metrics,
            options=dict(options or {}),
        )

    def verify(
        self,
        variables: ShootingVariables,
        *,
        options: Mapping[str, Any] | None = None,
        independent: bool = False,
        independence_source: str = "unspecified",
    ) -> ShootingVerification:
        key = _candidate_key(variables, options) + (
            bool(independent),
            str(independence_source),
        )
        if self.use_cache and key in self._verification:
            return self._verification[key]
        self._counts["verification_attempt"] += 1
        result = verify_shooting_system(
            self.system,
            variables,
            options=options,
            independent=independent,
            independence_source=independence_source,
        )
        self._counts["verification"] += 1
        if self.use_cache:
            self._verification[key] = result
        return result

    def cost_gradient(
        self, variables: ShootingVariables, *, options: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        self._counts["cost_gradient_attempt"] += 1
        value = cost_gradient_system(self.system, variables, options=options)
        self._counts["cost_gradient"] += 1
        return value

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        self._counts["bounds_attempt"] += 1
        value = shooting_bounds_system(self.system)
        if value is not None:
            self._counts["bounds"] += 1
        return value
