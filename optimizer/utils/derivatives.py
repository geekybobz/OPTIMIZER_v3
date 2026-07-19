"""Finite-difference derivative checks and fallbacks.

Why this file exists
--------------------
The library is built around analytically derived systems: the user derives the action,
variation, dynamics, costate equations, and gradient.  That is powerful, but it also
means a small sign, scale, ``dt``, or channel-ordering mistake can make every optimizer
look wrong.  This module gives a standard way to compare analytical hooks against
finite-difference directional checks.

How it fits the architecture
----------------------------
- systems still own analytical ``gradient`` and optional ``jacobian`` hooks.
- optimizers use analytical gradients during normal runs.
- diagnostics and repairs can call finite-difference fallbacks when optional Jacobian
  hooks are not available yet.
- tests and notebooks can call ``opt.verify_gradient`` before trusting a new system.

What this file deliberately does not do
---------------------------------------
It does not replace analytical gradients as the normal optimizer path.  Full
finite-difference derivatives are expensive and should be used for checks, fallbacks,
and small debugging runs only.

Reviewer invariants
-------------------
- finite-difference controls preserve the original ``ControlSpec``.
- scalar metrics are checked before being used as derivative targets.
- residual Jacobians are returned with shape ``(n_residuals, n_control_values)``.
- verification reports include absolute and relative errors, not just pass/fail.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.system_olgs import (
    evaluate_system,
    gradient_system,
    optional_jacobian,
    optional_residuals,
    validate_controls_for_system,
)


def _controls_from_flat(reference: Controls, flat: np.ndarray, *, name: str) -> Controls:
    """Rebuild controls from a flat vector using the reference layout."""

    return Controls.from_flat(reference.spec, np.asarray(flat, dtype=reference.spec.dtype), name=name)


def _scalar_metric(metrics: Mapping[str, Any], metric: str) -> float:
    """Extract a finite scalar metric from a system evaluation."""

    if metric not in metrics:
        raise KeyError(f"metrics do not include {metric!r}.")
    value = np.asarray(metrics[metric])
    if value.shape != ():
        raise ValueError(f"metric {metric!r} must be scalar for finite differences.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"metric {metric!r} must be finite.")
    return out


def _relative_error(a: np.ndarray, b: np.ndarray, *, atol: float) -> float:
    """Return a scale-aware relative error for two arrays."""

    numerator = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(atol))
    return numerator / denominator


def _direction_set(size: int, directions: int | np.ndarray, *, seed: int | None) -> np.ndarray:
    """Return normalized finite directions in flat control space."""

    if isinstance(directions, np.ndarray):
        raw = np.asarray(directions, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] != int(size):
            raise ValueError(f"directions must have shape (n_dirs, {size}).")
    else:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(int(directions), int(size)))
    norms = np.linalg.norm(raw, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("directions must be finite nonzero vectors.")
    return raw / norms[:, None]


def finite_difference_gradient(
    system: Any,
    controls: Controls,
    *,
    metric: str = "J",
    eps: float = 1.0e-6,
    method: str = "central",
) -> Controls:
    """Return a full finite-difference gradient for a scalar metric.

    This computes one coordinate derivative per control value.  It is intentionally
    explicit and therefore expensive for large controls.  Use ``verify_gradient`` with
    random directions for cheaper checks during normal development.
    """

    validate_controls_for_system(system, controls)
    eps = float(eps)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    base_flat = controls.flatten(copy=True)
    gradient = np.zeros_like(base_flat, dtype=float)
    base_value = None
    if method == "forward":
        base_value = _scalar_metric(evaluate_system(system, controls), metric)

    for index in range(base_flat.size):
        direction = np.zeros_like(base_flat)
        direction[index] = 1.0
        plus = _controls_from_flat(controls, base_flat + eps * direction, name="fd_plus")
        plus_value = _scalar_metric(evaluate_system(system, plus), metric)
        if method == "central":
            minus = _controls_from_flat(controls, base_flat - eps * direction, name="fd_minus")
            minus_value = _scalar_metric(evaluate_system(system, minus), metric)
            gradient[index] = (plus_value - minus_value) / (2.0 * eps)
        else:
            assert base_value is not None
            gradient[index] = (plus_value - base_value) / eps

    return _controls_from_flat(controls, gradient, name=f"finite_difference_{metric}_gradient")


def verify_gradient(
    system: Any,
    controls: Controls,
    *,
    metric: str = "J",
    eps: float = 1.0e-6,
    directions: int | np.ndarray = 8,
    seed: int | None = 12345,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Check analytical gradient against finite-difference directional derivatives."""

    analytical = gradient_system(system, controls).flatten(copy=False).astype(float, copy=False)
    dirs = _direction_set(controls.spec.size, directions, seed=seed)
    base_flat = controls.flatten(copy=True)
    rows: list[dict[str, float]] = []
    abs_errors = []
    rel_errors = []

    for row, direction in enumerate(dirs):
        plus = _controls_from_flat(controls, base_flat + float(eps) * direction, name="grad_check_plus")
        minus = _controls_from_flat(controls, base_flat - float(eps) * direction, name="grad_check_minus")
        plus_value = _scalar_metric(evaluate_system(system, plus), metric)
        minus_value = _scalar_metric(evaluate_system(system, minus), metric)
        finite_difference = (plus_value - minus_value) / (2.0 * float(eps))
        analytical_directional = float(np.dot(analytical, direction))
        abs_error = abs(analytical_directional - finite_difference)
        rel_error = abs_error / max(abs(analytical_directional), abs(finite_difference), float(atol))
        rows.append(
            {
                "direction": float(row),
                "analytical": analytical_directional,
                "finite_difference": float(finite_difference),
                "absolute_error": float(abs_error),
                "relative_error": float(rel_error),
            }
        )
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)

    max_abs = float(np.max(abs_errors)) if abs_errors else 0.0
    max_rel = float(np.max(rel_errors)) if rel_errors else 0.0
    return {
        "kind": "gradient_check",
        "metric": metric,
        "eps": float(eps),
        "n_directions": int(dirs.shape[0]),
        "gradient_norm": float(np.linalg.norm(analytical)),
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "rtol": float(rtol),
        "atol": float(atol),
        "passed": bool(max_abs <= float(atol) or max_rel <= float(rtol)),
        "directions": rows,
    }


def finite_difference_jacobian(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    eps: float = 1.0e-6,
    method: str = "central",
) -> np.ndarray:
    """Return a finite-difference Jacobian of ``system.residuals``."""

    validate_controls_for_system(system, controls)
    eps = float(eps)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    base_flat = controls.flatten(copy=True)
    base_residuals = optional_residuals(system, controls, name=residuals)
    jacobian = np.zeros((base_residuals.size, base_flat.size), dtype=float)

    for index in range(base_flat.size):
        direction = np.zeros_like(base_flat)
        direction[index] = 1.0
        plus = _controls_from_flat(controls, base_flat + eps * direction, name="fd_jac_plus")
        plus_residuals = optional_residuals(system, plus, name=residuals)
        if method == "central":
            minus = _controls_from_flat(controls, base_flat - eps * direction, name="fd_jac_minus")
            minus_residuals = optional_residuals(system, minus, name=residuals)
            jacobian[:, index] = (plus_residuals - minus_residuals) / (2.0 * eps)
        else:
            jacobian[:, index] = (plus_residuals - base_residuals) / eps

    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference Jacobian produced non-finite values.")
    return jacobian


def get_jacobian(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    fallback: bool = True,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, str]:
    """Return ``(jacobian, source)`` using analytical hook or finite differences."""

    try:
        return optional_jacobian(system, controls, name=residuals), "analytical"
    except AttributeError:
        if not fallback:
            raise
        return finite_difference_jacobian(system, controls, residuals=residuals, eps=eps), "finite_difference"


def verify_jacobian(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    eps: float = 1.0e-6,
    directions: int | np.ndarray = 8,
    seed: int | None = 12345,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-7,
    fallback: bool = False,
) -> dict[str, Any]:
    """Check analytical residual Jacobian against finite-difference directions."""

    jacobian, source = get_jacobian(
        system,
        controls,
        residuals=residuals,
        fallback=fallback,
        eps=eps,
    )
    dirs = _direction_set(controls.spec.size, directions, seed=seed)
    base_flat = controls.flatten(copy=True)
    rows: list[dict[str, float]] = []
    abs_errors = []
    rel_errors = []

    for row, direction in enumerate(dirs):
        plus = _controls_from_flat(controls, base_flat + float(eps) * direction, name="jac_check_plus")
        minus = _controls_from_flat(controls, base_flat - float(eps) * direction, name="jac_check_minus")
        fd_directional = (
            optional_residuals(system, plus, name=residuals)
            - optional_residuals(system, minus, name=residuals)
        ) / (2.0 * float(eps))
        analytical_directional = jacobian @ direction
        abs_error = float(np.linalg.norm(analytical_directional - fd_directional))
        rel_error = _relative_error(analytical_directional, fd_directional, atol=float(atol))
        rows.append(
            {
                "direction": float(row),
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "analytical_norm": float(np.linalg.norm(analytical_directional)),
                "finite_difference_norm": float(np.linalg.norm(fd_directional)),
            }
        )
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)

    max_abs = float(np.max(abs_errors)) if abs_errors else 0.0
    max_rel = float(np.max(rel_errors)) if rel_errors else 0.0
    return {
        "kind": "jacobian_check",
        "residuals": residuals,
        "jacobian_source": source,
        "eps": float(eps),
        "n_directions": int(dirs.shape[0]),
        "jacobian_shape": list(jacobian.shape),
        "jacobian_norm": float(np.linalg.norm(jacobian)),
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "rtol": float(rtol),
        "atol": float(atol),
        "passed": bool(max_abs <= float(atol) or max_rel <= float(rtol)),
        "directions": rows,
    }
