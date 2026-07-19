"""Numerical fallbacks for optional OLGS derivatives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.system_olgs.validation import (
    evaluate_system,
    gradient_system,
    optional_hessian,
    optional_hvp,
    optional_jacobian,
    optional_residuals,
    validate_controls_for_system,
)


def _controls_from_flat(reference: Controls, flat: np.ndarray, *, name: str) -> Controls:
    return Controls.from_flat(reference.spec, np.asarray(flat, dtype=reference.spec.dtype), name=name)


def _scalar_metric(metrics: Mapping[str, Any], metric: str) -> float:
    if metric not in metrics:
        raise KeyError(f"metrics do not include {metric!r}.")
    value = np.asarray(metrics[metric])
    if value.shape != ():
        raise ValueError(f"metric {metric!r} must be scalar for finite differences.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"metric {metric!r} must be finite.")
    return out


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


def finite_difference_hessian(
    system: Any,
    controls: Controls,
    *,
    metric: str = "J",
    eps: float = 1.0e-5,
    method: str = "central",
) -> np.ndarray:
    """Return a dense finite-difference Hessian for a scalar metric."""

    validate_controls_for_system(system, controls)
    eps = float(eps)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")
    if method not in {"central", "gradient"}:
        raise ValueError("method must be 'central' or 'gradient'.")

    base_flat = controls.flatten(copy=True)
    size = base_flat.size
    hessian = np.zeros((size, size), dtype=float)

    if method == "gradient":
        for index in range(size):
            direction = np.zeros_like(base_flat)
            direction[index] = 1.0
            plus = _controls_from_flat(controls, base_flat + eps * direction, name="fd_hess_plus")
            minus = _controls_from_flat(controls, base_flat - eps * direction, name="fd_hess_minus")
            g_plus = gradient_system(system, plus).flatten(copy=False).astype(float, copy=False)
            g_minus = gradient_system(system, minus).flatten(copy=False).astype(float, copy=False)
            hessian[:, index] = (g_plus - g_minus) / (2.0 * eps)
    else:
        base_value = _scalar_metric(evaluate_system(system, controls), metric)
        for i in range(size):
            ei = np.zeros_like(base_flat)
            ei[i] = 1.0
            f_plus = _scalar_metric(
                evaluate_system(system, _controls_from_flat(controls, base_flat + eps * ei, name="fd_hess_i_plus")),
                metric,
            )
            f_minus = _scalar_metric(
                evaluate_system(system, _controls_from_flat(controls, base_flat - eps * ei, name="fd_hess_i_minus")),
                metric,
            )
            hessian[i, i] = (f_plus - 2.0 * base_value + f_minus) / (eps * eps)
            for j in range(i + 1, size):
                ej = np.zeros_like(base_flat)
                ej[j] = 1.0
                f_pp = _scalar_metric(
                    evaluate_system(
                        system,
                        _controls_from_flat(controls, base_flat + eps * ei + eps * ej, name="fd_hess_pp"),
                    ),
                    metric,
                )
                f_pm = _scalar_metric(
                    evaluate_system(
                        system,
                        _controls_from_flat(controls, base_flat + eps * ei - eps * ej, name="fd_hess_pm"),
                    ),
                    metric,
                )
                f_mp = _scalar_metric(
                    evaluate_system(
                        system,
                        _controls_from_flat(controls, base_flat - eps * ei + eps * ej, name="fd_hess_mp"),
                    ),
                    metric,
                )
                f_mm = _scalar_metric(
                    evaluate_system(
                        system,
                        _controls_from_flat(controls, base_flat - eps * ei - eps * ej, name="fd_hess_mm"),
                    ),
                    metric,
                )
                value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
                hessian[i, j] = value
                hessian[j, i] = value

    if not np.all(np.isfinite(hessian)):
        raise ValueError("finite-difference Hessian produced non-finite values.")
    return 0.5 * (hessian + hessian.T)


def finite_difference_hvp(
    system: Any,
    controls: Controls,
    vector: Controls | np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> Controls:
    """Return a finite-difference Hessian-vector product using gradients."""

    validate_controls_for_system(system, controls)
    eps = float(eps)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")
    if isinstance(vector, Controls):
        if vector.spec.keys != controls.spec.keys or vector.spec.control_dim != controls.spec.control_dim:
            raise ValueError("vector controls must match controls layout.")
        v = vector.flatten(copy=False).astype(float, copy=False)
    else:
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.shape != (controls.spec.size,):
            raise ValueError(f"flat vector must have shape ({controls.spec.size},), got {v.shape}.")

    base_flat = controls.flatten(copy=True)
    plus = _controls_from_flat(controls, base_flat + eps * v, name="fd_hvp_plus")
    minus = _controls_from_flat(controls, base_flat - eps * v, name="fd_hvp_minus")
    g_plus = gradient_system(system, plus).flatten(copy=False).astype(float, copy=False)
    g_minus = gradient_system(system, minus).flatten(copy=False).astype(float, copy=False)
    return _controls_from_flat(controls, (g_plus - g_minus) / (2.0 * eps), name="fd_hvp")


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


def get_hessian(
    system: Any,
    controls: Controls,
    *,
    metric: str = "J",
    fallback: bool = True,
    eps: float = 1.0e-5,
) -> tuple[np.ndarray, str]:
    """Return ``(hessian, source)`` using analytical hook or finite differences."""

    try:
        return optional_hessian(system, controls), "analytical"
    except AttributeError:
        if not fallback:
            raise
        return finite_difference_hessian(system, controls, metric=metric, eps=eps), "finite_difference"


def get_hvp(
    system: Any,
    controls: Controls,
    vector: Controls | np.ndarray,
    *,
    fallback: bool = True,
    eps: float = 1.0e-6,
) -> tuple[Controls, str]:
    """Return ``(hvp, source)`` using analytical hook or finite differences."""

    try:
        return optional_hvp(system, controls, vector), "analytical"
    except AttributeError:
        if not fallback:
            raise
        return finite_difference_hvp(system, controls, vector, eps=eps), "finite_difference"
