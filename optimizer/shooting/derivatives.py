"""Explicit finite-difference and derivative-check tools for OLSS."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.shooting.evaluator import ShootingEvaluator
from optimizer.system_olss import ShootingVariables, get_shooting_system_spec


def finite_difference_shooting_jacobian(
    system: Any,
    variables: ShootingVariables,
    *,
    eps: float = 1.0e-6,
    options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    evaluator = ShootingEvaluator(system, use_cache=False)
    base = variables.flatten(copy=False)
    rows = evaluator.residuals(variables, options=options).size
    jacobian = np.empty((rows, variables.spec.size), dtype=float)
    for column in range(variables.spec.size):
        scale = max(1.0, abs(float(base[column])))
        step = eps * scale
        plus = base.copy()
        minus = base.copy()
        plus[column] += step
        minus[column] -= step
        r_plus = evaluator.residuals(variables.with_flat(plus), options=options)
        r_minus = evaluator.residuals(variables.with_flat(minus), options=options)
        jacobian[:, column] = (r_plus - r_minus) / (2.0 * step)
    return jacobian


def verify_shooting_jacobian(
    system: Any,
    variables: ShootingVariables,
    *,
    eps: float = 1.0e-6,
    atol: float = 1.0e-7,
    rtol: float = 1.0e-5,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = str(
        get_shooting_system_spec(system)["derivatives"]["jacobian"]["source"]
    )
    comparison_eps = float(eps)
    comparison_strategy = "central_difference"
    if source == "numerical-declared":
        comparison_eps = 10.0 * float(eps)
        comparison_strategy = "separated_central_difference_step"
    analytical = ShootingEvaluator(system).jacobian(variables, options=options)
    numerical = finite_difference_shooting_jacobian(
        system, variables, eps=comparison_eps, options=options
    )
    error = analytical - numerical
    max_abs = float(np.max(np.abs(error))) if error.size else 0.0
    denominator = np.maximum(np.abs(numerical), float(atol))
    max_rel = float(np.max(np.abs(error) / denominator)) if error.size else 0.0
    passed = bool(np.allclose(analytical, numerical, atol=float(atol), rtol=float(rtol)))
    return {
        "passed": passed,
        "shape": list(analytical.shape),
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "eps": float(eps),
        "comparison_eps": comparison_eps,
        "comparison_strategy": comparison_strategy,
        "declared_source": source,
        "independent": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "analytical": analytical,
        "numerical": numerical,
    }
