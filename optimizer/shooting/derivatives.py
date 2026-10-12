"""Explicit finite-difference and derivative-check tools for OLSS."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.shooting.evaluator import ShootingEvaluator
from optimizer.core.numerics import difference_jacobian
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
    return difference_jacobian(
        lambda flat: evaluator.residuals(variables.with_flat(flat), options=options),
        variables.flatten(copy=False), eps=eps, scales=variables.scales, relative=True,
    )


# Ratio between the two comparison steps used for a numerical-declared Jacobian.
# An irrational ratio keeps at least one step away from any common project step.
_SECOND_STEP_RATIO = 1.6180339887498949


def verify_shooting_jacobian(
    system: Any,
    variables: ShootingVariables,
    *,
    eps: float = 1.0e-6,
    atol: float = 1.0e-7,
    rtol: float = 1.0e-5,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the declared Jacobian with central finite differences.

    A numerical-declared Jacobian is itself a finite difference, so one
    comparison step could reproduce the project's own stencil and agree to
    roundoff without checking anything.  Such a Jacobian is compared at two
    separated steps; at least one differs from the project's step, and the
    check passes only when both comparisons agree.
    """

    source = str(
        get_shooting_system_spec(system)["derivatives"]["jacobian"]["source"]
    )
    if source == "numerical-declared":
        steps = (10.0 * float(eps), 10.0 * float(eps) * _SECOND_STEP_RATIO)
        comparison_strategy = "separated_central_difference_step"
    else:
        steps = (float(eps),)
        comparison_strategy = "central_difference"
    analytical = ShootingEvaluator(system).jacobian(variables, options=options)
    comparisons = []
    for step in steps:
        numerical = finite_difference_shooting_jacobian(
            system, variables, eps=step, options=options
        )
        error = analytical - numerical
        denominator = np.maximum(np.abs(numerical), float(atol))
        comparisons.append(
            {
                "numerical": numerical,
                "max_abs": float(np.max(np.abs(error))) if error.size else 0.0,
                "max_rel": float(np.max(np.abs(error) / denominator)) if error.size else 0.0,
                "passed": bool(np.allclose(analytical, numerical, atol=float(atol), rtol=float(rtol))),
            }
        )
    return {
        "passed": all(item["passed"] for item in comparisons),
        "shape": list(analytical.shape),
        "max_abs_error": max(item["max_abs"] for item in comparisons),
        "max_relative_error": max(item["max_rel"] for item in comparisons),
        "eps": float(eps),
        "comparison_eps": steps[0],
        "comparison_steps": list(steps),
        "step_max_abs_errors": [item["max_abs"] for item in comparisons],
        "comparison_strategy": comparison_strategy,
        "declared_source": source,
        "independent": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "analytical": analytical,
        "numerical": comparisons[0]["numerical"],
    }
