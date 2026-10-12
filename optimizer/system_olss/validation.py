"""Validated numerical transactions for OLSS systems."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optimizer.system_olss.contract import (
    get_shooting_system_spec,
    validate_shooting_spec,
)
from optimizer.system_olss.results import ShootingVerification, VerificationCheck
from optimizer.system_olss.variables import ShootingVariables, _validate_shooting_spec_compatibility


def validate_variables_for_shooting_system(
    system: Any, variables: ShootingVariables
) -> ShootingVariables:
    if not isinstance(variables, ShootingVariables):
        raise TypeError("OLSS numerical calls require ShootingVariables.")
    expected = validate_shooting_spec(system)
    _validate_shooting_spec_compatibility(expected, variables.spec)
    if variables.values.shape != (expected.size,) or not np.all(np.isfinite(variables.values)):
        raise ValueError("ShootingVariables must have the declared shape and finite values.")
    return variables


def _options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("Shooting evaluation options must be a mapping or None.")
    return {str(key): value for key, value in options.items()}


def residuals_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    validate_variables_for_shooting_system(system, variables)
    value = np.asarray(system.residuals(variables, options=_options(options)), dtype=float).reshape(-1)
    expected = int(get_shooting_system_spec(system)["residuals"]["size"])
    if value.shape != (expected,):
        raise ValueError(f"OLSS residuals must have shape ({expected},), got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError("OLSS residuals must be finite.")
    return value


def _dense_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "toarray") and callable(value.toarray):
        value = value.toarray()
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("OLSS dense Jacobian must be two-dimensional.")
    return matrix


def jacobian_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    validate_variables_for_shooting_system(system, variables)
    matrix = _dense_matrix(system.jacobian(variables, options=_options(options)))
    spec = get_shooting_system_spec(system)
    expected = (int(spec["residuals"]["size"]), variables.spec.size)
    if matrix.shape != expected:
        raise ValueError(f"OLSS Jacobian must have shape {expected}, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("OLSS Jacobian must be finite.")
    return matrix


def residuals_and_jacobian_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hook = getattr(system, "residuals_and_jacobian", None)
    if not callable(hook):
        return (
            residuals_system(system, variables, options=options),
            jacobian_system(system, variables, options=options),
        )
    validate_variables_for_shooting_system(system, variables)
    residuals, jacobian = hook(variables, options=_options(options))
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    jacobian = _dense_matrix(jacobian)
    spec = get_shooting_system_spec(system)
    expected_residuals = (int(spec["residuals"]["size"]),)
    expected_jacobian = (expected_residuals[0], variables.spec.size)
    if residuals.shape != expected_residuals or jacobian.shape != expected_jacobian:
        raise ValueError(
            "OLSS residuals_and_jacobian returned inconsistent shapes: "
            f"{residuals.shape}, {jacobian.shape}; expected "
            f"{expected_residuals}, {expected_jacobian}."
        )
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("OLSS combined residuals and Jacobian must be finite.")
    return residuals, jacobian


def _validate_metrics(system: Any, metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError("OLSS evaluate() must return a metrics mapping.")
    payload = {str(key): value for key, value in metrics.items()}
    spec = get_shooting_system_spec(system)
    declared = spec["metrics"]
    unknown = sorted(set(payload) - set(declared))
    if unknown and not spec["allow_additional_metrics"]:
        raise KeyError(f"OLSS evaluate returned undeclared metrics: {', '.join(unknown)}.")
    missing = sorted(
        key for key, declaration in declared.items() if declaration["required"] and key not in payload
    )
    if missing:
        raise KeyError(f"OLSS evaluate omitted required metrics: {', '.join(missing)}.")
    for key, value in payload.items():
        if key not in declared:
            continue
        array = np.asarray(value)
        expected = tuple(int(item) for item in declared[key].get("shape", ()))
        if array.shape != expected:
            raise ValueError(f"OLSS metric {key!r} must have shape {expected}, got {array.shape}.")
        if not declared[key].get("allow_nonfinite", False):
            try:
                finite = bool(np.all(np.isfinite(array)))
            except TypeError:
                finite = True
            if not finite:
                raise ValueError(f"OLSS metric {key!r} must be finite.")
    return payload


def evaluate_shooting_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_variables_for_shooting_system(system, variables)
    return _validate_metrics(system, system.evaluate(variables, options=_options(options)))


def verify_shooting_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
    independent: bool = False,
    independence_source: str = "unspecified",
) -> ShootingVerification:
    validate_variables_for_shooting_system(system, variables)
    requested = _options(options)
    value = system.verify(variables, options=requested)
    if isinstance(value, ShootingVerification):
        result = value
    elif isinstance(value, Mapping):
        raw_checks = value.get("checks") or ()
        result = ShootingVerification(
            residuals=np.asarray(value.get("residuals"), dtype=float).reshape(-1),
            metrics=dict(value.get("metrics") or {}),
            options=dict(value.get("options") or requested),
            numerical_success=bool(value.get("numerical_success", True)),
            message=str(value.get("message", "verified")),
            checks=tuple(
                check
                if isinstance(check, VerificationCheck)
                else VerificationCheck(**dict(check))
                for check in raw_checks
            ),
        )
    else:
        raise TypeError("OLSS verify() must return ShootingVerification or a mapping.")
    expected = int(get_shooting_system_spec(system)["residuals"]["size"])
    if result.residuals.shape != (expected,) or not np.all(np.isfinite(result.residuals)):
        raise ValueError("OLSS verification residuals have invalid shape or values.")
    metrics = _validate_metrics(system, result.metrics)
    return ShootingVerification(
        residuals=result.residuals.copy(),
        metrics=metrics,
        options=dict(result.options),
        numerical_success=result.numerical_success,
        message=result.message,
        independent=bool(independent),
        independence_source=str(independence_source),
        checks=result.checks,
    )


def roots_equivalent_system(
    system: Any,
    left: ShootingVariables,
    right: ShootingVariables,
    *,
    tolerance: float,
) -> bool | None:
    """Call an optional system-owned root-equivalence predicate.

    ``None`` means the system does not define a project symmetry and the
    multistart driver should use its scale-aware numerical default.
    """

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Root-equivalence tolerance must be finite and nonnegative.")
    hook = getattr(system, "roots_equivalent", None)
    if not callable(hook):
        return None
    validate_variables_for_shooting_system(system, left)
    validate_variables_for_shooting_system(system, right)
    value = hook(left, right, tolerance=tolerance)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("OLSS roots_equivalent() must return bool.")
    return bool(value)


def cost_gradient_system(
    system: Any,
    variables: ShootingVariables,
    *,
    options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    hook = getattr(system, "cost_gradient", None)
    if not callable(hook):
        raise TypeError("OLSS system does not provide cost_gradient().")
    validate_variables_for_shooting_system(system, variables)
    value = np.asarray(hook(variables, options=_options(options)), dtype=float).reshape(-1)
    if value.shape != (variables.spec.size,) or not np.all(np.isfinite(value)):
        raise ValueError("OLSS cost gradient has invalid shape or values.")
    return value


def shooting_bounds_system(system: Any) -> tuple[np.ndarray, np.ndarray] | None:
    hook = getattr(system, "shooting_bounds", None)
    if not callable(hook):
        return None
    lower, upper = hook()
    size = validate_shooting_spec(system).size
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if lower.shape != (size,) or upper.shape != (size,):
        raise ValueError("OLSS shooting bounds must match the flat variable size.")
    if np.any(lower > upper) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("OLSS shooting bounds are invalid.")
    return lower, upper
