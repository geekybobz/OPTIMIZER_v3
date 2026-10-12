"""Validation helpers for OLGS systems."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls, _validate_control_spec_compatibility
from optimizer.system_olgs.contract import ControlBounds, Metrics, OLGSystem, ValueGradient
from optimizer.system_olgs.specs import validate_declared_metrics


def validate_control_spec(system: OLGSystem) -> ControlSpec:
    """Call and validate ``system.control_spec()``."""

    spec = system.control_spec()
    if not isinstance(spec, ControlSpec):
        raise TypeError("system.control_spec() must return ControlSpec.")
    return spec


def validate_controls_for_system(system: OLGSystem, controls: Controls) -> ControlSpec:
    """Validate that controls match the system's control spec."""

    spec = validate_control_spec(system)
    if not isinstance(controls, Controls):
        raise TypeError("controls must be a Controls object.")
    _validate_control_spec_compatibility(spec, controls.spec)
    if controls.as_matrix(copy=False).shape != spec.shape or not np.all(
        np.isfinite(controls.as_matrix(copy=False))
    ):
        raise ValueError("Controls must have the declared shape and finite values.")
    return spec


def validate_metrics(
    metrics: Metrics,
    *,
    system_spec: dict[str, Any] | None = None,
    objective_metric: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize an evaluation metrics mapping.

    A bare metrics mapping retains the legacy ``J`` default.  A system-owned
    specification takes precedence, so a system may declare (for example)
    ``fidelity`` as its objective without manufacturing a duplicate ``J``.
    """

    from collections.abc import Mapping

    if not isinstance(metrics, Mapping):
        raise TypeError("system.evaluate(...) must return a metrics mapping.")
    declared_objective = "J" if objective_metric is None else str(objective_metric)
    if system_spec is not None:
        declared_objective = str(system_spec.get("objective", {}).get("metric_id", declared_objective))
    if declared_objective not in metrics:
        raise KeyError(
            "system.evaluate(...) metrics must include the declared objective "
            f"metric {declared_objective!r}."
        )
    objective_value = float(metrics[declared_objective])
    if not np.isfinite(objective_value):
        raise ValueError(f"metric {declared_objective!r} must be finite.")
    out = dict(metrics)
    out[declared_objective] = objective_value
    if system_spec is not None:
        out = validate_declared_metrics(out, system_spec)
        out[declared_objective] = objective_value
    return out


def evaluate_system(
    system: OLGSystem,
    controls: Controls,
    *,
    system_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate controls, evaluate the system, and normalize metrics."""

    validate_controls_for_system(system, controls)
    return validate_metrics(system.evaluate(controls), system_spec=system_spec)


def gradient_system(system: OLGSystem, controls: Controls) -> Controls:
    """Validate controls and return a gradient matching the system layout."""

    spec = validate_controls_for_system(system, controls)
    gradient = system.gradient(controls)
    if not isinstance(gradient, Controls):
        raise TypeError("system.gradient(...) must return Controls.")
    _validate_control_spec_compatibility(spec, gradient.spec)
    if not np.all(np.isfinite(gradient.as_matrix(copy=False))):
        raise ValueError("system.gradient(...) returned non-finite values.")
    return gradient


def value_and_gradient_system(
    system: Any,
    controls: Controls,
    *,
    system_spec: dict[str, Any] | None = None,
) -> ValueGradient:
    """Call and validate an optional combined objective/gradient hook."""

    hook = getattr(system, "value_and_gradient", None)
    if not callable(hook):
        raise AttributeError("system does not provide value_and_gradient(...).")
    spec = validate_controls_for_system(system, controls)
    raw = hook(controls)
    if isinstance(raw, ValueGradient):
        metrics, gradient = raw.metrics, raw.gradient
    elif isinstance(raw, tuple) and len(raw) == 2:
        metrics, gradient = raw
    else:
        raise TypeError("system.value_and_gradient(...) must return ValueGradient or (metrics, gradient).")
    normalized = validate_metrics(metrics, system_spec=system_spec)
    if not isinstance(gradient, Controls):
        raise TypeError("system.value_and_gradient(...) gradient must be Controls.")
    _validate_control_spec_compatibility(spec, gradient.spec)
    if not np.all(np.isfinite(gradient.as_matrix(copy=False))):
        raise ValueError("system.value_and_gradient(...) returned non-finite gradient values.")
    return ValueGradient(metrics=normalized, gradient=gradient)


def optional_smooth_gradient(system: Any, controls: Controls) -> Controls:
    """Call and validate the smooth component gradient of a composite objective."""

    hook = getattr(system, "smooth_gradient", None)
    if not callable(hook):
        raise AttributeError("system does not provide smooth_gradient(...).")
    spec = validate_controls_for_system(system, controls)
    gradient = hook(controls)
    if not isinstance(gradient, Controls):
        raise TypeError("system.smooth_gradient(...) must return Controls.")
    _validate_control_spec_compatibility(spec, gradient.spec)
    if not np.all(np.isfinite(gradient.as_matrix(copy=False))):
        raise ValueError("system.smooth_gradient(...) returned non-finite values.")
    return gradient


def optional_prox(system: Any, controls: Controls, step_size: float) -> Controls:
    """Call and validate an optional proximal operator."""

    hook = getattr(system, "prox", None)
    if not callable(hook):
        raise AttributeError("system does not provide prox(...).")
    spec = validate_controls_for_system(system, controls)
    alpha = float(step_size)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("prox step_size must be finite and positive.")
    result = hook(controls, alpha)
    if not isinstance(result, Controls):
        raise TypeError("system.prox(...) must return Controls.")
    _validate_control_spec_compatibility(spec, result.spec)
    if not np.all(np.isfinite(result.as_matrix(copy=False))):
        raise ValueError("system.prox(...) returned non-finite values.")
    return result


def optional_control_bounds(system: Any) -> ControlBounds:
    """Call and validate optional elementwise bounds against the system layout."""

    hook = getattr(system, "control_bounds", None)
    if not callable(hook):
        raise AttributeError("system does not provide control_bounds().")
    spec = validate_control_spec(system)
    raw = hook()
    if isinstance(raw, ControlBounds):
        lower, upper = raw.lower, raw.upper
    elif isinstance(raw, tuple) and len(raw) == 2:
        lower_raw, upper_raw = raw
        lower = lower_raw if isinstance(lower_raw, Controls) else Controls.from_matrix(spec, lower_raw, name="lower_bounds")
        upper = upper_raw if isinstance(upper_raw, Controls) else Controls.from_matrix(spec, upper_raw, name="upper_bounds")
    else:
        raise TypeError("system.control_bounds() must return ControlBounds or (lower, upper).")
    for label, value in (("lower", lower), ("upper", upper)):
        _validate_control_spec_compatibility(spec, value.spec)
    if np.any(lower.as_matrix(copy=False) > upper.as_matrix(copy=False)):
        raise ValueError("system.control_bounds() has lower values above upper values.")
    return ControlBounds(lower=lower, upper=upper)


def optional_control_transfer(
    system: Any,
    controls: Controls,
    *,
    operation: str,
    options: Mapping[str, Any] | None = None,
) -> tuple[Controls, dict[str, Any]]:
    """Call one explicit system-owned feasibility-preserving transfer hook."""

    hook = getattr(system, "transfer_controls", None)
    if not callable(hook):
        raise AttributeError("system does not provide transfer_controls(...).")
    validate_controls_for_system(system, controls)
    if not str(operation):
        raise ValueError("transfer operation must be non-empty.")
    raw = hook(controls, operation=str(operation), options=dict(options or {}))
    if isinstance(raw, tuple) and len(raw) == 2:
        transferred, metadata = raw
    else:
        transferred, metadata = raw, {}
    if not isinstance(transferred, Controls):
        raise TypeError("system.transfer_controls(...) must return Controls or (Controls, metadata).")
    validate_controls_for_system(system, transferred)
    if not isinstance(metadata, Mapping):
        raise TypeError("system.transfer_controls(...) metadata must be a mapping.")
    return transferred, dict(metadata)


def optional_residuals(system: Any, controls: Controls, *, name: str = "hard") -> np.ndarray:
    """Call optional residual hook and validate a 1D finite vector."""

    hook = getattr(system, "residuals", None)
    if not callable(hook):
        raise AttributeError("system does not provide residuals(...).")
    validate_controls_for_system(system, controls)
    residuals = np.asarray(hook(controls, name=name), dtype=float).reshape(-1)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("system.residuals(...) returned non-finite values.")
    return residuals


def optional_jacobian(system: Any, controls: Controls, *, name: str = "hard") -> np.ndarray:
    """Call optional Jacobian hook and validate a finite 2D matrix."""

    hook = getattr(system, "jacobian", None)
    if not callable(hook):
        raise AttributeError("system does not provide jacobian(...).")
    spec = validate_controls_for_system(system, controls)
    jacobian = np.asarray(hook(controls, name=name), dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("system.jacobian(...) must return a 2D matrix.")
    if jacobian.shape[1] != spec.size:
        raise ValueError(
            f"system.jacobian(...) second dimension must be {spec.size}, got {jacobian.shape[1]}."
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("system.jacobian(...) returned non-finite values.")
    return jacobian


def optional_hessian(system: Any, controls: Controls) -> np.ndarray:
    """Call optional dense Hessian hook and validate its shape."""

    hook = getattr(system, "hessian", None)
    if not callable(hook):
        raise AttributeError("system does not provide hessian(...).")
    spec = validate_controls_for_system(system, controls)
    hessian = np.asarray(hook(controls), dtype=float)
    expected = (spec.size, spec.size)
    if hessian.shape != expected:
        raise ValueError(f"system.hessian(...) must return shape {expected}, got {hessian.shape}.")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("system.hessian(...) returned non-finite values.")
    return hessian


def optional_hvp(system: Any, controls: Controls, vector: Controls | np.ndarray) -> Controls:
    """Call optional Hessian-vector product hook and validate Controls output."""

    hook = getattr(system, "hvp", None)
    if not callable(hook):
        raise AttributeError("system does not provide hvp(...).")
    spec = validate_controls_for_system(system, controls)
    result = hook(controls, vector)
    if isinstance(result, Controls):
        hvp = result
    else:
        hvp = Controls.from_flat(spec, np.asarray(result, dtype=spec.dtype).reshape(-1), name="hvp")
    _validate_control_spec_compatibility(spec, hvp.spec)
    if not np.all(np.isfinite(hvp.as_matrix(copy=False))):
        raise ValueError("system.hvp(...) returned non-finite values.")
    return hvp


def optional_plot_data(
    system: Any,
    controls: Controls,
    *,
    kind: str,
    **options: Any,
) -> Any:
    """Call and validate an explicit, system-owned visualization data product."""

    hook = getattr(system, "plot_data", None)
    if not callable(hook):
        raise AttributeError("system does not provide plot_data(...).")
    validate_controls_for_system(system, controls)
    if not str(kind):
        raise ValueError("plot_data kind must be non-empty.")
    from optimizer.visualization.models import coerce_plot_data

    result = coerce_plot_data(hook(controls, kind=str(kind), **options))
    if result.kind != str(kind):
        raise ValueError(
            f"system.plot_data(...) returned kind {result.kind!r}; expected {str(kind)!r}."
        )
    return result
