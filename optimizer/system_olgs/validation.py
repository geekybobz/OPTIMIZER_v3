"""Validation helpers for OLGS systems."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs.contract import Metrics, OLGSystem


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
    if controls.spec.keys != spec.keys or controls.spec.control_dim != spec.control_dim:
        raise ValueError(
            "controls do not match system control_spec: "
            f"system keys={spec.keys}, dim={spec.control_dim}; "
            f"controls keys={controls.spec.keys}, dim={controls.spec.control_dim}."
        )
    return spec


def validate_metrics(metrics: Metrics) -> dict[str, Any]:
    """Validate and normalize an evaluation metrics mapping."""

    from collections.abc import Mapping

    if not isinstance(metrics, Mapping):
        raise TypeError("system.evaluate(...) must return a metrics mapping.")
    if "J" not in metrics:
        raise KeyError("system.evaluate(...) metrics must include key 'J'.")
    j_value = float(metrics["J"])
    if not np.isfinite(j_value):
        raise ValueError("metric 'J' must be finite.")
    out = dict(metrics)
    out["J"] = j_value
    return out


def evaluate_system(system: OLGSystem, controls: Controls) -> dict[str, Any]:
    """Validate controls, evaluate the system, and normalize metrics."""

    validate_controls_for_system(system, controls)
    return validate_metrics(system.evaluate(controls))


def gradient_system(system: OLGSystem, controls: Controls) -> Controls:
    """Validate controls and return a gradient matching the system layout."""

    spec = validate_controls_for_system(system, controls)
    gradient = system.gradient(controls)
    if not isinstance(gradient, Controls):
        raise TypeError("system.gradient(...) must return Controls.")
    if gradient.spec.keys != spec.keys or gradient.spec.control_dim != spec.control_dim:
        raise ValueError("system.gradient(...) returned controls with the wrong layout.")
    if not np.all(np.isfinite(gradient.as_matrix(copy=False))):
        raise ValueError("system.gradient(...) returned non-finite values.")
    return gradient


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
    if hvp.spec.keys != spec.keys or hvp.spec.control_dim != spec.control_dim:
        raise ValueError("system.hvp(...) returned controls with the wrong layout.")
    if not np.all(np.isfinite(hvp.as_matrix(copy=False))):
        raise ValueError("system.hvp(...) returned non-finite values.")
    return hvp
