"""System contract helpers for OPTIMIZER v3.

Why this file exists
--------------------
The optimizer library needs a stable way to talk to many different physical systems.
The downstream systems already derive the objective and gradient analytically, often
with cost prefactors such as ``control_weight``, ``lambda2``, ``lambda4``, and
``energy_weight`` embedded in the dynamics and costate equations.  Therefore v3 should
not wrap those systems with an external objective builder.

This file defines the small optimizer-facing contract instead:

    control_spec() -> ControlSpec
    evaluate(controls) -> metrics with "J"
    gradient(controls) -> Controls
    with_params(**updates) -> new system

Advanced systems may also provide residuals, Jacobians, Hessians, Hessian-vector
products, or simulation helpers.  Those hooks are optional because not every optimizer
needs them.

How it fits the architecture
----------------------------
- controls.py defines the control layout.
- system.py defines what a physical model must expose to the optimizer.
- result/state will later store evaluated metrics and warmstart state.
- the engine will call the validation helpers here before running optimizers.
- diagnostics and repair tools will use optional residual/Jacobian helpers.

What this file deliberately does not do
---------------------------------------
It does not implement propagation, compute a physical objective, build gradients by
finite differences as the normal path, or hold run state.  It only defines and checks
the boundary between a system and the optimizer library.

Reviewer invariants
-------------------
- ``evaluate`` must return a mapping containing finite scalar ``J``.
- ``gradient`` must return ``Controls`` with the same layout as ``control_spec``.
- controls passed to a system must match the system's keys and control dimension.
- optional residuals must be finite 1D arrays.
- optional Jacobians must be finite 2D arrays with second dimension equal to the
  flattened control size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from optimizer.controls import ControlSpec, Controls


Metrics = Mapping[str, Any]


@runtime_checkable
class OptimizerSystem(Protocol):
    """Protocol every optimizer-facing system should satisfy."""

    def control_spec(self) -> ControlSpec:
        """Return the controls expected by this system."""

    def evaluate(self, controls: Controls) -> Metrics:
        """Evaluate controls and return metrics including scalar key ``J``."""

    def gradient(self, controls: Controls) -> Controls:
        """Return the analytical gradient/update signal in the same layout."""

    def with_params(self, **updates: Any) -> "OptimizerSystem":
        """Return an equivalent system with updated cost/model params."""


@dataclass(frozen=True)
class SystemProbe:
    """Summary of which system hooks are present."""

    has_control_spec: bool
    has_evaluate: bool
    has_gradient: bool
    has_with_params: bool
    has_residuals: bool
    has_jacobian: bool
    has_hessian: bool
    has_hvp: bool
    has_simulate: bool

    @property
    def required_ok(self) -> bool:
        """Whether the system has the minimum hooks for first-order optimization."""

        return (
            self.has_control_spec
            and self.has_evaluate
            and self.has_gradient
            and self.has_with_params
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "has_control_spec": self.has_control_spec,
            "has_evaluate": self.has_evaluate,
            "has_gradient": self.has_gradient,
            "has_with_params": self.has_with_params,
            "has_residuals": self.has_residuals,
            "has_jacobian": self.has_jacobian,
            "has_hessian": self.has_hessian,
            "has_hvp": self.has_hvp,
            "has_simulate": self.has_simulate,
            "required_ok": self.required_ok,
        }


def probe_system(system: Any) -> SystemProbe:
    """Inspect hook presence without calling expensive methods."""

    return SystemProbe(
        has_control_spec=callable(getattr(system, "control_spec", None)),
        has_evaluate=callable(getattr(system, "evaluate", None)),
        has_gradient=callable(getattr(system, "gradient", None)),
        has_with_params=callable(getattr(system, "with_params", None)),
        has_residuals=callable(getattr(system, "residuals", None)),
        has_jacobian=callable(getattr(system, "jacobian", None)),
        has_hessian=callable(getattr(system, "hessian", None)),
        has_hvp=callable(getattr(system, "hvp", None)),
        has_simulate=callable(getattr(system, "simulate", None)),
    )


def require_system(system: Any) -> OptimizerSystem:
    """Return system when required hooks are present, else raise clearly."""

    probe = probe_system(system)
    if probe.required_ok:
        return system
    missing = [
        name
        for name, ok in (
            ("control_spec", probe.has_control_spec),
            ("evaluate", probe.has_evaluate),
            ("gradient", probe.has_gradient),
            ("with_params", probe.has_with_params),
        )
        if not ok
    ]
    raise TypeError(f"System is missing required optimizer hooks: {', '.join(missing)}.")


def validate_control_spec(system: OptimizerSystem) -> ControlSpec:
    """Call and validate ``system.control_spec()``."""

    spec = system.control_spec()
    if not isinstance(spec, ControlSpec):
        raise TypeError("system.control_spec() must return ControlSpec.")
    return spec


def validate_controls_for_system(system: OptimizerSystem, controls: Controls) -> ControlSpec:
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


def evaluate_system(system: OptimizerSystem, controls: Controls) -> dict[str, Any]:
    """Validate controls, evaluate the system, and normalize metrics."""

    validate_controls_for_system(system, controls)
    return validate_metrics(system.evaluate(controls))


def gradient_system(system: OptimizerSystem, controls: Controls) -> Controls:
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
