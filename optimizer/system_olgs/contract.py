"""OLGS protocol and hook discovery.

OLGS means Open Loop Gradient System.  This module defines the optimizer-facing
contract for systems that provide analytical objective gradients.  The required
surface is intentionally small:

    control_spec()
    evaluate(controls)
    gradient(controls)
    with_secondary(**updates)

During migration, ``with_params`` is accepted as a compatibility update hook.  New
systems should implement ``with_secondary`` because secondary params are the
curriculum/objective weights that optimizers commonly change between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from optimizer.controls import ControlSpec, Controls


Metrics = Mapping[str, Any]


@runtime_checkable
class OLGSystem(Protocol):
    """Protocol for the target OLGS public surface."""

    def control_spec(self) -> ControlSpec:
        """Return the controls expected by this system."""

    def evaluate(self, controls: Controls) -> Metrics:
        """Evaluate controls and return metrics including scalar key ``J``."""

    def gradient(self, controls: Controls) -> Controls:
        """Return analytical ``dJ/du`` in the same control layout."""

    def with_secondary(self, **updates: Any) -> "OLGSystem":
        """Return an equivalent system with updated secondary parameters."""


OptimizerSystem = OLGSystem


@dataclass(frozen=True)
class SystemProbe:
    """Summary of required, recommended, and optional OLGS hooks."""

    has_control_spec: bool
    has_evaluate: bool
    has_gradient: bool
    has_with_secondary: bool
    has_with_params: bool
    has_forward_prop: bool
    has_back_prop: bool
    has_metrics: bool
    has_describe: bool
    has_residuals: bool
    has_jacobian: bool
    has_hessian: bool
    has_hvp: bool
    has_second_derivative: bool
    has_simulate: bool
    has_cache_reset: bool
    has_cache_status: bool

    @property
    def has_update_hook(self) -> bool:
        """Whether the system can return a param-updated copy."""

        return self.has_with_secondary or self.has_with_params

    @property
    def required_ok(self) -> bool:
        """Whether the system has the minimum hooks for OLGS optimization."""

        return (
            self.has_control_spec
            and self.has_evaluate
            and self.has_gradient
            and self.has_update_hook
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "has_control_spec": self.has_control_spec,
            "has_evaluate": self.has_evaluate,
            "has_gradient": self.has_gradient,
            "has_with_secondary": self.has_with_secondary,
            "has_with_params": self.has_with_params,
            "has_update_hook": self.has_update_hook,
            "has_forward_prop": self.has_forward_prop,
            "has_back_prop": self.has_back_prop,
            "has_metrics": self.has_metrics,
            "has_describe": self.has_describe,
            "has_residuals": self.has_residuals,
            "has_jacobian": self.has_jacobian,
            "has_hessian": self.has_hessian,
            "has_hvp": self.has_hvp,
            "has_second_derivative": self.has_second_derivative,
            "has_simulate": self.has_simulate,
            "has_cache_reset": self.has_cache_reset,
            "has_cache_status": self.has_cache_status,
            "required_ok": self.required_ok,
        }


def probe_system(system: Any) -> SystemProbe:
    """Inspect OLGS hook presence without calling expensive methods."""

    return SystemProbe(
        has_control_spec=callable(getattr(system, "control_spec", None)),
        has_evaluate=callable(getattr(system, "evaluate", None)),
        has_gradient=callable(getattr(system, "gradient", None)),
        has_with_secondary=callable(getattr(system, "with_secondary", None)),
        has_with_params=callable(getattr(system, "with_params", None)),
        has_forward_prop=callable(getattr(system, "forward_prop", None)),
        has_back_prop=callable(getattr(system, "back_prop", None)),
        has_metrics=callable(getattr(system, "metrics", None)),
        has_describe=callable(getattr(system, "describe", None)),
        has_residuals=callable(getattr(system, "residuals", None)),
        has_jacobian=callable(getattr(system, "jacobian", None)),
        has_hessian=callable(getattr(system, "hessian", None)),
        has_hvp=callable(getattr(system, "hvp", None)),
        has_second_derivative=callable(getattr(system, "second_derivative", None)),
        has_simulate=callable(getattr(system, "simulate", None)),
        has_cache_reset=callable(getattr(system, "cache_reset", None)),
        has_cache_status=callable(getattr(system, "cache_status", None)),
    )


def get_secondary_update_hook(system: Any) -> Any:
    """Return the preferred secondary-param update hook."""

    hook = getattr(system, "with_secondary", None)
    if callable(hook):
        return hook
    hook = getattr(system, "with_params", None)
    if callable(hook):
        return hook
    raise TypeError("System does not provide with_secondary(...) or with_params(...).")


def require_system(system: Any) -> OLGSystem:
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
            ("with_secondary or with_params", probe.has_update_hook),
        )
        if not ok
    ]
    raise TypeError(f"System is missing required OLGS hooks: {', '.join(missing)}.")
