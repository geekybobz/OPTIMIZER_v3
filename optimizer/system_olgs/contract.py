"""OLGS protocol and hook discovery.

OLGS means Open Loop Gradient System.  This module defines the optimizer-facing
contract for systems that provide analytical objective gradients.  The required
surface is intentionally small:

    system_spec()
    control_spec()
    evaluate(controls)
    gradient(controls)
    with_secondary(**updates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs.specs import SYSTEM_SPEC_SCHEMA, validate_system_spec


Metrics = Mapping[str, Any]
TrackingSpec = Mapping[str, Any]
TRACKING_SPEC_SCHEMA = "optimizer.tracking-spec/v1"


@dataclass(frozen=True)
class ValueGradient:
    """Validated system-owned objective metrics and analytical gradient pair."""

    metrics: Mapping[str, Any]
    gradient: Controls


@dataclass(frozen=True)
class ControlBounds:
    """Elementwise lower and upper limits in the system control layout."""

    lower: Controls
    upper: Controls


@runtime_checkable
class SupportsValueAndGradient(Protocol):
    """Optional capability for sharing work between value and gradient calls."""

    def value_and_gradient(self, controls: Controls) -> ValueGradient | tuple[Metrics, Controls]: ...


@runtime_checkable
class SupportsEqualityConstraints(Protocol):
    """Optional hard equality-constraint capability."""

    def residuals(self, controls: Controls, *, name: str = "hard") -> np.ndarray: ...

    def jacobian(self, controls: Controls, *, name: str = "hard") -> np.ndarray: ...


@runtime_checkable
class SupportsBounds(Protocol):
    """Optional elementwise control-bound capability."""

    def control_bounds(self) -> ControlBounds | tuple[Any, Any]: ...


@runtime_checkable
class SupportsControlTransfer(Protocol):
    """Optional system-owned feasibility-preserving control transformation."""

    def transfer_controls(
        self,
        controls: Controls,
        *,
        operation: str,
        options: Mapping[str, Any],
    ) -> Controls | tuple[Controls, Mapping[str, Any]]: ...


@runtime_checkable
class SupportsHVP(Protocol):
    """Optional Hessian-vector product capability for second-order methods."""

    def hvp(self, controls: Controls, vector: Controls | np.ndarray) -> Controls | np.ndarray: ...


@runtime_checkable
class SupportsProximalObjective(Protocol):
    """Optional composite-objective capability for proximal methods."""

    def smooth_gradient(self, controls: Controls) -> Controls: ...

    def prox(self, controls: Controls, step_size: float) -> Controls: ...


@runtime_checkable
class SupportsPlotData(Protocol):
    """Optional explicit data-product capability for visualization tools."""

    def plot_data(self, controls: Controls, *, kind: str, **options: Any) -> Any: ...


@runtime_checkable
class OLGSystem(Protocol):
    """Protocol for the target OLGS public surface."""

    def system_spec(self) -> Mapping[str, Any]:
        """Return complete static system, metric, and derivative metadata."""

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

    has_system_spec: bool
    has_control_spec: bool
    has_evaluate: bool
    has_gradient: bool
    has_with_secondary: bool
    has_forward_prop: bool
    has_back_prop: bool
    has_metrics: bool
    has_describe: bool
    has_tracking_spec: bool
    has_constraint_spec: bool
    has_value_and_gradient: bool
    has_residuals: bool
    has_jacobian: bool
    has_control_bounds: bool
    has_control_transfer: bool
    has_hessian: bool
    has_hvp: bool
    has_smooth_gradient: bool
    has_prox: bool
    has_second_derivative: bool
    has_simulate: bool
    has_plot_data: bool
    has_cache_reset: bool
    has_cache_status: bool

    @property
    def has_update_hook(self) -> bool:
        """Whether the system can return a param-updated copy."""

        return self.has_with_secondary

    @property
    def required_ok(self) -> bool:
        """Whether the system has the minimum hooks for OLGS optimization."""

        return (
            self.has_system_spec
            and self.has_control_spec
            and self.has_evaluate
            and self.has_gradient
            and self.has_update_hook
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "has_system_spec": self.has_system_spec,
            "has_control_spec": self.has_control_spec,
            "has_evaluate": self.has_evaluate,
            "has_gradient": self.has_gradient,
            "has_with_secondary": self.has_with_secondary,
            "has_update_hook": self.has_update_hook,
            "has_forward_prop": self.has_forward_prop,
            "has_back_prop": self.has_back_prop,
            "has_metrics": self.has_metrics,
            "has_describe": self.has_describe,
            "has_tracking_spec": self.has_tracking_spec,
            "has_constraint_spec": self.has_constraint_spec,
            "has_value_and_gradient": self.has_value_and_gradient,
            "has_residuals": self.has_residuals,
            "has_jacobian": self.has_jacobian,
            "has_control_bounds": self.has_control_bounds,
            "has_control_transfer": self.has_control_transfer,
            "has_hessian": self.has_hessian,
            "has_hvp": self.has_hvp,
            "has_smooth_gradient": self.has_smooth_gradient,
            "has_prox": self.has_prox,
            "has_second_derivative": self.has_second_derivative,
            "has_simulate": self.has_simulate,
            "has_plot_data": self.has_plot_data,
            "has_cache_reset": self.has_cache_reset,
            "has_cache_status": self.has_cache_status,
            "required_ok": self.required_ok,
        }


def probe_system(system: Any) -> SystemProbe:
    """Inspect OLGS hook presence without calling expensive methods."""

    return SystemProbe(
        has_system_spec=callable(getattr(system, "system_spec", None)),
        has_control_spec=callable(getattr(system, "control_spec", None)),
        has_evaluate=callable(getattr(system, "evaluate", None)),
        has_gradient=callable(getattr(system, "gradient", None)),
        has_with_secondary=callable(getattr(system, "with_secondary", None)),
        has_forward_prop=callable(getattr(system, "forward_prop", None)),
        has_back_prop=callable(getattr(system, "back_prop", None)),
        has_metrics=callable(getattr(system, "metrics", None)),
        has_describe=callable(getattr(system, "describe", None)),
        has_tracking_spec=callable(getattr(system, "tracking_spec", None)),
        has_constraint_spec=callable(getattr(system, "constraint_spec", None)),
        has_value_and_gradient=callable(getattr(system, "value_and_gradient", None)),
        has_residuals=callable(getattr(system, "residuals", None)),
        has_jacobian=callable(getattr(system, "jacobian", None)),
        has_control_bounds=callable(getattr(system, "control_bounds", None)),
        has_control_transfer=callable(getattr(system, "transfer_controls", None)),
        has_hessian=callable(getattr(system, "hessian", None)),
        has_hvp=callable(getattr(system, "hvp", None)),
        has_smooth_gradient=callable(getattr(system, "smooth_gradient", None)),
        has_prox=callable(getattr(system, "prox", None)),
        has_second_derivative=callable(getattr(system, "second_derivative", None)),
        has_simulate=callable(getattr(system, "simulate", None)),
        has_plot_data=callable(getattr(system, "plot_data", None)),
        has_cache_reset=callable(getattr(system, "cache_reset", None)),
        has_cache_status=callable(getattr(system, "cache_status", None)),
    )


def get_system_spec(system: Any) -> dict[str, Any]:
    """Return the mandatory static system specification without numerical work."""

    hook = getattr(system, "system_spec", None)
    if not callable(hook):
        raise TypeError("System does not provide required system_spec().")
    value = hook()
    if not isinstance(value, Mapping):
        raise TypeError("System system_spec() must return a mapping.")
    return validate_system_spec(value)


def get_tracking_spec(system: Any) -> dict[str, Any] | None:
    """Return optional static tracking metadata declared by a system.

    tracking_spec() is deliberately not part of the required OLGS protocol. When
    present, it must return a mapping of system-chosen metric identifiers and
    metadata. It must not evaluate controls or trigger numerical work.
    """

    hook = getattr(system, "tracking_spec", None)
    if not callable(hook):
        spec = get_system_spec(system)
        return {
            "schema_version": TRACKING_SPEC_SCHEMA,
            "objective": {
                "metric_id": spec["objective"]["metric_id"],
                "direction": spec["objective"]["direction"],
            },
            "metrics": dict(spec["metrics"]),
        }
    value = hook()
    if not isinstance(value, Mapping):
        raise TypeError("System tracking_spec() must return a mapping.")
    payload = dict(value)
    if "objective" in payload and not isinstance(payload["objective"], Mapping):
        raise TypeError("System tracking_spec()['objective'] must be a mapping when provided.")
    if "metrics" in payload and not isinstance(payload["metrics"], Mapping):
        raise TypeError("System tracking_spec()['metrics'] must be a mapping when provided.")
    return payload


def _static_mapping_hook(system: Any, name: str) -> dict[str, Any]:
    """Capture an optional static hook without turning an invalid hook into I/O."""

    hook = getattr(system, name, None)
    if not callable(hook):
        return {"available": False}
    try:
        if name == "system_spec":
            value = get_system_spec(system)
        elif name == "tracking_spec":
            value = get_tracking_spec(system)
        else:
            value = hook()
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"System {name}() must return a mapping.")
    except Exception as exc:
        return {
            "available": True,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "valid": True,
        "value": None if value is None else dict(value),
    }


def contract_info() -> dict[str, Any]:
    """Return the generic OLGS contract without inspecting a concrete system."""

    return {
        "schema_version": "optimizer.system-contract/v1",
        "required_hooks": [
            "system_spec",
            "control_spec",
            "evaluate",
            "gradient",
            "with_secondary",
        ],
        "optional_static_hooks": {
            "describe": "Legacy compact description; system_spec is canonical.",
            "tracking_spec": (
                f"Returns optional metric metadata with schema {TRACKING_SPEC_SCHEMA}; "
                "it must not evaluate controls."
            ),
            "constraint_spec": "Returns static constraint labels/scales without numerical work.",
        },
        "optional_numerical_hooks": [
            "value_and_gradient",
            "forward_prop",
            "back_prop",
            "metrics",
            "residuals",
            "jacobian",
            "control_bounds",
            "transfer_controls",
            "hessian",
            "hvp",
            "smooth_gradient",
            "prox",
            "second_derivative",
            "simulate",
            "plot_data",
        ],
        "system_spec_schema": SYSTEM_SPEC_SCHEMA,
        "tracking_spec_schema": TRACKING_SPEC_SCHEMA,
        "capability_bundles": {
            "value_and_gradient": ["value_and_gradient"],
            "equality_constraints": ["residuals", "jacobian"],
            "bounds": ["control_bounds"],
            "control_transfer": ["transfer_controls"],
            "second_order": ["hvp"],
            "proximal": ["smooth_gradient", "prox"],
            "plot_data": ["plot_data"],
        },
    }


def describe_system(system: Any) -> dict[str, Any]:
    """Describe declared system capabilities without evaluating controls.

    The optional hooks are called only because the caller explicitly requested
    self-description. A bad optional hook is reported as invalid rather than
    affecting the optimizer's required OLGS contract.
    """

    probe = probe_system(system)
    system_spec = _static_mapping_hook(system, "system_spec")
    base_complete = probe.required_ok and system_spec.get("valid") is True
    return {
        "schema_version": "optimizer.system-info/v1",
        "type": {
            "module": type(system).__module__,
            "qualname": type(system).__qualname__,
        },
        "probe": probe.to_dict(),
        "system_spec": system_spec,
        "capabilities": {
            "base_olgs": {"available": probe.required_ok, "complete": base_complete},
            "value_and_gradient": {"available": probe.has_value_and_gradient},
            "equality_constraints": {
                "residuals": probe.has_residuals,
                "jacobian": probe.has_jacobian,
                "complete": probe.has_residuals and probe.has_jacobian,
            },
            "bounds": {"available": probe.has_control_bounds},
            "control_transfer": {"available": probe.has_control_transfer},
            "second_order": {
                "hvp": probe.has_hvp,
                "hessian": probe.has_hessian,
                "newton_cg_ready": probe.has_hvp,
            },
            "proximal": {
                "smooth_gradient": probe.has_smooth_gradient,
                "prox": probe.has_prox,
                "complete": probe.has_smooth_gradient and probe.has_prox,
            },
            "plot_data": {"available": probe.has_plot_data},
        },
        "description": _static_mapping_hook(system, "describe"),
        "tracking_spec": _static_mapping_hook(system, "tracking_spec"),
        "constraint_spec": _static_mapping_hook(system, "constraint_spec"),
    }


def get_secondary_update_hook(system: Any) -> Any:
    """Return the preferred secondary-param update hook."""

    hook = getattr(system, "with_secondary", None)
    if callable(hook):
        return hook
    raise TypeError("System does not provide with_secondary(...).")


def require_system(system: Any) -> OLGSystem:
    """Return a system whose required hooks and static specification are valid."""

    probe = probe_system(system)
    if probe.required_ok:
        get_system_spec(system)
        return system
    missing = [
        name
        for name, ok in (
            ("system_spec", probe.has_system_spec),
            ("control_spec", probe.has_control_spec),
            ("evaluate", probe.has_evaluate),
            ("gradient", probe.has_gradient),
            ("with_secondary", probe.has_update_hook),
        )
        if not ok
    ]
    raise TypeError(f"System is missing required OLGS hooks: {', '.join(missing)}.")
