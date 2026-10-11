"""OLSS protocol, static discovery, and contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from optimizer.system_olss.specs import (
    SHOOTING_SYSTEM_SPEC_SCHEMA,
    validate_shooting_system_spec,
)
from optimizer.system_olss.variables import ShootingSpec, ShootingVariables


ShootingOptions = Mapping[str, Any]
ShootingMetrics = Mapping[str, Any]


@runtime_checkable
class OpenLoopShootingSystem(Protocol):
    def system_spec(self) -> Mapping[str, Any]: ...

    def shooting_spec(self) -> ShootingSpec: ...

    def residuals(
        self, variables: ShootingVariables, *, options: ShootingOptions | None = None
    ) -> np.ndarray: ...

    def jacobian(
        self, variables: ShootingVariables, *, options: ShootingOptions | None = None
    ) -> Any: ...

    def evaluate(
        self, variables: ShootingVariables, *, options: ShootingOptions | None = None
    ) -> ShootingMetrics: ...

    def verify(self, variables: ShootingVariables, *, options: ShootingOptions | None = None) -> Any: ...

    def with_secondary(self, **updates: Any) -> "OpenLoopShootingSystem": ...


@dataclass(frozen=True)
class ShootingSystemProbe:
    has_system_spec: bool
    has_shooting_spec: bool
    has_residuals: bool
    has_jacobian: bool
    has_evaluate: bool
    has_verify: bool
    has_with_secondary: bool
    has_residuals_and_jacobian: bool
    has_shooting_bounds: bool
    has_cost_gradient: bool
    has_recover_controls: bool
    has_forward_prop: bool
    has_simulate: bool
    has_plot_data: bool
    has_sparse_jacobian: bool
    has_jacobian_operator: bool
    has_roots_equivalent: bool
    has_cache_reset: bool
    has_cache_status: bool

    @property
    def required_ok(self) -> bool:
        return all(
            (
                self.has_system_spec,
                self.has_shooting_spec,
                self.has_residuals,
                self.has_jacobian,
                self.has_evaluate,
                self.has_verify,
                self.has_with_secondary,
            )
        )

    def to_dict(self) -> dict[str, bool]:
        payload = dict(vars(self))
        payload["required_ok"] = self.required_ok
        return payload


def probe_shooting_system(system: Any) -> ShootingSystemProbe:
    has = lambda name: callable(getattr(system, name, None))
    return ShootingSystemProbe(
        has_system_spec=has("system_spec"),
        has_shooting_spec=has("shooting_spec"),
        has_residuals=has("residuals"),
        has_jacobian=has("jacobian"),
        has_evaluate=has("evaluate"),
        has_verify=has("verify"),
        has_with_secondary=has("with_secondary"),
        has_residuals_and_jacobian=has("residuals_and_jacobian"),
        has_shooting_bounds=has("shooting_bounds"),
        has_cost_gradient=has("cost_gradient"),
        has_recover_controls=has("recover_controls"),
        has_forward_prop=has("forward_prop"),
        has_simulate=has("simulate"),
        has_plot_data=has("plot_data"),
        has_sparse_jacobian=has("sparse_jacobian"),
        has_jacobian_operator=has("jacobian_operator"),
        has_roots_equivalent=has("roots_equivalent"),
        has_cache_reset=has("cache_reset"),
        has_cache_status=has("cache_status"),
    )


def get_shooting_system_spec(system: Any) -> dict[str, Any]:
    hook = getattr(system, "system_spec", None)
    if not callable(hook):
        raise TypeError("OLSS system does not provide system_spec().")
    return validate_shooting_system_spec(hook())


def validate_shooting_spec(system: Any) -> ShootingSpec:
    hook = getattr(system, "shooting_spec", None)
    if not callable(hook):
        raise TypeError("OLSS system does not provide shooting_spec().")
    value = hook()
    if not isinstance(value, ShootingSpec):
        raise TypeError("OLSS shooting_spec() must return ShootingSpec.")
    declared = get_shooting_system_spec(system)["variables"]
    if int(declared["size"]) != value.size:
        raise ValueError("OLSS system_spec variables.size does not match shooting_spec().")
    return value


def require_shooting_system(system: Any) -> OpenLoopShootingSystem:
    probe = probe_shooting_system(system)
    if not probe.required_ok:
        missing = [
            name.removeprefix("has_")
            for name, present in probe.to_dict().items()
            if name.startswith("has_")
            and name
            in {
                "has_system_spec",
                "has_shooting_spec",
                "has_residuals",
                "has_jacobian",
                "has_evaluate",
                "has_verify",
                "has_with_secondary",
            }
            and not present
        ]
        raise TypeError(f"Object does not satisfy OLSS contract; missing: {', '.join(missing)}.")
    get_shooting_system_spec(system)
    validate_shooting_spec(system)
    return system


def describe_shooting_system(system: Any) -> dict[str, Any]:
    spec = get_shooting_system_spec(system)
    variable_spec = validate_shooting_spec(system)
    return {
        "family": "olss",
        "schema_version": SHOOTING_SYSTEM_SPEC_SCHEMA,
        "system_spec": {"available": True, "value": spec},
        "shooting_spec": variable_spec.to_dict(),
        "probe": probe_shooting_system(system).to_dict(),
        "description": {
            "available": True,
            "value": {
                "kind": "OLSS",
                "class": system.__class__.__name__,
                "identity": dict(spec["identity"]),
            },
        },
    }


def shooting_contract_info() -> dict[str, Any]:
    return {
        "family": "olss",
        "schema_version": SHOOTING_SYSTEM_SPEC_SCHEMA,
        "required": [
            "system_spec",
            "shooting_spec",
            "residuals",
            "jacobian",
            "evaluate",
            "verify",
            "with_secondary",
        ],
        "optional": [
            "residuals_and_jacobian",
            "shooting_bounds",
            "cost_gradient",
            "recover_controls",
            "sparse_jacobian",
            "jacobian_operator",
            "roots_equivalent",
            "simulate",
            "plot_data",
        ],
    }
