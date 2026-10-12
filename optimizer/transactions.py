"""Family-aware request/response adapter for shared core, CLI, and MCP layers.

A transaction decodes its payload against the system's own layout, calls the
system through the same validated helpers the numerical engines use, and returns
success or a structured failure.  Inputs may be live objects or their plain
``to_dict()`` forms, so a request that passed through JSON executes unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np


def _plain(value: Any) -> Any:
    from optimizer.controls import Controls
    from optimizer.result import controls_to_dict

    if isinstance(value, Controls):
        return _plain(controls_to_dict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "system_spec") and callable(value.system_spec):
        return {
            "system_type": type(value).__name__,
            "system_spec": _plain(value.system_spec()),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class SystemRequest:
    """One explicit operation against an OLGS or OLSS system."""

    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not str(self.operation).strip():
            raise ValueError("operation must be non-empty.")
        object.__setattr__(self, "operation", str(self.operation))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "options", dict(self.options))
        if self.family is not None and self.family not in {"olgs", "olss"}:
            raise ValueError("family must be 'olgs', 'olss', or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "family": self.family,
            "operation": self.operation,
            "payload": _plain(self.payload),
            "options": _plain(self.options),
        }


@dataclass(frozen=True)
class SystemResponse:
    """Stable response envelope for a system transaction."""

    request_id: str
    family: str
    operation: str
    ok: bool
    result: Any = None
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "family": self.family,
            "operation": self.operation,
            "ok": self.ok,
            "result": _plain(self.result),
            "error": None if self.error is None else _plain(self.error),
        }


class _TransactionFailure(Exception):
    """A failure with a stable transaction error type."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _family(system: Any) -> str:
    hook = getattr(system, "system_spec", None)
    if not callable(hook):
        raise TypeError("System transaction requires system_spec().")
    value = hook()
    family = value.get("family") if isinstance(value, Mapping) else None
    if family in {None, "olgs"}:
        return "olgs"
    if family == "olss":
        return "olss"
    raise ValueError(f"Unsupported system family {family!r}.")


# ----------------------------------------------------------------------
# Payload decoding against the system's own layout
# ----------------------------------------------------------------------


def _decode_controls(system: Any, value: Any, *, field_name: str = "controls") -> Any:
    from optimizer.controls import Controls

    if isinstance(value, Controls):
        return value
    spec = system.control_spec()
    try:
        if isinstance(value, Mapping):
            if "matrix" in value:
                declared = value.get("spec")
                if isinstance(declared, Mapping) and (
                    list(declared.get("keys", spec.keys)) != list(spec.keys)
                    or int(declared.get("control_dim", spec.control_dim)) != spec.control_dim
                ):
                    raise ValueError("payload control layout does not match system.control_spec().")
                return Controls.from_matrix(
                    spec,
                    np.asarray(value["matrix"], dtype=spec.dtype),
                    name=value.get("name"),
                    meta=value.get("meta"),
                )
            if "flat" in value:
                return Controls.from_flat(spec, np.asarray(value["flat"], dtype=spec.dtype))
            return Controls.from_dict(spec, value)
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=spec.dtype)
            if array.ndim == 2:
                return Controls.from_matrix(spec, array)
            if array.ndim == 1:
                return Controls.from_flat(spec, array)
    except (KeyError, TypeError, ValueError) as exc:
        raise _TransactionFailure("invalid_payload", f"{field_name}: {exc}") from exc
    raise _TransactionFailure(
        "invalid_payload",
        f"{field_name} must be Controls, a controls mapping, or an array; got {type(value).__name__}.",
    )


def _decode_variables(system: Any, value: Any) -> Any:
    from optimizer.system_olss import ShootingVariables

    if isinstance(value, ShootingVariables):
        return value
    spec = system.shooting_spec()
    try:
        if isinstance(value, Mapping):
            if "values" in value:
                declared = value.get("spec")
                if isinstance(declared, Mapping) and "blocks" in declared:
                    names = [block.get("name") for block in declared["blocks"]]
                    if names != list(spec.names) or int(declared.get("size", spec.size)) != spec.size:
                        raise ValueError("payload variable layout does not match system.shooting_spec().")
                return ShootingVariables.from_flat(
                    spec,
                    value["values"],
                    name=str(value.get("name", "shooting")),
                    metadata=value.get("metadata"),
                )
            return ShootingVariables.from_blocks(spec, value)
        if isinstance(value, (list, tuple, np.ndarray)):
            return ShootingVariables.from_flat(spec, value)
    except (KeyError, TypeError, ValueError) as exc:
        raise _TransactionFailure("invalid_payload", f"variables: {exc}") from exc
    raise _TransactionFailure(
        "invalid_payload",
        f"variables must be ShootingVariables, a variables mapping, or an array; got {type(value).__name__}.",
    )


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise _TransactionFailure("invalid_payload", f"payload requires {key!r}.")
    return payload[key]


# ----------------------------------------------------------------------
# Operation table: one validated handler per supported system call
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Operation:
    handler: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any]
    hooks: tuple[str, ...] = ()
    static: bool = False
    accepts_options: bool = False


def _olgs_operations() -> dict[str, _Operation]:
    from optimizer.result import controls_to_dict
    from optimizer.system_olgs import (
        evaluate_system,
        get_system_spec,
        gradient_system,
        optional_control_bounds,
        optional_hessian,
        optional_hvp,
        optional_jacobian,
        optional_prox,
        optional_residuals,
        optional_smooth_gradient,
        value_and_gradient_system,
    )

    def controls(system: Any, payload: Mapping[str, Any]) -> Any:
        return _decode_controls(system, _required(payload, "controls"))

    def value_and_gradient(system: Any, payload: Mapping[str, Any], _options: Mapping[str, Any]) -> Any:
        pair = value_and_gradient_system(system, controls(system, payload), system_spec=get_system_spec(system))
        return {"metrics": dict(pair.metrics), "gradient": pair.gradient}

    def control_bounds(system: Any, _payload: Mapping[str, Any], _options: Mapping[str, Any]) -> Any:
        bounds = optional_control_bounds(system)
        return {"lower": controls_to_dict(bounds.lower), "upper": controls_to_dict(bounds.upper)}

    return {
        "system_spec": _Operation(lambda system, payload, options: get_system_spec(system), static=True),
        "control_spec": _Operation(lambda system, payload, options: system.control_spec(), static=True),
        "evaluate": _Operation(
            lambda system, payload, options: evaluate_system(
                system, controls(system, payload), system_spec=get_system_spec(system)
            ),
            hooks=("evaluate",),
        ),
        "gradient": _Operation(
            lambda system, payload, options: gradient_system(system, controls(system, payload)),
            hooks=("gradient",),
        ),
        "value_and_gradient": _Operation(value_and_gradient, hooks=("value_and_gradient",)),
        "residuals": _Operation(
            lambda system, payload, options: optional_residuals(
                system, controls(system, payload), name=str(payload.get("name", "hard"))
            ),
            hooks=("residuals",),
        ),
        "jacobian": _Operation(
            lambda system, payload, options: optional_jacobian(
                system, controls(system, payload), name=str(payload.get("name", "hard"))
            ),
            hooks=("jacobian",),
        ),
        "hvp": _Operation(
            lambda system, payload, options: optional_hvp(
                system,
                controls(system, payload),
                _decode_controls(system, _required(payload, "vector"), field_name="vector"),
            ),
            hooks=("hvp",),
        ),
        "hessian": _Operation(
            lambda system, payload, options: optional_hessian(system, controls(system, payload)),
            hooks=("hessian",),
        ),
        "smooth_gradient": _Operation(
            lambda system, payload, options: optional_smooth_gradient(system, controls(system, payload)),
            hooks=("smooth_gradient",),
        ),
        "prox": _Operation(
            lambda system, payload, options: optional_prox(
                system, controls(system, payload), float(_required(payload, "step_size"))
            ),
            hooks=("prox",),
        ),
        "control_bounds": _Operation(control_bounds, hooks=("control_bounds",)),
    }


def _olss_operations() -> dict[str, _Operation]:
    from optimizer.system_olss import (
        cost_gradient_system,
        evaluate_shooting_system,
        get_shooting_system_spec,
        jacobian_system,
        residuals_and_jacobian_system,
        residuals_system,
        shooting_bounds_system,
        verify_shooting_system,
    )

    def variables(system: Any, payload: Mapping[str, Any]) -> Any:
        return _decode_variables(system, _required(payload, "variables"))

    def residuals_and_jacobian(system: Any, payload: Mapping[str, Any], options: Mapping[str, Any]) -> Any:
        residuals, jacobian = residuals_and_jacobian_system(system, variables(system, payload), options=options)
        return {"residuals": residuals, "jacobian": jacobian}

    def shooting_bounds(system: Any, _payload: Mapping[str, Any], _options: Mapping[str, Any]) -> Any:
        lower, upper = shooting_bounds_system(system)
        return {"lower": lower, "upper": upper}

    return {
        "system_spec": _Operation(lambda system, payload, options: get_shooting_system_spec(system), static=True),
        "shooting_spec": _Operation(lambda system, payload, options: system.shooting_spec(), static=True),
        "residuals": _Operation(
            lambda system, payload, options: residuals_system(system, variables(system, payload), options=options),
            hooks=("residuals",),
            accepts_options=True,
        ),
        "jacobian": _Operation(
            lambda system, payload, options: jacobian_system(system, variables(system, payload), options=options),
            hooks=("jacobian",),
            accepts_options=True,
        ),
        "residuals_and_jacobian": _Operation(
            residuals_and_jacobian, hooks=("residuals", "jacobian"), accepts_options=True
        ),
        "evaluate": _Operation(
            lambda system, payload, options: evaluate_shooting_system(
                system, variables(system, payload), options=options
            ),
            hooks=("evaluate",),
            accepts_options=True,
        ),
        "verify": _Operation(
            lambda system, payload, options: verify_shooting_system(
                system,
                variables(system, payload),
                options=options,
                independence_source="transaction_request",
            ),
            hooks=("verify",),
            accepts_options=True,
        ),
        "cost_gradient": _Operation(
            lambda system, payload, options: cost_gradient_system(system, variables(system, payload), options=options),
            hooks=("cost_gradient",),
            accepts_options=True,
        ),
        "shooting_bounds": _Operation(shooting_bounds, hooks=("shooting_bounds",)),
    }


_OPERATIONS: dict[str, Callable[[], dict[str, _Operation]]] = {
    "olgs": _olgs_operations,
    "olss": _olss_operations,
}


def _execute(system: Any, family: str, request: SystemRequest) -> Any:
    payload = dict(request.payload)
    options = dict(request.options)
    if request.operation == "with_secondary":
        if options:
            raise _TransactionFailure("invalid_payload", "with_secondary does not accept options.")
        return system.with_secondary(**payload)
    table = _OPERATIONS[family]()
    operation = table.get(request.operation)
    if operation is None:
        supported = ", ".join(sorted([*table, "with_secondary"]))
        raise _TransactionFailure(
            "unsupported_operation",
            f"Unsupported {family} operation {request.operation!r}; supported: {supported}.",
        )
    if operation.static and (payload or options):
        raise _TransactionFailure("invalid_payload", f"{request.operation} does not accept payload or options.")
    if options and not operation.accepts_options:
        raise _TransactionFailure("invalid_payload", f"{request.operation} does not accept options.")
    missing = [hook for hook in operation.hooks if not callable(getattr(system, hook, None))]
    if missing:
        raise _TransactionFailure(
            "capability_missing",
            f"System does not provide {', '.join(missing)}(); {request.operation!r} is unavailable.",
        )
    return operation.handler(system, payload, options)


def transact(system: Any, request: SystemRequest | Mapping[str, Any]) -> SystemResponse:
    """Execute one validated system operation and return success or structured failure."""

    if not isinstance(request, SystemRequest):
        request = SystemRequest(**dict(request))
    family = _family(system)
    if request.family is not None and request.family != family:
        return SystemResponse(
            request.request_id,
            family,
            request.operation,
            False,
            error={"type": "family_mismatch", "message": f"request expects {request.family}, system is {family}"},
        )
    try:
        result = _execute(system, family, request)
        return SystemResponse(request.request_id, family, request.operation, True, result=result)
    except _TransactionFailure as exc:
        return SystemResponse(
            request.request_id,
            family,
            request.operation,
            False,
            error={"type": exc.kind, "message": str(exc)},
        )
    except Exception as exc:  # transaction boundary turns failures into response evidence
        return SystemResponse(
            request.request_id,
            family,
            request.operation,
            False,
            error={"type": type(exc).__name__, "message": str(exc)},
        )


class SystemTransactionAdapter:
    """Bind one project system to the shared request/response transaction shape."""

    def __init__(self, system: Any) -> None:
        self.system = system
        self.family = _family(system)

    def execute(self, request: SystemRequest | Mapping[str, Any]) -> SystemResponse:
        return transact(self.system, request)
