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
from optimizer.families import system_family, _family_from_spec
from optimizer.serialization import to_plain


@dataclass(frozen=True)
class SystemRequest:
    """One explicit operation against an OLGS or OLSS system."""

    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string.")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string.")
        if not isinstance(self.payload, Mapping) or not isinstance(self.options, Mapping):
            raise TypeError("payload and options must be mappings.")
        if any(not isinstance(key, str) for key in (*self.payload, *self.options)):
            raise TypeError("payload and options keys must be strings.")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "options", dict(self.options))
        if self.family is not None and self.family not in {"olgs", "olss"}:
            raise ValueError("family must be 'olgs', 'olss', or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "family": self.family,
            "operation": self.operation,
            "payload": to_plain(self.payload),
            "options": to_plain(self.options),
        }


@dataclass(frozen=True)
class SystemResponse:
    """Stable response envelope for a system transaction."""

    request_id: str
    family: str | None
    operation: str | None
    ok: bool
    result: Any = None
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "family": self.family,
            "operation": self.operation,
            "ok": self.ok,
            "result": to_plain(self.result),
            "error": None if self.error is None else to_plain(self.error),
        }


class _TransactionFailure(Exception):
    """A failure with a stable transaction error type."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ----------------------------------------------------------------------
# Payload decoding against the system's own layout
# ----------------------------------------------------------------------


def _keys(value: Mapping[str, Any], required: set[str], allowed: set[str]) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(f"fields: missing={sorted(missing)}, unknown={sorted(unknown, key=str)}.")


def _array(value: Any, dtype: Any) -> np.ndarray:
    """Decode scalar or array complex JSON without broadcasting or lossy casts."""

    def decode(item: Any) -> Any:
        if isinstance(item, Mapping):
            _keys(item, {"real", "imag"}, {"real", "imag"})
            real, imag = np.asarray(item["real"], dtype=float), np.asarray(item["imag"], dtype=float)
            if real.shape != imag.shape:
                raise ValueError("complex real and imag shapes must match.")
            return real + 1j * imag
        if isinstance(item, (list, tuple)):
            return [decode(part) for part in item]
        return item

    array = np.asarray(decode(value))
    target = np.dtype(dtype)
    if target.kind not in "fciu":
        raise ValueError("numerical inputs require a real or complex numerical dtype.")
    if np.iscomplexobj(array) and target.kind != "c":
        raise ValueError("complex input requires a complex control layout; use an explicit conversion.")
    # Reject fractional or out-of-range values when targeting integral controls.
    if target.kind in "iu":
        limits = np.iinfo(target)
        for item in array.flat:
            integer = int(item)
            if integer != item or not limits.min <= integer <= limits.max:
                raise ValueError("input cannot be represented by the integral control dtype.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=target)
    if not np.all(np.isfinite(result)):
        raise ValueError("numerical inputs must be finite.")
    return result


def _control_layout(spec: Any, declared: Any) -> None:
    from optimizer.controls import ControlSpec, _validate_control_spec_compatibility

    if not isinstance(declared, Mapping):
        raise ValueError("declared control spec must be a mapping.")
    _keys(declared, {"keys", "control_dim", "dtype", "dt"},
          {"keys", "control_dim", "dtype", "dt", "n_controls", "shape", "meta"})
    if not isinstance(declared["keys"], (list, tuple)) or any(
        not isinstance(key, str) for key in declared["keys"]
    ):
        raise ValueError("control keys must be an ordered list of strings.")
    dim = declared["control_dim"]
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
        raise ValueError("control_dim must be an integer.")
    actual = ControlSpec(tuple(declared["keys"]), dim, dtype=declared["dtype"], dt=declared["dt"])
    _validate_control_spec_compatibility(spec, actual)
    if "shape" in declared and list(declared["shape"]) != list(actual.shape):
        raise ValueError("declared control shape is inconsistent.")
    if "n_controls" in declared and declared["n_controls"] != actual.n_controls:
        raise ValueError("declared channel count is inconsistent.")


def _shooting_layout(spec: Any, declared: Any) -> None:
    from optimizer.system_olss.variables import (
        ShootingBlock, ShootingSpec, _validate_shooting_spec_compatibility,
    )

    if not isinstance(declared, Mapping):
        raise ValueError("declared shooting spec must be a mapping.")
    _keys(declared, {"blocks", "size", "dtype"}, {"blocks", "size", "dtype", "metadata"})
    if not isinstance(declared["blocks"], (list, tuple)):
        raise ValueError("shooting blocks must be an ordered list.")
    blocks = []
    for block in declared["blocks"]:
        if not isinstance(block, Mapping):
            raise ValueError("shooting blocks must be mappings.")
        _keys(block, {"name", "shape", "size", "unit", "scale", "role"},
              {"name", "shape", "size", "unit", "scale", "role", "meaning", "metadata"})
        if not isinstance(block["shape"], (list, tuple)) or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) for n in block["shape"]
        ):
            raise ValueError("block shape must be a list of integers.")
        if not isinstance(block["name"], str) or not isinstance(block["role"], str) or (
            block["unit"] is not None and not isinstance(block["unit"], str)
        ):
            raise ValueError("block names, roles, and units must be strings (unit may be null).")
        actual = ShootingBlock(block["name"], tuple(block["shape"]), unit=block["unit"],
                               scale=block["scale"], role=block["role"])
        if block["size"] != actual.size:
            raise ValueError("declared block size is inconsistent.")
        blocks.append(actual)
    actual_spec = ShootingSpec(tuple(blocks), dtype=declared["dtype"])
    if declared["size"] != actual_spec.size:
        raise ValueError("declared shooting size is inconsistent.")
    _validate_shooting_spec_compatibility(spec, actual_spec)


def _decode_controls(system: Any, value: Any, *, field_name: str = "controls") -> Any:
    from optimizer.controls import Controls, _validate_control_spec_compatibility
    from optimizer.system_olgs.validation import validate_control_spec

    spec = validate_control_spec(system)
    try:
        if isinstance(value, Controls):
            _validate_control_spec_compatibility(spec, value.spec)
            return value.copy()
        if isinstance(value, Mapping):
            if set(value) == set(spec.keys):
                return Controls.from_matrix(spec, np.stack([_array(value[key], spec.dtype) for key in spec.keys]))
            representation = "matrix" if "matrix" in value else "flat" if "flat" in value else None
            if representation is None:
                raise ValueError("controls mapping must match the channels or contain matrix/flat.")
            _keys(value, {representation}, {representation, "spec", "name", "meta"})
            if "spec" in value:
                _control_layout(spec, value["spec"])
            constructor = Controls.from_matrix if representation == "matrix" else Controls.from_flat
            return constructor(spec, _array(value[representation], spec.dtype),
                               name=value.get("name"), meta=value.get("meta"))
        if isinstance(value, (list, tuple, np.ndarray)):
            array = _array(value, spec.dtype)
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
    from optimizer.system_olss.contract import validate_shooting_spec
    from optimizer.system_olss.variables import _validate_shooting_spec_compatibility

    spec = validate_shooting_spec(system)
    try:
        if isinstance(value, ShootingVariables):
            _validate_shooting_spec_compatibility(spec, value.spec)
            return value.copy()
        if isinstance(value, Mapping):
            if set(value) == set(spec.names):
                return ShootingVariables.from_blocks(spec, {key: _array(item, spec.dtype) for key, item in value.items()})
            if "values" in value:
                _keys(value, {"values"}, {"values", "spec", "name", "metadata"})
                if "spec" in value:
                    _shooting_layout(spec, value["spec"])
                return ShootingVariables.from_flat(
                    spec,
                    _array(value["values"], spec.dtype),
                    name=str(value.get("name", "shooting")),
                    metadata=value.get("metadata"),
                )
            raise ValueError("variables mapping must match the blocks or contain values.")
        if isinstance(value, (list, tuple, np.ndarray)):
            return ShootingVariables.from_flat(spec, _array(value, spec.dtype))
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
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


def _olgs_operations(system_spec: Mapping[str, Any]) -> dict[str, _Operation]:
    from optimizer.result import controls_to_dict
    from optimizer.system_olgs import (
        evaluate_system,
        gradient_system,
        optional_control_bounds,
        optional_hessian,
        optional_hvp,
        optional_jacobian,
        optional_prox,
        optional_residuals,
        optional_smooth_gradient,
        value_and_gradient_system,
        validate_control_spec,
    )

    def controls(system: Any, payload: Mapping[str, Any]) -> Any:
        return _decode_controls(system, _required(payload, "controls"))

    def value_and_gradient(system: Any, payload: Mapping[str, Any], _options: Mapping[str, Any]) -> Any:
        pair = value_and_gradient_system(system, controls(system, payload), system_spec=dict(system_spec))
        return {"metrics": dict(pair.metrics), "gradient": pair.gradient}

    def control_bounds(system: Any, _payload: Mapping[str, Any], _options: Mapping[str, Any]) -> Any:
        bounds = optional_control_bounds(system)
        return {"lower": controls_to_dict(bounds.lower), "upper": controls_to_dict(bounds.upper)}

    return {
        "system_spec": _Operation(lambda system, payload, options: dict(system_spec), static=True),
        "control_spec": _Operation(lambda system, payload, options: validate_control_spec(system), static=True),
        "evaluate": _Operation(
            lambda system, payload, options: evaluate_system(
                system, controls(system, payload), system_spec=dict(system_spec)
            ),
            hooks=("evaluate",),
            required=("controls",),
        ),
        "gradient": _Operation(
            lambda system, payload, options: gradient_system(system, controls(system, payload)),
            hooks=("gradient",),
            required=("controls",),
        ),
        "value_and_gradient": _Operation(value_and_gradient, hooks=("value_and_gradient",), required=("controls",)),
        "residuals": _Operation(
            lambda system, payload, options: optional_residuals(
                system, controls(system, payload), name=str(payload.get("name", "hard"))
            ),
            hooks=("residuals",),
            required=("controls",), optional=("name",),
        ),
        "jacobian": _Operation(
            lambda system, payload, options: optional_jacobian(
                system, controls(system, payload), name=str(payload.get("name", "hard"))
            ),
            hooks=("jacobian",),
            required=("controls",), optional=("name",),
        ),
        "hvp": _Operation(
            lambda system, payload, options: optional_hvp(
                system,
                controls(system, payload),
                _decode_controls(system, _required(payload, "vector"), field_name="vector"),
            ),
            hooks=("hvp",),
            required=("controls", "vector"),
        ),
        "hessian": _Operation(
            lambda system, payload, options: optional_hessian(system, controls(system, payload)),
            hooks=("hessian",),
            required=("controls",),
        ),
        "smooth_gradient": _Operation(
            lambda system, payload, options: optional_smooth_gradient(system, controls(system, payload)),
            hooks=("smooth_gradient",),
            required=("controls",),
        ),
        "prox": _Operation(
            lambda system, payload, options: optional_prox(
                system, controls(system, payload), float(_required(payload, "step_size"))
            ),
            hooks=("prox",),
            required=("controls", "step_size"),
        ),
        "control_bounds": _Operation(control_bounds, hooks=("control_bounds",)),
    }


def _olss_operations(system_spec: Mapping[str, Any]) -> dict[str, _Operation]:
    from optimizer.system_olss import (
        cost_gradient_system,
        evaluate_shooting_system,
        jacobian_system,
        residuals_and_jacobian_system,
        residuals_system,
        shooting_bounds_system,
        verify_shooting_system,
        validate_shooting_spec,
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
        "system_spec": _Operation(lambda system, payload, options: dict(system_spec), static=True),
        "shooting_spec": _Operation(lambda system, payload, options: validate_shooting_spec(system), static=True),
        "residuals": _Operation(
            lambda system, payload, options: residuals_system(system, variables(system, payload), options=options),
            hooks=("residuals",),
            accepts_options=True,
            required=("variables",),
        ),
        "jacobian": _Operation(
            lambda system, payload, options: jacobian_system(system, variables(system, payload), options=options),
            hooks=("jacobian",),
            accepts_options=True,
            required=("variables",),
        ),
        "residuals_and_jacobian": _Operation(
            residuals_and_jacobian, hooks=("residuals", "jacobian"), accepts_options=True,
            required=("variables",),
        ),
        "evaluate": _Operation(
            lambda system, payload, options: evaluate_shooting_system(
                system, variables(system, payload), options=options
            ),
            hooks=("evaluate",),
            accepts_options=True,
            required=("variables",),
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
            required=("variables",),
        ),
        "cost_gradient": _Operation(
            lambda system, payload, options: cost_gradient_system(system, variables(system, payload), options=options),
            hooks=("cost_gradient",),
            accepts_options=True,
            required=("variables",),
        ),
        "shooting_bounds": _Operation(shooting_bounds, hooks=("shooting_bounds",)),
    }


_OPERATIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, _Operation]]] = {
    "olgs": _olgs_operations,
    "olss": _olss_operations,
}


def _execute(system: Any, family: str, request: SystemRequest, system_spec: Mapping[str, Any]) -> Any:
    payload = dict(request.payload)
    options = dict(request.options)
    if request.operation == "with_secondary":
        if options:
            raise _TransactionFailure("invalid_payload", "with_secondary does not accept options.")
        if not callable(getattr(system, "with_secondary", None)):
            raise _TransactionFailure("capability_missing", "System does not provide with_secondary().")
        return system.with_secondary(**payload)
    table = _OPERATIONS[family](system_spec)
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
    try:
        _keys(payload, set(operation.required), set(operation.required + operation.optional))
    except ValueError as exc:
        raise _TransactionFailure("invalid_payload", str(exc)) from exc
    missing = [hook for hook in operation.hooks if not callable(getattr(system, hook, None))]
    if missing:
        raise _TransactionFailure(
            "capability_missing",
            f"System does not provide {', '.join(missing)}(); {request.operation!r} is unavailable.",
        )
    return operation.handler(system, payload, options)


def transact(system: Any, request: SystemRequest | Mapping[str, Any]) -> SystemResponse:
    """Execute one validated system operation and return success or structured failure."""

    request_id = uuid4().hex
    operation_name = None
    try:
        if isinstance(request, (SystemRequest, Mapping)):
            raw = vars(request) if isinstance(request, SystemRequest) else request
            if isinstance(raw.get("request_id"), str) and raw["request_id"].strip():
                request_id = raw["request_id"]
            if isinstance(raw.get("operation"), str) and raw["operation"].strip():
                operation_name = raw["operation"]
        if not isinstance(request, SystemRequest):
            if not isinstance(request, Mapping):
                raise TypeError("request must be SystemRequest or a mapping.")
            request = SystemRequest(**dict(request))
        # Revalidate live requests too, including mappings modified after construction.
        request = SystemRequest(**vars(request))
        request_id = request.request_id
    except Exception as exc:
        return SystemResponse(request_id, None, operation_name, False,
                              error={"type": "invalid_request", "message": str(exc)})
    try:
        hook = getattr(system, "system_spec", None)
        if not callable(hook):
            raise TypeError("System must provide system_spec().")
        raw_spec = hook()
        family = _family_from_spec(raw_spec)
        if family == "olgs":
            from optimizer.system_olgs.specs import validate_system_spec
            system_spec = validate_system_spec(raw_spec)
        else:
            from optimizer.system_olss.specs import validate_shooting_system_spec
            system_spec = validate_shooting_system_spec(raw_spec)
    except Exception as exc:
        return SystemResponse(request.request_id, None, request.operation, False,
                              error={"type": "invalid_system", "message": str(exc)})
    if request.family is not None and request.family != family:
        return SystemResponse(
            request.request_id,
            family,
            request.operation,
            False,
            error={"type": "family_mismatch", "message": f"request expects {request.family}, system is {family}"},
        )
    try:
        result = _execute(system, family, request, system_spec)
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
        self.family = system_family(system)

    def execute(self, request: SystemRequest | Mapping[str, Any]) -> SystemResponse:
        return transact(self.system, request)
