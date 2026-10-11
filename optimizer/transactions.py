"""Family-aware request/response adapter for shared core, CLI, and MCP layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
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


def transact(system: Any, request: SystemRequest | Mapping[str, Any]) -> SystemResponse:
    """Execute a direct-method transaction and return success or structured failure."""

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
        payload = dict(request.payload)
        options = dict(request.options)
        if request.operation in {"system_spec", "control_spec", "shooting_spec"}:
            if payload or options:
                raise ValueError(f"{request.operation} does not accept payload or options.")
            result = getattr(system, request.operation)()
        elif request.operation == "with_secondary":
            result = system.with_secondary(**payload)
        elif family == "olgs" and request.operation in {"evaluate", "gradient"}:
            controls = payload.get("controls")
            result = getattr(system, request.operation)(controls)
        elif family == "olss" and request.operation in {"residuals", "jacobian", "evaluate", "verify"}:
            variables = payload.get("variables")
            result = getattr(system, request.operation)(variables, options=options)
        else:
            raise KeyError(f"Unsupported {family} operation {request.operation!r}.")
        return SystemResponse(request.request_id, family, request.operation, True, result=result)
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
