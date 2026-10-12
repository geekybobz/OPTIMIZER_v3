"""Typed foreground requests; no transport parser or history writer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from optimizer.serialization import to_plain


def _name(value: Any, field_name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string.")


def _compact_recipe(value: Any, depth: int = 0) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if value is None or isinstance(value, (str, bool, int, float, complex, np.generic)):
        return value
    if depth >= 4:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return {str(k): _compact_recipe(v, depth + 1) for k, v in list(value.items())[:32]}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def _transport_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(state)
    if "contexts" in out:
        out["contexts"] = {name: {**entry, "recipe": _compact_recipe(entry.get("recipe"))}
                           for name, entry in out["contexts"].items()}
    return out


@dataclass(frozen=True)
class ObjectRef:
    """Reference to one retained object, optionally a declared data field."""

    object_id: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        _name(self.object_id, "object_id")
        if not isinstance(self.path, tuple) or any(not (
            isinstance(part, str) and part and not part.startswith("_") or
            type(part) is int and part >= 0) for part in self.path):
            raise ValueError("path must contain public data names or nonnegative sequence indices.")


@dataclass(frozen=True)
class OperationRequest:
    """Native keyword arguments and separate session/activation options."""

    operation: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    context: str | None = None
    bind: str | None = None
    activate: bool | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        for name in ("operation", "request_id", "context", "bind"):
            _name(getattr(self, name), name, optional=name in {"context", "bind"})
        if not isinstance(self.arguments, Mapping) or any(not isinstance(k, str) for k in self.arguments):
            raise TypeError("arguments must be a mapping with string keys.")
        if self.activate is not None and type(self.activate) is not bool:
            raise TypeError("activate must be bool or None.")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class OperationResponse:
    """Native typed result plus compact transport representation."""

    request_id: str
    operation: str | None
    status: str
    result: Any = None
    object_refs: tuple[ObjectRef, ...] = ()
    state_before: Mapping[str, Any] = field(default_factory=dict)
    state_after: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        # Never walk a numerical result or call its full-array serializer here.
        return to_plain({
            "request_id": self.request_id, "operation": self.operation,
            "status": self.status, "object_refs": [
                {"object_id": ref.object_id, "path": list(ref.path)} for ref in self.object_refs
            ], "state_before": _transport_state(self.state_before), "state_after": _transport_state(self.state_after),
            "error": self.error, "metadata": dict(self.metadata),
        })


class _ServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
