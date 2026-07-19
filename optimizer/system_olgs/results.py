"""Structured OLGS result containers.

Systems can keep returning dicts where that is convenient.  These containers are
small public shapes for code that wants to make the computation lifecycle explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.system_olgs.validation import validate_metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Controls):
        return {
            "name": value.name,
            "spec": value.spec.to_dict(),
            "matrix": value.as_matrix(copy=True).tolist(),
            "meta": dict(value.meta),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ForwardResult:
    controls: Controls
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"controls": _json_safe(self.controls), "data": _json_safe(self.data)}


@dataclass(frozen=True)
class BackwardResult:
    controls: Controls
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"controls": _json_safe(self.controls), "data": _json_safe(self.data)}


@dataclass(frozen=True)
class SystemEvaluation:
    controls: Controls
    metrics: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        controls: Controls,
        metrics: Mapping[str, Any],
        *,
        data: Mapping[str, Any] | None = None,
    ) -> "SystemEvaluation":
        return cls(controls=controls, metrics=validate_metrics(metrics), data=dict(data or {}))

    @property
    def J(self) -> float:
        return float(self.metrics["J"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": _json_safe(self.controls),
            "metrics": _json_safe(self.metrics),
            "data": _json_safe(self.data),
        }


@dataclass(frozen=True)
class GradientResult:
    controls: Controls
    gradient: Controls
    metrics: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "controls": _json_safe(self.controls),
            "gradient": _json_safe(self.gradient),
            "data": _json_safe(self.data),
        }
        if self.metrics is not None:
            payload["metrics"] = _json_safe(validate_metrics(self.metrics))
        return payload


@dataclass(frozen=True)
class SimulationResult:
    controls: Controls
    metrics: dict[str, Any]
    trajectories: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def J(self) -> float:
        return float(self.metrics["J"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": _json_safe(self.controls),
            "metrics": _json_safe(validate_metrics(self.metrics)),
            "trajectories": _json_safe(self.trajectories),
            "data": _json_safe(self.data),
        }
