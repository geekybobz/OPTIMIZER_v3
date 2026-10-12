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
from optimizer.serialization import to_plain


@dataclass(frozen=True)
class ForwardResult:
    controls: Controls
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"controls": to_plain(self.controls), "data": to_plain(self.data)}


@dataclass(frozen=True)
class BackwardResult:
    controls: Controls
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"controls": to_plain(self.controls), "data": to_plain(self.data)}


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
            "controls": to_plain(self.controls),
            "metrics": to_plain(self.metrics),
            "data": to_plain(self.data),
        }


@dataclass(frozen=True)
class GradientResult:
    controls: Controls
    gradient: Controls
    metrics: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "controls": to_plain(self.controls),
            "gradient": to_plain(self.gradient),
            "data": to_plain(self.data),
        }
        if self.metrics is not None:
            payload["metrics"] = to_plain(validate_metrics(self.metrics))
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
            "controls": to_plain(self.controls),
            "metrics": to_plain(validate_metrics(self.metrics)),
            "trajectories": to_plain(self.trajectories),
            "data": to_plain(self.data),
        }
