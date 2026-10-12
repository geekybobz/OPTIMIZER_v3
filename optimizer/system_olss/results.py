"""Public OLSS evaluation and solver evidence containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optimizer.system_olss.variables import ShootingVariables


def _plain(value: Any) -> Any:
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
class ShootingEvaluation:
    variables: ShootingVariables
    residuals: np.ndarray
    metrics: Mapping[str, Any]
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals, dtype=float).reshape(-1).copy()
        residuals.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "options", dict(self.options))

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residuals))

    @property
    def residual_max_abs(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables.to_dict(),
            "residuals": self.residuals.tolist(),
            "residual_norm": self.residual_norm,
            "residual_max_abs": self.residual_max_abs,
            "metrics": _plain(self.metrics),
            "options": _plain(self.options),
        }


@dataclass(frozen=True)
class VerificationCheck:
    """One named acceptance check attached to independent verification."""

    name: str
    passed: bool
    value: Any = None
    tolerance: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "passed": bool(self.passed),
            "value": _plain(self.value),
            "tolerance": _plain(self.tolerance),
            "message": str(self.message),
        }


@dataclass(frozen=True)
class ShootingVerification:
    residuals: np.ndarray
    metrics: Mapping[str, Any]
    options: Mapping[str, Any] = field(default_factory=dict)
    numerical_success: bool = True
    message: str = "verified"
    independent: bool = False
    independence_source: str = "unspecified"
    checks: tuple[VerificationCheck, ...] = ()

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals, dtype=float).reshape(-1).copy()
        residuals.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "independent", bool(self.independent))
        object.__setattr__(self, "independence_source", str(self.independence_source))
        normalized = []
        for check in self.checks:
            normalized.append(
                check
                if isinstance(check, VerificationCheck)
                else VerificationCheck(**dict(check))
            )
        object.__setattr__(self, "checks", tuple(normalized))

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residuals))

    @property
    def residual_max_abs(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "residuals": self.residuals.tolist(),
            "residual_norm": self.residual_norm,
            "residual_max_abs": self.residual_max_abs,
            "metrics": _plain(self.metrics),
            "options": _plain(self.options),
            "numerical_success": bool(self.numerical_success),
            "message": self.message,
            "independent": bool(self.independent),
            "independence_source": self.independence_source,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class ShootingResult:
    variables: ShootingVariables
    residuals: np.ndarray
    metrics: dict[str, Any]
    solver_converged: bool
    verified: bool
    qualified: bool
    iterations: int
    method: str
    stop_reason: str
    jacobian_source: str
    verification: ShootingVerification | None = None
    counters: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    verification_tolerance: float | None = None
    verification_tolerance_source: str | None = None

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residuals))

    @property
    def residual_max_abs(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables.to_dict(),
            "residuals": self.residuals.tolist(),
            "residual_norm": self.residual_norm,
            "residual_max_abs": self.residual_max_abs,
            "metrics": _plain(self.metrics),
            "solver_converged": bool(self.solver_converged),
            "verified": bool(self.verified),
            "qualified": bool(self.qualified),
            "iterations": int(self.iterations),
            "method": self.method,
            "stop_reason": self.stop_reason,
            "jacobian_source": self.jacobian_source,
            "verification": None if self.verification is None else self.verification.to_dict(),
            "verification_tolerance": _plain(self.verification_tolerance),
            "verification_tolerance_source": self.verification_tolerance_source,
            "counters": _plain(self.counters),
            "history": _plain(self.history),
            "error": self.error,
        }


@dataclass
class MultiStartResult:
    results: list[ShootingResult]
    qualified: list[ShootingResult]
    unique_qualified: list[ShootingResult]
    ranked: list[ShootingResult]
    selection_metric: str | None
    selection_direction: str | None
    selection_source: str | None = None

    @property
    def best(self) -> ShootingResult | None:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "qualified_count": len(self.qualified),
            "unique_qualified_count": len(self.unique_qualified),
            "ranked": [result.to_dict() for result in self.ranked],
            "selection_metric": self.selection_metric,
            "selection_direction": self.selection_direction,
            "selection_source": self.selection_source,
        }


@dataclass(frozen=True)
class ContinuationPoint:
    variables: ShootingVariables
    residual_norm: float
    tangent: np.ndarray
    corrector_iterations: int
    verified: bool
    verification_independent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables.to_dict(),
            "residual_norm": _plain(self.residual_norm),
            "tangent": _plain(self.tangent),
            "corrector_iterations": int(self.corrector_iterations),
            "verified": bool(self.verified),
            "verification_independent": bool(self.verification_independent),
        }


@dataclass
class ShootingContinuationResult:
    points: list[ContinuationPoint]
    completed_steps: int
    requested_steps: int
    stop_reason: str
    step_size: float
    step_history: list[dict[str, Any]] = field(default_factory=list)
    verification_tolerance: float | None = None
    verification_tolerance_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [point.to_dict() for point in self.points],
            "completed_steps": int(self.completed_steps),
            "requested_steps": int(self.requested_steps),
            "stop_reason": self.stop_reason,
            "step_size": _plain(self.step_size),
            "step_history": _plain(self.step_history),
            "verification_tolerance": _plain(self.verification_tolerance),
            "verification_tolerance_source": self.verification_tolerance_source,
        }


@dataclass(frozen=True)
class ParameterContinuationPoint:
    parameter: float
    result: ShootingResult
    attempted_step: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": _plain(self.parameter),
            "attempted_step": _plain(self.attempted_step),
            "result": self.result.to_dict(),
        }


@dataclass
class ParameterContinuationResult:
    parameter: str
    target: float
    points: list[ParameterContinuationPoint]
    stop_reason: str
    reached_target: bool
    step_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final(self) -> ShootingResult | None:
        return self.points[-1].result if self.points else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "target": _plain(self.target),
            "points": [point.to_dict() for point in self.points],
            "stop_reason": self.stop_reason,
            "reached_target": bool(self.reached_target),
            "step_history": _plain(self.step_history),
        }
