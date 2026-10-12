"""Shared continuation attempts and adaptive step policy; no physics ownership."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from optimizer.core.iteration import drive
from optimizer.core.numerics import residual_step
from optimizer.core.runtime import EvaluatorBase


@dataclass
class StepSizeController:
    current: float
    minimum: float
    maximum: float
    shrink: float = 0.5
    grow: float = 1.5

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.current, self.minimum, self.maximum, self.shrink, self.grow)):
            raise ValueError("Continuation step settings must be finite.")
        if not 0 < self.minimum <= self.maximum or self.current <= 0:
            raise ValueError("Continuation requires positive step sizes and min_step <= max_step.")
        if not 0 < self.shrink < 1 or self.grow < 1:
            raise ValueError("shrink must be in (0, 1) and grow must be >= 1.")
        self.current = min(self.current, self.maximum)

    def reject(self) -> bool:
        self.current *= self.shrink
        return self.current >= self.minimum

    def accept(self, *, fast=True):
        if fast:
            self.current = min(self.maximum, self.current * self.grow)

    def landing(self, current, target):
        distance = abs(target - current)
        step = min(self.current, distance)
        return (target if step == distance else current + np.copysign(step, target - current)), step


@dataclass
class ContinuationCoordinator:
    """One attempt shape across staged, natural and pseudo-arclength routes."""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, index, accepted, reason, **facts):
        row = dict(index=int(index), accepted=bool(accepted), reason=str(reason), **facts)
        self.attempts.append(row)
        return row

    def run(self, advance, *, limit, budget_reason="completed"):
        def adapted(index):
            before = len(self.attempts)
            reason = advance(index)
            for row in self.attempts[before:]:
                row.setdefault("index", index)
                row.setdefault("accepted", bool(row.get("qualified", False)))
                row.setdefault("reason", row.get("stop_reason", "accepted" if row["accepted"] else "corrector_failed"))
            return reason
        return drive(adapted, maxiter=limit, budget_reason=budget_reason)


def correct_root(initial, evaluate, *, maxiter, tolerance, scales=None):
    """Shared full Newton corrector for arclength constraints and exact landing."""
    current = np.asarray(initial, dtype=float).copy()
    iterations = 0
    error = None

    def advance(iteration):
        nonlocal current, iterations, error
        iterations = iteration
        outcome = EvaluatorBase.attempt(evaluate, current)
        if not outcome.ok:
            error = outcome.error
            return "evaluation_failed"
        residuals, jacobian = outcome.value
        if np.linalg.norm(residuals) <= tolerance:
            return "root_tolerance"
        if iteration == maxiter:
            return "maxiter"
        outcome = EvaluatorBase.attempt(residual_step, jacobian, residuals, scales=scales)
        if not outcome.ok:
            error = outcome.error
            return "singular_step"
        current += outcome.value
    reason = drive(advance, maxiter=maxiter, terminal=True)
    return current, iterations, error, reason == "root_tolerance"
