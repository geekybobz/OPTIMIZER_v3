"""One bounded iteration lifecycle for numerical methods and coordinators.

Adapters perform one mathematical iteration and return a named stop or None.
The engine owns iteration budgets and cooperative boundaries. Inner linear
algebra and backtracking remain finite operations within one iteration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from optimizer.core.runtime import EvaluatorBase


def drive(advance: Callable[[int], str | None], *, maxiter: int,
          before: Callable[[int], str | None] | None = None,
          boundary: Callable[[int], None] | None = None,
          terminal: bool = False, budget_reason: str = "maxiter") -> str:
    """Run a bounded chunk; terminal=True allows a final evidence-only visit."""
    if int(maxiter) < 0:
        raise ValueError("maxiter must be nonnegative.")
    for iteration in range(int(maxiter) + int(terminal)):
        reason = before(iteration) if before is not None else None
        if reason is not None:
            return reason
        if boundary is not None and iteration < int(maxiter):
            boundary(iteration)
        reason = advance(iteration)
        if reason is not None:
            return reason
    return budget_reason


@dataclass
class TrialSelection:
    value: Any
    evidence: Any
    merit: float
    accepted: bool = False
    scale: float = 0.0
    attempted: int = 0
    failed: int = 0
    error: str | None = None
    bound_stalled: bool = False


def backtrack(current, evidence, delta, *, evaluate, merit, attempts, shrink,
              bounds=None, initial_scale=1.0, name="trial", accept=None):
    """One failure-aware bounded trial search, shared by residual adapters."""
    selected = TrialSelection(current, evidence, float(merit(evidence)))
    scale = float(initial_scale)
    base = current.flatten(copy=False)
    for _ in range(int(attempts)):
        values = base + scale * delta
        if bounds is not None:
            values = np.clip(values, bounds[0], bounds[1])
            if np.array_equal(values, base):
                selected.bound_stalled = True
                break
        selected.attempted += 1
        outcome = EvaluatorBase.attempt(lambda: (current.with_flat(values, name=name)))
        if outcome.ok:
            trial = outcome.value
            outcome = EvaluatorBase.attempt(evaluate, trial)
        if not outcome.ok:
            selected.failed += 1
            selected.error = outcome.error
            scale *= shrink
            continue
        trial_merit = float(merit(outcome.value))
        accepted = trial_merit < selected.merit if accept is None else accept(outcome.value, trial_merit)
        if accepted:
            selected.value, selected.evidence, selected.merit = trial, outcome.value, trial_merit
            selected.accepted, selected.scale = True, scale
            break
        scale *= shrink
    return selected
