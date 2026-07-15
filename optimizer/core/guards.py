"""Reusable multi-metric accept/reject guards.

Why this file exists
--------------------
Single-metric acceptance is not enough for serious training.  A candidate can reduce
``J`` while damaging fidelity, energy, or a hard-condition metric.  Guards let callers
state direct metric requirements and plug the resulting callable into ``run_chunk`` or
future optimizer wrappers.

How it fits the architecture
----------------------------
- ``core.engine`` already supports custom accept functions.
- this module builds those accept functions from simple metric rules.
- optimizers stay focused on proposing controls.
- curriculum and repair workflows can use the same guard logic.

What this file deliberately does not do
---------------------------------------
It does not know what ``F_norm2`` or ``energy`` physically means.  It only compares
metric values using caller-provided rules.

Reviewer invariants
-------------------
- every failed rule is reported with metric, operator, value, and threshold.
- improve checks and hard requirements are separated.
- the returned object is callable with the engine's accept-function signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optimizer.core.engine import AcceptanceDecision, StepProposal
from optimizer.result import Evaluation
from optimizer.state import RunState


RuleSpec = tuple[str, float] | tuple[str, float, float]


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    """Return a finite scalar metric for guard comparisons."""

    if key not in metrics:
        raise KeyError(f"metrics do not include {key!r}.")
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f"metric {key!r} must be scalar for guard checks.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"metric {key!r} must be finite.")
    return out


def _compare(value: float, op: str, threshold: float, tolerance: float) -> bool:
    """Evaluate one scalar comparison rule."""

    op = str(op)
    if op == "<":
        return value < threshold + tolerance
    if op == "<=":
        return value <= threshold + tolerance
    if op == ">":
        return value > threshold - tolerance
    if op == ">=":
        return value >= threshold - tolerance
    if op == "==":
        return abs(value - threshold) <= tolerance
    if op == "!=":
        return abs(value - threshold) > tolerance
    raise ValueError(f"unsupported guard operator {op!r}.")


@dataclass(frozen=True)
class MetricGuard:
    """Callable accept rule for multi-metric training safeguards."""

    improve: str = "J"
    mode: str = "min"
    tolerance: float = 0.0
    require: Mapping[str, RuleSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if float(self.tolerance) < 0.0 or not np.isfinite(float(self.tolerance)):
            raise ValueError("tolerance must be finite and nonnegative.")

    def __call__(
        self,
        current: Evaluation,
        trial: Evaluation,
        proposal: StepProposal,
        state: RunState,
    ) -> AcceptanceDecision:
        """Return engine-compatible acceptance decision."""

        del proposal, state
        failures: list[dict[str, Any]] = []
        current_value = _metric(current.metrics, self.improve)
        trial_value = _metric(trial.metrics, self.improve)
        improvement = current_value - trial_value if self.mode == "min" else trial_value - current_value
        improved = improvement >= -float(self.tolerance)
        if not improved:
            failures.append(
                {
                    "kind": "improve",
                    "metric": self.improve,
                    "mode": self.mode,
                    "current": current_value,
                    "trial": trial_value,
                    "improvement": improvement,
                    "tolerance": float(self.tolerance),
                }
            )

        requirement_records = []
        for metric, rule in dict(self.require).items():
            if len(rule) == 2:
                op, threshold = rule
                rule_tolerance = float(self.tolerance)
            elif len(rule) == 3:
                op, threshold, rule_tolerance = rule
            else:
                raise ValueError("require rules must be (op, threshold) or (op, threshold, tolerance).")
            value = _metric(trial.metrics, metric)
            passed = _compare(value, str(op), float(threshold), float(rule_tolerance))
            record = {
                "kind": "require",
                "metric": metric,
                "operator": str(op),
                "threshold": float(threshold),
                "tolerance": float(rule_tolerance),
                "trial": value,
                "passed": bool(passed),
            }
            requirement_records.append(record)
            if not passed:
                failures.append(record)

        accepted = not failures
        return AcceptanceDecision(
            accepted=accepted,
            reason="accepted" if accepted else "guard_failed",
            technical={
                "guard": {
                    "improve": self.improve,
                    "mode": self.mode,
                    "current_value": current_value,
                    "trial_value": trial_value,
                    "improvement": improvement,
                    "requirements": requirement_records,
                    "failures": failures,
                }
            },
        )


def metric_guard(
    *,
    improve: str = "J",
    mode: str = "min",
    tolerance: float = 0.0,
    require: Mapping[str, RuleSpec] | None = None,
) -> MetricGuard:
    """Return a reusable engine accept function."""

    return MetricGuard(
        improve=improve,
        mode=mode,
        tolerance=tolerance,
        require=dict(require or {}),
    )
