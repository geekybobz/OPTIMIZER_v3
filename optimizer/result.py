"""Public evaluation and optimizer result containers.

Why this file exists
--------------------
The optimizer library needs stable public output objects before the engine exists.
Users should be able to inspect the final controls, metrics, stop reason, iteration
count, warmstart state, and trace/checkpoint context in a predictable way.  Later
modules should not invent separate result shapes for each optimizer.

This module defines two public containers:

    Evaluation       a single system evaluation snapshot
    OptimizerResult  the public return value from opt.adam, opt.line_search, etc.

How it fits the architecture
----------------------------
- ``system.evaluate`` returns metric dictionaries; the engine can wrap them as
  ``Evaluation`` snapshots when useful.
- optimizers and the engine will return ``OptimizerResult``.
- warmstart logic reads ``OptimizerResult.state`` or safe public fields.
- logs/checkpoints can consume ``to_dict`` payloads later.

What this file deliberately does not do
---------------------------------------
It does not run an optimizer, choose best controls, write checkpoint files, or decide
metric acceptance.  It only standardizes public data shape and lightweight export.

Reviewer invariants
-------------------
- ``metrics`` always contains a finite scalar declared objective when using the
  constructors here.  ``J`` remains a legacy convenience only for systems that
  declare and return it.
- exported dictionaries are JSON-friendly for common scalar/list/dict payloads.
- controls export includes both spec metadata and matrix values.
- optimizer-private run state stays in ``RunState`` and is not flattened accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.state import RunState, WarmStartState
from optimizer.system_olgs import validate_metrics


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-friendly structures."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Controls):
        return controls_to_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def controls_to_dict(controls: Controls) -> dict[str, Any]:
    """Serialize controls into a plain dictionary for result exports."""

    return {
        "name": controls.name,
        "spec": controls.spec.to_dict(),
        "matrix": controls.as_matrix(copy=True).tolist(),
        "meta": dict(controls.meta),
    }


@dataclass(frozen=True)
class Evaluation:
    """Snapshot of one system evaluation."""

    controls: Controls
    metrics: dict[str, Any]
    system_state: dict[str, Any] | None = None
    objective_metric: str = "J"

    @classmethod
    def from_metrics(
        cls,
        controls: Controls,
        metrics: Mapping[str, Any],
        *,
        system_state: Mapping[str, Any] | None = None,
        objective_metric: str = "J",
    ) -> "Evaluation":
        """Validate metrics and create an evaluation snapshot."""

        return cls(
            controls=controls,
            metrics=validate_metrics(metrics, objective_metric=objective_metric),
            system_state=None if system_state is None else dict(system_state),
            objective_metric=str(objective_metric),
        )

    @property
    def objective_value(self) -> float:
        """Return the scalar selected by this evaluation's declared objective."""

        return float(self.metrics[self.objective_metric])

    @property
    def J(self) -> float:
        """Legacy accessor for a metric literally named ``J``."""

        return float(self.metrics["J"])

    def to_dict(self, *, include_controls: bool = True) -> dict[str, Any]:
        """Return a JSON-friendly evaluation payload."""

        payload: dict[str, Any] = {"metrics": _json_safe(self.metrics)}
        if include_controls:
            payload["controls"] = controls_to_dict(self.controls)
        if self.system_state is not None:
            payload["system_state"] = _json_safe(self.system_state)
        return payload


@dataclass
class OptimizerResult:
    """Public result returned by an optimizer call."""

    controls: Controls
    metrics: dict[str, Any]
    stop_reason: str
    iterations: int
    optimizer: str
    objective_metric: str = "J"
    state: RunState | None = None
    trace: Any | None = None
    system_params: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    tracking_failures: tuple[dict[str, str], ...] = ()

    @classmethod
    def from_state(
        cls,
        state: RunState,
        *,
        stop_reason: str,
        optimizer: str | None = None,
        trace: Any | None = None,
        tracking_failures: tuple[dict[str, str], ...] = (),
        objective_metric: str = "J",
    ) -> "OptimizerResult":
        """Create a public result from a final run state."""

        return cls(
            controls=state.controls,
            metrics=validate_metrics(state.metrics, objective_metric=objective_metric),
            stop_reason=str(stop_reason),
            iterations=int(state.iteration),
            optimizer=optimizer or state.optimizer_name or "unknown",
            objective_metric=str(objective_metric),
            state=state,
            trace=trace,
            system_params=dict(state.system_params),
            trace_id=state.trace_id,
            checkpoint_ids=dict(state.checkpoint_ids),
            tracking_failures=tuple(dict(failure) for failure in tracking_failures),
        )

    @property
    def J(self) -> float:
        """Legacy accessor for a metric literally named ``J``."""

        return float(self.metrics["J"])

    @property
    def objective_value(self) -> float:
        """Return the final value of the declared objective metric."""

        return float(self.metrics[self.objective_metric])

    def warmstart(self, *, target_optimizer: str | None = None) -> WarmStartState:
        """Return safe warmstart state for another optimizer call."""

        return WarmStartState.from_result(self, target_optimizer=target_optimizer)

    def to_dict(
        self,
        *,
        include_state: bool = False,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-friendly result payload.

        ``include_state`` intentionally keeps optimizer-private state shallow.  Full
        checkpoint persistence belongs to the future logs/checkpoint layer.
        """

        payload: dict[str, Any] = {
            "controls": controls_to_dict(self.controls),
            "metrics": _json_safe(self.metrics),
            "objective_metric": self.objective_metric,
            "objective_value": self.objective_value,
            "J": None if "J" not in self.metrics else float(self.metrics["J"]),
            "stop_reason": self.stop_reason,
            "iterations": int(self.iterations),
            "optimizer": self.optimizer,
            "system_params": _json_safe(self.system_params),
            "trace_id": self.trace_id,
            "checkpoint_ids": dict(self.checkpoint_ids),
            "tracking_complete": not self.tracking_failures,
            "tracking_failures": [dict(failure) for failure in self.tracking_failures],
        }
        if include_state and self.state is not None:
            payload["state"] = {
                "iteration": int(self.state.iteration),
                "global_iteration": int(self.state.global_iteration),
                "step_size": self.state.step_size,
                "optimizer_name": self.state.optimizer_name,
                "stop_reason": self.state.stop_reason,
                "best_metrics": _json_safe(self.state.best_metrics),
            }
        if include_trace and self.trace is not None:
            to_dict = getattr(self.trace, "to_dict", None)
            payload["trace"] = _json_safe(to_dict() if callable(to_dict) else self.trace)
        return payload
