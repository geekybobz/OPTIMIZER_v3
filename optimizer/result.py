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
- ``metrics`` always contains finite scalar ``J`` when using the constructors here.
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

    @classmethod
    def from_metrics(
        cls,
        controls: Controls,
        metrics: Mapping[str, Any],
        *,
        system_state: Mapping[str, Any] | None = None,
    ) -> "Evaluation":
        """Validate metrics and create an evaluation snapshot."""

        return cls(
            controls=controls,
            metrics=validate_metrics(metrics),
            system_state=None if system_state is None else dict(system_state),
        )

    @property
    def J(self) -> float:
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
    state: RunState | None = None
    trace: Any | None = None
    system_params: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    blackbox_path: str | None = None
    blackbox_run_id: str | None = None

    @classmethod
    def from_state(
        cls,
        state: RunState,
        *,
        stop_reason: str,
        optimizer: str | None = None,
        trace: Any | None = None,
        blackbox: Any | None = None,
    ) -> "OptimizerResult":
        """Create a public result from a final run state."""

        blackbox_path = None if blackbox is None else str(getattr(blackbox, "run_dir", ""))
        blackbox_run_id = None if blackbox is None else str(getattr(blackbox, "run_id", ""))
        return cls(
            controls=state.controls,
            metrics=validate_metrics(state.metrics),
            stop_reason=str(stop_reason),
            iterations=int(state.iteration),
            optimizer=optimizer or state.optimizer_name or "unknown",
            state=state,
            trace=trace,
            system_params=dict(state.system_params),
            trace_id=state.trace_id,
            checkpoint_ids=dict(state.checkpoint_ids),
            blackbox_path=blackbox_path,
            blackbox_run_id=blackbox_run_id,
        )

    @property
    def J(self) -> float:
        return float(self.metrics["J"])

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
            "J": float(self.metrics["J"]),
            "stop_reason": self.stop_reason,
            "iterations": int(self.iterations),
            "optimizer": self.optimizer,
            "system_params": _json_safe(self.system_params),
            "trace_id": self.trace_id,
            "checkpoint_ids": dict(self.checkpoint_ids),
            "blackbox_path": self.blackbox_path,
            "blackbox_run_id": self.blackbox_run_id,
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
