"""Public evaluation and optimizer result containers.

Why this file exists
--------------------
The optimizer library needs stable public output objects before the engine exists.
Users should be able to inspect the final controls, metrics, stop reason, iteration
count, warmstart state, and compact numerical diagnostics in a predictable way.  Later
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
import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
from optimizer.controls import Controls
from optimizer.state import RunState, WarmStartState
from optimizer.system_olgs import validate_metrics
from optimizer.serialization import to_plain
from optimizer.core.record import RunRecord

def controls_to_dict(controls: Controls) -> dict[str, Any]:
    """Serialize controls into a plain dictionary for result exports."""
    return controls.to_dict()

@dataclass(frozen=True)
class Evaluation:
    """Snapshot of one system evaluation."""
    controls: Controls
    metrics: dict[str, Any]
    system_state: dict[str, Any] | None = None
    objective_metric: str = 'J'

    @classmethod
    def from_metrics(cls, controls: Controls, metrics: Mapping[str, Any], *, system_state: Mapping[str, Any] | None=None, objective_metric: str='J') -> 'Evaluation':
        """Validate metrics and create an evaluation snapshot."""
        return cls(controls=controls, metrics=validate_metrics(metrics, objective_metric=objective_metric), system_state=None if system_state is None else dict(system_state), objective_metric=str(objective_metric))

    @property
    def objective_value(self) -> float:
        """Return the scalar selected by this evaluation's declared objective."""
        return float(self.metrics[self.objective_metric])

    @property
    def J(self) -> float:
        """Legacy accessor for a metric literally named ``J``."""
        return float(self.metrics['J'])

    def to_dict(self, *, include_controls: bool=True) -> dict[str, Any]:
        """Return a JSON-friendly evaluation payload."""
        payload: dict[str, Any] = {'metrics': to_plain(self.metrics)}
        if include_controls:
            payload['controls'] = controls_to_dict(self.controls)
        if self.system_state is not None:
            payload['system_state'] = to_plain(self.system_state)
        return payload

    def copy(self, *, name: str | None=None) -> 'Evaluation':
        """Return an independent evaluation snapshot safe for public consumers."""
        return Evaluation(controls=self.controls.copy(name=self.controls.name if name is None else name), metrics=_copy.deepcopy(self.metrics), system_state=None if self.system_state is None else _copy.deepcopy(self.system_state), objective_metric=self.objective_metric)

@dataclass
class OptimizerResult:
    """Public result returned by an optimizer call."""
    controls: Controls
    metrics: dict[str, Any]
    stop_reason: str
    iterations: int
    optimizer: str
    objective_metric: str = 'J'
    state: RunState | None = None
    system_params: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    run_record: RunRecord | None = None

    @classmethod
    def from_state(cls, state: RunState, *, stop_reason: str, optimizer: str | None=None, objective_metric: str='J', diagnostics: dict[str, Any] | None=None) -> 'OptimizerResult':
        """Create a public result from a final run state."""
        return cls(controls=state.controls.copy(name=state.controls.name), metrics=validate_metrics(state.metrics, objective_metric=objective_metric), stop_reason=str(stop_reason), iterations=int(state.iteration), optimizer=optimizer or state.optimizer_name or 'unknown', objective_metric=str(objective_metric), state=state, system_params=dict(state.system_params), diagnostics=dict(diagnostics or {}))

    @property
    def counters(self) -> dict[str, int]:
        return dict(self.diagnostics.get("counters", {}))

    @property
    def J(self) -> float:
        """Legacy accessor for a metric literally named ``J``."""
        return float(self.metrics['J'])

    @property
    def objective_value(self) -> float:
        """Return the final value of the declared objective metric."""
        return float(self.metrics[self.objective_metric])

    def warmstart(self, *, target_optimizer: str | None=None) -> WarmStartState:
        """Return safe warmstart state for another optimizer call."""
        return WarmStartState.from_result(self, target_optimizer=target_optimizer)

    def to_dict(self, *, include_state: bool=False) -> dict[str, Any]:
        """Return a JSON-friendly result payload.

        ``include_state`` intentionally keeps optimizer-private state shallow.  Full
        checkpoint persistence belongs to the future logs/checkpoint layer.
        """
        payload: dict[str, Any] = {'controls': controls_to_dict(self.controls), 'metrics': to_plain(self.metrics), 'objective_metric': self.objective_metric, 'objective_value': to_plain(self.objective_value), 'J': None if 'J' not in self.metrics else to_plain(float(self.metrics['J'])), 'stop_reason': self.stop_reason, 'iterations': int(self.iterations), 'optimizer': self.optimizer, 'system_params': to_plain(self.system_params), 'diagnostics': to_plain(self.diagnostics)}
        payload["run_record"] = None if self.run_record is None else self.run_record.to_dict()
        if include_state and self.state is not None:
            payload['state'] = {'iteration': int(self.state.iteration), 'global_iteration': int(self.state.global_iteration), 'step_size': to_plain(self.state.step_size), 'optimizer_name': self.state.optimizer_name, 'stop_reason': self.state.stop_reason, 'best_metrics': to_plain(self.state.best_metrics)}
        return payload
