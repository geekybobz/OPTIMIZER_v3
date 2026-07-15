"""In-memory checkpoint snapshots.

Why this file exists
--------------------
Curriculum optimization changes system cost prefactors between short optimizer
chunks.  If a new chunk damages fidelity, robustness, or energy too much, the user
needs to restore a known-good state such as ``stage_start`` or ``best_safe``.

This module defines the restorable checkpoint object used by ``Trace``.  It is an
in-memory snapshot for Phase 4.  The object is deliberately serialization-ready, but
actual disk persistence belongs to a later persistence layer.

How it fits the architecture
----------------------------
- ``Trace.checkpoint(...)`` creates these snapshots.
- ``Trace.restore(label)`` returns copied controls and optional copied ``RunState``.
- the future engine will save labels such as ``latest``, ``accepted``, and ``best_J``.
- future logs/checkpoint persistence can serialize the ``to_dict`` payload.

What this file deliberately does not do
---------------------------------------
It does not write files, choose checkpoint cadence, or decide whether a checkpoint is
good.  It only captures enough in-memory state to restore.

Reviewer invariants
-------------------
- controls are copied when a checkpoint is created.
- run state is copied into an independent ``RunState`` when provided.
- restore returns fresh copies so later mutation does not corrupt the checkpoint.
- labels are simple strings; label history is managed by ``Trace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4

from optimizer.controls import Controls
from optimizer.result import controls_to_dict
from optimizer.state import RunState


def _copy_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow plain-dict copy."""

    return dict(payload or {})


def clone_run_state(state: RunState) -> RunState:
    """Create an independent copy of a run state for checkpoint storage."""

    cloned = RunState(
        controls=state.controls.copy(name=state.controls.name),
        metrics=dict(state.metrics),
        iteration=int(state.iteration),
        global_iteration=int(state.global_iteration),
        step_size=state.step_size,
        optimizer_name=state.optimizer_name,
        optimizer_state=dict(state.optimizer_state),
        best_controls=None if state.best_controls is None else state.best_controls.copy(name="best"),
        best_metrics=None if state.best_metrics is None else dict(state.best_metrics),
        stop_reason=state.stop_reason,
        trace_id=state.trace_id,
        checkpoint_ids=dict(state.checkpoint_ids),
        system_params=dict(state.system_params),
    )
    return cloned


@dataclass(frozen=True)
class Checkpoint:
    """Restorable snapshot of controls plus optional run state."""

    id: str
    label: str
    controls: Controls
    state: RunState | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    system_params: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    stage: str | None = None
    previous_id: str | None = None
    random_state: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        label: str,
        controls: Controls,
        state: RunState | None = None,
        metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        optimizer_state: Mapping[str, Any] | None = None,
        iteration: int | None = None,
        stage: str | None = None,
        previous_id: str | None = None,
        random_state: Mapping[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> "Checkpoint":
        copied_state = None if state is None else clone_run_state(state)
        source_metrics = metrics if metrics is not None else (state.metrics if state is not None else {})
        source_params = (
            system_params if system_params is not None else (state.system_params if state is not None else {})
        )
        source_optimizer_state = (
            optimizer_state
            if optimizer_state is not None
            else (state.optimizer_state if state is not None else {})
        )
        source_iteration = iteration if iteration is not None else (state.iteration if state is not None else 0)
        return cls(
            id=checkpoint_id or uuid4().hex,
            label=str(label),
            controls=controls.copy(name=controls.name),
            state=copied_state,
            metrics=_copy_mapping(source_metrics),
            system_params=_copy_mapping(source_params),
            optimizer_state=_copy_mapping(source_optimizer_state),
            iteration=int(source_iteration),
            stage=stage,
            previous_id=previous_id,
            random_state=None if random_state is None else dict(random_state),
        )

    def restore(self) -> tuple[Controls, RunState | None]:
        """Return independent controls/state copies."""

        controls = self.controls.copy(name=self.controls.name)
        state = None if self.state is None else clone_run_state(self.state)
        return controls, state

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-focused export payload."""

        return {
            "id": self.id,
            "label": self.label,
            "controls": controls_to_dict(self.controls),
            "metrics": dict(self.metrics),
            "system_params": dict(self.system_params),
            "optimizer_state_keys": sorted(self.optimizer_state),
            "iteration": int(self.iteration),
            "stage": self.stage,
            "previous_id": self.previous_id,
            "has_state": self.state is not None,
            "has_random_state": self.random_state is not None,
        }

