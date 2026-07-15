"""Trace record dataclasses for optimizer runs.

Why this file exists
--------------------
The optimizer needs a technical history that is cheaper and lighter than full
checkpoints.  During curriculum-style work, the user needs to inspect questions like:

    Which stage changed the system weights?
    Did fidelity break after a lambda4 ramp?
    Did the step size collapse?
    Was a trial accepted or rejected?

This module defines small immutable record objects for those answers.  They are
plain data containers and are intentionally independent of any file format.

How it fits the architecture
----------------------------
- ``Trace`` stores these records in order.
- the future engine will emit one ``IterationRecord`` per iteration and one
  ``ChunkRecord`` per optimizer chunk.
- reports can later turn these records into tables, CSV, JSONL, or notebook displays.
- checkpoints are separate and heavier; records only describe what happened.

What this file deliberately does not do
---------------------------------------
It does not write logs to disk, restore controls, evaluate systems, or decide whether
a chunk is good.  It only defines structured trace payloads.

Reviewer invariants
-------------------
- records are immutable dataclasses.
- metrics and technical values are shallow-copied into plain dictionaries.
- ``to_dict`` output is predictable and JSON-friendly for ordinary scalar payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _copy_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow plain-dict copy for trace payloads."""

    return dict(payload or {})


@dataclass(frozen=True)
class IterationRecord:
    """Lightweight record for one optimizer iteration."""

    run_id: str
    optimizer: str
    iteration: int
    global_iteration: int
    metrics: dict[str, Any] = field(default_factory=dict)
    system_params: dict[str, Any] = field(default_factory=dict)
    technical: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    accepted: bool | None = None
    reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        optimizer: str,
        iteration: int,
        global_iteration: int,
        metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        technical: Mapping[str, Any] | None = None,
        stage: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
    ) -> "IterationRecord":
        return cls(
            run_id=str(run_id),
            optimizer=str(optimizer),
            iteration=int(iteration),
            global_iteration=int(global_iteration),
            metrics=_copy_mapping(metrics),
            system_params=_copy_mapping(system_params),
            technical=_copy_mapping(technical),
            stage=stage,
            accepted=accepted,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "optimizer": self.optimizer,
            "iteration": self.iteration,
            "global_iteration": self.global_iteration,
            "stage": self.stage,
            "accepted": self.accepted,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "system_params": dict(self.system_params),
            "technical": dict(self.technical),
        }


@dataclass(frozen=True)
class ChunkRecord:
    """Lightweight record for one optimizer chunk."""

    run_id: str
    optimizer: str
    chunk: int
    start_iteration: int
    end_iteration: int
    start_metrics: dict[str, Any] = field(default_factory=dict)
    end_metrics: dict[str, Any] = field(default_factory=dict)
    system_params: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    accepted: bool | None = None
    reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        optimizer: str,
        chunk: int,
        start_iteration: int,
        end_iteration: int,
        start_metrics: Mapping[str, Any] | None = None,
        end_metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        stage: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
    ) -> "ChunkRecord":
        return cls(
            run_id=str(run_id),
            optimizer=str(optimizer),
            chunk=int(chunk),
            start_iteration=int(start_iteration),
            end_iteration=int(end_iteration),
            start_metrics=_copy_mapping(start_metrics),
            end_metrics=_copy_mapping(end_metrics),
            system_params=_copy_mapping(system_params),
            stage=stage,
            accepted=accepted,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "optimizer": self.optimizer,
            "chunk": self.chunk,
            "start_iteration": self.start_iteration,
            "end_iteration": self.end_iteration,
            "stage": self.stage,
            "accepted": self.accepted,
            "reason": self.reason,
            "start_metrics": dict(self.start_metrics),
            "end_metrics": dict(self.end_metrics),
            "system_params": dict(self.system_params),
        }


@dataclass(frozen=True)
class EventRecord:
    """General trace event for non-iteration actions such as rollback."""

    run_id: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> "EventRecord":
        return cls(run_id=str(run_id), event=str(event), payload=_copy_mapping(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "event": self.event, "payload": dict(self.payload)}

