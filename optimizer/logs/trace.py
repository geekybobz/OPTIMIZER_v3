"""Trace ledger for records, checkpoints, and rollback.

Why this file exists
--------------------
The optimizer needs a single object that can collect technical records and hold
rollback points.  This is especially important for the planned curriculum workflow:
run a short chunk, inspect metrics, change system prefactors, and restore
``stage_start`` if the new chunk breaks important metrics.

This module provides that ledger.  Phase 4 keeps it in memory.  Later phases can add
file persistence without changing the basic user-facing ideas.

How it fits the architecture
----------------------------
- the future engine will call ``record_iteration`` and ``record_chunk``.
- curriculum/manual workflows will call ``checkpoint`` before changing system params.
- repairs and guards can call ``event`` to record rollback or rejection reasons.
- later reports can consume ``to_dict``.

What this file deliberately does not do
---------------------------------------
It does not run optimization, decide acceptance, or write log files.  It only stores
records and checkpoint snapshots with label-based lookup.

Reviewer invariants
-------------------
- label restore uses the latest checkpoint for that label.
- checkpoint history is preserved even when labels are reused.
- restore returns copies, not references to stored checkpoints.
- records and checkpoints remain separate concepts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from optimizer.controls import Controls
from optimizer.logs.checkpoint import Checkpoint
from optimizer.logs.records import ChunkRecord, EventRecord, IterationRecord
from optimizer.state import RunState


@dataclass
class Trace:
    """In-memory trace ledger for one run."""

    run_id: str
    iteration_records: list[IterationRecord] = field(default_factory=list)
    chunk_records: list[ChunkRecord] = field(default_factory=list)
    event_records: list[EventRecord] = field(default_factory=list)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    labels: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def record_iteration(
        self,
        *,
        optimizer: str,
        iteration: int,
        global_iteration: int,
        metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        technical: Mapping[str, Any] | None = None,
        stage: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
    ) -> IterationRecord:
        """Append and return one iteration record."""

        record = IterationRecord.create(
            run_id=self.run_id,
            optimizer=optimizer,
            iteration=iteration,
            global_iteration=global_iteration,
            metrics=metrics,
            system_params=system_params,
            technical=technical,
            stage=stage,
            accepted=accepted,
            reason=reason,
        )
        self.iteration_records.append(record)
        return record

    def record_chunk(
        self,
        *,
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
    ) -> ChunkRecord:
        """Append and return one chunk record."""

        record = ChunkRecord.create(
            run_id=self.run_id,
            optimizer=optimizer,
            chunk=chunk,
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            start_metrics=start_metrics,
            end_metrics=end_metrics,
            system_params=system_params,
            stage=stage,
            accepted=accepted,
            reason=reason,
        )
        self.chunk_records.append(record)
        return record

    def event(self, event: str, **payload: Any) -> EventRecord:
        """Append a general trace event."""

        record = EventRecord.create(run_id=self.run_id, event=event, payload=payload)
        self.event_records.append(record)
        return record

    def checkpoint(
        self,
        label: str,
        controls: Controls,
        state: RunState | None = None,
        *,
        metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        optimizer_state: Mapping[str, Any] | None = None,
        iteration: int | None = None,
        stage: str | None = None,
        random_state: Mapping[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint and register it under a label."""

        previous_id = self.labels[label][-1] if self.labels.get(label) else None
        checkpoint = Checkpoint.create(
            label=label,
            controls=controls,
            state=state,
            metrics=metrics,
            system_params=system_params,
            optimizer_state=optimizer_state,
            iteration=iteration,
            stage=stage,
            previous_id=previous_id,
            random_state=random_state,
        )
        self.checkpoints[checkpoint.id] = checkpoint
        self.labels[label].append(checkpoint.id)
        if state is not None:
            state.checkpoint_ids[label] = checkpoint.id
        return checkpoint

    def latest_checkpoint_id(self, label: str) -> str:
        """Return the latest checkpoint id for a label."""

        ids = self.labels.get(label)
        if not ids:
            raise KeyError(f"No checkpoint found for label {label!r}.")
        return ids[-1]

    def get_checkpoint(self, label_or_id: str) -> Checkpoint:
        """Return checkpoint by explicit id or latest label."""

        if label_or_id in self.checkpoints:
            return self.checkpoints[label_or_id]
        checkpoint_id = self.latest_checkpoint_id(label_or_id)
        return self.checkpoints[checkpoint_id]

    def restore(self, label_or_id: str) -> tuple[Controls, RunState | None]:
        """Restore controls and optional state from checkpoint id or label."""

        checkpoint = self.get_checkpoint(label_or_id)
        controls, state = checkpoint.restore()
        self.event("restore", checkpoint_id=checkpoint.id, label=checkpoint.label)
        return controls, state

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-focused trace payload."""

        return {
            "run_id": self.run_id,
            "iterations": [record.to_dict() for record in self.iteration_records],
            "chunks": [record.to_dict() for record in self.chunk_records],
            "events": [record.to_dict() for record in self.event_records],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints.values()],
            "labels": {label: list(ids) for label, ids in self.labels.items()},
        }

