"""Append-only journal writer for the canonical optimization run model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from optimizer.controls import Controls
from optimizer.tracker.artifacts import CheckpointReference, save_checkpoint
from optimizer.tracker.model import (
    AnalysisRecord,
    InterventionRecord,
    IterationRecord,
    LiveSnapshot,
    RunReceipt,
    ScalarSnapshot,
    TerminalRecord,
    json_safe,
    utc_now,
)


class RunJournal:
    """Persist caller-supplied run facts without importing or calling a system.

    ``timeline.jsonl`` is the causal source of truth.  ``live.json`` is an
    atomically replaced convenience snapshot for a small local observer, and
    ``terminal.json`` makes the selected final result direct to locate.
    """

    def __init__(self, run_dir: Path, receipt: RunReceipt) -> None:
        self.run_dir = run_dir
        self.receipt = receipt
        self._timeline_path = run_dir / "timeline.jsonl"
        self._sequence = 0
        self._started_at = perf_counter()
        self._last_counters: dict[str, int] = {}
        self._latest_iteration: int | None = None
        self._latest_method: str | None = None

    @classmethod
    def create(cls, run_dir: str | Path, receipt: RunReceipt) -> "RunJournal":
        """Create a new run directory and write its immutable small receipt."""

        if not isinstance(receipt, RunReceipt):
            raise TypeError("receipt must be a RunReceipt.")
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=False)
        journal = cls(path, receipt)
        journal._write_json_atomically(path / "run_receipt.json", receipt.to_dict())
        return journal

    @staticmethod
    def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _append(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "schema_version": self.receipt.schema_version,
            "run_id": self.receipt.run_id,
            "event_id": f"{self.receipt.run_id}:{self._sequence:06d}",
            "sequence": self._sequence,
            "created_utc": utc_now(),
            "elapsed_s": perf_counter() - self._started_at,
            "kind": str(kind),
            "payload": json_safe(payload),
        }
        with self._timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def _require_nonregressing(self, snapshot: ScalarSnapshot) -> None:
        for name, count in snapshot.counters.items():
            previous = self._last_counters.get(name)
            if previous is not None and count < previous:
                raise ValueError(
                    f"Counter {name!r} regressed from {previous} to {count}; "
                    "the causal journal would become inconsistent."
                )

    @staticmethod
    def _require_transition(before: ScalarSnapshot, after: ScalarSnapshot) -> None:
        for name, count in before.counters.items():
            if after.counters.get(name, count) < count:
                raise ValueError(f"Counter {name!r} regressed within one record.")

    def _commit_counters(self, snapshot: ScalarSnapshot) -> None:
        self._last_counters.update(snapshot.counters)

    def update_live(self, snapshot: LiveSnapshot, *, last_event_id: str | None = None) -> None:
        """Replace the observer-only progress snapshot without making a system call."""

        if not isinstance(snapshot, LiveSnapshot):
            raise TypeError("snapshot must be a LiveSnapshot.")
        self._write_json_atomically(
            self.run_dir / "live.json",
            {
                "schema_version": self.receipt.schema_version,
                "run_id": self.receipt.run_id,
                "updated_utc": utc_now(),
                "last_event_id": last_event_id,
                **snapshot.to_dict(),
            },
        )

    def record_start(self, initial: ScalarSnapshot, *, method: str, phase: str = "initialization") -> dict[str, Any]:
        """Record the measured initial state before any optimizer iteration."""

        if not isinstance(initial, ScalarSnapshot):
            raise TypeError("initial must be a ScalarSnapshot.")
        self._require_nonregressing(initial)
        event = self._append("run_start", {"method": str(method), "phase": str(phase), "initial": initial.to_dict()})
        self._commit_counters(initial)
        self._latest_iteration = 0
        self._latest_method = str(method)
        self.update_live(
            LiveSnapshot.create(
                status="running",
                phase=str(phase),
                iteration=0,
                method=str(method),
                current=initial,
                best=initial,
            ),
            last_event_id=event["event_id"],
        )
        return event

    def record_iteration(self, record: IterationRecord) -> dict[str, Any]:
        """Append one raw outer-iteration transition and refresh ``live.json``."""

        if not isinstance(record, IterationRecord):
            raise TypeError("record must be an IterationRecord.")
        self._require_nonregressing(record.before)
        self._require_transition(record.before, record.after)
        self._require_nonregressing(record.after)
        event = self._append("iteration", record.to_dict())
        self._commit_counters(record.after)
        self._latest_iteration = record.iteration
        self._latest_method = record.method
        self.update_live(
            LiveSnapshot.create(
                status="running",
                phase=record.phase or "optimization",
                iteration=record.iteration,
                method=record.method,
                current=record.after,
                best=record.best_after or record.after,
            ),
            last_event_id=event["event_id"],
        )
        return event

    def record_analysis(self, record: AnalysisRecord) -> dict[str, Any]:
        """Append a derived analysis tied to an explicit raw-iteration window."""

        if not isinstance(record, AnalysisRecord):
            raise TypeError("record must be an AnalysisRecord.")
        analysis_state = ScalarSnapshot.create(counters=record.counters)
        self._require_nonregressing(analysis_state)
        event = self._append("analysis", record.to_dict())
        self._commit_counters(analysis_state)
        return event

    def record_intervention(self, record: InterventionRecord) -> dict[str, Any]:
        """Append an action, its causal evidence link, and its measured outcome."""

        if not isinstance(record, InterventionRecord):
            raise TypeError("record must be an InterventionRecord.")
        self._require_nonregressing(record.before)
        self._require_transition(record.before, record.after)
        self._require_nonregressing(record.after)
        event = self._append("intervention", record.to_dict())
        self._commit_counters(record.after)
        self.update_live(
            LiveSnapshot.create(
                status="running",
                phase="intervention",
                iteration=self._latest_iteration,
                method=self._latest_method,
                current=record.after,
                best=record.after,
            ),
            last_event_id=event["event_id"],
        )
        return event

    def save_checkpoint(
        self,
        *,
        checkpoint_id: str,
        label: str,
        controls: Controls,
        optimizer_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointReference:
        """Save controls only at an explicit caller-selected checkpoint."""

        reference = save_checkpoint(
            self.run_dir,
            checkpoint_id=checkpoint_id,
            label=label,
            controls=controls,
            optimizer_state=optimizer_state,
            metadata=metadata,
        )
        self._append("checkpoint", {"checkpoint": reference.to_dict()})
        return reference

    def record_terminal(self, record: TerminalRecord) -> dict[str, Any]:
        """Append and index the terminal result; no optimality claim is invented."""

        if not isinstance(record, TerminalRecord):
            raise TypeError("record must be a TerminalRecord.")
        self._require_nonregressing(record.final)
        event = self._append("run_end", record.to_dict())
        self._commit_counters(record.final)
        self._write_json_atomically(
            self.run_dir / "terminal.json",
            {
                "schema_version": self.receipt.schema_version,
                "run_id": self.receipt.run_id,
                "event_id": event["event_id"],
                "created_utc": event["created_utc"],
                "elapsed_s": event["elapsed_s"],
                **record.to_dict(),
            },
        )
        self.update_live(
            LiveSnapshot.create(
                status=record.status,
                phase="terminal",
                iteration=self._latest_iteration,
                method=self._latest_method,
                current=record.final,
                best=record.final,
            ),
            last_event_id=event["event_id"],
        )
        return event
