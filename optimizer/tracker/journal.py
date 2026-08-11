"""Append-only journal writer for the canonical optimization run model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

from optimizer.controls import Controls
from optimizer.tracker.artifacts import CheckpointReference, save_checkpoint
from optimizer.tracker.model import (
    AnalysisRecord,
    ChunkRecord,
    InterventionRecord,
    IterationRecord,
    LiveSnapshot,
    RunReceipt,
    ScalarSnapshot,
    TerminalRecord,
    compact_json,
    json_safe,
    utc_now,
)


_RUN_JOURNAL_CREATE_TOKEN = object()


class RunJournal:
    """Persist caller-supplied run facts without importing or calling a system.

    ``timeline.jsonl`` is the causal source of truth.  ``live.json`` is an
    atomically replaced convenience snapshot for a small local observer, and
    ``terminal.json`` makes the selected final result direct to locate.
    """

    def __init__(
        self,
        run_dir: Path,
        receipt: RunReceipt,
        *,
        _create_token: object | None = None,
    ) -> None:
        if _create_token is not _RUN_JOURNAL_CREATE_TOKEN:
            raise RuntimeError(
                "RunJournal instances must be created with RunJournal.create(); "
                "reopening a live journal is not supported."
            )
        self.run_dir = run_dir
        self.receipt = receipt
        self._timeline_path = run_dir / "timeline.jsonl"
        self._writer_path = run_dir / "writer.json"
        self._lock_path = run_dir / ".writer.lock"
        self._writer_id = uuid4().hex
        self._writer_host = socket.gethostname()
        self._sequence = 0
        self._last_event_id: str | None = None
        self._created_at = perf_counter()
        self._started_at = self._created_at
        self._last_heartbeat_at = 0.0
        self._directory_synced_paths: set[Path] = set()
        self._process_start_token = self._read_process_start_token(os.getpid())
        self._last_counters: dict[str, int] = {}
        self._latest_iteration: int | None = None
        self._latest_method: str | None = None
        self._current_snapshot: ScalarSnapshot | None = None
        self._best_snapshot: ScalarSnapshot | None = None
        self._live_status = "created"
        self._live_phase = "initialization"
        self._lifecycle = "created"

    @classmethod
    def create(cls, run_dir: str | Path, receipt: RunReceipt) -> "RunJournal":
        """Create a new run directory and write its immutable small receipt."""

        if not isinstance(receipt, RunReceipt):
            raise TypeError("receipt must be a RunReceipt.")
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=False)
        journal = cls(path, receipt, _create_token=_RUN_JOURNAL_CREATE_TOKEN)
        journal._claim_writer()
        journal._write_json_atomically(path / "run_receipt.json", receipt.to_dict())
        return journal

    def _writer_payload(
        self,
        *,
        status: str,
        last_event_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.receipt.schema_version,
            "run_id": self.receipt.run_id,
            "writer_id": self._writer_id,
            "pid": os.getpid(),
            "host": self._writer_host,
            "process_start_token": self._process_start_token,
            "status": str(status),
            "lifecycle": self._lifecycle,
            "updated_utc": utc_now(),
            "last_event_id": last_event_id,
        }

    def _claim_writer(self) -> None:
        """Claim one exclusive local writer token for this run directory."""

        payload = self._writer_payload(status="live")
        with self._lock_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._write_json_atomically(self._writer_path, payload)

    def _heartbeat(
        self,
        event: Mapping[str, Any],
        *,
        status: str = "live",
        force: bool = False,
    ) -> None:
        now = perf_counter()
        if status == "live" and not force and now - self._last_heartbeat_at < 0.5:
            return
        self._write_json_atomically(
            self._writer_path,
            self._writer_payload(status=status, last_event_id=str(event["event_id"])),
        )
        self._last_heartbeat_at = now

    def _release_writer(self, event: Mapping[str, Any]) -> None:
        self._heartbeat(event, status="closed", force=True)
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def _require_lifecycle(self, *allowed: str) -> None:
        if self._lifecycle not in allowed:
            expected = ", ".join(allowed)
            raise RuntimeError(
                f"Run journal lifecycle is {self._lifecycle!r}; expected {expected}."
            )

    @staticmethod
    def _read_process_start_token(pid: int) -> str | None:
        """Return a stable local process-start token without sending a signal."""

        try:
            completed = subprocess.run(
                ["ps", "-p", str(int(pid)), "-o", "lstart="],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        token = completed.stdout.strip()
        return token or None

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _write_json_atomically(self, path: Path, payload: Mapping[str, Any]) -> None:
        existed = path.exists()
        temporary = path.with_suffix(path.suffix + f".{self._writer_id}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if not existed and path not in self._directory_synced_paths:
            self._fsync_directory(path.parent)
            self._directory_synced_paths.add(path)

    def _append(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        compact_payload = compact_json(payload)
        next_sequence = self._sequence + 1
        event = {
            "schema_version": self.receipt.schema_version,
            "run_id": self.receipt.run_id,
            "event_id": f"{self.receipt.run_id}:{next_sequence:06d}",
            "sequence": next_sequence,
            "created_utc": utc_now(),
            "elapsed_s": perf_counter() - self._started_at,
            "kind": str(kind),
            "payload": compact_payload,
        }
        timeline_existed = self._timeline_path.exists()
        with self._timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not timeline_existed:
            self._fsync_directory(self._timeline_path.parent)
        self._sequence = next_sequence
        self._last_event_id = str(event["event_id"])
        self._heartbeat(
            event,
            force=str(kind) in {"run_start", "checkpoint", "run_end", "tracking_failure"},
        )
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
        self._require_lifecycle("running")
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

    def _publish_live(
        self,
        event: Mapping[str, Any],
        *,
        status: str | None = None,
        phase: str | None = None,
    ) -> None:
        """Refresh the replaceable display snapshot from already-recorded facts."""

        if status is not None:
            self._live_status = str(status)
        if phase is not None:
            self._live_phase = str(phase)
        if self._current_snapshot is None:
            return
        self.update_live(
            LiveSnapshot.create(
                status=self._live_status,
                phase=self._live_phase,
                iteration=self._latest_iteration,
                method=self._latest_method,
                current=self._current_snapshot,
                best=self._best_snapshot,
            ),
            last_event_id=str(event["event_id"]),
        )

    def record_start(
        self,
        initial: ScalarSnapshot,
        *,
        method: str,
        phase: str = "initialization",
        chunk: int = 0,
        system_params: Mapping[str, Any] | None = None,
        api_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the measured initial state before any optimizer iteration."""

        if not isinstance(initial, ScalarSnapshot):
            raise TypeError("initial must be a ScalarSnapshot.")
        self._require_lifecycle("created")
        if not str(method):
            raise ValueError("method must be a non-empty string.")
        self._require_nonregressing(initial)
        setup_s = perf_counter() - self._created_at
        self._started_at = perf_counter()
        event = self._append(
            "run_start",
            {
                "method": str(method),
                "phase": str(phase),
                "chunk": int(chunk),
                "system_params": dict(system_params or {}),
                "api_context": dict(api_context or {}),
                "setup_s": setup_s,
                "initial": initial.to_dict(),
            },
        )
        self._lifecycle = "running"
        self._heartbeat(event, force=True)
        self._commit_counters(initial)
        self._latest_iteration = 0
        self._latest_method = str(method)
        self._current_snapshot = initial
        self._best_snapshot = initial
        self._publish_live(event, status="running", phase=str(phase))
        return event

    def record_iteration(self, record: IterationRecord) -> dict[str, Any]:
        """Append one raw outer-iteration transition and refresh ``live.json``."""

        if not isinstance(record, IterationRecord):
            raise TypeError("record must be an IterationRecord.")
        self._require_lifecycle("running")
        if self._latest_iteration is not None and record.iteration <= self._latest_iteration:
            raise ValueError("iteration must strictly increase within one journal lifecycle.")
        self._require_nonregressing(record.before)
        self._require_transition(record.before, record.after)
        self._require_nonregressing(record.after)
        event = self._append("iteration", record.to_dict())
        self._commit_counters(record.after)
        self._latest_iteration = record.iteration
        self._latest_method = record.method
        self._current_snapshot = record.after
        if record.best_after is not None:
            self._best_snapshot = record.best_after
        elif self._best_snapshot is None:
            self._best_snapshot = record.after
        self._publish_live(event, status="running", phase=record.phase or "optimization")
        return event

    def record_analysis(self, record: AnalysisRecord) -> dict[str, Any]:
        """Append a derived analysis tied to an explicit raw-iteration window."""

        if not isinstance(record, AnalysisRecord):
            raise TypeError("record must be an AnalysisRecord.")
        self._require_lifecycle("running")
        analysis_state = ScalarSnapshot.create(counters=record.counters)
        self._require_nonregressing(analysis_state)
        event = self._append("analysis", record.to_dict())
        self._commit_counters(analysis_state)
        self._publish_live(event)
        return event

    def record_intervention(self, record: InterventionRecord) -> dict[str, Any]:
        """Append an action, its causal evidence link, and its measured outcome."""

        if not isinstance(record, InterventionRecord):
            raise TypeError("record must be an InterventionRecord.")
        self._require_lifecycle("running")
        self._require_nonregressing(record.before)
        self._require_transition(record.before, record.after)
        self._require_nonregressing(record.after)
        event = self._append("intervention", record.to_dict())
        self._commit_counters(record.after)
        self._current_snapshot = record.after
        if self._best_snapshot is None:
            self._best_snapshot = record.after
        self._publish_live(event, status="running", phase="intervention")
        return event

    def record_chunk(self, record: ChunkRecord) -> dict[str, Any]:
        """Append one completed chunk with resolved method, phase, and system facts."""

        if not isinstance(record, ChunkRecord):
            raise TypeError("record must be a ChunkRecord.")
        self._require_lifecycle("running")
        event = self._append("chunk", record.to_dict())
        self._latest_method = record.method
        self._live_phase = record.phase or self._live_phase
        self._publish_live(event)
        return event

    def save_checkpoint(
        self,
        *,
        checkpoint_id: str,
        label: str,
        controls: Controls,
        optimizer_state: Mapping[str, Any] | None = None,
        snapshot: ScalarSnapshot | Mapping[str, Any] | None = None,
        api_context: Mapping[str, Any] | None = None,
        reason: str = "manual_pin",
        source_event_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointReference:
        """Save exact controls, metrics, optimizer state, and API context."""

        self._require_lifecycle("running")
        if isinstance(snapshot, ScalarSnapshot):
            snapshot_payload = snapshot.to_dict()
        elif snapshot is None:
            snapshot_payload = (
                {} if self._current_snapshot is None else self._current_snapshot.to_dict()
            )
        else:
            snapshot_payload = json_safe(snapshot)
        resolved_api_context = {
            "receipt": dict(self.receipt.api_context),
            "checkpoint": dict(api_context or {}),
        }
        checkpoint_metadata = {
            **dict(metadata or {}),
            "reason": str(reason),
            "source_event_id": self._last_event_id if source_event_id is None else source_event_id,
            "method": self._latest_method,
            "phase": self._live_phase,
            "iteration": self._latest_iteration,
            "metric_contract": {
                "objective_metric": self.receipt.objective_metric,
                "direction": self.receipt.direction,
            },
        }

        reference = save_checkpoint(
            self.run_dir,
            checkpoint_id=checkpoint_id,
            label=label,
            controls=controls,
            optimizer_state=optimizer_state,
            snapshot=snapshot_payload,
            api_context=resolved_api_context,
            metadata=checkpoint_metadata,
        )
        event = self._append("checkpoint", {"checkpoint": reference.to_dict()})
        self._publish_live(event)
        return reference

    def record_terminal(self, record: TerminalRecord) -> dict[str, Any]:
        """Append and index the terminal result; no optimality claim is invented."""

        if not isinstance(record, TerminalRecord):
            raise TypeError("record must be a TerminalRecord.")
        self._require_lifecycle("running")
        self._require_nonregressing(record.final)
        event = self._append("run_end", record.to_dict())
        self._commit_counters(record.final)
        self._current_snapshot = record.final
        if record.best is not None:
            self._best_snapshot = record.best
        elif self._best_snapshot is None:
            self._best_snapshot = record.final
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
        self._publish_live(event, status=record.status, phase="terminal")
        self._lifecycle = "ended"
        self._release_writer(event)
        return event

    def record_tracking_failure(self, failure: Mapping[str, Any]) -> dict[str, Any]:
        """Close an incomplete evidence stream after a contained observer failure."""

        self._require_lifecycle("created", "running")
        if self._lifecycle == "created":
            self._write_json_atomically(
                self._writer_path,
                self._writer_payload(status="failed", last_event_id=None),
            )
            self._lifecycle = "failed"
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
            return {"kind": "tracking_failure", "payload": {"failure": dict(failure)}}
        event = self._append("tracking_failure", {"failure": dict(failure)})
        self._publish_live(event, status="tracking_failed", phase="terminal")
        self._lifecycle = "failed"
        self._heartbeat(event, status="failed", force=True)
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        return event

    def record_observer_failure(self, failure: Mapping[str, Any]) -> dict[str, Any]:
        """Record another passive sink's failure while this journal remains healthy."""

        self._require_lifecycle("running")
        event = self._append("observer_failure", {"failure": dict(failure)})
        self._publish_live(event)
        return event
