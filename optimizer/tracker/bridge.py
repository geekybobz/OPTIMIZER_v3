"""Passive adapter from generic engine events to the durable tracker journal.

This module is the only tracker component that knows about the core observer
contract.  It receives compact engine facts and never receives a system,
controls, gradient, evaluator, or diagnostic hook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
import re

from optimizer.core.observer import (
    CheckpointEvent,
    ChunkEvent,
    IterationEvent,
    ObserverAdapter,
    RunStartEvent,
    SnapshotFacts,
    TerminalEvent,
)
from optimizer.tracker.journal import RunJournal
from optimizer.tracker.model import (
    ChunkRecord,
    IterationRecord,
    ScalarSnapshot,
    TerminalRecord,
    json_safe,
)


_SKIP = object()
_MAX_INLINE_SEQUENCE = 64
_MAX_INLINE_STRING = 8192
_MAX_BRIDGE_BYTES = 12000


@dataclass(frozen=True)
class CheckpointPolicy:
    """Bound durable checkpoint storage without dropping iteration evidence.

    The engine can emit frequent ``latest``, ``accepted`` and ``best_*``
    checkpoints for in-memory traces.  Persisting all of them as control arrays
    is inappropriate for long scientific runs.  This policy keeps the durable
    journal sparse by default while the timeline still records every attempted
    iteration.
    """

    save_all: bool = False
    persist_start: bool = True
    persist_selected_best: bool = True
    persist_failure: bool = True
    persist_final: bool = True

    def accepts(self, label: str, *, reason: str | None = None) -> bool:
        """Return whether one already-emitted checkpoint should be durable."""

        if self.save_all:
            return True
        name = str(label)
        if name == "chunk_start":
            return self.persist_start
        if name == "chunk_end":
            return self.persist_final
        if name.startswith("failure_"):
            return self.persist_failure
        if str(reason) == "terminal_selected_best":
            return self.persist_selected_best
        return False


def _bounded_string(value: str) -> str:
    if len(value) <= _MAX_INLINE_STRING:
        return value
    marker = f"...[truncated {len(value) - 8000} chars]"
    prefix_length = _MAX_INLINE_STRING - len(marker)
    marker = f"...[truncated {len(value) - prefix_length} chars]"
    prefix_length = _MAX_INLINE_STRING - len(marker)
    return value[:prefix_length] + marker


def _scalar_or_skip(value: Any) -> Any:
    """Keep one portable scalar and decline arrays or unsupported objects."""

    try:
        safe = json_safe(value)
    except (TypeError, ValueError):
        return _SKIP
    if isinstance(safe, str):
        return _bounded_string(safe)
    if safe is None or isinstance(safe, (bool, int, float)):
        return safe
    return _SKIP


def _compact_value(value: Any) -> Any:
    """Keep bounded scalar structure while refusing arrays and large sequences."""

    scalar = _scalar_or_skip(value)
    if scalar is not _SKIP:
        return scalar
    if isinstance(value, Mapping):
        return _compact_mapping(value)
    if (
        isinstance(value, (list, tuple))
        and len(value) <= _MAX_INLINE_SEQUENCE
        and all(isinstance(item, Mapping) for item in value)
    ):
        compacted: list[Any] = []
        for item in value:
            compact = _compact_value(item)
            if compact is _SKIP:
                return _SKIP
            compacted.append(compact)
        return compacted
    return _SKIP


def _compact_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Retain bounded structured technical evidence, never control-sized data."""

    result: dict[str, Any] = {}
    omitted: list[str] = []
    items = tuple((payload or {}).items())
    for index, (key, value) in enumerate(items):
        if index >= 128:
            omitted.extend(str(item_key) for item_key, _ in items[index:])
            break
        compact_key = _bounded_string(str(key))[:256]
        compact = _compact_value(value)
        if compact is not _SKIP:
            result[compact_key] = compact
            encoded_size = len(
                json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )
            if encoded_size <= _MAX_BRIDGE_BYTES:
                continue
            result.pop(compact_key, None)
        omitted.append(compact_key)
    if omitted:
        names = sorted(set(omitted))
        result["_tracker_omitted_fields"] = names[:32]
        if len(names) > 32:
            result["_tracker_omitted_count"] = len(names)
    return result


def _metric_values(payload: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    metrics: dict[str, Any] = {}
    omitted: list[str] = []
    for name, value in payload.items():
        scalar = _scalar_or_skip(value)
        if scalar is _SKIP:
            omitted.append(str(name))
        else:
            metrics[str(name)] = scalar
    return metrics, tuple(sorted(set(omitted)))


def _portable_large_integers(value: Any) -> Any:
    """Tag integers that JavaScript cannot represent exactly."""

    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        return {"__integer_decimal__": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _portable_large_integers(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_large_integers(child) for child in value]
    return value


def _snapshot(facts: SnapshotFacts) -> ScalarSnapshot:
    """Convert existing engine facts into a compact tracker snapshot."""

    metrics, omitted = _metric_values(facts.metrics)
    sources: dict[str, str] = {}
    for name in metrics:
        key = str(name)
        source = facts.metric_sources.get(key)
        if source is not None:
            sources[key] = str(source)
    return ScalarSnapshot.create(
        metrics=metrics,
        metric_sources=sources,
        counters=facts.counters,
        omitted_metrics=omitted,
    )


class JournalObserver(ObserverAdapter):
    """Write one `run_chunk` lifecycle into a fresh :class:`RunJournal`.

    A journal observer is intentionally one-shot.  A later segment coordinator
    will own multi-chunk/campaign continuation rather than pretending that a
    new core evaluator is a transparent journal resume.
    """

    def __init__(
        self,
        journal: RunJournal,
        *,
        save_all_checkpoints: bool = False,
        checkpoint_policy: CheckpointPolicy | None = None,
    ) -> None:
        """Create a one-shot bridge.

        ``save_all_checkpoints=True`` is a compatibility alias for
        ``CheckpointPolicy(save_all=True)`` and should be used only for bounded
        runs.  The default policy retains initial, terminal-selected-best,
        failure, and final controls: durable storage is therefore bounded per
        chunk while scalar iteration evidence remains complete.
        """

        if not isinstance(journal, RunJournal):
            raise TypeError("journal must be a RunJournal.")
        self.journal = journal
        if checkpoint_policy is not None and save_all_checkpoints:
            raise ValueError("use checkpoint_policy or save_all_checkpoints, not both.")
        self.checkpoint_policy = checkpoint_policy or CheckpointPolicy(
            save_all=bool(save_all_checkpoints)
        )
        self._started = False
        self._ended = False
        self._checkpoint_count = 0
        self._checkpoint_references: dict[str, dict[str, Any]] = {}
        self._best_checkpoint_reference: dict[str, Any] | None = None

    def _is_durable_checkpoint(self, label: str, *, reason: str) -> bool:
        return self.checkpoint_policy.accepts(label, reason=reason)

    def on_start(self, event: RunStartEvent) -> None:
        if self._started:
            raise RuntimeError("JournalObserver can record only one engine chunk.")
        self.journal.record_start(
            _snapshot(event.initial),
            method=event.optimizer,
            phase=event.phase,
            chunk=event.chunk,
            system_params=_compact_mapping(event.system_params),
            api_context=_compact_mapping(event.api_context),
        )
        self._started = True

    def on_iteration(self, event: IterationEvent) -> None:
        if not self._started or self._ended:
            raise RuntimeError("JournalObserver received an iteration outside its run lifecycle.")
        self.journal.record_iteration(
            IterationRecord.create(
                iteration=event.iteration,
                method=event.optimizer,
                before=_snapshot(event.before),
                trial=None if event.trial is None else _snapshot(event.trial),
                after=_snapshot(event.after),
                best_after=None if event.best_after is None else _snapshot(event.best_after),
                accepted=bool(event.accepted),
                reason=event.reason or "unspecified",
                proposal=_compact_mapping(event.technical),
                phase=event.stage,
                global_iteration=event.global_iteration,
                chunk=event.chunk,
                system_params=_compact_mapping(event.system_params),
            )
        )

    def on_chunk(self, event: ChunkEvent) -> None:
        if not self._started or self._ended:
            raise RuntimeError("JournalObserver received a chunk outside its run lifecycle.")
        start_metrics, start_omitted = _metric_values(event.start_metrics)
        end_metrics, end_omitted = _metric_values(event.end_metrics)
        self.journal.record_chunk(
            ChunkRecord.create(
                method=event.optimizer,
                chunk=event.chunk,
                start_iteration=event.start_iteration,
                end_iteration=event.end_iteration,
                start_metrics=start_metrics,
                end_metrics=end_metrics,
                start_omitted_metrics=start_omitted,
                end_omitted_metrics=end_omitted,
                system_params=_compact_mapping(event.system_params),
                phase=event.stage,
                accepted=event.accepted,
                reason=event.reason,
            )
        )

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        if not self._started or self._ended:
            raise RuntimeError("JournalObserver received a checkpoint outside its run lifecycle.")
        if not self._is_durable_checkpoint(event.label, reason=event.reason):
            return
        self._checkpoint_count += 1
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", event.label).strip("._") or "checkpoint"
        checkpoint_id = (
            f"{stem}-c{event.chunk:04d}-i{event.global_iteration:08d}-"
            f"{self._checkpoint_count:04d}"
        )
        reference = self.journal.save_checkpoint(
            checkpoint_id=checkpoint_id,
            label=stem,
            controls=event.controls,
            optimizer_state=event.optimizer_state,
            snapshot=_snapshot(event.snapshot),
            api_context=_compact_mapping(event.api_context),
            reason=event.reason,
            metadata={
                "optimizer": event.optimizer,
                "stage": event.stage,
                "chunk": event.chunk,
                "iteration": event.iteration,
                "global_iteration": event.global_iteration,
                "system_params": _compact_mapping(event.system_params),
                "random_state": _compact_mapping(_portable_large_integers(event.random_state)),
                "random_state_source": event.random_state_source,
                "random_state_integer_encoding": "tagged_decimal_v1",
            },
        )
        reference_payload = reference.to_dict()
        self._checkpoint_references[event.label] = reference_payload
        if event.label == "chunk_start" and self._best_checkpoint_reference is None:
            self._best_checkpoint_reference = reference_payload
        if event.reason == "terminal_selected_best":
            self._best_checkpoint_reference = reference_payload

    def on_terminal(self, event: TerminalEvent) -> None:
        if not self._started or self._ended:
            raise RuntimeError("JournalObserver received a terminal event outside its run lifecycle.")
        final_reference = self._checkpoint_references.get("chunk_end")
        selection_fallback: str | None = None
        if self.journal.receipt.selection_rule == "best_finite_value":
            selected_checkpoint = self._best_checkpoint_reference
            if selected_checkpoint is None:
                selected_checkpoint = final_reference
                if final_reference is not None:
                    selection_fallback = "best_checkpoint_unavailable_used_chunk_end"
        else:
            selected_checkpoint = final_reference
            if selected_checkpoint is None:
                selected_checkpoint = self._best_checkpoint_reference
                if selected_checkpoint is not None:
                    selection_fallback = "final_checkpoint_unavailable_used_best"
        if selected_checkpoint is None:
            selection_fallback = "no_durable_checkpoint_available"
        self.journal.record_terminal(
            TerminalRecord.create(
                status=event.status,
                reason=event.reason,
                final=_snapshot(event.final),
                best=None if event.best is None else _snapshot(event.best),
                selected_checkpoint=selected_checkpoint,
                selection_fallback=selection_fallback,
                method=event.optimizer,
                phase=event.stage,
                iteration=event.iteration,
                global_iteration=event.global_iteration,
                chunk=event.chunk,
                system_params=_compact_mapping(event.system_params),
                certification=_compact_mapping(event.certification),
            )
        )
        self._ended = True

    def on_observer_failure(self, failure: Mapping[str, Any], *, fatal: bool = False) -> None:
        """Best-effort closure when this or another passive sink fails."""

        if self._ended:
            return
        if fatal:
            self.journal.record_tracking_failure(_compact_mapping(failure))
            self._ended = True
        else:
            self.journal.record_observer_failure(_compact_mapping(failure))
