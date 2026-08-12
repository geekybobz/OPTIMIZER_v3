"""Passive event contracts emitted by the shared optimization engine.

Observers receive facts the engine has already computed.  They are deliberately
not given a system object, controls, gradients, or an evaluator, so attaching an
observer cannot add numerical work by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from optimizer.controls import Controls


def _copy_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


@dataclass(frozen=True)
class SnapshotFacts:
    """Compact facts known at one point in the engine lifecycle."""

    metrics: dict[str, Any] = field(default_factory=dict)
    metric_sources: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        metrics: Mapping[str, Any] | None = None,
        metric_sources: Mapping[str, str] | None = None,
        counters: Mapping[str, int] | None = None,
    ) -> "SnapshotFacts":
        return cls(
            metrics=_copy_mapping(metrics),
            metric_sources={str(name): str(source) for name, source in (metric_sources or {}).items()},
            counters={str(name): int(value) for name, value in (counters or {}).items()},
        )


@dataclass(frozen=True)
class RunStartEvent:
    """The first validated state of one engine chunk."""

    optimizer: str
    phase: str
    initial: SnapshotFacts
    chunk: int = 0
    stage: str | None = None
    system_params: dict[str, Any] = field(default_factory=dict)
    api_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationEvent:
    """One attempted outer iteration, including a rejected or failed trial."""

    optimizer: str
    iteration: int
    global_iteration: int
    stage: str | None
    system_params: dict[str, Any]
    before: SnapshotFacts
    trial: SnapshotFacts | None
    after: SnapshotFacts
    best_after: SnapshotFacts | None
    technical: dict[str, Any]
    accepted: bool | None
    reason: str | None
    chunk: int = 0


@dataclass(frozen=True)
class ChunkEvent:
    """A completed engine chunk, used by the in-memory compatibility trace."""

    optimizer: str
    chunk: int
    start_iteration: int
    end_iteration: int
    start_metrics: dict[str, Any]
    end_metrics: dict[str, Any]
    system_params: dict[str, Any]
    stage: str | None
    accepted: bool | None
    reason: str | None


@dataclass(frozen=True)
class CheckpointEvent:
    """An engine-selected reproducibility point containing already-owned state."""

    label: str
    reason: str
    controls: Controls
    snapshot: SnapshotFacts
    optimizer: str
    optimizer_state: dict[str, Any]
    system_params: dict[str, Any]
    api_context: dict[str, Any]
    iteration: int
    global_iteration: int
    chunk: int
    stage: str | None = None
    random_state: dict[str, Any] = field(default_factory=dict)
    random_state_source: str = "not_supplied"


@dataclass(frozen=True)
class TerminalEvent:
    """The final fact produced by one engine chunk."""

    optimizer: str
    status: str
    reason: str
    final: SnapshotFacts
    best: SnapshotFacts | None
    iteration: int
    global_iteration: int
    stage: str | None
    chunk: int = 0
    system_params: dict[str, Any] = field(default_factory=dict)
    certification: dict[str, Any] = field(default_factory=dict)


class RunObserver(Protocol):
    """Passive sink for the engine lifecycle.

    Implementations receive only values that already exist in the engine.  A
    durable journal, an in-memory trace, and a future policy-shadow recorder
    can therefore share the same event stream without adding system calls.
    """

    def on_start(self, event: RunStartEvent) -> None: ...

    def on_iteration(self, event: IterationEvent) -> None: ...

    def on_chunk(self, event: ChunkEvent) -> None: ...

    def on_checkpoint(self, event: CheckpointEvent) -> None: ...

    def on_terminal(self, event: TerminalEvent) -> None: ...


class ObserverAdapter:
    """No-op base class for observers interested in only some event kinds."""

    def on_start(self, event: RunStartEvent) -> None:
        del event

    def on_iteration(self, event: IterationEvent) -> None:
        del event

    def on_chunk(self, event: ChunkEvent) -> None:
        del event

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        del event

    def on_terminal(self, event: TerminalEvent) -> None:
        del event


@dataclass(frozen=True)
class ObserverFailure:
    """One contained passive-sink failure.

    Observer failures are evidence failures, not numerical optimizer failures.
    The dispatcher therefore records a compact description and quarantines the
    offending sink for the rest of the chunk.
    """

    observer: str
    callback: str
    error_type: str
    message: str

    @classmethod
    def create(cls, observer: RunObserver, callback: str, error: Exception) -> "ObserverFailure":
        message = str(error)
        if len(message) > 2048:
            omitted = len(message) - 2000
            message = f"{message[:2000]}...[truncated {omitted} chars]"
        return cls(
            observer=f"{type(observer).__module__}.{type(observer).__qualname__}",
            callback=str(callback),
            error_type=type(error).__name__,
            message=message,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "observer": self.observer,
            "callback": self.callback,
            "error_type": self.error_type,
            "message": self.message,
        }


class ObserverDispatcher:
    """Safely fan out lifecycle facts while quarantining failed observers."""

    def __init__(self, observers: Iterable[RunObserver]) -> None:
        self._active = list(observers)
        self._failures: list[ObserverFailure] = []

    @property
    def active(self) -> tuple[RunObserver, ...]:
        return tuple(self._active)

    @property
    def failures(self) -> tuple[dict[str, str], ...]:
        return tuple(failure.to_dict() for failure in self._failures)

    def emit(self, callback: str, event: Any) -> tuple[RunObserver, ...]:
        """Deliver one event and return the observers that accepted it."""

        delivered: list[RunObserver] = []
        contained: list[tuple[RunObserver, ObserverFailure]] = []
        for observer in tuple(self._active):
            try:
                getattr(observer, callback)(event)
            except Exception as exc:  # noqa: BLE001 - passive sinks cannot abort optimization.
                self._active.remove(observer)
                failure = ObserverFailure.create(observer, callback, exc)
                self._failures.append(failure)
                contained.append((observer, failure))
            else:
                delivered.append(observer)
        for observer, failure in contained:
            self._notify_failure(observer, failure)
        return tuple(delivered)

    def _notify_failure(self, failed: RunObserver, failure: ObserverFailure) -> None:
        """Best-effort evidence closure for journals that remain writable."""

        payload = failure.to_dict()
        candidates = (failed, *tuple(self._active))
        seen: set[int] = set()
        for observer in candidates:
            identity = id(observer)
            if identity in seen:
                continue
            seen.add(identity)
            callback = getattr(observer, "on_observer_failure", None)
            if not callable(callback):
                continue
            try:
                callback(payload, fatal=observer is failed)
            except Exception:
                # The original failure is already retained on the result. A
                # secondary evidence-write failure must remain non-fatal too.
                continue


def normalize_observers(observers: Iterable[RunObserver] | None) -> tuple[RunObserver, ...]:
    """Validate a caller-provided observer collection before a run begins."""

    if observers is None:
        return ()
    if isinstance(observers, (str, bytes)):
        raise TypeError("observers must be an iterable of RunObserver instances.")
    try:
        values = tuple(observers)
    except TypeError as exc:
        raise TypeError("observers must be an iterable of RunObserver instances.") from exc
    required = ("on_start", "on_iteration", "on_chunk", "on_checkpoint", "on_terminal")
    for observer in values:
        missing = [name for name in required if not callable(getattr(observer, name, None))]
        if missing:
            raise TypeError(
                f"Observer {type(observer).__name__} is missing callback(s): {', '.join(missing)}."
            )
    return values


def emit_initial_failure(
    dispatcher: ObserverDispatcher,
    start_event: RunStartEvent,
    terminal_event: TerminalEvent,
) -> None:
    """Emit a consistent all-starts-before-any-terminal failure lifecycle."""

    dispatcher.emit("on_start", start_event)
    dispatcher.emit("on_terminal", terminal_event)


def classify_terminal_status(reason: str, *, degenerate: bool = False) -> str:
    """Map a stop reason to an evidence status without claiming optimality."""

    if str(reason) == "interrupted":
        return "interrupted"
    if degenerate or str(reason) in {"no_usable_candidates"}:
        return "degenerate"
    if str(reason) in {
        "gradient_failed",
        "smooth_gradient_failed",
        "initial_evaluation_failed",
        "nonfinite",
        "nonfinite_trial",
        "proposal_failed",
        "restoration_failed",
    }:
        return "failed"
    return "completed"
