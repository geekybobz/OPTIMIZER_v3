"""Canonical, replayable records for one optimization run.

The tracker model preserves values supplied by an optimizer/controller.  It is
deliberately unaware of systems, objective functions, and decision policies:
recording a run must never add an evaluation, gradient, or diagnostic call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "optimizer.tracker/v3"
_DIRECTIONS = {"minimize", "maximize"}
_MAX_EVENT_DEPTH = 8
_MAX_EVENT_MAPPING = 128
_MAX_EVENT_SEQUENCE = 64
_MAX_EVENT_STRING = 8192
_MAX_EVENT_BYTES = 16384
_SELECTION_RULES = {"best_finite_value", "final_value"}


def utc_now() -> str:
    """Return an unambiguous UTC timestamp suitable for durable records."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert common scientific-Python values into portable strict JSON data."""

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Tracker records cannot contain non-finite float values.")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-safe.")


def compact_json(value: Any) -> Any:
    """Return strict JSON data only when it is safe for an inline event payload.

    Checkpoint arrays and other heavy state belong in artifacts.  Keeping this
    validation in the canonical model lets the journal enforce the same bound
    for bridge-generated and directly submitted records.
    """

    safe = json_safe(value)

    def validate(item: Any, *, path: str, depth: int) -> None:
        if depth > _MAX_EVENT_DEPTH:
            raise ValueError(f"Tracker event payload exceeds maximum depth at {path}.")
        if isinstance(item, str):
            if len(item) > _MAX_EVENT_STRING:
                raise ValueError(f"Tracker event string is too large at {path}.")
            return
        if item is None or isinstance(item, (bool, int, float)):
            return
        if isinstance(item, dict):
            if len(item) > _MAX_EVENT_MAPPING:
                raise ValueError(f"Tracker event mapping is too large at {path}.")
            for key, child in item.items():
                validate(child, path=f"{path}.{key}", depth=depth + 1)
            return
        if isinstance(item, list):
            if len(item) > _MAX_EVENT_SEQUENCE:
                raise ValueError(
                    f"Tracker event sequence is too large at {path}; save it as an artifact."
                )
            for index, child in enumerate(item):
                validate(child, path=f"{path}[{index}]", depth=depth + 1)
            return
        raise TypeError(f"Unsupported tracker event value at {path}: {type(item).__name__}.")

    validate(safe, path="$", depth=0)
    encoded_size = len(json.dumps(safe, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if encoded_size > _MAX_EVENT_BYTES:
        raise ValueError(
            f"Tracker event payload is too large: {encoded_size} bytes exceeds "
            f"the {_MAX_EVENT_BYTES}-byte inline limit; save heavy state as an artifact."
        )
    return safe


def _nonempty(value: str, *, field: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def _mapping(value: Mapping[str, Any] | None, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return json_safe(value)


def _scalar_metrics(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep high-volume arrays in checkpoint artifacts, not per-iteration logs."""

    metrics = _mapping(value, field="metrics")
    for name, item in metrics.items():
        if item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError(
                f"Metric {name!r} must be a JSON scalar; save vectors or arrays as an artifact."
            )
    return metrics


def _counter_mapping(value: Mapping[str, int] | None) -> dict[str, int]:
    counters: dict[str, int] = {}
    for name, count in (value or {}).items():
        if isinstance(count, bool):
            raise ValueError(f"Counter {name!r} must be a non-negative integer.")
        integer = int(count)
        if integer < 0 or integer != count:
            raise ValueError(f"Counter {name!r} must be a non-negative integer.")
        counters[str(name)] = integer
    return counters


@dataclass(frozen=True)
class SourceReference:
    """A small pointer to an authority owned outside the tracker.

    The problem statement and system remain authoritative in their own files.
    This reference only says exactly which one a run used.
    """

    identifier: str
    path: str | None = None
    content_hash: str | None = None
    version: str | None = None

    @classmethod
    def create(
        cls,
        identifier: str,
        *,
        path: str | None = None,
        content_hash: str | None = None,
        version: str | None = None,
    ) -> "SourceReference":
        return cls(
            identifier=_nonempty(identifier, field="reference identifier"),
            path=None if path is None else str(path),
            content_hash=None if content_hash is None else str(content_hash),
            version=None if version is None else str(version),
        )

    @classmethod
    def from_value(cls, value: "SourceReference | Mapping[str, Any]") -> "SourceReference":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("A source reference must be a SourceReference or mapping.")
        return cls.create(
            str(value["identifier"]),
            path=value.get("path"),
            content_hash=value.get("content_hash"),
            version=value.get("version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "path": self.path,
            "content_hash": self.content_hash,
            "version": self.version,
        }


@dataclass(frozen=True)
class MetricDescriptor:
    """Declared meaning of one scalar metric saved by the tracker."""

    metric_id: str
    label: str
    role: str = "diagnostic"
    direction: str | None = None
    unit: str | None = None
    source: str | None = None

    @classmethod
    def create(
        cls,
        metric_id: str,
        *,
        label: str | None = None,
        role: str = "diagnostic",
        direction: str | None = None,
        unit: str | None = None,
        source: str | None = None,
    ) -> "MetricDescriptor":
        normalized_direction = None if direction is None else str(direction)
        if normalized_direction is not None and normalized_direction not in _DIRECTIONS:
            raise ValueError(f"metric direction must be one of {sorted(_DIRECTIONS)!r}.")
        return cls(
            metric_id=_nonempty(metric_id, field="metric_id"),
            label=_nonempty(label or metric_id, field="metric label"),
            role=_nonempty(role, field="metric role"),
            direction=normalized_direction,
            unit=None if unit is None else str(unit),
            source=None if source is None else str(source),
        )

    @classmethod
    def from_value(
        cls,
        metric_id: str,
        value: "MetricDescriptor | Mapping[str, Any]",
    ) -> "MetricDescriptor":
        if isinstance(value, cls):
            if value.metric_id != str(metric_id):
                raise ValueError("Metric descriptor key and metric_id must agree.")
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Metric descriptors must be MetricDescriptor instances or mappings.")
        return cls.create(
            metric_id,
            label=value.get("label"),
            role=str(value.get("role", "diagnostic")),
            direction=value.get("direction"),
            unit=value.get("unit"),
            source=value.get("source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "label": self.label,
            "role": self.role,
            "direction": self.direction,
            "unit": self.unit,
            "source": self.source,
        }


@dataclass(frozen=True)
class ScalarSnapshot:
    """A compact state view at one causal point in a run."""

    metrics: dict[str, Any] = field(default_factory=dict)
    metric_sources: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    omitted_metrics: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        metrics: Mapping[str, Any] | None = None,
        metric_sources: Mapping[str, str] | None = None,
        counters: Mapping[str, int] | None = None,
        omitted_metrics: Iterable[str] | None = None,
    ) -> "ScalarSnapshot":
        metric_values = _scalar_metrics(metrics)
        sources = {str(name): str(source) for name, source in (metric_sources or {}).items()}
        unknown_sources = set(sources).difference(metric_values)
        if unknown_sources:
            raise ValueError(
                "metric_sources may only name saved metrics; unknown "
                f"metric(s): {sorted(unknown_sources)!r}."
            )
        return cls(
            metrics=metric_values,
            metric_sources=sources,
            counters=_counter_mapping(counters),
            omitted_metrics=tuple(sorted({str(name) for name in (omitted_metrics or ())})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "metric_sources": dict(self.metric_sources),
            "counters": dict(self.counters),
            "omitted_metrics": list(self.omitted_metrics),
        }


@dataclass(frozen=True)
class RunReceipt:
    """Small immutable receipt for one run, without redefining its problem."""

    run_id: str
    objective_label: str
    direction: str
    references: dict[str, SourceReference]
    objective_metric: str = "J"
    selection_rule: str = "best_finite_value"
    metric_descriptors: dict[str, MetricDescriptor] = field(default_factory=dict)
    api_context: dict[str, Any] = field(default_factory=dict)
    method_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    created_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        objective_label: str,
        direction: str,
        references: Mapping[str, SourceReference | Mapping[str, Any]],
        objective_metric: str | None = None,
        selection_rule: str = "best_finite_value",
        metric_descriptors: Mapping[str, MetricDescriptor | Mapping[str, Any]] | None = None,
        api_context: Mapping[str, Any] | None = None,
        method_config: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> "RunReceipt":
        direction = str(direction)
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)!r}.")
        selection_rule = str(selection_rule)
        if selection_rule not in _SELECTION_RULES:
            raise ValueError(f"selection_rule must be one of {sorted(_SELECTION_RULES)!r}.")
        source_references = {
            _nonempty(role, field="reference role"): SourceReference.from_value(reference)
            for role, reference in references.items()
        }
        missing = {"problem", "system"}.difference(source_references)
        if missing:
            raise ValueError(f"references must contain {sorted(missing)!r}.")
        metric_id = _nonempty(objective_metric or objective_label, field="objective_metric")
        descriptors = {
            str(name): MetricDescriptor.from_value(str(name), descriptor)
            for name, descriptor in (metric_descriptors or {}).items()
        }
        objective_descriptor = descriptors.get(metric_id)
        if objective_descriptor is None:
            descriptors[metric_id] = MetricDescriptor.create(
                metric_id,
                label=objective_label,
                role="objective",
                direction=direction,
            )
        elif (
            objective_descriptor.direction is not None
            and objective_descriptor.direction != direction
        ):
            raise ValueError("Objective descriptor direction conflicts with run direction.")
        elif objective_descriptor.direction is None:
            descriptors[metric_id] = MetricDescriptor.create(
                metric_id,
                label=objective_descriptor.label,
                role=objective_descriptor.role,
                direction=direction,
                unit=objective_descriptor.unit,
                source=objective_descriptor.source,
            )
        return cls(
            run_id=_nonempty(run_id, field="run_id"),
            objective_label=_nonempty(objective_label, field="objective_label"),
            direction=direction,
            references=source_references,
            objective_metric=metric_id,
            selection_rule=selection_rule,
            metric_descriptors=descriptors,
            api_context=_mapping(api_context, field="api_context"),
            method_config=_mapping(method_config, field="method_config"),
            provenance=_mapping(provenance, field="provenance"),
            seed=None if seed is None else int(seed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "objective": {
                "metric_id": self.objective_metric,
                "label": self.objective_label,
                "direction": self.direction,
            },
            "selection": {
                "metric_id": self.objective_metric,
                "direction": self.direction,
                "rule": self.selection_rule,
            },
            "metrics": {
                name: descriptor.to_dict()
                for name, descriptor in self.metric_descriptors.items()
            },
            "references": {role: value.to_dict() for role, value in self.references.items()},
            "api_context": dict(self.api_context),
            "method_config": dict(self.method_config),
            "provenance": dict(self.provenance),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class IterationRecord:
    """The raw before/proposal/trial/after record for one outer iteration."""

    iteration: int
    method: str
    before: ScalarSnapshot
    after: ScalarSnapshot
    accepted: bool
    reason: str
    trial: ScalarSnapshot | None = None
    proposal: dict[str, Any] = field(default_factory=dict)
    phase: str | None = None
    best_after: ScalarSnapshot | None = None
    global_iteration: int | None = None
    chunk: int = 0
    system_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        iteration: int,
        method: str,
        before: ScalarSnapshot,
        after: ScalarSnapshot,
        accepted: bool,
        reason: str,
        trial: ScalarSnapshot | None = None,
        proposal: Mapping[str, Any] | None = None,
        phase: str | None = None,
        best_after: ScalarSnapshot | None = None,
        global_iteration: int | None = None,
        chunk: int = 0,
        system_params: Mapping[str, Any] | None = None,
    ) -> "IterationRecord":
        if int(iteration) < 0 or int(iteration) != iteration:
            raise ValueError("iteration must be a non-negative integer.")
        if not isinstance(before, ScalarSnapshot) or not isinstance(after, ScalarSnapshot):
            raise TypeError("before and after must be ScalarSnapshot instances.")
        if trial is not None and not isinstance(trial, ScalarSnapshot):
            raise TypeError("trial must be a ScalarSnapshot when supplied.")
        if best_after is not None and not isinstance(best_after, ScalarSnapshot):
            raise TypeError("best_after must be a ScalarSnapshot when supplied.")
        if global_iteration is not None and (
            int(global_iteration) < 0 or int(global_iteration) != global_iteration
        ):
            raise ValueError("global_iteration must be a non-negative integer when supplied.")
        if int(chunk) < 0 or int(chunk) != chunk:
            raise ValueError("chunk must be a non-negative integer.")
        return cls(
            iteration=int(iteration),
            method=_nonempty(method, field="method"),
            before=before,
            after=after,
            accepted=bool(accepted),
            reason=_nonempty(reason, field="reason"),
            trial=trial,
            proposal=_mapping(proposal, field="proposal"),
            phase=None if phase is None else str(phase),
            best_after=best_after,
            global_iteration=None if global_iteration is None else int(global_iteration),
            chunk=int(chunk),
            system_params=_mapping(system_params, field="system_params"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "global_iteration": self.global_iteration,
            "chunk": self.chunk,
            "method": self.method,
            "phase": self.phase,
            "system_params": dict(self.system_params),
            "before": self.before.to_dict(),
            "trial": None if self.trial is None else self.trial.to_dict(),
            "technical": dict(self.proposal),
            "after": self.after.to_dict(),
            "best_after": None if self.best_after is None else self.best_after.to_dict(),
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChunkRecord:
    """One completed optimizer chunk with its resolved method and system facts."""

    method: str
    chunk: int
    start_iteration: int
    end_iteration: int
    start_metrics: dict[str, Any]
    end_metrics: dict[str, Any]
    start_omitted_metrics: tuple[str, ...] = ()
    end_omitted_metrics: tuple[str, ...] = ()
    system_params: dict[str, Any] = field(default_factory=dict)
    phase: str | None = None
    accepted: bool | None = None
    reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        method: str,
        chunk: int,
        start_iteration: int,
        end_iteration: int,
        start_metrics: Mapping[str, Any],
        end_metrics: Mapping[str, Any],
        start_omitted_metrics: Iterable[str] | None = None,
        end_omitted_metrics: Iterable[str] | None = None,
        system_params: Mapping[str, Any] | None = None,
        phase: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
    ) -> "ChunkRecord":
        chunk_value = int(chunk)
        start = int(start_iteration)
        end = int(end_iteration)
        if chunk_value < 0 or chunk_value != chunk:
            raise ValueError("chunk must be a non-negative integer.")
        if start < 0 or end < start or start != start_iteration or end != end_iteration:
            raise ValueError("chunk iteration window must be non-negative and ordered.")
        return cls(
            method=_nonempty(method, field="method"),
            chunk=chunk_value,
            start_iteration=start,
            end_iteration=end,
            start_metrics=_scalar_metrics(start_metrics),
            end_metrics=_scalar_metrics(end_metrics),
            start_omitted_metrics=tuple(
                sorted({str(name) for name in (start_omitted_metrics or ())})
            ),
            end_omitted_metrics=tuple(
                sorted({str(name) for name in (end_omitted_metrics or ())})
            ),
            system_params=_mapping(system_params, field="system_params"),
            phase=None if phase is None else str(phase),
            accepted=None if accepted is None else bool(accepted),
            reason=None if reason is None else str(reason),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "chunk": self.chunk,
            "window": {
                "start_iteration": self.start_iteration,
                "end_iteration": self.end_iteration,
            },
            "start_metrics": dict(self.start_metrics),
            "end_metrics": dict(self.end_metrics),
            "omitted_metrics": {
                "start": list(self.start_omitted_metrics),
                "end": list(self.end_omitted_metrics),
            },
            "system_params": dict(self.system_params),
            "phase": self.phase,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnalysisRecord:
    """A derived, explicitly scheduled analysis of a raw iteration window."""

    analysis_code: str
    start_iteration: int
    end_iteration: int
    inputs: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        analysis_code: str,
        start_iteration: int,
        end_iteration: int,
        inputs: Mapping[str, Any] | None = None,
        findings: Mapping[str, Any] | None = None,
        counters: Mapping[str, int] | None = None,
    ) -> "AnalysisRecord":
        start = int(start_iteration)
        end = int(end_iteration)
        if start < 0 or end < start or start != start_iteration or end != end_iteration:
            raise ValueError("analysis iteration window must be non-negative and ordered.")
        return cls(
            analysis_code=_nonempty(analysis_code, field="analysis_code"),
            start_iteration=start,
            end_iteration=end,
            inputs=_mapping(inputs, field="analysis inputs"),
            findings=_mapping(findings, field="analysis findings"),
            counters=_counter_mapping(counters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_code": self.analysis_code,
            "window": {"start_iteration": self.start_iteration, "end_iteration": self.end_iteration},
            "inputs": dict(self.inputs),
            "findings": dict(self.findings),
            "counters": dict(self.counters),
        }


@dataclass(frozen=True)
class InterventionRecord:
    """An action between normal iterations, linked to its numerical evidence."""

    action_code: str
    reason: str
    before: ScalarSnapshot
    after: ScalarSnapshot
    outcome_code: str
    parameters: dict[str, Any] = field(default_factory=dict)
    analysis_event_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        action_code: str,
        reason: str,
        before: ScalarSnapshot,
        after: ScalarSnapshot,
        outcome_code: str,
        parameters: Mapping[str, Any] | None = None,
        analysis_event_id: str | None = None,
    ) -> "InterventionRecord":
        if not isinstance(before, ScalarSnapshot) or not isinstance(after, ScalarSnapshot):
            raise TypeError("before and after must be ScalarSnapshot instances.")
        return cls(
            action_code=_nonempty(action_code, field="action_code"),
            reason=_nonempty(reason, field="reason"),
            before=before,
            after=after,
            outcome_code=_nonempty(outcome_code, field="outcome_code"),
            parameters=_mapping(parameters, field="parameters"),
            analysis_event_id=None if analysis_event_id is None else str(analysis_event_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_code": self.action_code,
            "reason": self.reason,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "outcome_code": self.outcome_code,
            "parameters": dict(self.parameters),
            "analysis_event_id": self.analysis_event_id,
        }


@dataclass(frozen=True)
class TerminalRecord:
    """The selected terminal result and any explicitly requested certificate."""

    status: str
    reason: str
    final: ScalarSnapshot
    best: ScalarSnapshot | None = None
    selected_checkpoint: dict[str, Any] | None = None
    selection_fallback: str | None = None
    certification: dict[str, Any] = field(default_factory=dict)
    method: str | None = None
    phase: str | None = None
    iteration: int | None = None
    global_iteration: int | None = None
    chunk: int | None = None
    system_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        status: str,
        reason: str,
        final: ScalarSnapshot,
        best: ScalarSnapshot | None = None,
        selected_checkpoint: Mapping[str, Any] | None = None,
        selection_fallback: str | None = None,
        certification: Mapping[str, Any] | None = None,
        method: str | None = None,
        phase: str | None = None,
        iteration: int | None = None,
        global_iteration: int | None = None,
        chunk: int | None = None,
        system_params: Mapping[str, Any] | None = None,
    ) -> "TerminalRecord":
        if not isinstance(final, ScalarSnapshot):
            raise TypeError("final must be a ScalarSnapshot.")
        if best is not None and not isinstance(best, ScalarSnapshot):
            raise TypeError("best must be a ScalarSnapshot when supplied.")
        for field_name, value in (
            ("iteration", iteration),
            ("global_iteration", global_iteration),
            ("chunk", chunk),
        ):
            if value is not None and (int(value) < 0 or int(value) != value):
                raise ValueError(f"{field_name} must be a non-negative integer when supplied.")
        return cls(
            status=_nonempty(status, field="status"),
            reason=_nonempty(reason, field="reason"),
            final=final,
            best=best,
            selected_checkpoint=None
            if selected_checkpoint is None
            else _mapping(selected_checkpoint, field="selected_checkpoint"),
            selection_fallback=None if selection_fallback is None else str(selection_fallback),
            certification=_mapping(certification, field="certification"),
            method=None if method is None else str(method),
            phase=None if phase is None else str(phase),
            iteration=None if iteration is None else int(iteration),
            global_iteration=None if global_iteration is None else int(global_iteration),
            chunk=None if chunk is None else int(chunk),
            system_params=_mapping(system_params, field="system_params"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "final": self.final.to_dict(),
            "best": None if self.best is None else self.best.to_dict(),
            "selected_checkpoint": None
            if self.selected_checkpoint is None
            else dict(self.selected_checkpoint),
            "selection_fallback": self.selection_fallback,
            "certification": dict(self.certification),
            "method": self.method,
            "phase": self.phase,
            "iteration": self.iteration,
            "global_iteration": self.global_iteration,
            "chunk": self.chunk,
            "system_params": dict(self.system_params),
        }


@dataclass(frozen=True)
class LiveSnapshot:
    """Replaceable progress view for a local observer, not the scientific source."""

    status: str
    phase: str
    iteration: int | None
    method: str | None
    current: ScalarSnapshot | None
    best: ScalarSnapshot | None

    @classmethod
    def create(
        cls,
        *,
        status: str,
        phase: str,
        iteration: int | None = None,
        method: str | None = None,
        current: ScalarSnapshot | None = None,
        best: ScalarSnapshot | None = None,
    ) -> "LiveSnapshot":
        if iteration is not None and (int(iteration) < 0 or int(iteration) != iteration):
            raise ValueError("live iteration must be a non-negative integer when supplied.")
        if current is not None and not isinstance(current, ScalarSnapshot):
            raise TypeError("current must be a ScalarSnapshot when supplied.")
        if best is not None and not isinstance(best, ScalarSnapshot):
            raise TypeError("best must be a ScalarSnapshot when supplied.")
        return cls(
            status=_nonempty(status, field="live status"),
            phase=_nonempty(phase, field="live phase"),
            iteration=None if iteration is None else int(iteration),
            method=None if method is None else str(method),
            current=current,
            best=best,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "iteration": self.iteration,
            "method": self.method,
            "current": None if self.current is None else self.current.to_dict(),
            "best": None if self.best is None else self.best.to_dict(),
        }
