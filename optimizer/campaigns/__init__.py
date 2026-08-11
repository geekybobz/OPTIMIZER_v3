"""Durable, system-semantic multi-stage optimization campaigns.

Campaigns coordinate existing optimizers.  The system remains authoritative for
control layout, parameters, metric meaning, and objective direction through its
required ``system_spec()`` API.  A campaign never guesses those facts from a
class name, a conventional ``J`` field, or caller prose.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.guesses import random_smooth_guess
from optimizer.optimizers import (
    adagrad,
    adam,
    cma_es,
    lbfgs,
    line_search,
    momentum,
    newton_cg,
    nonlinear_cg,
    projected_gradient,
    proximal_gradient,
    rmsprop,
    trust_region,
)
from optimizer.result import OptimizerResult
from optimizer.system_olgs import (
    get_system_spec,
    optional_control_bounds,
    optional_control_transfer,
    probe_system,
    require_system,
    validate_controls_for_system,
    with_secondary,
)
from optimizer.tracker import (
    CheckpointPolicy,
    JournalObserver,
    MetricDescriptor,
    RunJournal,
    RunReceipt,
    SourceReference,
    load_checkpoint,
    read_terminal,
)
from optimizer.utils.parallel import ParallelConfig, parallel_map


OptimizerCallable = Callable[..., OptimizerResult]
SystemFactory = Callable[[], Any]
SystemResolver = Callable[["SystemReference"], Any]

_CAMPAIGN_SCHEMA = "optimizer.campaigns/v2"
_OPTIMIZERS: dict[str, OptimizerCallable] = {
    "adam": adam,
    "adagrad": adagrad,
    "cma_es": cma_es,
    "lbfgs": lbfgs,
    "line_search": line_search,
    "momentum": momentum,
    "newton_cg": newton_cg,
    "nonlinear_cg": nonlinear_cg,
    "ncg": nonlinear_cg,
    "projected_gradient": projected_gradient,
    "proximal_gradient": proximal_gradient,
    "rmsprop": rmsprop,
    "trust_region": trust_region,
}


def _json_safe(value: Any) -> Any:
    """Return strict portable values; campaign provenance must not use repr()."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (builtins.list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Campaign evidence is not JSON-safe: {type(value).__name__}.")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a small campaign-owned record."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON mapping at {path}.")
    return dict(value)


def _compact_error(error: BaseException) -> dict[str, str]:
    message = str(error)
    if len(message) > 2048:
        message = f"{message[:2000]}...[truncated {len(message) - 2000} chars]"
    return {"type": type(error).__name__, "message": message}


@dataclass(frozen=True)
class SelectionSpec:
    """System-declared scalar used for campaign ranking and validation."""

    metric_id: str
    label: str
    direction: str

    @classmethod
    def from_system_spec(cls, spec: Mapping[str, Any]) -> "SelectionSpec":
        objective = dict(spec["objective"])
        metric_id = str(objective["metric_id"])
        direction = str(objective["direction"])
        if direction not in {"minimize", "maximize"}:
            raise ValueError("system_spec objective direction must be minimize or maximize.")
        metrics = dict(spec["metrics"])
        metric = dict(metrics[metric_id])
        return cls(metric_id=metric_id, label=str(metric.get("label", metric_id)), direction=direction)

    def value(self, metrics: Mapping[str, Any]) -> float:
        if self.metric_id not in metrics:
            raise KeyError(f"metrics do not include declared campaign objective {self.metric_id!r}.")
        value = float(metrics[self.metric_id])
        if not np.isfinite(value):
            raise ValueError(f"campaign objective {self.metric_id!r} must be finite.")
        return value

    def key(self, value: float, candidate_id: str) -> tuple[float, str]:
        return (float(value) if self.direction == "minimize" else -float(value), str(candidate_id))

    def to_dict(self) -> dict[str, str]:
        return {"metric_id": self.metric_id, "label": self.label, "direction": self.direction}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionSpec":
        return cls(
            metric_id=str(value["metric_id"]),
            label=str(value["label"]),
            direction=str(value["direction"]),
        )


@dataclass(frozen=True)
class SystemReference:
    """Immutable, system-owned semantic identity of one campaign stage."""

    system_id: str
    fingerprint: str
    system_spec: dict[str, Any]

    @classmethod
    def from_system(cls, system: Any) -> "SystemReference":
        spec = get_system_spec(system)
        fingerprint = sha256(_canonical_bytes(spec)).hexdigest()
        return cls(
            system_id=str(spec["identity"]["id"]),
            fingerprint=fingerprint,
            system_spec=_json_safe(spec),
        )

    def selection(self) -> SelectionSpec:
        return SelectionSpec.from_system_spec(self.system_spec)

    def assert_matches(self, system: Any) -> None:
        actual = SystemReference.from_system(system)
        if actual.fingerprint != self.fingerprint:
            raise ValueError(
                "Resolved system does not match the candidate's declared system fingerprint: "
                f"expected {self.fingerprint}, got {actual.fingerprint}."
            )

    def source_reference(self) -> SourceReference:
        identity = dict(self.system_spec["identity"])
        source_path = identity.get("source_path")
        return SourceReference.create(
            self.system_id,
            path=None if source_path is None else str(source_path),
            content_hash=self.fingerprint,
            version=str(self.system_spec["schema_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "fingerprint": self.fingerprint,
            "system_spec": _json_safe(self.system_spec),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SystemReference":
        reference = cls(
            system_id=str(value["system_id"]),
            fingerprint=str(value["fingerprint"]),
            system_spec=_json_safe(dict(value["system_spec"])),
        )
        expected = sha256(_canonical_bytes(reference.system_spec)).hexdigest()
        if reference.fingerprint != expected:
            raise ValueError("Campaign system reference fingerprint does not match its system_spec.")
        if reference.system_id != str(reference.system_spec["identity"]["id"]):
            raise ValueError("Campaign system reference system_id does not match its system_spec.")
        return reference


@dataclass(frozen=True)
class MethodSpec:
    """One named optimizer and fully declared bounded options."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).lower()
        if name not in _OPTIMIZERS:
            raise KeyError(f"Unknown campaign optimizer {self.name!r}; valid names are {sorted(_OPTIMIZERS)}.")
        options = dict(self.options)
        if "maxiter" not in options or int(options["maxiter"]) < 1:
            raise ValueError("MethodSpec.options must include an integer maxiter >= 1.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", options)

    @property
    def optimizer(self) -> OptimizerCallable:
        return _OPTIMIZERS[self.name]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "options": _json_safe(dict(self.options))}


@dataclass(frozen=True)
class CandidateRef:
    """One selected stage result tied to an exact system and checkpoint."""

    candidate_id: str
    mode: str
    stage_dir: Path
    result: OptimizerResult
    system: SystemReference
    selection: SelectionSpec
    checkpoint: dict[str, Any]
    source_id: str | None = None

    @property
    def objective_value(self) -> float:
        return self.selection.value(self.result.metrics)

    @property
    def J(self) -> float:
        """Compatibility accessor; campaign selection never assumes it is primary."""

        return self.result.J

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "stage_dir": str(self.stage_dir),
            "source_id": self.source_id,
            "system": self.system.to_dict(),
            "selection": self.selection.to_dict(),
            "checkpoint": _json_safe(self.checkpoint),
            "optimizer": self.result.optimizer,
            "objective_value": self.objective_value,
            "J": self.result.J,
            "iterations": self.result.iterations,
            "stop_reason": self.result.stop_reason,
            "metrics": _json_safe(self.result.metrics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, stage_dir: Path) -> "CandidateRef":
        system = SystemReference.from_dict(dict(value["system"]))
        selection = SelectionSpec.from_dict(dict(value["selection"]))
        checkpoint = _json_safe(dict(value["checkpoint"]))
        data = load_checkpoint(stage_dir, checkpoint, verify=True)
        metrics = _json_safe(dict(value["metrics"]))
        selection.value(metrics)
        result = OptimizerResult(
            controls=data.controls,
            metrics=metrics,
            stop_reason=str(value["stop_reason"]),
            iterations=int(value["iterations"]),
            optimizer=str(value["optimizer"]),
            state=None,
            system_params=dict(data.metadata.get("system_params", {})),
            checkpoint_ids={"selected": data.reference.checkpoint_id},
        )
        return cls(
            candidate_id=str(value["candidate_id"]),
            mode=str(value["mode"]),
            stage_dir=stage_dir,
            result=result,
            system=system,
            selection=selection,
            checkpoint=checkpoint,
            source_id=None if value.get("source_id") is None else str(value["source_id"]),
        )


@dataclass(frozen=True)
class ModeResult:
    """Candidates produced by a mode, sorted by the system-owned objective."""

    mode: str
    candidates: tuple[CandidateRef, ...]
    selected: CandidateRef

    @property
    def best(self) -> CandidateRef:
        return self.selected

    def top(self, count: int) -> tuple[CandidateRef, ...]:
        if int(count) < 1:
            raise ValueError("count must be >= 1.")
        return self.candidates[: int(count)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected": self.selected.candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ValidationResult:
    """Explicit validation against the candidate's producing system."""

    source_id: str
    system_fingerprint: str
    metrics: Mapping[str, Any]
    gradient_norm: float | None
    counters: Mapping[str, int]
    artifact: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "system_fingerprint": self.system_fingerprint,
            "metrics": _json_safe(self.metrics),
            "gradient_norm": self.gradient_norm,
            "counters": dict(self.counters),
            "artifact": str(self.artifact),
        }


@dataclass(frozen=True)
class _StageRequest:
    stage_dir: Path
    stage_id: str
    mode: str
    controls: Controls
    method: MethodSpec
    system_factory: SystemFactory
    system: SystemReference
    selection: SelectionSpec
    source_id: str | None = None
    warmstart: Any | None = None


def _metric_descriptors(system: SystemReference) -> dict[str, MetricDescriptor]:
    descriptors: dict[str, MetricDescriptor] = {}
    selection = system.selection()
    for metric_id, declaration in dict(system.system_spec["metrics"]).items():
        payload = dict(declaration)
        descriptors[str(metric_id)] = MetricDescriptor.create(
            str(metric_id),
            label=str(payload.get("label", metric_id)),
            role=str(payload.get("role", "diagnostic")),
            direction=selection.direction if str(metric_id) == selection.metric_id else None,
            unit=None if payload.get("unit") is None else str(payload["unit"]),
            source="system_spec",
        )
    return descriptors


def _execute_stage(request: _StageRequest) -> CandidateRef:
    """Run one isolated system/method pair and close its stage journal."""

    system = request.system_factory()
    require_system(system)
    request.system.assert_matches(system)
    receipt = RunReceipt.create(
        run_id=request.stage_id,
        objective_label=request.selection.label,
        objective_metric=request.selection.metric_id,
        direction=request.selection.direction,
        references={"problem": request.system.source_reference(), "system": request.system.source_reference()},
        metric_descriptors=_metric_descriptors(request.system),
        method_config=request.method.to_dict(),
        api_context={
            "campaign_mode": request.mode,
            "source_id": request.source_id,
            "system_fingerprint": request.system.fingerprint,
        },
        provenance={"coordinator": "optimizer.campaigns", "system": request.system.to_dict()},
    )
    journal = RunJournal.create(request.stage_dir, receipt)
    options = dict(request.method.options)
    options.update(
        {
            "observers": [JournalObserver(journal, checkpoint_policy=CheckpointPolicy())],
            "stage": request.mode,
            "create_trace": False,
            "accept_metric": request.selection.metric_id,
            "accept_mode": "min" if request.selection.direction == "minimize" else "max",
        }
    )
    if request.warmstart is not None:
        options["warmstart"] = request.warmstart
    result = request.method.optimizer(system, request.controls, **options)
    terminal = read_terminal(request.stage_dir)
    checkpoint = terminal.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Stage completed without a durable selected checkpoint.")
    candidate = CandidateRef(
        candidate_id=request.stage_id,
        mode=request.mode,
        stage_dir=request.stage_dir,
        result=result,
        system=request.system,
        selection=request.selection,
        checkpoint=_json_safe(dict(checkpoint)),
        source_id=request.source_id,
    )
    _write_json(request.stage_dir / "stage_result.json", candidate.to_dict())
    return candidate


def _same_layout(controls: Controls, system: Any) -> bool:
    spec = system.control_spec()
    return (
        controls.spec.keys == spec.keys
        and controls.spec.control_dim == spec.control_dim
        and controls.spec.dtype == spec.dtype
        and controls.spec.dt == spec.dt
    )


def _fourier_compress(controls: Controls, terms: int) -> Controls:
    if int(terms) < 1:
        raise ValueError("terms must be >= 1.")
    matrix = controls.as_matrix(copy=True)
    spectrum = np.fft.rfft(matrix, axis=1)
    spectrum[:, int(terms) + 1 :] = 0.0
    compressed = np.fft.irfft(spectrum, n=matrix.shape[1], axis=1)
    return Controls.from_matrix(
        controls.spec,
        compressed,
        name="fourier_compressed",
        meta={**controls.meta, "campaign_transfer": "generic_fourier_low_pass", "fourier_terms": int(terms)},
    )


def _preflight_controls(system: Any, controls: Controls) -> Controls:
    """Check layout and declared bounds without evaluating the objective."""

    validate_controls_for_system(system, controls)
    if callable(getattr(system, "control_bounds", None)):
        bounds = optional_control_bounds(system)
        matrix = controls.as_matrix(copy=False)
        if np.any(matrix < bounds.lower.as_matrix(copy=False)) or np.any(matrix > bounds.upper.as_matrix(copy=False)):
            raise ValueError("Campaign candidate violates system-declared control bounds.")
    return controls


def _transfer_controls(system: Any, controls: Controls, *, operation: str, options: Mapping[str, Any]) -> Controls:
    """Use an explicit system-owned feasibility-preserving transfer hook."""

    try:
        transferred, _metadata = optional_control_transfer(
            system,
            controls,
            operation=operation,
            options=options,
        )
    except AttributeError as exc:
        raise ValueError(
            "Challenge transformations require system.transfer_controls(...) when the system declares "
            "constraints or control_bounds()."
        ) from exc
    return _preflight_controls(system, transferred)


class Campaign:
    """Durable coordinator for explicitly selected, system-semantic stages."""

    def __init__(
        self,
        system: Any,
        root: str | Path,
        *,
        system_factory: SystemFactory | None = None,
        campaign_id: str | None = None,
    ) -> None:
        require_system(system)
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f"Campaign root already exists: {self.root}")
        self.root.mkdir(parents=True)
        (self.root / "stages").mkdir()
        (self.root / "diagnostics").mkdir()
        self.campaign_id = campaign_id or uuid4().hex
        self.system = system
        self.system_factory = system_factory
        base = SystemReference.from_system(system)
        self._base_system = base
        self._systems: dict[str, Any] = {base.fingerprint: system}
        self._resolver: SystemResolver | None = None
        self._stage_number = 0
        self._events = 0
        self._terminal_status: str | None = None
        _write_json(
            self.root / "campaign_receipt.json",
            {
                "schema_version": _CAMPAIGN_SCHEMA,
                "campaign_id": self.campaign_id,
                "base_system": base.to_dict(),
                "selection": base.selection().to_dict(),
                "parallel_requires_system_factory": True,
            },
        )
        self._write_live("running")
        self._event("campaign_created", {"system": base.to_dict(), "selection": base.selection().to_dict()})

    @classmethod
    def open(cls, root: str | Path, *, system_resolver: SystemResolver) -> "Campaign":
        """Open persisted campaign evidence without guessing how systems rebuild."""

        if not callable(system_resolver):
            raise TypeError("Campaign.open requires callable system_resolver(SystemReference).")
        path = Path(root)
        receipt = _read_json(path / "campaign_receipt.json")
        if receipt.get("schema_version") != _CAMPAIGN_SCHEMA:
            raise ValueError("Unsupported campaign receipt schema.")
        instance = cls.__new__(cls)
        instance.root = path
        instance.campaign_id = str(receipt["campaign_id"])
        instance.system = None
        instance.system_factory = None
        instance._base_system = SystemReference.from_dict(dict(receipt["base_system"]))
        instance._systems = {}
        instance._resolver = system_resolver
        instance._stage_number = len(builtins.list((path / "stages").glob("*/stage_result.json")))
        timeline_path = path / "campaign_timeline.jsonl"
        instance._events = len(timeline_path.read_text(encoding="utf-8").splitlines()) if timeline_path.exists() else 0
        terminal_path = path / "campaign_terminal.json"
        instance._terminal_status = _read_json(terminal_path)["status"] if terminal_path.exists() else None
        return instance

    def _write_live(self, status: str) -> None:
        _write_json(
            self.root / "campaign_live.json",
            {
                "schema_version": _CAMPAIGN_SCHEMA,
                "campaign_id": self.campaign_id,
                "status": str(status),
                "last_sequence": self._events,
            },
        )

    def _event(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._events += 1
        event = {"sequence": self._events, "kind": str(kind), "payload": _json_safe(payload)}
        with (self.root / "campaign_timeline.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if self._terminal_status is None:
            self._write_live("running")

    def _close(self, status: str, *, reason: str, details: Mapping[str, Any] | None = None) -> Path:
        if self._terminal_status is not None:
            return self.root / "campaign_terminal.json"
        self._terminal_status = str(status)
        payload = {
            "schema_version": _CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id,
            "status": str(status),
            "reason": str(reason),
            "details": _json_safe(dict(details or {})),
            "last_sequence": self._events,
        }
        _write_json(self.root / "campaign_terminal.json", payload)
        self._write_live(str(status))
        return self.root / "campaign_terminal.json"

    def close(self) -> Path:
        """Explicitly mark the current durable campaign evidence complete."""

        self._event("campaign_completed", {"stage_count": self._stage_number})
        return self._close("completed", reason="explicit_close")

    def _assert_open(self) -> None:
        if self._terminal_status is not None:
            raise RuntimeError(f"Campaign is terminal ({self._terminal_status}); open a new campaign for more stages.")

    def _allocate_stage(self, mode: str, suffix: str) -> tuple[str, Path]:
        self._stage_number += 1
        safe_suffix = "".join(character if character.isalnum() or character in "-_" else "_" for character in suffix)
        stage_id = f"{self._stage_number:03d}_{mode}_{safe_suffix}"
        return stage_id, self.root / "stages" / stage_id

    def _register_system(self, system: Any) -> SystemReference:
        reference = SystemReference.from_system(system)
        self._systems[reference.fingerprint] = system
        return reference

    def _resolve_system(self, reference: SystemReference) -> Any:
        current = self._systems.get(reference.fingerprint)
        if current is not None:
            reference.assert_matches(current)
            return current
        if self._resolver is None:
            raise RuntimeError(
                "This campaign does not have the candidate system in memory. Reopen with "
                "Campaign.open(..., system_resolver=...) to validate or refine persisted stages."
            )
        resolved = self._resolver(reference)
        require_system(resolved)
        reference.assert_matches(resolved)
        self._systems[reference.fingerprint] = resolved
        return resolved

    def _factory_for(self, reference: SystemReference, parallel: ParallelConfig | None) -> SystemFactory:
        config = parallel or ParallelConfig()
        if config.backend != "serial" and config.resolved_workers() > 1:
            if self.system_factory is None:
                raise ValueError("Parallel campaign modes require system_factory for isolated worker systems.")
            return self.system_factory
        system = self._resolve_system(reference)
        return lambda: system

    def _run(
        self,
        *,
        mode: str,
        entries: Sequence[tuple[Controls, MethodSpec, str | None, Any | None]],
        system: SystemReference,
        parallel: ParallelConfig | None = None,
    ) -> ModeResult:
        self._assert_open()
        selection = system.selection()
        requests: list[_StageRequest] = []
        factory = self._factory_for(system, parallel)
        for index, (controls, method, source_id, warmstart) in enumerate(entries):
            stage_id, stage_dir = self._allocate_stage(mode, f"{index:03d}_{method.name}")
            requests.append(_StageRequest(stage_dir, stage_id, mode, controls, method, factory, system, selection, source_id, warmstart))
        self._event(
            "mode_declared",
            {
                "mode": mode,
                "candidate_ids": [request.stage_id for request in requests],
                "system_fingerprint": system.fingerprint,
                "selection": selection.to_dict(),
                "parallel": (parallel or ParallelConfig()).__dict__,
            },
        )
        started = perf_counter()
        try:
            results = parallel_map(_execute_stage, requests, config=parallel)
        except KeyboardInterrupt:
            self._event("mode_interrupted", {"mode": mode, "candidate_ids": [request.stage_id for request in requests]})
            self._close("interrupted", reason="keyboard_interrupt", details={"mode": mode})
            raise
        except Exception as exc:
            failure = _compact_error(exc)
            self._event("mode_failed", {"mode": mode, "candidate_ids": [request.stage_id for request in requests], "error": failure})
            self._close("failed", reason="stage_execution_failed", details={"mode": mode, "error": failure})
            raise
        elapsed_s = perf_counter() - started
        ordered = tuple(sorted(results, key=lambda item: selection.key(item.objective_value, item.candidate_id)))
        if not ordered:
            error = {"type": "RuntimeError", "message": "Campaign mode returned no candidates."}
            self._event("mode_failed", {"mode": mode, "error": error})
            self._close("failed", reason="no_candidates", details={"mode": mode})
            raise RuntimeError(error["message"])
        selected = ordered[0]
        self._event(
            "mode_completed",
            {
                "mode": mode,
                "elapsed_s": elapsed_s,
                "candidate_ids": [candidate.candidate_id for candidate in ordered],
                "selected": selected.candidate_id,
                "selection": selection.to_dict(),
            },
        )
        _write_json(self.root / "selected.json", selected.to_dict())
        return ModeResult(mode=mode, candidates=ordered, selected=selected)

    def scout(self, guesses: Sequence[Controls], *, method: MethodSpec, parallel: ParallelConfig | None = None) -> ModeResult:
        values = tuple(guesses)
        if not values:
            raise ValueError("scout requires at least one explicit guess.")
        for controls in values:
            _preflight_controls(self._resolve_system(self._base_system), controls)
        return self._run(
            mode="scout",
            entries=[(guess, method, None, None) for guess in values],
            system=self._base_system,
            parallel=parallel,
        )

    def portfolio(self, sources: Sequence[CandidateRef], *, methods: Sequence[MethodSpec], parallel: ParallelConfig | None = None) -> ModeResult:
        if not sources or not methods:
            raise ValueError("portfolio requires at least one source and one method.")
        reference = sources[0].system
        if any(source.system.fingerprint != reference.fingerprint for source in sources):
            raise ValueError("portfolio requires candidates produced by the same declared system.")
        entries = [
            (source.result.controls.copy(name="portfolio_source"), method, source.candidate_id, None)
            for source in sources
            for method in methods
        ]
        return self._run(mode="portfolio", entries=entries, system=reference, parallel=parallel)

    def refine(self, source: CandidateRef, *, method: MethodSpec) -> ModeResult:
        warmstart = source.result.warmstart(target_optimizer=method.name)
        return self._run(
            mode="refine",
            entries=[(source.result.controls.copy(name="refine_source"), method, source.candidate_id, warmstart)],
            system=source.system,
        )

    def challenge(
        self,
        source: CandidateRef,
        *,
        method: MethodSpec,
        variants: int = 10,
        perturbation: float = 0.02,
        fourier_terms: int | None = None,
        seed: int = 0,
        parallel: ParallelConfig | None = None,
    ) -> ModeResult:
        if int(variants) < 2:
            raise ValueError("challenge variants must be >= 2, including the unmodified reference.")
        if not np.isfinite(float(perturbation)) or float(perturbation) < 0.0:
            raise ValueError("perturbation must be finite and nonnegative.")
        system = self._resolve_system(source.system)
        spec = get_system_spec(system)
        constrained = bool(spec.get("constraints")) or callable(getattr(system, "control_bounds", None))
        has_transfer = probe_system(system).has_control_transfer
        if constrained and (float(perturbation) > 0.0 or fourier_terms is not None) and not has_transfer:
            raise ValueError(
                "Challenge cannot alter a constrained system without its explicit transfer_controls system API."
            )
        base = _preflight_controls(system, source.result.controls.copy(name="challenge_reference"))
        candidates: list[Controls] = [base]
        if fourier_terms is not None:
            candidate = (
                _transfer_controls(system, base, operation="fourier_low_pass", options={"terms": int(fourier_terms)})
                if has_transfer
                else _preflight_controls(system, _fourier_compress(base, int(fourier_terms)))
            )
            candidates.append(candidate)
        scale = max(float(np.max(np.abs(base.as_matrix(copy=False)))), 1.0)
        rng = np.random.default_rng(seed)
        while len(candidates) < int(variants):
            noise = rng.normal(scale=float(perturbation) * scale, size=base.spec.shape)
            if has_transfer:
                candidate = _transfer_controls(
                    system,
                    base,
                    operation="perturbation",
                    options={"noise": noise, "seed": int(seed), "scale": float(perturbation)},
                )
            else:
                candidate = _preflight_controls(
                    system,
                    Controls.from_matrix(
                        base.spec,
                        base.as_matrix(copy=True) + noise,
                        name=f"challenge_perturb_{len(candidates):02d}",
                        meta={**base.meta, "campaign_transfer": "generic_unconstrained_perturbation"},
                    ),
                )
            candidates.append(candidate)
        self._event(
            "challenge_declared",
            {
                "source_id": source.candidate_id,
                "system_fingerprint": source.system.fingerprint,
                "constrained": constrained,
                "system_transfer_used": has_transfer,
                "fourier_terms": fourier_terms,
                "perturbation": float(perturbation),
                "seed": int(seed),
            },
        )
        return self._run(
            mode="challenge",
            entries=[(candidate, method, source.candidate_id, None) for candidate in candidates],
            system=source.system,
            parallel=parallel,
        )

    def validate(self, source: CandidateRef, *, gradient: bool = True) -> ValidationResult:
        """Evaluate a candidate only under its exact producing system."""

        system = self._resolve_system(source.system)
        evaluator = SystemEvaluator(system)
        evaluation = evaluator.evaluate(source.result.controls)
        gradient_norm = evaluator.gradient(source.result.controls).norm() if gradient else None
        artifact = self.root / "diagnostics" / f"validate_{source.candidate_id}.json"
        result = ValidationResult(
            source_id=source.candidate_id,
            system_fingerprint=source.system.fingerprint,
            metrics=dict(evaluation.metrics),
            gradient_norm=gradient_norm,
            counters=evaluator.counters(),
            artifact=artifact,
        )
        _write_json(artifact, result.to_dict())
        self._event("validation_completed", result.to_dict())
        return result

    def continue_parameter(self, source: CandidateRef, *, updates: Mapping[str, Any], method: MethodSpec) -> ModeResult:
        """Continue from a candidate with an explicit system-owned parameter update."""

        current = self._resolve_system(source.system)
        next_system = with_secondary(current, **dict(updates))
        require_system(next_system)
        if not _same_layout(source.result.controls, next_system):
            raise ValueError("Continuation changed control layout; provide a declared transfer before continuing.")
        next_reference = self._register_system(next_system)
        self._event(
            "continuation_declared",
            {
                "source_id": source.candidate_id,
                "source_system_fingerprint": source.system.fingerprint,
                "next_system_fingerprint": next_reference.fingerprint,
                "updates": _json_safe(dict(updates)),
            },
        )
        return self._run(
            mode="continue",
            entries=[(source.result.controls.copy(name="continuation_source"), method, source.candidate_id, None)],
            system=next_reference,
        )

    def load_candidate(self, candidate_id: str) -> CandidateRef:
        """Reload one persisted selected artifact with SHA-256 verification."""

        token = str(candidate_id)
        for stage_file in sorted((self.root / "stages").glob("*/stage_result.json")):
            payload = _read_json(stage_file)
            if str(payload.get("candidate_id")) == token:
                return CandidateRef.from_dict(payload, stage_dir=stage_file.parent)
        raise KeyError(f"Unknown campaign candidate {token!r}.")

    def report(self) -> dict[str, Any]:
        """Read campaign records only; this method never resolves or calls a system."""

        stage_files = sorted((self.root / "stages").glob("*/stage_result.json"))
        stages = [_read_json(path) for path in stage_files]
        timeline_path = self.root / "campaign_timeline.jsonl"
        timeline = [
            json.loads(line)
            for line in timeline_path.read_text(encoding="utf-8").splitlines()
            if line
        ] if timeline_path.exists() else []
        receipt = _read_json(self.root / "campaign_receipt.json")
        selection = SelectionSpec.from_dict(dict(receipt["selection"]))
        objectives = [selection.value(dict(stage["metrics"])) for stage in stages]
        best_value = (
            min(objectives) if selection.direction == "minimize" else max(objectives)
        ) if objectives else None
        return {
            "schema_version": "optimizer.campaign-report/v2",
            "campaign_id": self.campaign_id,
            "status": self._terminal_status or "running",
            "selection": selection.to_dict(),
            "stage_count": len(stages),
            "best_objective": best_value,
            "best_J": best_value if selection.metric_id == "J" else None,
            "stages": stages,
            "timeline": timeline,
            "selected": _read_json(self.root / "selected.json") if (self.root / "selected.json").exists() else None,
        }


def create(
    system: Any,
    root: str | Path,
    *,
    system_factory: SystemFactory | None = None,
    campaign_id: str | None = None,
) -> Campaign:
    return Campaign(system, root, system_factory=system_factory, campaign_id=campaign_id)


def open(root: str | Path, *, system_resolver: SystemResolver) -> Campaign:
    """Open campaign evidence with an explicit caller-owned system resolver."""

    return Campaign.open(root, system_resolver=system_resolver)


def random_smooth_guesses(
    system: Any,
    count: int,
    *,
    amplitude: Any,
    seed: int = 0,
    **options: Any,
) -> tuple[Controls, ...]:
    if int(count) < 1:
        raise ValueError("count must be >= 1.")
    return tuple(
        random_smooth_guess(system, amplitude=amplitude, seed=int(seed) + index, **options)
        for index in range(int(count))
    )


def info() -> dict[str, Any]:
    return {
        "schema_version": "optimizer.campaigns-info/v2",
        "modes": ["scout", "portfolio", "refine", "challenge", "validate", "continue"],
        "parallel_modes": ["scout", "portfolio", "challenge"],
        "invariants": {
            "system_owned_objective_selection": True,
            "candidate_system_fingerprint": True,
            "one_tracker_journal_per_stage": True,
            "report_makes_no_numerical_calls": True,
            "parallel_requires_isolated_system_factory": True,
            "reopen_requires_explicit_system_resolver": True,
            "challenge_requires_system_transfer_when_constrained": True,
        },
    }


def list(*, h: bool = False) -> dict[str, Any] | str:
    payload = info()
    if not h:
        return payload
    return "\n".join(
        ["optimizer.campaigns", "Modes:"]
        + [f"- {mode}" for mode in payload["modes"]]
        + ["Parallel modes: " + ", ".join(payload["parallel_modes"])]
    )


__all__ = [
    "Campaign",
    "CandidateRef",
    "MethodSpec",
    "ModeResult",
    "SelectionSpec",
    "SystemReference",
    "ValidationResult",
    "create",
    "info",
    "list",
    "open",
    "random_smooth_guesses",
]
