"""Explicit continuation/curriculum coordination across optimizer segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.logs.trace import Trace
from optimizer.result import OptimizerResult
from optimizer.system_olgs import get_secondary_update_hook, require_system


OptimizerCallable = Callable[..., OptimizerResult]
SystemFactory = Callable[[Any], Any]
ControlTransfer = Callable[[Any, Any, Controls], Controls]
ObserverFactory = Callable[[int, "ContinuationStage"], Sequence[Any] | None]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if callable(value):
        return getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass(frozen=True)
class StageGate:
    """Explicit evidence required before advancing a continuation stage."""

    objective_target: float | None = None
    objective_mode: str = "min"
    feasibility_tolerance: float | None = None
    residual_name: str = "hard"
    stationarity_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.objective_mode not in {"min", "max"}:
            raise ValueError("objective_mode must be 'min' or 'max'.")
        for name, value in (
            ("objective_target", self.objective_target),
            ("feasibility_tolerance", self.feasibility_tolerance),
            ("stationarity_tolerance", self.stationarity_tolerance),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided.")
        if self.feasibility_tolerance is not None and float(self.feasibility_tolerance) < 0.0:
            raise ValueError("feasibility_tolerance must be nonnegative.")
        if self.stationarity_tolerance is not None and float(self.stationarity_tolerance) < 0.0:
            raise ValueError("stationarity_tolerance must be nonnegative.")


@dataclass(frozen=True)
class ContinuationStage:
    """One declared system transition and bounded optimizer segment."""

    label: str
    optimizer: str | OptimizerCallable
    secondary_updates: Mapping[str, Any] = field(default_factory=dict)
    optimizer_options: Mapping[str, Any] = field(default_factory=dict)
    gate: StageGate = field(default_factory=StageGate)
    system_factory: SystemFactory | None = None
    control_transfer: ControlTransfer | None = None

    def __post_init__(self) -> None:
        if not str(self.label):
            raise ValueError("continuation stage label must be non-empty.")
        if not isinstance(self.optimizer, str) and not callable(self.optimizer):
            raise TypeError("continuation stage optimizer must be a registered name or callable.")
        object.__setattr__(self, "secondary_updates", dict(self.secondary_updates))
        object.__setattr__(self, "optimizer_options", dict(self.optimizer_options))


@dataclass(frozen=True)
class StageResult:
    index: int
    stage: ContinuationStage
    system: Any
    result: OptimizerResult
    gate_passed: bool
    gate_metrics: Mapping[str, Any]
    transfer: str
    parent_checkpoint_ids: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.stage.label,
            "optimizer": self.result.optimizer,
            "secondary_updates": _json_safe(self.stage.secondary_updates),
            "optimizer_options": _json_safe(self.stage.optimizer_options),
            "result": self.result.to_dict(),
            "gate_passed": self.gate_passed,
            "gate_metrics": _json_safe(self.gate_metrics),
            "transfer": self.transfer,
            "parent_checkpoint_ids": dict(self.parent_checkpoint_ids),
        }


@dataclass(frozen=True)
class ContinuationResult:
    """Result chain from a continuation plan."""

    stages: tuple[StageResult, ...]
    final_result: OptimizerResult
    final_system: Any
    stop_reason: str
    plan_id: str

    @property
    def controls(self) -> Controls:
        return self.final_result.controls

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.final_result.metrics)

    @property
    def J(self) -> float:
        return self.final_result.J

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stop_reason": self.stop_reason,
            "stages": [stage.to_dict() for stage in self.stages],
            "final_result": self.final_result.to_dict(),
        }


def _registered_optimizer(name: str) -> OptimizerCallable:
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

    methods: dict[str, OptimizerCallable] = {
        "adagrad": adagrad,
        "adam": adam,
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
    key = str(name).lower()
    if key not in methods:
        raise KeyError(f"Unknown continuation optimizer {name!r}; valid names are {sorted(methods)}.")
    return methods[key]


def _coerce_stage(value: ContinuationStage | Mapping[str, Any]) -> ContinuationStage:
    if isinstance(value, ContinuationStage):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each continuation stage must be ContinuationStage or a mapping.")
    payload = dict(value)
    gate = payload.get("gate")
    if isinstance(gate, Mapping):
        payload["gate"] = StageGate(**dict(gate))
    return ContinuationStage(**payload)


def _same_layout(left: ControlSpec, right: ControlSpec) -> bool:
    if left.keys != right.keys or left.control_dim != right.control_dim:
        return False
    if left.dt is None or right.dt is None:
        return left.dt is right.dt
    return bool(np.isclose(left.dt, right.dt, rtol=0.0, atol=np.finfo(float).eps * max(1.0, abs(left.dt))))


def _transition_controls(
    previous_system: Any,
    next_system: Any,
    controls: Controls,
    transfer: ControlTransfer | None,
) -> tuple[Controls, str]:
    next_spec = next_system.control_spec()
    if transfer is not None:
        result = transfer(previous_system, next_system, controls)
        if not isinstance(result, Controls):
            raise TypeError("continuation control transfer must return Controls.")
        return result, getattr(transfer, "__qualname__", getattr(transfer, "__name__", "callable"))
    if not _same_layout(controls.spec, next_spec):
        raise ValueError(
            "continuation changed the control layout or time spacing; provide an explicit control_transfer."
        )
    return Controls.from_matrix(
        next_spec,
        controls.as_matrix(copy=True),
        name="continuation_transfer",
        meta=dict(controls.meta),
    ), "compatible_copy"


def _stage_gate(system: Any, result: OptimizerResult, gate: StageGate) -> tuple[bool, dict[str, Any]]:
    facts: dict[str, Any] = {"J": result.J}
    passed = True
    if gate.objective_target is not None:
        target = float(gate.objective_target)
        objective_passed = result.J <= target if gate.objective_mode == "min" else result.J >= target
        facts.update({"objective_target": target, "objective_passed": objective_passed})
        passed = passed and objective_passed
    if gate.feasibility_tolerance is not None:
        evaluator = SystemEvaluator(system)
        residual = evaluator.residuals(result.controls, name=gate.residual_name)
        residual_norm = float(np.linalg.norm(residual))
        feasible = residual_norm <= float(gate.feasibility_tolerance)
        facts.update(
            {
                "residual_name": gate.residual_name,
                "residual_norm": residual_norm,
                "feasibility_passed": feasible,
                "gate_counters": evaluator.counters(),
            }
        )
        passed = passed and feasible
    if gate.stationarity_tolerance is not None:
        state = result.state
        memory = {} if state is None else state.optimizer_state
        candidates = (
            memory.get("projected_kkt_norm"),
            memory.get("proximal_gradient_mapping_norm"),
            memory.get("gradient_norm"),
        )
        stationarity = next((float(value) for value in candidates if value is not None), None)
        source = "optimizer_state"
        if stationarity is None:
            evaluator = SystemEvaluator(system)
            stationarity = evaluator.gradient(result.controls).norm()
            facts["stationarity_gate_counters"] = evaluator.counters()
            source = "explicit_final_gradient"
        stationarity_passed = stationarity <= float(gate.stationarity_tolerance)
        facts.update(
            {
                "stationarity_norm": stationarity,
                "stationarity_source": source,
                "stationarity_passed": stationarity_passed,
            }
        )
        passed = passed and stationarity_passed
    return passed, facts


def continuation(
    system: Any,
    controls: Controls,
    *,
    stages: Sequence[ContinuationStage | Mapping[str, Any]],
    control_transfer: ControlTransfer | None = None,
    trace: Trace | None = None,
    observer_factory: ObserverFactory | None = None,
    stop_on_gate_failure: bool = True,
) -> ContinuationResult:
    """Run an explicit sequence of system transitions and optimizer segments."""

    require_system(system)
    if not stages:
        raise ValueError("continuation requires at least one stage.")
    active_system = system
    active_controls = controls
    active_trace = trace or Trace(run_id=uuid4().hex)
    stage_results: list[StageResult] = []
    stop_reason = "completed"
    plan_id = uuid4().hex
    parent_checkpoint_ids: dict[str, str] = {}
    for index, raw_stage in enumerate(stages):
        stage = _coerce_stage(raw_stage)
        next_system = stage.system_factory(active_system) if stage.system_factory is not None else active_system
        if stage.secondary_updates:
            next_system = get_secondary_update_hook(next_system)(**stage.secondary_updates)
        require_system(next_system)
        stage_controls, transfer_name = _transition_controls(
            active_system,
            next_system,
            active_controls,
            stage.control_transfer or control_transfer,
        )
        method = _registered_optimizer(stage.optimizer) if isinstance(stage.optimizer, str) else stage.optimizer
        options = dict(stage.optimizer_options)
        options.setdefault("stage", stage.label)
        options.setdefault("chunk", index)
        options.setdefault("trace", active_trace)
        options.setdefault("create_trace", False)
        caller_context = dict(options.get("api_context", {}) or {})
        caller_context["continuation"] = {
            "plan_id": plan_id,
            "stage_index": index,
            "stage_label": stage.label,
            "secondary_updates": dict(stage.secondary_updates),
            "transfer": transfer_name,
            "parent_checkpoint_ids": dict(parent_checkpoint_ids),
        }
        options["api_context"] = caller_context
        if observer_factory is not None and "observers" not in options:
            observers = observer_factory(index, stage)
            if observers is not None:
                options["observers"] = observers
        result = method(next_system, stage_controls, **options)
        gate_passed, gate_metrics = _stage_gate(next_system, result, stage.gate)
        stage_results.append(
            StageResult(
                index=index,
                stage=stage,
                system=next_system,
                result=result,
                gate_passed=gate_passed,
                gate_metrics=gate_metrics,
                transfer=transfer_name,
                parent_checkpoint_ids=dict(parent_checkpoint_ids),
            )
        )
        active_system = next_system
        active_controls = result.controls.copy(name="continuation_stage_result")
        parent_checkpoint_ids = dict(result.checkpoint_ids)
        if not gate_passed and stop_on_gate_failure:
            stop_reason = f"stage_gate_failed:{stage.label}"
            break
    final = stage_results[-1]
    return ContinuationResult(
        stages=tuple(stage_results),
        final_result=final.result,
        final_system=final.system,
        stop_reason=stop_reason,
        plan_id=plan_id,
    )
