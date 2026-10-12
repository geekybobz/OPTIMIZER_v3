"""Explicit continuation/curriculum coordination across optimizer segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from optimizer import catalog
from optimizer.controls import ControlSpec, Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.result import OptimizerResult
from optimizer.system_olgs import get_secondary_update_hook, require_system
from optimizer.serialization import to_plain


OptimizerCallable = Callable[..., OptimizerResult]
SystemFactory = Callable[[Any], Any]
ControlTransfer = Callable[[Any, Any, Controls], Controls]


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.stage.label,
            "optimizer": self.result.optimizer,
            "secondary_updates": to_plain(self.stage.secondary_updates),
            "optimizer_options": to_plain(self.stage.optimizer_options),
            "result": self.result.to_dict(),
            "gate_passed": self.gate_passed,
            "gate_metrics": to_plain(self.gate_metrics),
            "transfer": self.transfer,
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
    key = str(name).lower()
    names = catalog.exports("optimizers")
    if key not in names:
        raise KeyError(f"Unknown continuation optimizer {name!r}; valid names are {names}.")
    return catalog.group("optimizers").get(key).public_object()


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
    stop_on_gate_failure: bool = True,
) -> ContinuationResult:
    """Run an explicit sequence of system transitions and optimizer segments."""

    require_system(system)
    if not stages:
        raise ValueError("continuation requires at least one stage.")
    active_system = system
    active_controls = controls
    stage_results: list[StageResult] = []
    stop_reason = "completed"
    plan_id = uuid4().hex
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
            )
        )
        active_system = next_system
        active_controls = result.controls.copy(name="continuation_stage_result")
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
