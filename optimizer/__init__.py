"""Direct public API for OPTIMIZER v3.

Why this file exists
--------------------
The intended user experience is ``import optimizer as opt`` followed by direct calls.
This package initializer is therefore part of the design, not just import plumbing.
It exposes the current stable objects and forwards public functions through the
``OptimizerLibrary`` facade in ``library.py``.

How it fits the architecture
----------------------------
- research notebooks import this module as ``opt``.
- ``library.py`` owns public method dispatch and reserved names.
- implementation modules remain separately reviewable and testable.
- later phases can attach real optimizer implementations without changing imports.

What this file deliberately does not do
---------------------------------------
It does not contain algorithm logic.  It only re-exports types and delegates direct
calls to the default facade object.

Reviewer invariants
-------------------
- ``import optimizer as opt`` is enough for the public Phase 6 surface.
- method names planned for later phases are present but fail clearly.
- explicit ``system, controls`` style remains the reference usage.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import AcceptanceDecision, StepContext, StepProposal
from optimizer.core.guards import MetricGuard
from optimizer.core.parallel import ParallelConfig
from optimizer.library import DEFAULT_LIBRARY, MethodInfo, OptimizerContext, OptimizerLibrary
from optimizer.logs.checkpoint import Checkpoint
from optimizer.logs.records import ChunkRecord, EventRecord, IterationRecord
from optimizer.logs.trace import Trace
from optimizer.result import Evaluation, OptimizerResult
from optimizer.schedules import AdaptiveStepSchedule, ConstantSchedule
from optimizer.state import RunState, WarmStartState
from optimizer.system import (
    OptimizerSystem,
    SystemProbe,
    evaluate_system,
    gradient_system,
    optional_jacobian,
    optional_residuals,
    probe_system,
    require_system,
    validate_controls_for_system,
    validate_metrics,
)
from optimizer.utils.repairs import RepairResult


library = DEFAULT_LIBRARY


# ----------------------------------------------------------------------
# Phase 1-5 direct helpers
# ----------------------------------------------------------------------


def control_spec(keys: Iterable[str], control_dim: int, **kwargs: Any) -> ControlSpec:
    return library.control_spec(keys, control_dim, **kwargs)


def controls(spec: ControlSpec, values: Any, **kwargs: Any) -> Controls:
    return library.controls(spec, values, **kwargs)


def zeros(spec: ControlSpec, **kwargs: Any) -> Controls:
    return library.zeros(spec, **kwargs)


def constant(spec: ControlSpec, value: Any, **kwargs: Any) -> Controls:
    return library.constant(spec, value, **kwargs)


def evaluate(system: Any, controls: Controls, *, use_cache: bool = True) -> Evaluation:
    return library.evaluate(system, controls, use_cache=use_cache)


def gradient(system: Any, controls: Controls, *, use_cache: bool = True) -> Controls:
    return library.gradient(system, controls, use_cache=use_cache)


def run_chunk(
    system: Any,
    controls: Controls | None = None,
    *,
    step: Callable[[StepContext], StepProposal | Controls],
    optimizer_name: str,
    maxiter: int,
    **kwargs: Any,
) -> OptimizerResult:
    return library.run_chunk(
        system,
        controls,
        step=step,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        **kwargs,
    )


def trace(run_id: str | None = None) -> Trace:
    return library.trace(run_id=run_id)


def warmstart(result_or_state: Any, *, target_optimizer: str | None = None) -> WarmStartState:
    return library.warmstart(result_or_state, target_optimizer=target_optimizer)


def parallel_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    config: ParallelConfig | None = None,
) -> list[Any]:
    return library.parallel_map(function, items, config=config)


def context(system: Any, *, trace: Trace | str | None = None, **defaults: Any) -> OptimizerContext:
    return library.context(system, trace=trace, **defaults)


def bind(system: Any, *, trace: Trace | str | None = None, **defaults: Any) -> OptimizerContext:
    return library.bind(system, trace=trace, **defaults)


def methods() -> dict[str, MethodInfo]:
    return library.methods()


# ----------------------------------------------------------------------
# Reserved direct-call method names
# ----------------------------------------------------------------------


def adam(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.adam(*args, **kwargs)


def momentum(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.momentum(*args, **kwargs)


def line_search(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.line_search(*args, **kwargs)


def metric_report(system: Any, controls: Controls) -> dict[str, Any]:
    return library.metric_report(system, controls)


def diagnostic_report(system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
    return library.diagnostic_report(system, controls, **kwargs)


def finite_difference_gradient(system: Any, controls: Controls, **kwargs: Any) -> Controls:
    return library.finite_difference_gradient(system, controls, **kwargs)


def verify_gradient(system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
    return library.verify_gradient(system, controls, **kwargs)


def finite_difference_jacobian(system: Any, controls: Controls, **kwargs: Any) -> Any:
    return library.finite_difference_jacobian(system, controls, **kwargs)


def verify_jacobian(system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
    return library.verify_jacobian(system, controls, **kwargs)


def repair_newton(system: Any, controls: Controls, **kwargs: Any) -> RepairResult:
    return library.repair_newton(system, controls, **kwargs)


def project_gradient(system: Any, controls: Controls, gradient: Controls, **kwargs: Any) -> Any:
    return library.project_gradient(system, controls, gradient, **kwargs)


def nullspace_basis(system: Any, controls: Controls, **kwargs: Any) -> Any:
    return library.nullspace_basis(system, controls, **kwargs)


def metric_guard(**kwargs: Any) -> MetricGuard:
    return library.metric_guard(**kwargs)


def control_spectrum(controls: Controls, **kwargs: Any) -> dict[str, Any]:
    return library.control_spectrum(controls, **kwargs)


def smoothness_report(controls: Controls) -> dict[str, Any]:
    return library.smoothness_report(controls)


def constant_schedule(value: float) -> ConstantSchedule:
    return library.constant_schedule(value)


def adaptive_step_schedule(**kwargs: Any) -> AdaptiveStepSchedule:
    return library.adaptive_step_schedule(**kwargs)


def fourier_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.fourier_guess(*args, **kwargs)


def geometry_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.geometry_probe(*args, **kwargs)


__all__ = [
    "AcceptanceDecision",
    "AdaptiveStepSchedule",
    "Checkpoint",
    "ChunkRecord",
    "ControlSpec",
    "Controls",
    "Evaluation",
    "EventRecord",
    "IterationRecord",
    "MethodInfo",
    "MetricGuard",
    "OptimizerContext",
    "OptimizerLibrary",
    "OptimizerResult",
    "OptimizerSystem",
    "ParallelConfig",
    "RepairResult",
    "RunState",
    "StepContext",
    "StepProposal",
    "SystemProbe",
    "Trace",
    "WarmStartState",
    "adam",
    "adaptive_step_schedule",
    "bind",
    "constant",
    "constant_schedule",
    "context",
    "control_spec",
    "control_spectrum",
    "controls",
    "diagnostic_report",
    "evaluate",
    "evaluate_system",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "fourier_guess",
    "geometry_probe",
    "gradient",
    "gradient_system",
    "library",
    "line_search",
    "metric_guard",
    "metric_report",
    "methods",
    "momentum",
    "nullspace_basis",
    "optional_jacobian",
    "optional_residuals",
    "parallel_map",
    "project_gradient",
    "probe_system",
    "repair_newton",
    "require_system",
    "run_chunk",
    "smoothness_report",
    "trace",
    "validate_controls_for_system",
    "validate_metrics",
    "verify_gradient",
    "verify_jacobian",
    "warmstart",
    "zeros",
]
