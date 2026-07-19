"""Public API for OPTIMIZER v3.

Why this file exists
--------------------
The intended user experience is ``import optimizer as opt`` followed by a clear,
reviewable call.  The preferred long-form style groups tools by role, for example
``opt.optimizers.adam(...)``, ``opt.utils.verify_gradient(...)``, and
``opt.guesses.fourier_guess(...)``.  Direct shortcuts such as ``opt.adam(...)`` remain
available for compact notebooks and scripts.

This package initializer is therefore part of the design, not just import plumbing.
It exposes stable namespace modules, stable data objects, and the direct convenience
functions forwarded through the ``OptimizerLibrary`` facade in ``library.py``.

How it fits the architecture
----------------------------
- research notebooks import this module as ``opt``.
- namespace modules own role-specific calls: optimizers, utils, guesses, schedules.
- ``library.py`` owns direct shortcut dispatch and bound-system contexts.
- implementation modules remain separately reviewable and testable.
- later phases can attach new methods without changing the import root.

What this file deliberately does not do
---------------------------------------
It does not contain algorithm logic.  It only re-exports types and delegates direct
calls to the default facade object.

Reviewer invariants
-------------------
- ``import optimizer as opt`` is enough for the current public surface.
- namespace calls and direct shortcuts point at the same implementation families.
- explicit ``system, controls`` style remains the reference usage.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from optimizer import blackbox, catalog, core, guesses, optimizers, schedules, utils
from optimizer.blackbox import BlackBoxPolicy, BlackBoxRun
from optimizer.catalog import base
from optimizer.controls import ControlSpec, Controls
from optimizer.core import (
    AcceptanceDecision,
    MetricGuard,
    StepContext,
    StepProposal,
    StoppingConfig,
    SystemEvaluator,
)
from optimizer.utils.parallel import ParallelConfig
from optimizer.library import DEFAULT_LIBRARY, MethodInfo, OptimizerContext, OptimizerLibrary
from optimizer.logs.checkpoint import Checkpoint
from optimizer.logs.records import ChunkRecord, EventRecord, IterationRecord
from optimizer.logs.trace import Trace
from optimizer.result import Evaluation, OptimizerResult
from optimizer.schedules import AdaptiveStepSchedule, ConstantSchedule
from optimizer.state import RunState, WarmStartState
from optimizer.system_olgs import (
    OLGS,
    OLGSystem,
    OptimizerSystem,
    SystemProbe,
    evaluate_system,
    finite_difference_hessian,
    finite_difference_hvp,
    gradient_system,
    optional_hessian,
    optional_hvp,
    optional_jacobian,
    optional_residuals,
    probe_system,
    require_system,
    validate_controls_for_system,
    validate_metrics,
    with_secondary,
)
from optimizer.utils.repairs import RepairResult


library = DEFAULT_LIBRARY
util = utils


# ----------------------------------------------------------------------
# Direct core helpers
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


def blackbox_start(*args: Any, **kwargs: Any) -> BlackBoxRun:
    return library.blackbox_start(*args, **kwargs)


def blackbox_analyze(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.blackbox_analyze(*args, **kwargs)


def diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.diagnostics(*args, **kwargs)


def blackbox_reset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.blackbox_reset(*args, **kwargs)


def blackbox_prune(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.blackbox_prune(*args, **kwargs)


def context(system: Any, *, trace: Trace | str | None = None, **defaults: Any) -> OptimizerContext:
    return library.context(system, trace=trace, **defaults)


def bind(system: Any, *, trace: Trace | str | None = None, **defaults: Any) -> OptimizerContext:
    return library.bind(system, trace=trace, **defaults)


def methods() -> dict[str, MethodInfo]:
    return library.methods()


# ----------------------------------------------------------------------
# Direct optimizer shortcuts
# ----------------------------------------------------------------------


def adam(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.adam(*args, **kwargs)


def momentum(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.momentum(*args, **kwargs)


def line_search(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.line_search(*args, **kwargs)


def adagrad(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.adagrad(*args, **kwargs)


def rmsprop(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.rmsprop(*args, **kwargs)


def lbfgs(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.lbfgs(*args, **kwargs)


def nonlinear_cg(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.nonlinear_cg(*args, **kwargs)


def ncg(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.ncg(*args, **kwargs)


def cma_es(*args: Any, **kwargs: Any) -> OptimizerResult:
    return library.cma_es(*args, **kwargs)


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


def constant_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.constant_guess(*args, **kwargs)


def ramp_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.ramp_guess(*args, **kwargs)


def sine_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.sine_guess(*args, **kwargs)


def cosine_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.cosine_guess(*args, **kwargs)


def gaussian_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.gaussian_guess(*args, **kwargs)


def sinc_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.sinc_guess(*args, **kwargs)


def fourier_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.fourier_guess(*args, **kwargs)


def random_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.random_guess(*args, **kwargs)


def random_smooth_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.random_smooth_guess(*args, **kwargs)


def random_fourier_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.random_fourier_guess(*args, **kwargs)


def scale_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.scale_guess(*args, **kwargs)


def mix_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.mix_guess(*args, **kwargs)


def perturb_guess(*args: Any, **kwargs: Any) -> Controls:
    return library.perturb_guess(*args, **kwargs)


def geometry_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return library.geometry_probe(*args, **kwargs)


_DISCOVERY_PATHS = ("beginner", "debug_gradient", "repair_residuals", "optimize_then_polish")


def _root_info_payload() -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for path_name in _DISCOVERY_PATHS:
        path_info = catalog.path(path_name)
        if isinstance(path_info, dict):
            paths[path_name] = path_info

    return {
        "name": "OPTIMIZER v3",
        "summary": "Namespace-first optimization toolbox for system-defined control problems.",
        "start_here": [
            "opt.info(h=True)",
            "opt.list(h=True)",
            "opt.search('gradient', h=True)",
            "opt.path('beginner', h=True)",
            "opt.optimizers.list(h=True)",
            "opt.guesses.list(h=True)",
            "opt.utils.list(h=True)",
            "opt.core.list(h=True)",
        ],
        "groups": catalog.groups(),
        "paths": paths,
        "examples": {
            "method_details": "opt.info('adam', h=True)",
            "namespace_details": "opt.optimizers.info('adam', h=True)",
            "debug_derivatives": "opt.utils.list(kind='derivative', h=True)",
            "guess_search": "opt.search('fourier', h=True)",
        },
    }


def _render_root_info(payload: dict[str, Any]) -> str:
    lines = [payload["name"], f"Summary: {payload['summary']}"]

    lines.append("Start here:")
    lines.extend(f"- {call}" for call in payload["start_here"])

    groups = payload["groups"]
    if isinstance(groups, dict):
        lines.append("Groups:")
        for name, data in groups.items():
            lines.append(f"- {name}: {data['summary']} ({data['count']} items)")

    lines.append("Paths:")
    for name, data in payload["paths"].items():
        lines.append(f"- {name}: {data['goal']}")

    lines.append("Examples:")
    for label, call in payload["examples"].items():
        lines.append(f"- {label}: {call}")

    return "\n".join(lines)


def list(*, h: bool = False) -> dict[str, Any] | str:
    """Return the full discovery catalog from the root ``opt`` namespace."""

    return catalog.list(h=h)


def info(name: str | None = None, *, h: bool = False) -> dict[str, Any] | str:
    """Return root discovery help or detailed metadata for one catalog item."""

    if name is not None:
        return catalog.info(name, h=h)

    payload = _root_info_payload()
    return _render_root_info(payload) if h else payload


def search(query: str, *, h: bool = False) -> dict[str, Any] | str:
    """Search the discovery catalog from the root ``opt`` namespace."""

    return catalog.search(query, h=h)


def path(name: str, *, h: bool = False) -> dict[str, Any] | str:
    """Return a focused workflow path such as ``beginner`` or ``debug_gradient``."""

    return catalog.path(name, h=h)


catalog.attach_root_helpers(globals())


__all__ = [
    "AcceptanceDecision",
    "AdaptiveStepSchedule",
    "BlackBoxPolicy",
    "BlackBoxRun",
    "Checkpoint",
    "ChunkRecord",
    "ControlSpec",
    "Controls",
    "Evaluation",
    "EventRecord",
    "IterationRecord",
    "MethodInfo",
    "MetricGuard",
    "OLGS",
    "OLGSystem",
    "OptimizerContext",
    "OptimizerLibrary",
    "OptimizerResult",
    "OptimizerSystem",
    "ParallelConfig",
    "RepairResult",
    "RunState",
    "StepContext",
    "StepProposal",
    "StoppingConfig",
    "SystemEvaluator",
    "SystemProbe",
    "Trace",
    "WarmStartState",
    "adagrad",
    "adam",
    "adaptive_step_schedule",
    "base",
    "bind",
    "blackbox",
    "blackbox_analyze",
    "blackbox_prune",
    "blackbox_reset",
    "blackbox_start",
    "cma_es",
    "constant",
    "constant_guess",
    "constant_schedule",
    "context",
    "catalog",
    "control_spec",
    "control_spectrum",
    "controls",
    "core",
    "cosine_guess",
    "diagnostic_report",
    "diagnostics",
    "evaluate",
    "evaluate_system",
    "finite_difference_hessian",
    "finite_difference_hvp",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "fourier_guess",
    "gaussian_guess",
    "geometry_probe",
    "gradient",
    "gradient_system",
    "guesses",
    "info",
    "library",
    "lbfgs",
    "line_search",
    "list",
    "metric_guard",
    "metric_report",
    "methods",
    "mix_guess",
    "momentum",
    "ncg",
    "nonlinear_cg",
    "nullspace_basis",
    "optimizers",
    "optional_jacobian",
    "optional_hessian",
    "optional_hvp",
    "optional_residuals",
    "parallel_map",
    "path",
    "perturb_guess",
    "project_gradient",
    "probe_system",
    "ramp_guess",
    "random_fourier_guess",
    "random_guess",
    "random_smooth_guess",
    "repair_newton",
    "require_system",
    "rmsprop",
    "run_chunk",
    "schedules",
    "scale_guess",
    "search",
    "sinc_guess",
    "sine_guess",
    "smoothness_report",
    "trace",
    "util",
    "utils",
    "validate_controls_for_system",
    "validate_metrics",
    "verify_gradient",
    "verify_jacobian",
    "warmstart",
    "with_secondary",
    "zeros",
]
