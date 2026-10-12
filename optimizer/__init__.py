"""Public API for OPTIMIZER v4.

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

from typing import Any as _Any, Callable as _Callable, Iterable as _Iterable

from optimizer import campaigns, catalog, core, guesses, jobs, optimizers, schedules, shooting, utils, visualization
from optimizer.catalog import base
from optimizer.controls import ControlSpec, Controls
from optimizer.continuation import ContinuationResult, ContinuationStage, StageGate, StageResult
from optimizer.core import (
    AcceptanceDecision,
    MetricGuard,
    StepContext,
    StepProposal,
    StoppingConfig,
    SystemEvaluator,
)
from optimizer.utils.parallel import ParallelConfig
from optimizer.library import DEFAULT_LIBRARY as _DEFAULT_LIBRARY, MethodInfo, OptimizerContext, OptimizerLibrary
from optimizer.result import Evaluation, OptimizerResult
from optimizer.schedules import (
    AdaptiveStepSchedule,
    ConstantSchedule,
    CosineDecaySchedule,
    ExponentialDecaySchedule,
    LinearWarmupSchedule,
)
from optimizer.state import RunState, WarmStartState
from optimizer.transactions import SystemRequest, SystemResponse, SystemTransactionAdapter, transact
from optimizer.serialization import to_plain
from optimizer.families import system_family
from optimizer.system_olgs import (
    ControlBounds,
    OLGS,
    OLGSystem,
    OptimizerSystem,
    SystemProbe,
    SupportsBounds,
    SupportsControlTransfer,
    SupportsEqualityConstraints,
    SupportsHVP,
    SupportsPlotData,
    SupportsProximalObjective,
    SupportsValueAndGradient,
    SYSTEM_SPEC_SCHEMA,
    TrackingSpec,
    ValueGradient,
    basic_system_spec,
    contract_info,
    describe_system,
    evaluate_system,
    finite_difference_hessian,
    finite_difference_hvp,
    get_system_spec,
    gradient_system,
    optional_hessian,
    optional_hvp,
    optional_jacobian,
    optional_plot_data,
    optional_control_bounds,
    optional_control_transfer,
    optional_prox,
    optional_residuals,
    optional_smooth_gradient,
    probe_system,
    require_system,
    validate_controls_for_system,
    validate_metrics,
    validate_system_spec,
    metric_spec,
    value_and_gradient_system,
    with_secondary,
)
from optimizer.system_olss import (
    ContinuationPoint,
    MultiStartResult,
    ParameterContinuationPoint,
    ParameterContinuationResult,
    OLSS,
    OpenLoopShootingSystem,
    SHOOTING_SYSTEM_SPEC_SCHEMA,
    ShootingBlock,
    ShootingContinuationResult,
    ShootingEvaluation,
    ShootingResult,
    ShootingSpec,
    ShootingSystemProbe,
    ShootingVariables,
    ShootingVerification,
    VerificationCheck,
    basic_shooting_system_spec,
    describe_shooting_system,
    get_shooting_system_spec,
    probe_shooting_system,
    require_shooting_system,
    shooting_contract_info,
    shooting_metric_spec,
    validate_shooting_system_spec,
)
from optimizer.shooting import (
    ShootingEvaluator,
    continue_parameter,
    compare_shooting_replay,
    continue_shooting,
    finite_difference_shooting_jacobian,
    solve_direct_shooting,
    solve_shooting,
    solve_shooting_multistart,
    random_shooting_seeds,
    perturb_shooting_seeds,
    refine_shooting_seeds,
    verify_shooting_jacobian,
)
from optimizer import operations
from optimizer.operations import ObjectRef, OperationRequest, OperationResponse, OperationService, Session


__version__ = "4.0.4"
from optimizer.visualization import PlotData, PlotLayer
from optimizer.utils.repairs import RepairResult


library = _DEFAULT_LIBRARY
util = utils


# ----------------------------------------------------------------------
# Direct core helpers
# ----------------------------------------------------------------------


def control_spec(keys: _Iterable[str], control_dim: int, **kwargs: _Any) -> ControlSpec:
    return library.control_spec(keys, control_dim, **kwargs)


def controls(spec: ControlSpec, values: _Any, **kwargs: _Any) -> Controls:
    return library.controls(spec, values, **kwargs)


def zeros(spec: ControlSpec, **kwargs: _Any) -> Controls:
    return library.zeros(spec, **kwargs)


def constant(spec: ControlSpec, value: _Any, **kwargs: _Any) -> Controls:
    return library.constant(spec, value, **kwargs)


def evaluate(system: _Any, controls: _Any, *, use_cache: bool = True) -> _Any:
    return library.evaluate(system, controls, use_cache=use_cache)


def gradient(system: _Any, controls: _Any, *, use_cache: bool = True) -> Controls:
    return library.gradient(system, controls, use_cache=use_cache)


def value_and_gradient(
    system: _Any,
    controls: Controls,
    *,
    use_cache: bool = True,
) -> tuple[Evaluation, Controls]:
    return library.value_and_gradient(system, controls, use_cache=use_cache)


def plot_data(system: _Any, controls: Controls, *, kind: str, **options: _Any) -> PlotData:
    """Return one explicit system-owned visualization data product."""

    return library.plot_data(system, controls, kind=kind, **options)


def run_chunk(
    system: _Any,
    controls: Controls | None = None,
    *,
    step: _Callable[[StepContext], StepProposal | Controls],
    optimizer_name: str,
    maxiter: int,
    **kwargs: _Any,
) -> OptimizerResult:
    return library.run_chunk(
        system,
        controls,
        step=step,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        **kwargs,
    )




def warmstart(result_or_state: _Any, *, target_optimizer: str | None = None) -> WarmStartState:
    return library.warmstart(result_or_state, target_optimizer=target_optimizer)


def parallel_map(
    function: _Callable[[_Any], _Any],
    items: _Iterable[_Any],
    *,
    config: ParallelConfig | None = None,
) -> list[_Any]:
    return library.parallel_map(function, items, config=config)


def context(system: _Any, **defaults: _Any) -> OptimizerContext:
    return library.context(system, **defaults)


def bind(system: _Any, **defaults: _Any) -> OptimizerContext:
    return library.bind(system, **defaults)


def methods() -> dict[str, MethodInfo]:
    return library.methods()


def system_info(system: _Any) -> dict[str, _Any]:
    """Return declared system capabilities without evaluating controls."""

    return library.system_info(system)


def context_info(system: _Any | None = None) -> dict[str, _Any]:
    """Return registered numerical capabilities and optional system metadata."""

    return library.context_info(system=system)


# ----------------------------------------------------------------------
# Direct optimizer shortcuts
# ----------------------------------------------------------------------


def adam(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.adam(*args, **kwargs)


def momentum(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.momentum(*args, **kwargs)


def line_search(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.line_search(*args, **kwargs)


def adagrad(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.adagrad(*args, **kwargs)


def rmsprop(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.rmsprop(*args, **kwargs)


def lbfgs(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.lbfgs(*args, **kwargs)


def nonlinear_cg(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.nonlinear_cg(*args, **kwargs)


def ncg(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.ncg(*args, **kwargs)


def cma_es(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.cma_es(*args, **kwargs)


def projected_gradient(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.projected_gradient(*args, **kwargs)


def newton_cg(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.newton_cg(*args, **kwargs)


def trust_region(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.trust_region(*args, **kwargs)


def proximal_gradient(*args: _Any, **kwargs: _Any) -> OptimizerResult:
    return library.proximal_gradient(*args, **kwargs)


def continuation(
    system: _Any,
    controls: Controls,
    *,
    stages: _Iterable[ContinuationStage | dict[str, _Any]],
    **kwargs: _Any,
) -> ContinuationResult:
    return library.continuation(system, controls, stages=stages, **kwargs)


def metric_report(system: _Any, controls: Controls) -> dict[str, _Any]:
    return library.metric_report(system, controls)


def diagnostic_report(system: _Any, controls: Controls, **kwargs: _Any) -> dict[str, _Any]:
    return library.diagnostic_report(system, controls, **kwargs)


def diagnose_run(result: _Any, **kwargs: _Any) -> dict[str, _Any]:
    return library.diagnose_run(result, **kwargs)


def curvature_probe(system: _Any, controls: Controls, **kwargs: _Any) -> dict[str, _Any]:
    return library.curvature_probe(system, controls, **kwargs)


def compare_methods(system: _Any, controls: Controls, methods: _Any, **kwargs: _Any) -> dict[str, _Any]:
    return library.compare_methods(system, controls, methods, **kwargs)


def shooting_geometry_probe(system: _Any, variables: _Any, **kwargs: _Any) -> dict[str, _Any]:
    return library.shooting_geometry_probe(system, variables, **kwargs)


def finite_difference_gradient(system: _Any, controls: Controls, **kwargs: _Any) -> Controls:
    return library.finite_difference_gradient(system, controls, **kwargs)


def verify_gradient(system: _Any, controls: Controls, **kwargs: _Any) -> dict[str, _Any]:
    return library.verify_gradient(system, controls, **kwargs)


def finite_difference_jacobian(system: _Any, controls: Controls, **kwargs: _Any) -> _Any:
    return library.finite_difference_jacobian(system, controls, **kwargs)


def verify_jacobian(system: _Any, controls: Controls, **kwargs: _Any) -> dict[str, _Any]:
    return library.verify_jacobian(system, controls, **kwargs)


def repair_newton(system: _Any, controls: Controls, **kwargs: _Any) -> RepairResult:
    return library.repair_newton(system, controls, **kwargs)


def project_gradient(system: _Any, controls: Controls, gradient: Controls, **kwargs: _Any) -> _Any:
    return library.project_gradient(system, controls, gradient, **kwargs)


def nullspace_basis(system: _Any, controls: Controls, **kwargs: _Any) -> _Any:
    return library.nullspace_basis(system, controls, **kwargs)


def metric_guard(**kwargs: _Any) -> MetricGuard:
    return library.metric_guard(**kwargs)


def control_spectrum(controls: Controls, **kwargs: _Any) -> dict[str, _Any]:
    return library.control_spectrum(controls, **kwargs)


def smoothness_report(controls: Controls) -> dict[str, Any]:
    return library.smoothness_report(controls)


def constant_schedule(value: float) -> ConstantSchedule:
    return library.constant_schedule(value)


def adaptive_step_schedule(**kwargs: Any) -> AdaptiveStepSchedule:
    return library.adaptive_step_schedule(**kwargs)


def exponential_decay_schedule(**kwargs: Any) -> ExponentialDecaySchedule:
    return library.exponential_decay_schedule(**kwargs)


def linear_warmup_schedule(**kwargs: Any) -> LinearWarmupSchedule:
    return library.linear_warmup_schedule(**kwargs)


def cosine_decay_schedule(**kwargs: Any) -> CosineDecaySchedule:
    return library.cosine_decay_schedule(**kwargs)


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


def _root_info_payload() -> dict[str, _Any]:
    paths: dict[str, _Any] = {}
    for path_name in _DISCOVERY_PATHS:
        path_info = catalog.path(path_name)
        if isinstance(path_info, dict):
            paths[path_name] = path_info

    return {
        "name": "OPTIMIZER v4",
        "version": __version__,
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


def _render_root_info(payload: dict[str, _Any]) -> str:
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


def list(*, h: bool = False) -> dict[str, _Any] | str:
    """Return the full discovery catalog from the root ``opt`` namespace."""

    return catalog.list(h=h)


def info(name: str | None = None, *, h: bool = False) -> dict[str, _Any] | str:
    """Return root discovery help or detailed metadata for one catalog item."""

    if name is not None:
        return catalog.info(name, h=h)

    payload = _root_info_payload()
    return _render_root_info(payload) if h else payload


def search(query: str, *, h: bool = False) -> dict[str, _Any] | str:
    """Search the discovery catalog from the root ``opt`` namespace."""

    return catalog.search(query, h=h)


def path(name: str, *, h: bool = False) -> dict[str, _Any] | str:
    """Return a focused workflow path such as ``beginner`` or ``debug_gradient``."""

    return catalog.path(name, h=h)


catalog.attach_root_helpers(globals())

# Importing submodules attaches implementation-module attributes to this package.
# Keep the root namespace limited to the explicit surface below; those modules
# remain available through their documented role namespaces.
for _implementation_name in ("annotations", "logs", "result", "state", "system_olgs"):
    globals().pop(_implementation_name, None)


# Role namespaces, discovery views, and discovery helpers. Every other root name
# is a catalog declaration: a base item or an item marked root=True.
_NAVIGATION = (
    "base",
    "campaigns",
    "catalog",
    "core",
    "guesses",
    "jobs",
    "info",
    "list",
    "optimizers",
    "path",
    "schedules",
    "search",
    "shooting",
    "util",
    "utils",
    "visualization",
)
__all__ = sorted({*catalog.exports("base"), *_NAVIGATION})


def __dir__() -> list[str]:
    """Expose only the deliberate package surface in interactive discovery."""

    return sorted(__all__)
