"""Public facade object for OPTIMIZER v3.

Why this file exists
--------------------
The user-facing library should support both the explicit role namespaces and compact
direct shortcuts.  Typical calls are ``opt.optimizers.adam(system, controls, ...)`` or
``opt.adam(system, controls, ...)``.  Internal modules stay organized by
responsibility, and this facade keeps the notebook/script entry point consistent.

This module builds that facade around ``OptimizerLibrary`` and ``OptimizerContext``.
The library object supports explicit calls such as ``opt.run_chunk(system, controls)``.
The context object supports bound-system calls such as
``ctx = opt.context(system); ctx.run_chunk(controls, ...)``.  Both styles delegate to
the same core implementation.

How it fits the architecture
----------------------------
- ``__init__.py`` re-exports role namespaces plus a default ``OptimizerLibrary``.
- low-level modules keep owning implementation details.
- optimizer/tool modules can be attached here without changing user import style.
- curriculum workflows can keep one bound context and advance weights through
  ``ctx.with_secondary(...)``.
- tests can exercise the public API through the same path notebooks will use.

What this file deliberately does not do
---------------------------------------
It does not contain algorithm math.  Adam, line search, guesses, repairs, and
diagnostics live in their role-specific modules.  This file defines the public doorway
and delegates to those implemented components.

Reviewer invariants
-------------------
- explicit ``system, controls`` calls remain the reference style.
- direct helpers return the same core objects as internal imports.
- bound-system convenience is explicit context state, not a mutable package global.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable, Iterable
from uuid import uuid4

from optimizer.blackbox import analyze as _blackbox_analyze
from optimizer.blackbox import diagnostics as _blackbox_diagnostics
from optimizer.blackbox import prune as _blackbox_prune
from optimizer.blackbox import reset as _blackbox_reset
from optimizer.blackbox import start as _blackbox_start
from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import StepFunction, run_chunk
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.guards import metric_guard as _metric_guard
from optimizer.guesses import (
    constant_guess as _constant_guess,
    cosine_guess as _cosine_guess,
    fourier_guess as _fourier_guess,
    gaussian_guess as _gaussian_guess,
    mix_guess as _mix_guess,
    perturb_guess as _perturb_guess,
    ramp_guess as _ramp_guess,
    random_fourier_guess as _random_fourier_guess,
    random_guess as _random_guess,
    random_smooth_guess as _random_smooth_guess,
    scale_guess as _scale_guess,
    sinc_guess as _sinc_guess,
    sine_guess as _sine_guess,
)
from optimizer.core.parallel import ParallelConfig, parallel_map
from optimizer.logs.trace import Trace
from optimizer.optimizers import adagrad as _adagrad
from optimizer.optimizers import adam as _adam
from optimizer.optimizers import cma_es as _cma_es
from optimizer.optimizers import lbfgs as _lbfgs
from optimizer.optimizers import line_search as _line_search
from optimizer.optimizers import momentum as _momentum
from optimizer.optimizers import ncg as _ncg
from optimizer.optimizers import nonlinear_cg as _nonlinear_cg
from optimizer.optimizers import rmsprop as _rmsprop
from optimizer.result import Evaluation, OptimizerResult
from optimizer.schedules import AdaptiveStepSchedule, ConstantSchedule
from optimizer.state import RunState, WarmStartState
from optimizer.system_olgs import get_secondary_update_hook, require_system
from optimizer.utils import (
    control_spectrum as _control_spectrum,
    diagnostic_report as _diagnostic_report,
    finite_difference_gradient as _finite_difference_gradient,
    finite_difference_jacobian as _finite_difference_jacobian,
    geometry_probe as _geometry_probe,
    metric_report as _metric_report,
    nullspace_basis as _nullspace_basis,
    project_gradient as _project_gradient,
    repair_newton as _repair_newton,
    smoothness_report as _smoothness_report,
    verify_gradient as _verify_gradient,
    verify_jacobian as _verify_jacobian,
)


def _system_params(system: Any) -> dict[str, Any]:
    """Return best-effort system params for display/logging convenience."""

    params = getattr(system, "params", None)
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    if is_dataclass(params):
        return asdict(params)
    if hasattr(params, "__dict__"):
        return dict(vars(params))
    return {}


@dataclass(frozen=True)
class MethodInfo:
    """Small public status record for a public method."""

    name: str
    status: str
    module: str
    phase: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "module": self.module,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class OptimizerContext:
    """Bound-system public facade for notebook and curriculum workflows."""

    system: Any
    library: "OptimizerLibrary"
    trace: Trace | None = None
    defaults: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        require_system(self.system)
        object.__setattr__(self, "defaults", dict(self.defaults or {}))

    # ------------------------------------------------------------------
    # Bound system and params
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        """Return a best-effort copy of current system params."""

        return _system_params(self.system)

    def with_secondary(self, **updates: Any) -> "OptimizerContext":
        """Return a new context whose system has updated secondary params."""

        hook = get_secondary_update_hook(self.system)
        return replace(self, system=hook(**updates))

    def with_trace(self, trace: Trace | str | None) -> "OptimizerContext":
        """Return a new context carrying the provided trace."""

        next_trace = self.library.trace(trace) if isinstance(trace, str) else trace
        return replace(self, trace=next_trace)

    def with_defaults(self, **defaults: Any) -> "OptimizerContext":
        """Return a new context with updated default keyword arguments."""

        merged = dict(self.defaults or {})
        merged.update(defaults)
        return replace(self, defaults=merged)

    # ------------------------------------------------------------------
    # Bound convenience helpers
    # ------------------------------------------------------------------

    def control_spec(self) -> ControlSpec:
        """Return the bound system's control spec."""

        return self.system.control_spec()

    def zeros(self, **kwargs: Any) -> Controls:
        """Create zero controls for the bound system."""

        return self.library.zeros(self.control_spec(), **kwargs)

    def constant(self, value: Any, **kwargs: Any) -> Controls:
        """Create constant controls for the bound system."""

        return self.library.constant(self.control_spec(), value, **kwargs)

    def evaluate(self, controls: Controls, *, use_cache: bool = True) -> Evaluation:
        """Evaluate controls against the bound system."""

        return self.library.evaluate(self.system, controls, use_cache=use_cache)

    def gradient(self, controls: Controls, *, use_cache: bool = True) -> Controls:
        """Return analytical gradient from the bound system."""

        return self.library.gradient(self.system, controls, use_cache=use_cache)

    def run_chunk(
        self,
        controls: Controls | None = None,
        *,
        step: StepFunction,
        optimizer_name: str,
        maxiter: int,
        **kwargs: Any,
    ) -> OptimizerResult:
        """Run one engine chunk against the bound system."""

        call_kwargs = dict(self.defaults or {})
        call_kwargs.update(kwargs)
        if self.trace is not None and "trace" not in call_kwargs:
            call_kwargs["trace"] = self.trace
            call_kwargs.setdefault("create_trace", False)
        return self.library.run_chunk(
            self.system,
            controls,
            step=step,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            **call_kwargs,
        )

    def warmstart(self, result_or_state: Any, *, target_optimizer: str | None = None) -> WarmStartState:
        """Create warmstart state through the shared public helper."""

        return self.library.warmstart(result_or_state, target_optimizer=target_optimizer)

    # ------------------------------------------------------------------
    # Bound optimizer methods
    # ------------------------------------------------------------------

    def adam(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.adam(self.system, controls, *args, **kwargs)

    def momentum(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.momentum(self.system, controls, *args, **kwargs)

    def line_search(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.line_search(self.system, controls, *args, **kwargs)

    def adagrad(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.adagrad(self.system, controls, *args, **kwargs)

    def rmsprop(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.rmsprop(self.system, controls, *args, **kwargs)

    def lbfgs(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.lbfgs(self.system, controls, *args, **kwargs)

    def nonlinear_cg(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.nonlinear_cg(self.system, controls, *args, **kwargs)

    def ncg(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.ncg(self.system, controls, *args, **kwargs)

    def cma_es(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        return self.library.cma_es(self.system, controls, *args, **kwargs)

    # ------------------------------------------------------------------
    # Bound diagnostics, repair, and training aids
    # ------------------------------------------------------------------

    def metric_report(self, controls: Controls) -> dict[str, Any]:
        """Return current metrics for the bound system."""

        return self.library.metric_report(self.system, controls)

    def diagnostic_report(self, controls: Controls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return technical diagnostics for the bound system."""

        return self.library.diagnostic_report(self.system, controls, *args, **kwargs)

    def geometry_probe(self, controls: Controls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return local residual/Jacobian geometry for the bound system."""

        return self.library.geometry_probe(self.system, controls, *args, **kwargs)

    def verify_gradient(self, controls: Controls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Check the bound system's analytical gradient."""

        return self.library.verify_gradient(self.system, controls, *args, **kwargs)

    def verify_jacobian(self, controls: Controls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Check the bound system's analytical residual Jacobian."""

        return self.library.verify_jacobian(self.system, controls, *args, **kwargs)

    def repair_newton(self, controls: Controls, *args: Any, **kwargs: Any) -> Any:
        """Repair bound-system residuals using Newton/LM updates."""

        return self.library.repair_newton(self.system, controls, *args, **kwargs)

    def project_gradient(self, controls: Controls, gradient: Controls, *args: Any, **kwargs: Any) -> Any:
        """Project a gradient against the bound system's hard residual geometry."""

        return self.library.project_gradient(self.system, controls, gradient, *args, **kwargs)

    # ------------------------------------------------------------------
    # Bound guess generators
    # ------------------------------------------------------------------

    def constant_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a constant guess for the bound system."""

        return self.library.constant_guess(self.system, *args, **kwargs)

    def ramp_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a ramp guess for the bound system."""

        return self.library.ramp_guess(self.system, *args, **kwargs)

    def sine_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a sine guess for the bound system."""

        return self.library.sine_guess(self.system, *args, **kwargs)

    def cosine_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a cosine guess for the bound system."""

        return self.library.cosine_guess(self.system, *args, **kwargs)

    def gaussian_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a Gaussian guess for the bound system."""

        return self.library.gaussian_guess(self.system, *args, **kwargs)

    def sinc_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a sinc guess for the bound system."""

        return self.library.sinc_guess(self.system, *args, **kwargs)

    def fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a Fourier-series guess for the bound system."""

        return self.library.fourier_guess(self.system, *args, **kwargs)

    def random_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a random guess for the bound system."""

        return self.library.random_guess(self.system, *args, **kwargs)

    def random_smooth_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a smoothed random guess for the bound system."""

        return self.library.random_smooth_guess(self.system, *args, **kwargs)

    def random_fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create a random Fourier guess for the bound system."""

        return self.library.random_fourier_guess(self.system, *args, **kwargs)

    def scale_guess(self, controls: Controls, *args: Any, **kwargs: Any) -> Controls:
        """Rescale existing controls."""

        return self.library.scale_guess(controls, *args, **kwargs)

    def mix_guess(self, guesses: Any, *args: Any, **kwargs: Any) -> Controls:
        """Mix compatible controls."""

        return self.library.mix_guess(guesses, *args, **kwargs)

    def perturb_guess(self, controls: Controls, *args: Any, **kwargs: Any) -> Controls:
        """Perturb existing controls for local restarts."""

        return self.library.perturb_guess(controls, *args, **kwargs)


class OptimizerLibrary:
    """Public direct-shortcut facade for optimizer users."""

    # ------------------------------------------------------------------
    # Implemented core helpers
    # ------------------------------------------------------------------

    def control_spec(self, keys: Iterable[str], control_dim: int, **kwargs: Any) -> ControlSpec:
        """Create a ``ControlSpec`` without importing the controls module directly."""

        return ControlSpec(tuple(keys), control_dim, **kwargs)

    def controls(self, spec: ControlSpec, values: Any, **kwargs: Any) -> Controls:
        """Create ``Controls`` from a matrix-like payload."""

        return Controls.from_matrix(spec, values, **kwargs)

    def zeros(self, spec: ControlSpec, **kwargs: Any) -> Controls:
        """Create zero controls from a public facade call."""

        return Controls.zeros(spec, **kwargs)

    def constant(self, spec: ControlSpec, value: Any, **kwargs: Any) -> Controls:
        """Create constant controls from a public facade call."""

        return Controls.constant(spec, value, **kwargs)

    def evaluate(self, system: Any, controls: Controls, *, use_cache: bool = True) -> Evaluation:
        """Evaluate a system through the validated public boundary."""

        return SystemEvaluator(system, use_cache=use_cache).evaluate(controls)

    def gradient(self, system: Any, controls: Controls, *, use_cache: bool = True) -> Controls:
        """Return the analytical system gradient through the public boundary."""

        return SystemEvaluator(system, use_cache=use_cache).gradient(controls)

    def run_chunk(
        self,
        system: Any,
        controls: Controls | None = None,
        *,
        step: StepFunction,
        optimizer_name: str,
        maxiter: int,
        **kwargs: Any,
    ) -> OptimizerResult:
        """Run the shared engine with an explicit method-specific step function."""

        return run_chunk(
            system,
            controls,
            step=step,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            **kwargs,
        )

    def trace(self, run_id: str | None = None) -> Trace:
        """Create an in-memory trace ledger."""

        return Trace(run_id=run_id or uuid4().hex)

    def warmstart(self, result_or_state: Any, *, target_optimizer: str | None = None) -> WarmStartState:
        """Create safe warmstart state from a result or active run state."""

        if isinstance(result_or_state, RunState):
            return WarmStartState.from_run_state(result_or_state, target_optimizer=target_optimizer)
        if hasattr(result_or_state, "warmstart"):
            return result_or_state.warmstart(target_optimizer=target_optimizer)
        return WarmStartState.from_result(result_or_state, target_optimizer=target_optimizer)

    def parallel_map(
        self,
        function: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        config: ParallelConfig | None = None,
    ) -> list[Any]:
        """Map independent work through the library's local parallel interface."""

        return parallel_map(function, items, config=config)

    def blackbox_start(self, *args: Any, **kwargs: Any) -> Any:
        """Create a numeric blackbox run ledger."""

        return _blackbox_start(*args, **kwargs)

    def blackbox_analyze(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Analyze a blackbox run folder without extra system calls."""

        return _blackbox_analyze(*args, **kwargs)

    def diagnostics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return historical diagnostics from a blackbox run folder."""

        return _blackbox_diagnostics(*args, **kwargs)

    def blackbox_reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Reset all or part of a blackbox run folder."""

        return _blackbox_reset(*args, **kwargs)

    def blackbox_prune(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Prune transient blackbox artifacts."""

        return _blackbox_prune(*args, **kwargs)

    def context(
        self,
        system: Any,
        *,
        trace: Trace | str | None = None,
        **defaults: Any,
    ) -> OptimizerContext:
        """Bind a system for notebook/curriculum convenience."""

        trace_obj = self.trace(trace) if isinstance(trace, str) else trace
        return OptimizerContext(system=system, library=self, trace=trace_obj, defaults=defaults)

    def bind(
        self,
        system: Any,
        *,
        trace: Trace | str | None = None,
        **defaults: Any,
    ) -> OptimizerContext:
        """Alias for ``context`` using action-oriented naming."""

        return self.context(system, trace=trace, **defaults)

    # ------------------------------------------------------------------
    # Direct optimizer shortcuts
    # ------------------------------------------------------------------

    def adam(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run Adam or one of its implemented variants."""

        return _adam(*args, **kwargs)

    def momentum(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run the momentum optimizer family."""

        return _momentum(*args, **kwargs)

    def line_search(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run gradient descent with line-search variants."""

        return _line_search(*args, **kwargs)

    def adagrad(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run AdaGrad with cumulative squared-gradient scaling."""

        return _adagrad(*args, **kwargs)

    def rmsprop(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run RMSProp with moving-average squared-gradient scaling."""

        return _rmsprop(*args, **kwargs)

    def lbfgs(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run limited-memory BFGS."""

        return _lbfgs(*args, **kwargs)

    def nonlinear_cg(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run nonlinear conjugate gradient."""

        return _nonlinear_cg(*args, **kwargs)

    def ncg(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run nonlinear conjugate gradient through the short alias."""

        return _ncg(*args, **kwargs)

    def cma_es(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Run compact CMA-ES style population search."""

        return _cma_es(*args, **kwargs)

    # ------------------------------------------------------------------
    # Diagnostics, repair, and training aids
    # ------------------------------------------------------------------

    def metric_report(self, system: Any, controls: Controls) -> dict[str, Any]:
        """Return current system metrics in a review-friendly payload."""

        return _metric_report(system, controls)

    def diagnostic_report(self, system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
        """Return metric/control/residual/gradient diagnostics."""

        return _diagnostic_report(system, controls, **kwargs)

    def geometry_probe(self, system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
        """Return local residual/Jacobian geometry diagnostics."""

        return _geometry_probe(system, controls, **kwargs)

    def finite_difference_gradient(self, system: Any, controls: Controls, **kwargs: Any) -> Controls:
        """Return finite-difference gradient for a scalar metric."""

        return _finite_difference_gradient(system, controls, **kwargs)

    def verify_gradient(self, system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
        """Compare analytical gradient with finite-difference directions."""

        return _verify_gradient(system, controls, **kwargs)

    def finite_difference_jacobian(self, system: Any, controls: Controls, **kwargs: Any) -> Any:
        """Return finite-difference residual Jacobian."""

        return _finite_difference_jacobian(system, controls, **kwargs)

    def verify_jacobian(self, system: Any, controls: Controls, **kwargs: Any) -> dict[str, Any]:
        """Compare analytical residual Jacobian with finite-difference directions."""

        return _verify_jacobian(system, controls, **kwargs)

    def repair_newton(self, system: Any, controls: Controls, **kwargs: Any) -> Any:
        """Repair residual violations using Newton/LM updates."""

        return _repair_newton(system, controls, **kwargs)

    def project_gradient(self, system: Any, controls: Controls, gradient: Controls, **kwargs: Any) -> Any:
        """Project a gradient into the local residual nullspace."""

        return _project_gradient(system, controls, gradient, **kwargs)

    def nullspace_basis(self, system: Any, controls: Controls, **kwargs: Any) -> Any:
        """Return local residual nullspace basis columns."""

        return _nullspace_basis(system, controls, **kwargs)

    def metric_guard(self, **kwargs: Any) -> Any:
        """Build a reusable multi-metric accept function."""

        return _metric_guard(**kwargs)

    def control_spectrum(self, controls: Controls, **kwargs: Any) -> dict[str, Any]:
        """Return FFT-based control spectrum diagnostics."""

        return _control_spectrum(controls, **kwargs)

    def smoothness_report(self, controls: Controls) -> dict[str, Any]:
        """Return finite-difference smoothness diagnostics."""

        return _smoothness_report(controls)

    def constant_schedule(self, value: float) -> ConstantSchedule:
        """Create a constant step-size schedule."""

        return ConstantSchedule(value)

    def adaptive_step_schedule(self, **kwargs: Any) -> AdaptiveStepSchedule:
        """Create a shrink/grow adaptive step-size schedule."""

        return AdaptiveStepSchedule(**kwargs)

    # ------------------------------------------------------------------
    # Guess generators
    # ------------------------------------------------------------------

    def constant_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create constant initial controls."""

        return _constant_guess(*args, **kwargs)

    def ramp_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create ramp-shaped initial controls."""

        return _ramp_guess(*args, **kwargs)

    def sine_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create sine-wave initial controls."""

        return _sine_guess(*args, **kwargs)

    def cosine_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create cosine-wave initial controls."""

        return _cosine_guess(*args, **kwargs)

    def gaussian_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create localized Gaussian initial controls."""

        return _gaussian_guess(*args, **kwargs)

    def sinc_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create sinc-shaped initial controls."""

        return _sinc_guess(*args, **kwargs)

    def fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create Fourier-series initial controls."""

        return _fourier_guess(*args, **kwargs)

    def random_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create raw random initial controls."""

        return _random_guess(*args, **kwargs)

    def random_smooth_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create smoothed random initial controls."""

        return _random_smooth_guess(*args, **kwargs)

    def random_fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Create random low-frequency Fourier initial controls."""

        return _random_fourier_guess(*args, **kwargs)

    def scale_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Rescale existing controls as a new guess."""

        return _scale_guess(*args, **kwargs)

    def mix_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Mix compatible control guesses."""

        return _mix_guess(*args, **kwargs)

    def perturb_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Perturb existing controls for local restarts."""

        return _perturb_guess(*args, **kwargs)

    # ------------------------------------------------------------------
    # Facade metadata
    # ------------------------------------------------------------------

    def methods(self) -> dict[str, MethodInfo]:
        """Return public method status for notebooks and review checks."""

        return {
            "run_chunk": MethodInfo("run_chunk", "implemented", "optimizer/core/engine.py", "Phase 5"),
            "evaluate": MethodInfo("evaluate", "implemented", "optimizer/core/evaluate.py", "Phase 5"),
            "gradient": MethodInfo("gradient", "implemented", "optimizer/core/evaluate.py", "Phase 5"),
            "trace": MethodInfo("trace", "implemented", "optimizer/logs/trace.py", "Phase 4"),
            "blackbox_start": MethodInfo("blackbox_start", "implemented", "optimizer/blackbox/run.py", "Phase 12"),
            "blackbox_analyze": MethodInfo(
                "blackbox_analyze",
                "implemented",
                "optimizer/blackbox/analysis.py",
                "Phase 12",
            ),
            "diagnostics": MethodInfo("diagnostics", "implemented", "optimizer/blackbox/analysis.py", "Phase 12"),
            "warmstart": MethodInfo("warmstart", "implemented", "optimizer/state.py", "Phase 3"),
            "parallel_map": MethodInfo("parallel_map", "implemented", "optimizer/core/parallel.py", "Phase 5"),
            "context": MethodInfo("context", "implemented", "optimizer/library.py", "Phase 6"),
            "bind": MethodInfo("bind", "implemented", "optimizer/library.py", "Phase 6"),
            "adam": MethodInfo("adam", "implemented", "optimizer/optimizers/adam.py", "Phase 7"),
            "momentum": MethodInfo("momentum", "implemented", "optimizer/optimizers/momentum.py", "Phase 7"),
            "line_search": MethodInfo(
                "line_search",
                "implemented",
                "optimizer/optimizers/line_search.py",
                "Phase 7",
            ),
            "adagrad": MethodInfo("adagrad", "implemented", "optimizer/optimizers/adaptive.py", "Phase 7"),
            "rmsprop": MethodInfo("rmsprop", "implemented", "optimizer/optimizers/adaptive.py", "Phase 7"),
            "lbfgs": MethodInfo("lbfgs", "implemented", "optimizer/optimizers/lbfgs.py", "Phase 7"),
            "nonlinear_cg": MethodInfo(
                "nonlinear_cg",
                "implemented",
                "optimizer/optimizers/nonlinear_cg.py",
                "Phase 7",
            ),
            "ncg": MethodInfo("ncg", "implemented", "optimizer/optimizers/nonlinear_cg.py", "Phase 7"),
            "cma_es": MethodInfo("cma_es", "implemented", "optimizer/optimizers/cma_es.py", "Phase 7"),
            "fourier_guess": MethodInfo(
                "fourier_guess",
                "implemented",
                "optimizer/guesses/harmonic.py",
                "Phase 9",
            ),
            "constant_guess": MethodInfo(
                "constant_guess",
                "implemented",
                "optimizer/guesses/simple.py",
                "Phase 9",
            ),
            "ramp_guess": MethodInfo("ramp_guess", "implemented", "optimizer/guesses/simple.py", "Phase 9"),
            "sine_guess": MethodInfo("sine_guess", "implemented", "optimizer/guesses/harmonic.py", "Phase 9"),
            "cosine_guess": MethodInfo(
                "cosine_guess",
                "implemented",
                "optimizer/guesses/harmonic.py",
                "Phase 9",
            ),
            "gaussian_guess": MethodInfo(
                "gaussian_guess",
                "implemented",
                "optimizer/guesses/harmonic.py",
                "Phase 9",
            ),
            "sinc_guess": MethodInfo("sinc_guess", "implemented", "optimizer/guesses/harmonic.py", "Phase 9"),
            "random_guess": MethodInfo("random_guess", "implemented", "optimizer/guesses/random.py", "Phase 9"),
            "random_smooth_guess": MethodInfo(
                "random_smooth_guess",
                "implemented",
                "optimizer/guesses/random.py",
                "Phase 9",
            ),
            "random_fourier_guess": MethodInfo(
                "random_fourier_guess",
                "implemented",
                "optimizer/guesses/random.py",
                "Phase 9",
            ),
            "scale_guess": MethodInfo(
                "scale_guess",
                "implemented",
                "optimizer/guesses/composite.py",
                "Phase 9",
            ),
            "mix_guess": MethodInfo(
                "mix_guess",
                "implemented",
                "optimizer/guesses/composite.py",
                "Phase 9",
            ),
            "perturb_guess": MethodInfo(
                "perturb_guess",
                "implemented",
                "optimizer/guesses/composite.py",
                "Phase 9",
            ),
            "repair_newton": MethodInfo(
                "repair_newton",
                "implemented",
                "optimizer/utils/repairs.py",
                "Phase 11",
            ),
            "geometry_probe": MethodInfo(
                "geometry_probe",
                "implemented",
                "optimizer/utils/diagnostics.py",
                "Phase 11",
            ),
            "diagnostic_report": MethodInfo(
                "diagnostic_report",
                "implemented",
                "optimizer/utils/diagnostics.py",
                "Phase 11",
            ),
            "verify_gradient": MethodInfo(
                "verify_gradient",
                "implemented",
                "optimizer/utils/derivatives.py",
                "Phase 11",
            ),
            "verify_jacobian": MethodInfo(
                "verify_jacobian",
                "implemented",
                "optimizer/utils/derivatives.py",
                "Phase 11",
            ),
            "project_gradient": MethodInfo(
                "project_gradient",
                "implemented",
                "optimizer/utils/geometry.py",
                "Phase 11",
            ),
            "metric_guard": MethodInfo(
                "metric_guard",
                "implemented",
                "optimizer/core/guards.py",
                "Phase 11",
            ),
        }


DEFAULT_LIBRARY = OptimizerLibrary()
