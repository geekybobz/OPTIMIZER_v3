"""Public facade object for OPTIMIZER v3.

Why this file exists
--------------------
The user-facing library should feel direct: ``import optimizer as opt`` followed by
calls such as ``opt.adam(system, controls, ...)`` or ``opt.run_chunk(...)``.  Internal
modules can stay organized by responsibility, but users should not need to know where
the engine, trace, state, controls, or later optimizers live.

This module builds that facade around ``OptimizerLibrary`` and ``OptimizerContext``.
The library object supports explicit calls such as ``opt.run_chunk(system, controls)``.
The context object supports bound-system calls such as
``ctx = opt.context(system); ctx.run_chunk(controls, ...)``.  Both styles delegate to
the same core implementation.

How it fits the architecture
----------------------------
- ``__init__.py`` re-exports a default ``OptimizerLibrary`` instance and its methods.
- low-level modules keep owning implementation details.
- future optimizer modules can be attached here without changing user import style.
- curriculum workflows can keep one bound context and advance weights through
  ``ctx.with_params(...)``.
- tests can exercise the public API through the same path notebooks will use.

What this file deliberately does not do
---------------------------------------
It does not implement Adam, line search, guesses, repairs, diagnostics, or modes.
Those belong to later phases.  This file only defines the public doorway and delegates
to implemented components.

Reviewer invariants
-------------------
- explicit ``system, controls`` calls remain the reference style.
- unfinished public methods fail loudly and name the phase/module that should add them.
- direct helpers return the same core objects as internal imports.
- bound-system convenience is explicit context state, not a mutable package global.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable, Iterable
from uuid import uuid4

from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import StepFunction, run_chunk
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.parallel import ParallelConfig, parallel_map
from optimizer.logs.trace import Trace
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.system import require_system


def _not_implemented(
    method: str,
    phase: str,
    module: str,
    *,
    prefix: str = "opt",
) -> NotImplementedError:
    """Build a clear error for public methods reserved for later phases."""

    return NotImplementedError(
        f"{prefix}.{method}(...) is reserved for {phase} and is not implemented yet. "
        f"Expected implementation module: {module}."
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
    """Small public status record for a direct-call method."""

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

    def with_params(self, **updates: Any) -> "OptimizerContext":
        """Return a new context whose system has updated params."""

        hook = getattr(self.system, "with_params", None)
        if not callable(hook):
            raise TypeError("Bound system does not provide with_params(...).")
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
    # Reserved optimizer/tool methods for later phases
    # ------------------------------------------------------------------

    def adam(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        del controls, args, kwargs
        raise _not_implemented("adam", "Phase 7", "optimizer/optimizers/adam.py", prefix="ctx")

    def momentum(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        del controls, args, kwargs
        raise _not_implemented("momentum", "Phase 7", "optimizer/optimizers/momentum.py", prefix="ctx")

    def line_search(self, controls: Any = None, *args: Any, **kwargs: Any) -> OptimizerResult:
        del controls, args, kwargs
        raise _not_implemented(
            "line_search",
            "Phase 7",
            "optimizer/optimizers/line_search.py",
            prefix="ctx",
        )

    def fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        del args, kwargs
        raise _not_implemented("fourier_guess", "Phase 9", "optimizer/guesses/fourier.py", prefix="ctx")

    def repair_newton(self, controls: Any = None, *args: Any, **kwargs: Any) -> Controls:
        del controls, args, kwargs
        raise _not_implemented("repair_newton", "Phase 11", "optimizer/utils/repairs.py", prefix="ctx")

    def geometry_probe(self, controls: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del controls, args, kwargs
        raise _not_implemented("geometry_probe", "Phase 11", "optimizer/utils/diagnostics.py", prefix="ctx")


class OptimizerLibrary:
    """Public direct-call facade for optimizer users."""

    # ------------------------------------------------------------------
    # Implemented Phase 1-5 helpers
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
    # Reserved direct-call method names for later phases
    # ------------------------------------------------------------------

    def adam(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Reserved direct Adam call; implementation starts in Phase 7."""

        del args, kwargs
        raise _not_implemented("adam", "Phase 7", "optimizer/optimizers/adam.py")

    def momentum(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Reserved direct momentum call; implementation starts in Phase 7."""

        del args, kwargs
        raise _not_implemented("momentum", "Phase 7", "optimizer/optimizers/momentum.py")

    def line_search(self, *args: Any, **kwargs: Any) -> OptimizerResult:
        """Reserved direct line-search call; implementation starts in Phase 7."""

        del args, kwargs
        raise _not_implemented("line_search", "Phase 7", "optimizer/optimizers/line_search.py")

    def fourier_guess(self, *args: Any, **kwargs: Any) -> Controls:
        """Reserved direct Fourier guess call; implementation starts in Phase 9."""

        del args, kwargs
        raise _not_implemented("fourier_guess", "Phase 9", "optimizer/guesses/fourier.py")

    def repair_newton(self, *args: Any, **kwargs: Any) -> Controls:
        """Reserved direct Newton repair call; implementation starts in Phase 11."""

        del args, kwargs
        raise _not_implemented("repair_newton", "Phase 11", "optimizer/utils/repairs.py")

    def geometry_probe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Reserved direct geometry diagnostic call; implementation starts in Phase 11."""

        del args, kwargs
        raise _not_implemented("geometry_probe", "Phase 11", "optimizer/utils/diagnostics.py")

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
            "warmstart": MethodInfo("warmstart", "implemented", "optimizer/state.py", "Phase 3"),
            "parallel_map": MethodInfo("parallel_map", "implemented", "optimizer/core/parallel.py", "Phase 5"),
            "context": MethodInfo("context", "implemented", "optimizer/library.py", "Phase 6"),
            "bind": MethodInfo("bind", "implemented", "optimizer/library.py", "Phase 6"),
            "adam": MethodInfo("adam", "reserved", "optimizer/optimizers/adam.py", "Phase 7"),
            "momentum": MethodInfo("momentum", "reserved", "optimizer/optimizers/momentum.py", "Phase 7"),
            "line_search": MethodInfo(
                "line_search",
                "reserved",
                "optimizer/optimizers/line_search.py",
                "Phase 7",
            ),
            "fourier_guess": MethodInfo(
                "fourier_guess",
                "reserved",
                "optimizer/guesses/fourier.py",
                "Phase 9",
            ),
            "repair_newton": MethodInfo(
                "repair_newton",
                "reserved",
                "optimizer/utils/repairs.py",
                "Phase 11",
            ),
            "geometry_probe": MethodInfo(
                "geometry_probe",
                "reserved",
                "optimizer/utils/diagnostics.py",
                "Phase 11",
            ),
        }


DEFAULT_LIBRARY = OptimizerLibrary()
