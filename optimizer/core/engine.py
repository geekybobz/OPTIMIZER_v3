"""Shared optimizer chunk engine.

Why this file exists
--------------------
OPTIMIZER v4 should not let every algorithm carry its own driver loop.  Adam,
momentum, gradient descent, line search, projected, proximal, and trust-region methods need the
same outer mechanics: evaluate current controls, request an analytical gradient,
build a trial control, evaluate the trial, accept or reject it, update run state,
record technical logs, create rollback checkpoints, and stop consistently.

This module owns that common loop.  A concrete optimizer supplies only a proposal
function that maps ``StepContext`` to ``StepProposal``.  The engine handles the rest.

How it fits the architecture
----------------------------
- ``controls.py`` defines the object being moved.
- ``system_olgs`` defines how metrics and gradients are produced analytically.
- ``state.py`` stores mutable chunk state and warmstart handoff state.
- ``result.py`` standardizes the returned result.
- ``logs/trace.py`` receives iteration/chunk records and checkpoints.
- optimizer modules remain small method-specific proposal builders.

What this file deliberately does not do
---------------------------------------
It does not implement Adam moments, line-search bracketing, Fourier guesses, Newton
repairs, or curriculum policy.  It provides the reusable engine those layers call.

Reviewer invariants
-------------------
- attempted iterations are counted whether the trial is accepted or rejected.
- accepted trials update current controls and metrics; rejected trials leave them.
- best-so-far is updated only from accepted finite metrics.
- trace/checkpoint integration is optional but consistent when a trace is present.
- all system calls go through ``SystemEvaluator`` validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import GradientOutcome, SystemEvaluator
from optimizer.core.observer import (
    CheckpointEvent,
    ChunkEvent,
    IterationEvent,
    ObserverDispatcher,
    RunObserver,
    RunStartEvent,
    SnapshotFacts,
    TerminalEvent,
    classify_terminal_status,
    emit_initial_failure,
    normalize_observers,
)
from optimizer.core.stopping import StopDecision, StopTracker, StoppingConfig
from optimizer.logs.trace import Trace
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.system_olgs import OptimizerSystem, validate_controls_for_system


@dataclass(frozen=True)
class StepContext:
    """Read-oriented context passed to a concrete optimizer proposal function."""

    system: OptimizerSystem
    evaluator: SystemEvaluator
    state: RunState
    evaluation: Evaluation
    gradient: Controls
    iteration: int
    stage: str | None = None


@dataclass(frozen=True)
class StepProposal:
    """Trial update proposed by one optimizer iteration."""

    controls: Controls
    step_size: float | None = None
    optimizer_state: Mapping[str, Any] | None = None
    optimizer_state_on_accept: Mapping[str, Any] | None = None
    optimizer_state_on_reject: Mapping[str, Any] | None = None
    step_size_on_accept: float | None = None
    step_size_on_reject: float | None = None
    technical: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class AcceptanceDecision:
    """Engine decision on whether a trial replaces the current controls."""

    accepted: bool
    reason: str
    technical: dict[str, Any] = field(default_factory=dict)
    optimizer_state_update: Mapping[str, Any] | None = None
    next_step_size: float | None = None


StepFunction = Callable[[StepContext], StepProposal | Controls]
AcceptFunction = Callable[[Evaluation, Evaluation, StepProposal, RunState], AcceptanceDecision | bool]
RandomStateSource = Callable[[], Mapping[str, Any]] | Mapping[str, Any]


# ----------------------------------------------------------------------
# Metric and acceptance helpers
# ----------------------------------------------------------------------


def _scalar_metric(metrics: Mapping[str, Any], key: str) -> float:
    """Return a metric as a scalar float for accept/reject logic."""

    if key not in metrics:
        raise KeyError(f"metrics do not include {key!r}.")
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f"metric {key!r} must be scalar for acceptance checks.")
    return float(value)


def _descent_gradient(gradient: Controls, *, objective_direction: str) -> Controls:
    """Convert the declared objective derivative into a descent direction.

    Systems always return the analytical derivative of their declared objective
    metric.  Concrete first-order methods are written as minimizers, so a
    maximization problem must receive the negative derivative as its descent
    direction.  Keeping that conversion here makes it a system-contract
    decision, rather than a campaign or optimizer-name convention.
    """

    if objective_direction == "minimize":
        return gradient
    if objective_direction == "maximize":
        return -gradient
    raise ValueError(f"unsupported declared objective direction: {objective_direction!r}")


def default_accept(
    current: Evaluation,
    trial: Evaluation,
    proposal: StepProposal,
    state: RunState,
    *,
    metric: str = "J",
    mode: str = "min",
    tolerance: float = 0.0,
) -> AcceptanceDecision:
    """Accept trial controls when the selected metric does not get worse."""

    del proposal, state
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")

    current_value = _scalar_metric(current.metrics, metric)
    trial_value = _scalar_metric(trial.metrics, metric)
    tolerance = float(tolerance)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("acceptance tolerance must be finite and >= 0.")

    if mode == "min":
        accepted = trial_value <= current_value + tolerance
        improvement = current_value - trial_value
    else:
        accepted = trial_value >= current_value - tolerance
        improvement = trial_value - current_value

    reason = "accepted" if accepted else "rejected_worse_metric"
    return AcceptanceDecision(
        accepted=accepted,
        reason=reason,
        technical={
            "accept_metric": metric,
            "accept_mode": mode,
            "current_value": current_value,
            "trial_value": trial_value,
            "improvement": improvement,
            "tolerance": tolerance,
        },
    )


def _normalize_acceptance(decision: AcceptanceDecision | bool) -> AcceptanceDecision:
    """Convert custom accept-function output into an ``AcceptanceDecision``."""

    if isinstance(decision, AcceptanceDecision):
        return decision
    if isinstance(decision, bool):
        return AcceptanceDecision(
            accepted=decision,
            reason="accepted" if decision else "rejected",
            technical={},
        )
    raise TypeError("accept function must return AcceptanceDecision or bool.")


def _proposal_state_update(proposal: StepProposal, *, accepted: bool) -> dict[str, Any]:
    """Return the optimizer-private state update for the decision branch.

    ``optimizer_state`` is the Phase 5 legacy field and still means "apply this
    update after the proposal is processed."  Real optimizers need finer control:
    Adam moments, momentum velocity, and future L-BFGS histories should usually
    advance only after an accepted control move.  The branch-specific fields let a
    method express that without forcing the engine to understand the algorithm.
    """

    branch_state = (
        proposal.optimizer_state_on_accept
        if accepted
        else proposal.optimizer_state_on_reject
    )
    if branch_state is not None:
        return dict(branch_state)
    return dict(proposal.optimizer_state or {})


def _proposal_step_size(proposal: StepProposal, *, accepted: bool) -> float | None:
    """Return the step-size value to retain for the decision branch."""

    branch_step_size = (
        proposal.step_size_on_accept
        if accepted
        else proposal.step_size_on_reject
    )
    if branch_step_size is not None:
        return float(branch_step_size)
    if proposal.step_size is not None:
        return float(proposal.step_size)
    return None


# ----------------------------------------------------------------------
# Run-state construction helpers
# ----------------------------------------------------------------------


def _extract_system_params(system: Any, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return system params for state/log context without imposing a class."""

    if explicit is not None:
        return dict(explicit)
    params = getattr(system, "params", None)
    if isinstance(params, Mapping):
        return dict(params)
    if params is not None and hasattr(params, "__dict__"):
        return dict(vars(params))
    return {}


def _start_controls(
    controls: Controls | None,
    state: RunState | None,
    warmstart: WarmStartState | None,
) -> Controls:
    """Choose the controls that start this chunk."""

    if state is not None:
        return state.controls.copy(name=state.controls.name)
    if warmstart is not None:
        return warmstart.controls.copy(name=warmstart.controls.name)
    if controls is not None:
        return controls.copy(name=controls.name)
    raise ValueError("run_chunk requires controls, state, or warmstart.")


def _start_step_size(
    explicit: float | None,
    state: RunState | None,
    warmstart: WarmStartState | None,
) -> float | None:
    """Choose the step-size estimate carried into this chunk."""

    if explicit is not None:
        return float(explicit)
    if state is not None and state.step_size is not None:
        return float(state.step_size)
    if warmstart is not None and warmstart.step_size is not None:
        return float(warmstart.step_size)
    return None


def _start_optimizer_state(
    state: RunState | None,
    warmstart: WarmStartState | None,
) -> dict[str, Any]:
    """Choose optimizer-private state carried into this chunk."""

    if state is not None:
        return dict(state.optimizer_state)
    if warmstart is not None and warmstart.optimizer_state is not None:
        return dict(warmstart.optimizer_state)
    return {}


def _build_state(
    *,
    controls: Controls,
    metrics: Mapping[str, Any],
    optimizer_name: str,
    step_size: float | None,
    optimizer_state: Mapping[str, Any],
    system_params: Mapping[str, Any],
    trace: Trace | None,
    previous_state: RunState | None,
    warmstart: WarmStartState | None,
) -> RunState:
    """Create the mutable chunk state used by the engine loop."""

    state = RunState.initial(
        controls,
        metrics=metrics,
        optimizer_name=optimizer_name,
        step_size=step_size,
        system_params=system_params,
        trace_id=None if trace is None else trace.run_id,
    )
    state.optimizer_state = dict(optimizer_state)

    if previous_state is not None:
        state.global_iteration = int(previous_state.global_iteration)
        state.checkpoint_ids = dict(previous_state.checkpoint_ids)
        if previous_state.best_controls is not None:
            state.best_controls = previous_state.best_controls.copy(name="best")
        if previous_state.best_metrics is not None:
            state.best_metrics = dict(previous_state.best_metrics)
    elif warmstart is not None:
        state.checkpoint_ids = dict(warmstart.checkpoint_ids)
    return state


# ----------------------------------------------------------------------
# Trace helpers
# ----------------------------------------------------------------------


def _ensure_trace(trace: Trace | None, *, create_trace: bool) -> Trace | None:
    """Return provided trace or create one when requested."""

    if trace is not None:
        return trace
    if create_trace:
        return Trace(run_id=uuid4().hex)
    return None


def _active_observers(
    trace: Trace | None,
    observers: Iterable[RunObserver] | None,
) -> ObserverDispatcher:
    """Combine caller observers with the compatibility trace sink."""

    active = list(normalize_observers(observers))
    if trace is not None and all(observer is not trace for observer in active):
        active.append(trace)
    return ObserverDispatcher(active)


def _snapshot_facts(
    metrics: Mapping[str, Any],
    *,
    evaluator: SystemEvaluator,
    state: RunState,
) -> SnapshotFacts:
    """Package facts already owned by the engine without numerical work."""

    return SnapshotFacts.create(
        metrics=metrics,
        metric_sources={str(name): "system.evaluate" for name in metrics},
        counters={
            **evaluator.counters(compact=True),
            "iteration": state.iteration,
            "global_iteration": state.global_iteration,
        },
    )


def _best_snapshot(
    *,
    evaluator: SystemEvaluator,
    state: RunState,
) -> SnapshotFacts | None:
    """Return the engine-owned best-so-far state, if one is available."""

    if state.best_metrics is None:
        return None
    return _snapshot_facts(state.best_metrics, evaluator=evaluator, state=state)


def _capture_random_state(source: RandomStateSource | None) -> tuple[dict[str, Any], str]:
    """Capture caller-owned stochastic state without allowing capture to abort a run."""

    if source is None:
        return {}, "not_supplied"
    try:
        value = source() if callable(source) else source
        if not isinstance(value, Mapping):
            raise TypeError("random_state must resolve to a mapping")
        return dict(value), "callable" if callable(source) else "mapping"
    except Exception as exc:  # noqa: BLE001 - reproducibility capture is passive evidence.
        message = str(exc)
        if len(message) > 1024:
            message = f"{message[:1000]}...[truncated {len(message) - 1000} chars]"
        return {
            "_capture_error": f"{type(exc).__name__}: {message}",
        }, "capture_failed"


def _record_start(
    observers: ObserverDispatcher,
    *,
    optimizer_name: str,
    stage: str | None,
    initial: SnapshotFacts,
    chunk: int,
    system_params: Mapping[str, Any],
    api_context: Mapping[str, Any],
) -> None:
    event = RunStartEvent(
        optimizer=str(optimizer_name),
        phase="initialization" if stage is None else str(stage),
        initial=initial,
        chunk=int(chunk),
        stage=stage,
        system_params=dict(system_params),
        api_context=dict(api_context),
    )
    observers.emit("on_start", event)


def _record_checkpoint(
    observers: ObserverDispatcher,
    label: str,
    state: RunState,
    *,
    evaluator: SystemEvaluator,
    optimizer_name: str,
    reason: str,
    stage: str | None,
    chunk: int,
    api_context: Mapping[str, Any],
    random_state: RandomStateSource | None = None,
    controls: Controls | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Emit an important control checkpoint without additional numerical work."""

    captured_random_state, random_state_source = _capture_random_state(random_state)
    event = CheckpointEvent(
        label=str(label),
        reason=str(reason),
        controls=(state.controls if controls is None else controls).copy(
            name=(state.controls if controls is None else controls).name
        ),
        snapshot=_snapshot_facts(
            state.metrics if metrics is None else metrics,
            evaluator=evaluator,
            state=state,
        ),
        optimizer=str(optimizer_name),
        optimizer_state=dict(state.optimizer_state),
        system_params=dict(state.system_params),
        api_context=dict(api_context),
        iteration=int(state.iteration),
        global_iteration=int(state.global_iteration),
        chunk=int(chunk),
        stage=stage,
        random_state=captured_random_state,
        random_state_source=random_state_source,
    )
    for observer in observers.emit("on_checkpoint", event):
        if isinstance(observer, Trace):
            state.checkpoint_ids[str(label)] = observer.latest_checkpoint_id(str(label))


def _record_iteration(
    observers: ObserverDispatcher,
    *,
    state: RunState,
    optimizer_name: str,
    technical: Mapping[str, Any],
    stage: str | None,
    chunk: int,
    accepted: bool | None,
    reason: str | None,
    before: SnapshotFacts,
    trial: SnapshotFacts | None,
    after: SnapshotFacts,
    best_after: SnapshotFacts | None,
) -> None:
    """Emit one already-computed outer-iteration transaction."""

    event = IterationEvent(
        optimizer=optimizer_name,
        iteration=state.iteration,
        global_iteration=state.global_iteration,
        stage=stage,
        system_params=dict(state.system_params),
        before=before,
        trial=trial,
        after=after,
        best_after=best_after,
        technical=dict(technical),
        accepted=accepted,
        reason=reason,
        chunk=int(chunk),
    )
    observers.emit("on_iteration", event)


def _terminal_status(reason: str) -> str:
    """Classify engine stops without claiming numerical optimality."""

    return classify_terminal_status(reason)


def _record_terminal(
    observers: ObserverDispatcher,
    *,
    state: RunState,
    evaluator: SystemEvaluator,
    optimizer_name: str,
    reason: str,
    stage: str | None,
    chunk: int,
) -> None:
    event = TerminalEvent(
        optimizer=str(optimizer_name),
        status=_terminal_status(reason),
        reason=str(reason),
        final=_snapshot_facts(state.metrics, evaluator=evaluator, state=state),
        best=_best_snapshot(evaluator=evaluator, state=state),
        iteration=state.iteration,
        global_iteration=state.global_iteration,
        stage=stage,
        chunk=int(chunk),
        system_params=dict(state.system_params),
    )
    observers.emit("on_terminal", event)


def _attempt_snapshot(evaluator: SystemEvaluator) -> SnapshotFacts:
    """Return counter evidence when no valid initial metric snapshot exists."""

    return SnapshotFacts.create(
        counters={
            **evaluator.counters(compact=True),
            "iteration": 0,
            "global_iteration": 0,
        }
    )


def _record_initial_failure(
    observers: ObserverDispatcher,
    *,
    evaluator: SystemEvaluator,
    optimizer_name: str,
    error: str | None,
    stage: str | None,
    chunk: int,
    system_params: Mapping[str, Any],
    api_context: Mapping[str, Any],
) -> None:
    """Close a failed initial evaluation as a complete observer lifecycle."""

    snapshot = _attempt_snapshot(evaluator)
    start_event = RunStartEvent(
        optimizer=str(optimizer_name),
        phase="initialization" if stage is None else str(stage),
        initial=snapshot,
        chunk=int(chunk),
        stage=stage,
        system_params=dict(system_params),
        api_context={**dict(api_context), "initial_error": error},
    )
    terminal_event = TerminalEvent(
        optimizer=str(optimizer_name),
        status="failed",
        reason="initial_evaluation_failed",
        final=snapshot,
        best=None,
        iteration=0,
        global_iteration=0,
        stage=stage,
        chunk=int(chunk),
        system_params=dict(system_params),
    )
    emit_initial_failure(observers, start_event, terminal_event)


def _record_chunk(
    observers: ObserverDispatcher,
    *,
    optimizer_name: str,
    chunk: int,
    start_iteration: int,
    end_iteration: int,
    start_metrics: Mapping[str, Any],
    end_metrics: Mapping[str, Any],
    system_params: Mapping[str, Any],
    stage: str | None,
    accepted: bool | None,
    reason: str | None,
) -> None:
    """Emit the completed chunk to all passive sinks."""

    event = ChunkEvent(
        optimizer=optimizer_name,
        chunk=chunk,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        start_metrics=start_metrics,
        end_metrics=end_metrics,
        system_params=system_params,
        stage=stage,
        accepted=accepted,
        reason=reason,
    )
    observers.emit("on_chunk", event)


def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)))


# ----------------------------------------------------------------------
# Main engine loop
# ----------------------------------------------------------------------


def run_chunk(
    system: Any,
    controls: Controls | None = None,
    *,
    step: StepFunction,
    optimizer_name: str,
    maxiter: int,
    state: RunState | None = None,
    warmstart: WarmStartState | None = None,
    trace: Trace | None = None,
    create_trace: bool = True,
    observers: Iterable[RunObserver] | None = None,
    step_size: float | None = None,
    stopping: StoppingConfig | None = None,
    target_value: float | None = None,
    target_metric: str | None = None,
    target_mode: str | None = None,
    stall_patience: int | None = None,
    stall_tolerance: float = 0.0,
    stall_metric: str | None = None,
    stall_mode: str | None = None,
    accept: AcceptFunction | None = None,
    accept_metric: str | None = None,
    accept_mode: str | None = None,
    accept_tolerance: float = 0.0,
    system_params: Mapping[str, Any] | None = None,
    api_context: Mapping[str, Any] | None = None,
    random_state: RandomStateSource | None = None,
    stage: str | None = None,
    chunk: int = 0,
    use_cache: bool = True,
    checkpoint_start: bool = True,
    checkpoint_latest: bool = True,
    checkpoint_accepted: bool = True,
    checkpoint_best: bool = True,
    checkpoint_final: bool = True,
    gradient_source: str = "gradient",
    prefer_value_and_gradient: bool = True,
) -> OptimizerResult:
    """Run one optimizer chunk using a method-specific proposal function."""

    if not callable(step):
        raise TypeError("step must be callable.")
    if random_state is not None and not callable(random_state) and not isinstance(random_state, Mapping):
        raise TypeError("random_state must be a mapping, callable, or None.")
    if gradient_source not in {"gradient", "smooth_gradient"}:
        raise ValueError("gradient_source must be 'gradient' or 'smooth_gradient'.")

    evaluator = SystemEvaluator(system, use_cache=use_cache)
    objective = evaluator.system_spec["objective"]
    objective_metric = str(objective["metric_id"])
    objective_direction = str(objective["direction"])
    resolved_target_metric = objective_metric if target_metric is None else str(target_metric)
    resolved_target_mode = (
        "le" if objective_direction == "minimize" else "ge"
    ) if target_mode is None else str(target_mode)
    resolved_stall_metric = objective_metric if stall_metric is None else str(stall_metric)
    resolved_stall_mode = (
        "min" if objective_direction == "minimize" else "max"
    ) if stall_mode is None else str(stall_mode)
    resolved_accept_metric = objective_metric if accept_metric is None else str(accept_metric)
    resolved_accept_mode = (
        "min" if objective_direction == "minimize" else "max"
    ) if accept_mode is None else str(accept_mode)
    if gradient_source == "smooth_gradient" and objective_direction == "maximize":
        raise ValueError(
            "smooth_gradient is currently defined only for minimization; "
            "a maximizing system must provide a full analytical gradient."
        )
    active_trace = _ensure_trace(trace, create_trace=create_trace)
    active_observers = _active_observers(active_trace, observers)
    active_controls = _start_controls(controls, state, warmstart)
    validate_controls_for_system(evaluator.system, active_controls)

    params = _extract_system_params(evaluator.system, system_params)
    stop_config = stopping or StoppingConfig(
        maxiter=maxiter,
        target_value=target_value,
        target_metric=resolved_target_metric,
        target_mode=resolved_target_mode,
        stall_patience=stall_patience,
        stall_tolerance=stall_tolerance,
        stall_metric=resolved_stall_metric,
        stall_mode=resolved_stall_mode,
    )
    resolved_api_context = {
        "run_chunk": {
            "optimizer_name": str(optimizer_name),
            "maxiter": int(maxiter),
            "step": _callable_name(step),
            "step_size": None if step_size is None else float(step_size),
            "target_value": None if target_value is None else float(target_value),
            "target_metric": resolved_target_metric,
            "target_mode": resolved_target_mode,
            "stall_patience": None if stall_patience is None else int(stall_patience),
            "stall_tolerance": float(stall_tolerance),
            "stall_metric": resolved_stall_metric,
            "stall_mode": resolved_stall_mode,
            "accept": _callable_name(accept) or "default_accept",
            "accept_metric": resolved_accept_metric,
            "accept_mode": resolved_accept_mode,
            "accept_tolerance": float(accept_tolerance),
            "system_objective": {
                "metric_id": objective_metric,
                "direction": objective_direction,
                "gradient_to_descent": "identity"
                if objective_direction == "minimize"
                else "negate",
            },
            "stage": stage,
            "chunk": int(chunk),
            "use_cache": bool(use_cache),
            "gradient_source": gradient_source,
            "prefer_value_and_gradient": bool(prefer_value_and_gradient),
            "state_supplied": state is not None,
            "warmstart_supplied": warmstart is not None,
            "trace_supplied": trace is not None,
            "observer_count": len(active_observers.active),
            "random_state_source": "not_supplied"
            if random_state is None
            else ("callable" if callable(random_state) else "mapping"),
            "stopping": {
                "maxiter": stop_config.maxiter,
                "target_value": stop_config.target_value,
                "target_metric": stop_config.target_metric,
                "target_mode": stop_config.target_mode,
                "stall_patience": stop_config.stall_patience,
                "stall_tolerance": stop_config.stall_tolerance,
                "stall_metric": stop_config.stall_metric,
                "stall_mode": stop_config.stall_mode,
                "check_finite": stop_config.check_finite,
            },
            "checkpoint_policy": {
                "start": bool(checkpoint_start),
                "latest": bool(checkpoint_latest),
                "accepted": bool(checkpoint_accepted),
                "best": bool(checkpoint_best),
                "final": bool(checkpoint_final),
            },
        },
        "caller": dict(api_context or {}),
    }

    prefetched_gradient: Controls | None = None
    initial_error: str | None = None
    initial_evaluation: Evaluation | None = None
    use_combined = (
        gradient_source == "gradient"
        and bool(prefer_value_and_gradient)
        and evaluator.capabilities.get("has_value_and_gradient", False)
    )
    if use_combined:
        combined_outcome = evaluator.try_value_and_gradient(active_controls)
        if combined_outcome.ok and combined_outcome.value_gradient is not None:
            initial_evaluation = Evaluation.from_metrics(
                active_controls.copy(name=active_controls.name),
                combined_outcome.value_gradient.metrics,
            )
            # Keep the system derivative in its native convention here.  The
            # common conversion to a descent direction happens immediately
            # before ``StepContext`` is constructed, for both combined and
            # standalone gradient paths.
            prefetched_gradient = combined_outcome.value_gradient.gradient
        else:
            initial_error = combined_outcome.error
    else:
        initial_outcome = evaluator.try_evaluate(active_controls)
        initial_evaluation = initial_outcome.evaluation
        initial_error = initial_outcome.error
    if initial_evaluation is None:
        _record_initial_failure(
            active_observers,
            evaluator=evaluator,
            optimizer_name=optimizer_name,
            error=initial_error,
            stage=stage,
            chunk=chunk,
            system_params=params,
            api_context=resolved_api_context,
        )
        raise ValueError(f"initial controls failed evaluation: {initial_error}")

    run_state = _build_state(
        controls=active_controls,
        metrics=initial_evaluation.metrics,
        optimizer_name=optimizer_name,
        step_size=_start_step_size(step_size, state, warmstart),
        optimizer_state=_start_optimizer_state(state, warmstart),
        system_params=params,
        trace=active_trace,
        previous_state=state,
        warmstart=warmstart,
    )
    start_metrics = dict(run_state.metrics)
    start_iteration = int(run_state.iteration)
    _record_start(
        active_observers,
        optimizer_name=optimizer_name,
        stage=stage,
        initial=_snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state),
        chunk=chunk,
        system_params=run_state.system_params,
        api_context=resolved_api_context,
    )

    tracker = StopTracker(stop_config, initial_metrics=run_state.metrics)

    if checkpoint_start:
        _record_checkpoint(
            active_observers,
            "chunk_start",
            run_state,
            evaluator=evaluator,
            optimizer_name=optimizer_name,
            reason="initial_valid_state",
            stage=stage,
            chunk=chunk,
            api_context=resolved_api_context,
            random_state=random_state,
        )

    initial_stop = tracker.check_initial_metrics(run_state.metrics)
    if initial_stop.stop:
        run_state.stop_reason = initial_stop.reason
        if checkpoint_final:
            _record_checkpoint(
                active_observers,
                "chunk_end",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=initial_stop.reason or "initial_stop",
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )
        _record_chunk(
            active_observers,
            optimizer_name=optimizer_name,
            chunk=chunk,
            start_iteration=start_iteration,
            end_iteration=run_state.iteration,
            start_metrics=start_metrics,
            end_metrics=run_state.metrics,
            system_params=run_state.system_params,
            stage=stage,
            accepted=None,
            reason=initial_stop.reason,
        )
        _record_terminal(
            active_observers,
            state=run_state,
            evaluator=evaluator,
            optimizer_name=optimizer_name,
            reason=initial_stop.reason or "stopped",
            stage=stage,
            chunk=chunk,
        )
        result = OptimizerResult.from_state(
            run_state,
            stop_reason=initial_stop.reason or "stopped",
            optimizer=optimizer_name,
            trace=active_trace,
            tracking_failures=active_observers.failures,
        )
        return result

    current_eval = initial_evaluation
    stop_reason = "maxiter"
    accepted_any = False

    def record_terminal_selection(reason: str) -> None:
        """Persist bounded selected evidence from already-owned run state."""

        if checkpoint_best and run_state.best_controls is not None and run_state.best_metrics is not None:
            _record_checkpoint(
                active_observers,
                f"best_{resolved_accept_metric}",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason="terminal_selected_best",
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
                controls=run_state.best_controls,
                metrics=run_state.best_metrics,
            )
        if checkpoint_final:
            _record_checkpoint(
                active_observers,
                "chunk_end",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=reason,
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )

    def close_interrupted_run() -> None:
        """Close passive evidence using the last valid state, without new numerics."""

        run_state.stop_reason = "interrupted"
        record_terminal_selection("interrupted")
        _record_chunk(
            active_observers,
            optimizer_name=optimizer_name,
            chunk=chunk,
            start_iteration=start_iteration,
            end_iteration=run_state.iteration,
            start_metrics=start_metrics,
            end_metrics=run_state.metrics,
            system_params=run_state.system_params,
            stage=stage,
            accepted=accepted_any,
            reason="interrupted",
        )
        _record_terminal(
            active_observers,
            state=run_state,
            evaluator=evaluator,
            optimizer_name=optimizer_name,
            reason="interrupted",
            stage=stage,
            chunk=chunk,
        )

    while True:
        budget_decision = tracker.check_before_iteration(run_state.iteration)
        if budget_decision.stop:
            stop_reason = budget_decision.reason or "maxiter"
            break

        before_snapshot = _snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state)

        try:
            if prefetched_gradient is not None:
                gradient_outcome = GradientOutcome(
                    gradient=prefetched_gradient,
                    ok=True,
                    reason="ok",
                )
                prefetched_gradient = None
            elif gradient_source == "smooth_gradient":
                gradient_outcome = evaluator.try_smooth_gradient(run_state.controls)
            else:
                gradient_outcome = evaluator.try_gradient(run_state.controls)
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        if gradient_outcome.ok and gradient_outcome.gradient is not None:
            gradient_outcome = GradientOutcome(
                gradient=_descent_gradient(
                    gradient_outcome.gradient,
                    objective_direction=objective_direction,
                ),
                ok=True,
                reason=gradient_outcome.reason,
                error=gradient_outcome.error,
            )
        if not gradient_outcome.ok or gradient_outcome.gradient is None:
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = "smooth_gradient_failed" if gradient_source == "smooth_gradient" else "gradient_failed"
            _record_iteration(
                active_observers,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={"error": gradient_outcome.error},
                stage=stage,
                chunk=chunk,
                accepted=False,
                reason=stop_reason,
                before=before_snapshot,
                trial=None,
                after=_snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state),
                best_after=_best_snapshot(evaluator=evaluator, state=run_state),
            )
            _record_checkpoint(
                active_observers,
                "failure_gradient",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=stop_reason,
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )
            break

        context = StepContext(
            system=evaluator.system,
            evaluator=evaluator,
            state=run_state,
            evaluation=current_eval,
            gradient=gradient_outcome.gradient,
            iteration=run_state.iteration,
            stage=stage,
        )

        try:
            raw_proposal = step(context)
            proposal = raw_proposal if isinstance(raw_proposal, StepProposal) else StepProposal(raw_proposal)
            validate_controls_for_system(evaluator.system, proposal.controls)
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        except Exception as exc:  # noqa: BLE001 - recorded as a clean engine stop.
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = "proposal_failed"
            _record_iteration(
                active_observers,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={"error": f"{type(exc).__name__}: {exc}"},
                stage=stage,
                chunk=chunk,
                accepted=False,
                reason=stop_reason,
                before=before_snapshot,
                trial=None,
                after=_snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state),
                best_after=_best_snapshot(evaluator=evaluator, state=run_state),
            )
            _record_checkpoint(
                active_observers,
                "failure_proposal",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=stop_reason,
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )
            break

        try:
            trial_outcome = evaluator.try_evaluate(proposal.controls)
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        if not trial_outcome.ok or trial_outcome.evaluation is None:
            run_state.iteration += 1
            run_state.global_iteration += 1
            run_state.optimizer_state.update(_proposal_state_update(proposal, accepted=False))
            rejected_step_size = _proposal_step_size(proposal, accepted=False)
            if rejected_step_size is not None:
                run_state.step_size = rejected_step_size
            stop_reason = "nonfinite_trial"
            _record_iteration(
                active_observers,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={
                    "proposal": dict(proposal.technical),
                    "error": trial_outcome.error,
                },
                stage=stage,
                chunk=chunk,
                accepted=False,
                reason=stop_reason,
                before=before_snapshot,
                trial=None,
                after=_snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state),
                best_after=_best_snapshot(evaluator=evaluator, state=run_state),
            )
            _record_checkpoint(
                active_observers,
                "failure_trial",
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=stop_reason,
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )
            break

        try:
            accept_function = accept
            if accept_function is None:
                accept_decision = default_accept(
                    current_eval,
                    trial_outcome.evaluation,
                    proposal,
                    run_state,
                    metric=resolved_accept_metric,
                    mode=resolved_accept_mode,
                    tolerance=accept_tolerance,
                )
            else:
                accept_decision = _normalize_acceptance(
                    accept_function(current_eval, trial_outcome.evaluation, proposal, run_state)
                )
        except KeyboardInterrupt:
            close_interrupted_run()
            raise

        technical = {
            "gradient_norm": gradient_outcome.gradient.norm(),
            "evaluation_count": evaluator.evaluation_count,
            "gradient_count": evaluator.gradient_count,
            "evaluation_attempt_count": evaluator.evaluation_attempt_count,
            "gradient_attempt_count": evaluator.gradient_attempt_count,
            "counters": evaluator.counters(compact=True),
            "proposal": dict(proposal.technical),
            "acceptance": dict(accept_decision.technical),
        }

        branch_state_update = _proposal_state_update(
            proposal,
            accepted=accept_decision.accepted,
        )
        if accept_decision.optimizer_state_update:
            branch_state_update.update(dict(accept_decision.optimizer_state_update))
        branch_step_size = _proposal_step_size(
            proposal,
            accepted=accept_decision.accepted,
        )
        if accept_decision.next_step_size is not None:
            branch_step_size = float(accept_decision.next_step_size)
        run_state.optimizer_state.update(branch_state_update)
        pending_checkpoints: list[tuple[str, str]] = []
        if accept_decision.accepted:
            run_state.update_current(
                proposal.controls.copy(name=proposal.controls.name),
                trial_outcome.evaluation.metrics,
                step_size=branch_step_size,
                iteration_increment=1,
            )
            current_eval = trial_outcome.evaluation
            accepted_any = True
            improved = run_state.update_best_by_metric(
                metric=resolved_accept_metric,
                mode=resolved_accept_mode,
            )
            if checkpoint_latest:
                pending_checkpoints.append(("latest", "accepted_latest"))
            if checkpoint_accepted:
                pending_checkpoints.append(("accepted", "accepted_step"))
            if checkpoint_best and improved:
                pending_checkpoints.append((f"best_{resolved_accept_metric}", "new_best_metric"))
        else:
            run_state.iteration += 1
            run_state.global_iteration += 1
            if branch_step_size is not None:
                run_state.step_size = branch_step_size

        _record_iteration(
            active_observers,
            state=run_state,
            optimizer_name=optimizer_name,
            technical=technical,
            stage=stage,
            chunk=chunk,
            accepted=accept_decision.accepted,
            reason=accept_decision.reason,
            before=before_snapshot,
            trial=_snapshot_facts(
                trial_outcome.evaluation.metrics,
                evaluator=evaluator,
                state=run_state,
            ),
            after=_snapshot_facts(run_state.metrics, evaluator=evaluator, state=run_state),
            best_after=_best_snapshot(evaluator=evaluator, state=run_state),
        )
        for checkpoint_label, checkpoint_reason in pending_checkpoints:
            _record_checkpoint(
                active_observers,
                checkpoint_label,
                run_state,
                evaluator=evaluator,
                optimizer_name=optimizer_name,
                reason=checkpoint_reason,
                stage=stage,
                chunk=chunk,
                api_context=resolved_api_context,
                random_state=random_state,
            )
        metric_decision: StopDecision = tracker.check_metrics(run_state.metrics)
        if metric_decision.stop:
            stop_reason = metric_decision.reason or "stopped"
            break
        if proposal.stop_reason is not None:
            stop_reason = str(proposal.stop_reason)
            break

    run_state.stop_reason = stop_reason
    record_terminal_selection(stop_reason)
    _record_chunk(
        active_observers,
        optimizer_name=optimizer_name,
        chunk=chunk,
        start_iteration=start_iteration,
        end_iteration=run_state.iteration,
        start_metrics=start_metrics,
        end_metrics=run_state.metrics,
        system_params=run_state.system_params,
        stage=stage,
        accepted=accepted_any,
        reason=stop_reason,
    )
    _record_terminal(
        active_observers,
        state=run_state,
        evaluator=evaluator,
        optimizer_name=optimizer_name,
        reason=stop_reason,
        stage=stage,
        chunk=chunk,
    )
    result = OptimizerResult.from_state(
        run_state,
        stop_reason=stop_reason,
        optimizer=optimizer_name,
        trace=active_trace,
        tracking_failures=active_observers.failures,
    )
    return result
