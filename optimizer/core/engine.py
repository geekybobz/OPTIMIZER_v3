"""Shared optimizer chunk engine.

Why this file exists
--------------------
OPTIMIZER v3 should not let every algorithm carry its own driver loop.  Adam,
momentum, gradient descent, line search, and later trust-region variants all need the
same outer mechanics: evaluate current controls, request an analytical gradient,
build a trial control, evaluate the trial, accept or reject it, update run state,
record technical logs, create rollback checkpoints, and stop consistently.

This module owns that common loop.  A concrete optimizer supplies only a proposal
function that maps ``StepContext`` to ``StepProposal``.  The engine handles the rest.

How it fits the architecture
----------------------------
- ``controls.py`` defines the object being moved.
- ``system.py`` defines how metrics and gradients are produced analytically.
- ``state.py`` stores mutable chunk state and warmstart handoff state.
- ``result.py`` standardizes the returned result.
- ``logs/trace.py`` receives iteration/chunk records and checkpoints.
- future optimizer modules become small method-specific proposal builders.

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
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.stopping import StopDecision, StopTracker, StoppingConfig
from optimizer.logs.trace import Trace
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.system import OptimizerSystem, validate_controls_for_system


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


@dataclass(frozen=True)
class AcceptanceDecision:
    """Engine decision on whether a trial replaces the current controls."""

    accepted: bool
    reason: str
    technical: dict[str, Any] = field(default_factory=dict)


StepFunction = Callable[[StepContext], StepProposal | Controls]
AcceptFunction = Callable[[Evaluation, Evaluation, StepProposal, RunState], AcceptanceDecision | bool]


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


def _record_checkpoint(
    trace: Trace | None,
    label: str,
    state: RunState,
    *,
    stage: str | None,
) -> None:
    """Create a trace checkpoint when tracing is enabled."""

    if trace is None:
        return
    trace.checkpoint(label, state.controls, state, stage=stage)


def _record_iteration(
    trace: Trace | None,
    *,
    state: RunState,
    optimizer_name: str,
    technical: Mapping[str, Any],
    stage: str | None,
    accepted: bool | None,
    reason: str | None,
) -> None:
    """Append one iteration record when tracing is enabled."""

    if trace is None:
        return
    trace.record_iteration(
        optimizer=optimizer_name,
        iteration=state.iteration,
        global_iteration=state.global_iteration,
        metrics=state.metrics,
        system_params=state.system_params,
        technical=technical,
        stage=stage,
        accepted=accepted,
        reason=reason,
    )


def _record_chunk(
    trace: Trace | None,
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
    """Append one chunk record when tracing is enabled."""

    if trace is None:
        return
    trace.record_chunk(
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
    step_size: float | None = None,
    stopping: StoppingConfig | None = None,
    target_value: float | None = None,
    target_metric: str = "J",
    stall_patience: int | None = None,
    stall_tolerance: float = 0.0,
    accept: AcceptFunction | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    system_params: Mapping[str, Any] | None = None,
    stage: str | None = None,
    chunk: int = 0,
    use_cache: bool = True,
    checkpoint_start: bool = True,
    checkpoint_latest: bool = True,
    checkpoint_accepted: bool = True,
    checkpoint_best: bool = True,
) -> OptimizerResult:
    """Run one optimizer chunk using a method-specific proposal function."""

    if not callable(step):
        raise TypeError("step must be callable.")

    evaluator = SystemEvaluator(system, use_cache=use_cache)
    active_trace = _ensure_trace(trace, create_trace=create_trace)
    active_controls = _start_controls(controls, state, warmstart)
    validate_controls_for_system(evaluator.system, active_controls)

    initial_outcome = evaluator.try_evaluate(active_controls)
    if not initial_outcome.ok or initial_outcome.evaluation is None:
        raise ValueError(f"initial controls failed evaluation: {initial_outcome.error}")

    params = _extract_system_params(evaluator.system, system_params)
    run_state = _build_state(
        controls=active_controls,
        metrics=initial_outcome.evaluation.metrics,
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

    stop_config = stopping or StoppingConfig(
        maxiter=maxiter,
        target_value=target_value,
        target_metric=target_metric,
        stall_patience=stall_patience,
        stall_tolerance=stall_tolerance,
    )
    tracker = StopTracker(stop_config, initial_metrics=run_state.metrics)

    if checkpoint_start:
        _record_checkpoint(active_trace, "chunk_start", run_state, stage=stage)

    initial_stop = tracker.check_initial_metrics(run_state.metrics)
    if initial_stop.stop:
        run_state.stop_reason = initial_stop.reason
        _record_chunk(
            active_trace,
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
        return OptimizerResult.from_state(
            run_state,
            stop_reason=initial_stop.reason or "stopped",
            optimizer=optimizer_name,
            trace=active_trace,
        )

    current_eval = initial_outcome.evaluation
    stop_reason = "maxiter"
    accepted_any = False

    while True:
        budget_decision = tracker.check_before_iteration(run_state.iteration)
        if budget_decision.stop:
            stop_reason = budget_decision.reason or "maxiter"
            break

        gradient_outcome = evaluator.try_gradient(run_state.controls)
        if not gradient_outcome.ok or gradient_outcome.gradient is None:
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = "gradient_failed"
            _record_iteration(
                active_trace,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={"error": gradient_outcome.error},
                stage=stage,
                accepted=False,
                reason=stop_reason,
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
        except Exception as exc:  # noqa: BLE001 - recorded as a clean engine stop.
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = "proposal_failed"
            _record_iteration(
                active_trace,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={"error": f"{type(exc).__name__}: {exc}"},
                stage=stage,
                accepted=False,
                reason=stop_reason,
            )
            break

        trial_outcome = evaluator.try_evaluate(proposal.controls)
        if not trial_outcome.ok or trial_outcome.evaluation is None:
            run_state.iteration += 1
            run_state.global_iteration += 1
            run_state.optimizer_state.update(_proposal_state_update(proposal, accepted=False))
            rejected_step_size = _proposal_step_size(proposal, accepted=False)
            if rejected_step_size is not None:
                run_state.step_size = rejected_step_size
            stop_reason = "nonfinite_trial"
            _record_iteration(
                active_trace,
                state=run_state,
                optimizer_name=optimizer_name,
                technical={
                    "proposal": dict(proposal.technical),
                    "error": trial_outcome.error,
                },
                stage=stage,
                accepted=False,
                reason=stop_reason,
            )
            break

        accept_function = accept
        if accept_function is None:
            accept_decision = default_accept(
                current_eval,
                trial_outcome.evaluation,
                proposal,
                run_state,
                metric=accept_metric,
                mode=accept_mode,
                tolerance=accept_tolerance,
            )
        else:
            accept_decision = _normalize_acceptance(
                accept_function(current_eval, trial_outcome.evaluation, proposal, run_state)
            )

        technical = {
            "gradient_norm": gradient_outcome.gradient.norm(),
            "evaluation_count": evaluator.evaluation_count,
            "gradient_count": evaluator.gradient_count,
            "proposal": dict(proposal.technical),
            "acceptance": dict(accept_decision.technical),
        }

        branch_state_update = _proposal_state_update(
            proposal,
            accepted=accept_decision.accepted,
        )
        branch_step_size = _proposal_step_size(
            proposal,
            accepted=accept_decision.accepted,
        )
        run_state.optimizer_state.update(branch_state_update)
        if accept_decision.accepted:
            run_state.update_current(
                proposal.controls.copy(name=proposal.controls.name),
                trial_outcome.evaluation.metrics,
                step_size=branch_step_size,
                iteration_increment=1,
            )
            current_eval = trial_outcome.evaluation
            accepted_any = True
            improved = run_state.update_best_by_metric(metric=accept_metric, mode=accept_mode)
            if checkpoint_latest:
                _record_checkpoint(active_trace, "latest", run_state, stage=stage)
            if checkpoint_accepted:
                _record_checkpoint(active_trace, "accepted", run_state, stage=stage)
            if checkpoint_best and improved:
                _record_checkpoint(active_trace, f"best_{accept_metric}", run_state, stage=stage)
        else:
            run_state.iteration += 1
            run_state.global_iteration += 1
            if branch_step_size is not None:
                run_state.step_size = branch_step_size

        _record_iteration(
            active_trace,
            state=run_state,
            optimizer_name=optimizer_name,
            technical=technical,
            stage=stage,
            accepted=accept_decision.accepted,
            reason=accept_decision.reason,
        )

        metric_decision: StopDecision = tracker.check_metrics(run_state.metrics)
        if metric_decision.stop:
            stop_reason = metric_decision.reason or "stopped"
            break

    run_state.stop_reason = stop_reason
    _record_chunk(
        active_trace,
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
    return OptimizerResult.from_state(
        run_state,
        stop_reason=stop_reason,
        optimizer=optimizer_name,
        trace=active_trace,
    )
