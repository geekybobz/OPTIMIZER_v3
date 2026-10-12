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
- Numerical result aggregates retain bounded diagnostic evidence.
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
- Interaction recording belongs to the frontend, outside this numerical engine.
- all system calls go through ``SystemEvaluator`` validation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4
import numpy as np
from optimizer.core.progress import _Progress
from optimizer.controls import Controls
from optimizer.core.evaluate import EvaluationOutcome, GradientOutcome, SystemEvaluator
from optimizer.core.stopping import StopDecision, StopTracker, StoppingConfig
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
    objective_direction: str
    iteration: int
    stage: str | None = None

    def try_descent_gradient(self, controls: Controls) -> GradientOutcome:
        """Return a trial gradient in this method's descent convention.

        ``SystemEvaluator`` exposes the native derivative of the system-owned
        objective.  Step functions are implemented as minimizers, so methods
        that need a gradient away from the current controls must use this
        helper instead of bypassing the engine's declared-direction conversion.
        """
        outcome = self.evaluator.try_gradient(controls)
        if not outcome.ok or outcome.gradient is None:
            return outcome
        return GradientOutcome(gradient=_descent_gradient(outcome.gradient, objective_direction=self.objective_direction), ok=True, reason=outcome.reason, error=outcome.error)

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

def _scalar_metric(metrics: Mapping[str, Any], key: str) -> float:
    """Return a metric as a scalar float for accept/reject logic."""
    if key not in metrics:
        raise KeyError(f'metrics do not include {key!r}.')
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f'metric {key!r} must be scalar for acceptance checks.')
    return float(value)

def _descent_gradient(gradient: Controls, *, objective_direction: str) -> Controls:
    """Convert the declared objective derivative into a descent direction.

    Systems always return the analytical derivative of their declared objective
    metric.  Concrete first-order methods are written as minimizers, so a
    maximization problem must receive the negative derivative as its descent
    direction.  Keeping that conversion here makes it a system-contract
    decision, rather than a campaign or optimizer-name convention.
    """
    if objective_direction == 'minimize':
        return gradient
    if objective_direction == 'maximize':
        return -gradient
    raise ValueError(f'unsupported declared objective direction: {objective_direction!r}')

def default_accept(current: Evaluation, trial: Evaluation, proposal: StepProposal, state: RunState, *, metric: str='J', mode: str='min', tolerance: float=0.0) -> AcceptanceDecision:
    """Accept trial controls when the selected metric does not get worse."""
    del proposal, state
    if mode not in {'min', 'max'}:
        raise ValueError("mode must be 'min' or 'max'.")
    current_value = _scalar_metric(current.metrics, metric)
    trial_value = _scalar_metric(trial.metrics, metric)
    tolerance = float(tolerance)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError('acceptance tolerance must be finite and >= 0.')
    if mode == 'min':
        accepted = trial_value <= current_value + tolerance
        improvement = current_value - trial_value
    else:
        accepted = trial_value >= current_value - tolerance
        improvement = trial_value - current_value
    reason = 'accepted' if accepted else 'rejected_worse_metric'
    return AcceptanceDecision(accepted=accepted, reason=reason, technical={'accept_metric': metric, 'accept_mode': mode, 'current_value': current_value, 'trial_value': trial_value, 'improvement': improvement, 'tolerance': tolerance})

def _normalize_acceptance(decision: AcceptanceDecision | bool) -> AcceptanceDecision:
    """Convert custom accept-function output into an ``AcceptanceDecision``."""
    if isinstance(decision, AcceptanceDecision):
        return decision
    if isinstance(decision, bool):
        return AcceptanceDecision(accepted=decision, reason='accepted' if decision else 'rejected', technical={})
    raise TypeError('accept function must return AcceptanceDecision or bool.')

def _proposal_state_update(proposal: StepProposal, *, accepted: bool) -> dict[str, Any]:
    """Return the optimizer-private state update for the decision branch.

    ``optimizer_state`` is the Phase 5 legacy field and still means "apply this
    update after the proposal is processed."  Real optimizers need finer control:
    Adam moments, momentum velocity, and future L-BFGS histories should usually
    advance only after an accepted control move.  The branch-specific fields let a
    method express that without forcing the engine to understand the algorithm.
    """
    branch_state = proposal.optimizer_state_on_accept if accepted else proposal.optimizer_state_on_reject
    if branch_state is not None:
        return dict(branch_state)
    return dict(proposal.optimizer_state or {})

def _proposal_step_size(proposal: StepProposal, *, accepted: bool) -> float | None:
    """Return the step-size value to retain for the decision branch."""
    branch_step_size = proposal.step_size_on_accept if accepted else proposal.step_size_on_reject
    if branch_step_size is not None:
        return float(branch_step_size)
    if proposal.step_size is not None:
        return float(proposal.step_size)
    return None

def _project_proposal_to_bounds(proposal: StepProposal, *, lower: np.ndarray, upper: np.ndarray) -> StepProposal:
    """Project one proposed control onto system-declared elementwise bounds."""
    matrix = proposal.controls.as_matrix(copy=False)
    clipped = np.clip(matrix, lower, upper)
    if np.array_equal(matrix, clipped):
        return proposal
    controls = Controls.from_matrix(proposal.controls.spec, clipped, name=proposal.controls.name, meta=proposal.controls.meta)
    technical = dict(proposal.technical)
    technical.update({'bounds_projected': True, 'bounds_projection_norm': float(np.linalg.norm(clipped - matrix))})
    return replace(proposal, controls=controls, technical=technical)

def _stationary_reason(evaluator: SystemEvaluator, controls: Controls, *, curvature_tolerance: float) -> tuple[str, dict[str, Any]]:
    """Distinguish a sampled non-minimum stationary point when HVP is available."""
    if not evaluator.capabilities.get('has_hvp', False):
        return ('gradient_converged', {'curvature_checked': False})
    from optimizer.core.curvature import curvature_evidence
    evidence = curvature_evidence(evaluator.system, controls, directions=min(16, controls.spec.size))
    evidence['curvature_checked'] = True
    if float(evidence['minimum_curvature']) < -float(curvature_tolerance):
        return ('stationary_nonminimum', evidence)
    return ('gradient_converged', evidence)

def _extract_system_params(system: Any, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return system params for state/log context without imposing a class."""
    if explicit is not None:
        return dict(explicit)
    params = getattr(system, 'params', None)
    if isinstance(params, Mapping):
        return dict(params)
    if params is not None and hasattr(params, '__dict__'):
        return dict(vars(params))
    return {}

def _start_controls(controls: Controls | None, state: RunState | None, warmstart: WarmStartState | None) -> Controls:
    """Choose the controls that start this chunk."""
    if state is not None:
        return state.controls.copy(name=state.controls.name)
    if warmstart is not None:
        return warmstart.controls.copy(name=warmstart.controls.name)
    if controls is not None:
        return controls.copy(name=controls.name)
    raise ValueError('run_chunk requires controls, state, or warmstart.')

def _start_step_size(explicit: float | None, state: RunState | None, warmstart: WarmStartState | None) -> float | None:
    """Choose the step-size estimate carried into this chunk."""
    if explicit is not None:
        return float(explicit)
    if state is not None and state.step_size is not None:
        return float(state.step_size)
    if warmstart is not None and warmstart.step_size is not None:
        return float(warmstart.step_size)
    return None

def _start_optimizer_state(state: RunState | None, warmstart: WarmStartState | None, *, optimizer_name: str) -> dict[str, Any]:
    """Choose optimizer-private state carried into this chunk."""
    if state is not None and state.optimizer_name == str(optimizer_name):
        return dict(state.optimizer_state)
    if warmstart is not None and warmstart.optimizer_state is not None:
        return dict(warmstart.optimizer_state)
    return {}

def _build_state(*, controls: Controls, metrics: Mapping[str, Any], optimizer_name: str, step_size: float | None, optimizer_state: Mapping[str, Any], system_params: Mapping[str, Any], previous_state: RunState | None, warmstart: WarmStartState | None) -> RunState:
    """Create the mutable chunk state used by the engine loop."""
    state = RunState.initial(controls, metrics=metrics, optimizer_name=optimizer_name, step_size=step_size, system_params=system_params)
    state.optimizer_state = dict(optimizer_state)
    if previous_state is not None:
        state.global_iteration = int(previous_state.global_iteration)
        if previous_state.best_controls is not None:
            state.best_controls = previous_state.best_controls.copy(name='best')
        if previous_state.best_metrics is not None:
            state.best_metrics = dict(previous_state.best_metrics)
    return state

def _terminal_status(reason: str) -> str:
    """Classify engine stops without claiming numerical optimality."""
    return classify_terminal_status(reason)

def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, '__qualname__', getattr(value, '__name__', type(value).__name__)))

def run_chunk(system: Any, controls: Controls | None=None, *, step: StepFunction, optimizer_name: str, maxiter: int, state: RunState | None=None, warmstart: WarmStartState | None=None, step_size: float | None=None, stopping: StoppingConfig | None=None, target_value: float | None=None, target_metric: str | None=None, target_mode: str | None=None, stall_patience: int | None=None, stall_tolerance: float=0.0, stall_metric: str | None=None, stall_mode: str | None=None, gradient_tolerance: float | None=None, stationary_curvature_tolerance: float=1e-10, accept: AcceptFunction | None=None, accept_metric: str | None=None, accept_mode: str | None=None, accept_tolerance: float=0.0, system_params: Mapping[str, Any] | None=None, stage: str | None=None, use_cache: bool=True, enforce_bounds: bool=True, failed_trial_retries: int=3, failed_trial_shrink: float=0.5, gradient_source: str='gradient', prefer_value_and_gradient: bool=True) -> OptimizerResult:
    """Run one optimizer chunk using a method-specific proposal function."""
    if not callable(step):
        raise TypeError('step must be callable.')
    if gradient_source not in {'gradient', 'smooth_gradient'}:
        raise ValueError("gradient_source must be 'gradient' or 'smooth_gradient'.")
    if gradient_tolerance is not None and (not np.isfinite(float(gradient_tolerance)) or float(gradient_tolerance) < 0.0):
        raise ValueError('gradient_tolerance must be finite and nonnegative, or None.')
    if not np.isfinite(float(stationary_curvature_tolerance)) or float(stationary_curvature_tolerance) < 0.0:
        raise ValueError('stationary_curvature_tolerance must be finite and nonnegative.')
    if int(failed_trial_retries) < 1:
        raise ValueError('failed_trial_retries must be >= 1.')
    if not 0.0 < float(failed_trial_shrink) < 1.0:
        raise ValueError('failed_trial_shrink must satisfy 0 < value < 1.')
    evaluator = SystemEvaluator(system, use_cache=use_cache)
    objective = evaluator.system_spec['objective']
    objective_metric = str(objective['metric_id'])
    objective_direction = str(objective['direction'])
    resolved_target_metric = objective_metric if target_metric is None else str(target_metric)
    resolved_target_mode = ('le' if objective_direction == 'minimize' else 'ge') if target_mode is None else str(target_mode)
    resolved_stall_metric = objective_metric if stall_metric is None else str(stall_metric)
    resolved_stall_mode = ('min' if objective_direction == 'minimize' else 'max') if stall_mode is None else str(stall_mode)
    resolved_accept_metric = objective_metric if accept_metric is None else str(accept_metric)
    resolved_accept_mode = ('min' if objective_direction == 'minimize' else 'max') if accept_mode is None else str(accept_mode)
    if gradient_source == 'smooth_gradient' and objective_direction == 'maximize':
        raise ValueError('smooth_gradient is currently defined only for minimization; a maximizing system must provide a full analytical gradient.')
    active_controls = _start_controls(controls, state, warmstart)
    validate_controls_for_system(evaluator.system, active_controls)
    bounds_arrays: tuple[np.ndarray, np.ndarray] | None = None
    initial_bounds_projection_norm = 0.0
    if bool(enforce_bounds) and evaluator.capabilities.get('has_control_bounds', False):
        bounds = evaluator.control_bounds()
        bounds_arrays = (bounds.lower.as_matrix(copy=False), bounds.upper.as_matrix(copy=False))
        active_matrix = active_controls.as_matrix(copy=False)
        projected_matrix = np.clip(active_matrix, bounds_arrays[0], bounds_arrays[1])
        if not np.array_equal(active_matrix, projected_matrix):
            initial_bounds_projection_norm = float(np.linalg.norm(projected_matrix - active_matrix))
            active_controls = Controls.from_matrix(active_controls.spec, projected_matrix, name=active_controls.name, meta=active_controls.meta)
    params = _extract_system_params(evaluator.system, system_params)
    if stopping is None:
        stop_config = StoppingConfig(maxiter=maxiter, target_value=target_value, target_metric=resolved_target_metric, target_mode=resolved_target_mode, stall_patience=stall_patience, stall_tolerance=stall_tolerance, stall_metric=resolved_stall_metric, stall_mode=resolved_stall_mode)
    else:
        stop_config = replace(stopping, target_metric=resolved_target_metric if stopping.target_metric is None else stopping.target_metric, target_mode=resolved_target_mode if stopping.target_mode is None else stopping.target_mode, stall_metric=resolved_stall_metric if stopping.stall_metric is None else stopping.stall_metric, stall_mode=resolved_stall_mode if stopping.stall_mode is None else stopping.stall_mode)
    prefetched_gradient: Controls | None = None
    initial_error: str | None = None
    initial_evaluation: Evaluation | None = None
    use_combined = gradient_source == 'gradient' and bool(prefer_value_and_gradient) and evaluator.capabilities.get('has_value_and_gradient', False)
    if use_combined:
        combined_outcome = evaluator.try_value_and_gradient(active_controls)
        if combined_outcome.ok and combined_outcome.value_gradient is not None:
            initial_evaluation = Evaluation.from_metrics(active_controls.copy(name=active_controls.name), combined_outcome.value_gradient.metrics, objective_metric=objective_metric)
            prefetched_gradient = combined_outcome.value_gradient.gradient
        else:
            initial_error = combined_outcome.error
    else:
        initial_outcome = evaluator.try_evaluate(active_controls)
        initial_evaluation = initial_outcome.evaluation
        initial_error = initial_outcome.error
    if initial_evaluation is None:
        raise ValueError(f'initial controls failed evaluation: {initial_error}')
    run_state = _build_state(controls=active_controls, metrics=initial_evaluation.metrics, optimizer_name=optimizer_name, step_size=_start_step_size(step_size, state, warmstart), optimizer_state=_start_optimizer_state(state, warmstart, optimizer_name=optimizer_name), system_params=params, previous_state=state, warmstart=warmstart)
    progress = _Progress(run_state.metrics, objective_metric, method=optimizer_name, budget=maxiter)
    start_metrics = dict(run_state.metrics)
    start_iteration = int(run_state.iteration)
    tracker = StopTracker(stop_config, initial_metrics=run_state.metrics)
    initial_stop = tracker.check_initial_metrics(run_state.metrics)
    if initial_stop.stop:
        run_state.stop_reason = initial_stop.reason
        result = OptimizerResult.from_state(run_state, stop_reason=initial_stop.reason or 'stopped', optimizer=optimizer_name, objective_metric=objective_metric, diagnostics=progress.finish(evaluator.counters()))
        return result
    current_eval = initial_evaluation
    stop_reason = 'maxiter'
    accepted_any = False
    consecutive_failed_trials = 0
    prefetch_trial_gradient = bool(use_combined and getattr(accept, 'accepts_every_finite_trial', False))

    def close_interrupted_run() -> None:
        """Close passive evidence using the last valid state, without new numerics."""
        run_state.stop_reason = 'interrupted'
    while True:
        budget_decision = tracker.check_before_iteration(run_state.iteration)
        if budget_decision.stop:
            stop_reason = budget_decision.reason or 'maxiter'
            break
        from optimizer.jobs.boundary import _checkpoint
        _checkpoint('olgs', run_state, run_state.global_iteration)
        try:
            if prefetched_gradient is not None:
                gradient_outcome = GradientOutcome(gradient=prefetched_gradient, ok=True, reason='ok')
                prefetched_gradient = None
            elif gradient_source == 'smooth_gradient':
                gradient_outcome = evaluator.try_smooth_gradient(run_state.controls)
            else:
                gradient_outcome = evaluator.try_gradient(run_state.controls)
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        if gradient_outcome.ok and gradient_outcome.gradient is not None:
            gradient_outcome = GradientOutcome(gradient=_descent_gradient(gradient_outcome.gradient, objective_direction=objective_direction), ok=True, reason=gradient_outcome.reason, error=gradient_outcome.error)
        if not gradient_outcome.ok or gradient_outcome.gradient is None:
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = 'smooth_gradient_failed' if gradient_source == 'smooth_gradient' else 'gradient_failed'
            progress.observe(run_state.metrics, accepted=False, failure=stop_reason)
            break
        gradient_norm = gradient_outcome.gradient.norm()
        if gradient_tolerance is not None and gradient_norm <= float(gradient_tolerance):
            stop_reason, curvature = _stationary_reason(evaluator, run_state.controls, curvature_tolerance=stationary_curvature_tolerance)
            run_state.optimizer_state.update({'gradient_norm': gradient_norm, 'stationary_evidence': curvature})
            break
        context = StepContext(system=evaluator.system, evaluator=evaluator, state=run_state, evaluation=current_eval, gradient=gradient_outcome.gradient, objective_direction=objective_direction, iteration=run_state.iteration, stage=stage)
        try:
            raw_proposal = step(context)
            proposal = raw_proposal if isinstance(raw_proposal, StepProposal) else StepProposal(raw_proposal)
            validate_controls_for_system(evaluator.system, proposal.controls)
            if bounds_arrays is not None:
                proposal = _project_proposal_to_bounds(proposal, lower=bounds_arrays[0], upper=bounds_arrays[1])
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        except Exception as exc:
            run_state.iteration += 1
            run_state.global_iteration += 1
            stop_reason = 'proposal_failed'
            progress.observe(run_state.metrics, accepted=False, failure=stop_reason)
            break
        trial_gradient: Controls | None = None
        try:
            if prefetch_trial_gradient:
                combined_trial = evaluator.try_value_and_gradient(proposal.controls)
                if combined_trial.ok and combined_trial.value_gradient is not None:
                    trial_gradient = combined_trial.value_gradient.gradient
                    trial_outcome = EvaluationOutcome(evaluation=Evaluation.from_metrics(proposal.controls.copy(name=proposal.controls.name), combined_trial.value_gradient.metrics, objective_metric=objective_metric), ok=True, reason='ok')
                else:
                    trial_outcome = EvaluationOutcome(evaluation=None, ok=False, reason=combined_trial.reason, error=combined_trial.error)
            else:
                trial_outcome = evaluator.try_evaluate(proposal.controls)
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        if not trial_outcome.ok or trial_outcome.evaluation is None:
            consecutive_failed_trials += 1
            run_state.iteration += 1
            run_state.global_iteration += 1
            run_state.optimizer_state.update(_proposal_state_update(proposal, accepted=False))
            rejected_step_size = _proposal_step_size(proposal, accepted=False)
            current_step_size = run_state.step_size
            if current_step_size is not None and (rejected_step_size is None or rejected_step_size >= current_step_size):
                rejected_step_size = current_step_size * float(failed_trial_shrink)
            if rejected_step_size is not None:
                run_state.step_size = rejected_step_size
            terminal_failure = consecutive_failed_trials >= int(failed_trial_retries)
            iteration_reason = 'nonfinite_trial' if terminal_failure else 'trial_evaluation_rejected'
            progress.observe(run_state.metrics, accepted=False, failure=iteration_reason)
            if terminal_failure:
                stop_reason = 'nonfinite_trial'
                break
            continue
        consecutive_failed_trials = 0
        try:
            accept_function = accept
            if accept_function is None:
                accept_decision = default_accept(current_eval, trial_outcome.evaluation, proposal, run_state, metric=resolved_accept_metric, mode=resolved_accept_mode, tolerance=accept_tolerance)
            else:
                accept_decision = _normalize_acceptance(accept_function(current_eval, trial_outcome.evaluation, proposal, run_state))
        except KeyboardInterrupt:
            close_interrupted_run()
            raise
        branch_state_update = _proposal_state_update(proposal, accepted=accept_decision.accepted)
        if accept_decision.optimizer_state_update:
            branch_state_update.update(dict(accept_decision.optimizer_state_update))
        branch_step_size = _proposal_step_size(proposal, accepted=accept_decision.accepted)
        if accept_decision.next_step_size is not None:
            branch_step_size = float(accept_decision.next_step_size)
        run_state.optimizer_state.update(branch_state_update)
        if accept_decision.accepted:
            run_state.update_current(proposal.controls.copy(name=proposal.controls.name), trial_outcome.evaluation.metrics, step_size=branch_step_size, iteration_increment=1)
            current_eval = trial_outcome.evaluation
            if trial_gradient is not None:
                prefetched_gradient = trial_gradient
            accepted_any = True
            improved = run_state.update_best_by_metric(metric=resolved_accept_metric, mode=resolved_accept_mode)
        else:
            run_state.iteration += 1
            run_state.global_iteration += 1
            if branch_step_size is not None:
                run_state.step_size = branch_step_size
        progress.observe(run_state.metrics, accepted=accept_decision.accepted, failure=accept_decision.reason)
        metric_decision: StopDecision = tracker.check_metrics(run_state.metrics)
        if metric_decision.stop:
            stop_reason = metric_decision.reason or 'stopped'
            break
        if proposal.stop_reason is not None:
            stop_reason = str(proposal.stop_reason)
            break
    run_state.stop_reason = stop_reason
    result = OptimizerResult.from_state(run_state, stop_reason=stop_reason, optimizer=optimizer_name, objective_metric=objective_metric, diagnostics=progress.finish(evaluator.counters()))
    return result
