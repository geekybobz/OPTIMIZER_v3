"""Root and constrained-cost solvers for OLSS systems."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from optimizer.core.iteration import drive, backtrack
from optimizer.core.progress import _report, _Progress
from optimizer.state import RunState
from optimizer.core.numerics import residual_step, tangent_projection

from optimizer.shooting.evaluator import (
    ShootingEvaluator,
    resolve_verification_request,
    resolve_verification_tolerance,
)
from optimizer.system_olss import ShootingResult, ShootingVariables


class _InitialEvaluationError(ValueError):
    """The starting candidate itself cannot be evaluated.

    Mirrors the OLGS engine: an invalid start raises, while failures after the
    start become named stop reasons.  Coordinators catch this one error type to
    record a failed seed or continuation stage without hiding programming errors.
    """


def _failure_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _initial_failure_result(
    system: Any,
    initial: ShootingVariables,
    exc: BaseException,
    *,
    method: str = "lm",
) -> ShootingResult:
    """Structured evidence for a candidate whose starting evaluation failed."""

    from optimizer.system_olss import get_shooting_system_spec

    spec = get_shooting_system_spec(system)
    return ShootingResult(
        variables=initial.copy(),
        residuals=np.full(int(spec["residuals"]["size"]), np.nan),
        metrics={},
        solver_converged=False,
        verified=False,
        qualified=False,
        iterations=0,
        method=str(method),
        stop_reason="initial_evaluation_failed",
        jacobian_source=str(spec["derivatives"]["jacobian"]["source"]),
        error=str(exc),
    )


def _solve_or_record(system: Any, initial: ShootingVariables, options: Mapping[str, Any]) -> ShootingResult:
    """Run one coordinator solve; an unusable start becomes a failed result."""

    try:
        return solve_shooting(system, initial, **dict(options))
    except _InitialEvaluationError as exc:
        return _initial_failure_result(system, initial, exc, method=dict(options).get("method", "lm"))




def solve_shooting(
    system: Any,
    initial: ShootingVariables,
    *,
    method: str = "lm",
    tolerance: float = 1.0e-10,
    verification_tolerance: float | None = None,
    maxiter: int = 40,
    damping: float = 1.0e-10,
    max_step_norm: float | None = None,
    line_search: bool = True,
    shrink: float = 0.5,
    max_backtracks: int = 12,
    options: Mapping[str, Any] | None = None,
    verification_options: Mapping[str, Any] | None = None,
    evaluator: ShootingEvaluator | None = None,
    previous_result: ShootingResult | None = None,
    on_iteration: Callable[[Mapping[str, Any]], None] | None = None,
    on_event: Callable[[Mapping[str, Any]], None] | None = None,
    step: Callable | None = None,
    record_limit: int = 256,
) -> ShootingResult:
    """Solve the OLSS residual equation and independently verify the result."""

    if step is not None and not callable(step):
        raise TypeError("step must be callable or None.")
    method = str(method).lower()
    if method not in {"newton", "lm", "damped"}:
        raise ValueError("method must be 'newton', 'lm', or 'damped'.")
    tolerance = float(tolerance)
    damping = float(damping)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Tolerances must be finite and nonnegative.")
    if verification_tolerance is not None and (
        not np.isfinite(float(verification_tolerance)) or float(verification_tolerance) < 0.0
    ):
        raise ValueError("Tolerances must be finite and nonnegative.")
    if damping < 0.0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and nonnegative.")
    if maxiter < 0 or max_backtracks < 1:
        raise ValueError("Invalid iteration or backtracking limit.")
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must satisfy 0 < shrink < 1.")
    if max_step_norm is not None and (
        not np.isfinite(float(max_step_norm)) or float(max_step_norm) <= 0.0
    ):
        raise ValueError("max_step_norm must be finite and positive when provided.")

    runtime = evaluator or ShootingEvaluator(system)
    verification_tolerance, tolerance_source = resolve_verification_tolerance(
        runtime.system_spec,
        working_tolerance=tolerance,
        verification_tolerance=verification_tolerance,
    )
    if previous_result is not None:
        initial = previous_result.variables
    current = initial.copy(name=initial.name)
    history: list[dict[str, Any]] = (
        [dict(row) for row in previous_result.history] if previous_result is not None else []
    )
    iteration_offset = (
        max((int(row.get("iteration", -1)) for row in history), default=-1) + 1
    )
    stop_reason = "maxiter"
    solver_converged = False

    def emit(event: str, payload: Mapping[str, Any]) -> None:
        message = {"event": event, **dict(payload)}
        if on_event is not None:
            on_event(message)
        if event == "iteration":
            if "accepted" in payload:
                progress.observe({"residual_norm": payload.get("trial_residual_norm", payload["residual_norm"])},
                                 accepted=payload["accepted"], failure=stop_reason,
                                 derivative_norm=payload.get("normal_gradient_norm"),
                                 step_norm=payload.get("scaled_step_norm"))
            _report(kind="iteration", method="solve_shooting",
                    iteration=int(payload["iteration"]) - iteration_offset + 1, total=int(maxiter) + 1,
                    metric="residual", value=float(payload["residual_norm"]))
            if on_iteration is not None:
                on_iteration(dict(payload))

    emit(
        "start",
        {
            "method": method,
            "maxiter": int(maxiter),
            "history_offset": iteration_offset,
            "resumed": previous_result is not None,
        },
    )

    bounds = runtime.bounds()
    error: str | None = None
    try:
        residuals, jacobian = runtime.residuals_and_jacobian(current, options=options)
    except Exception as exc:  # noqa: BLE001 - an invalid start raises, as in the OLGS engine.
        raise _InitialEvaluationError(
            f"initial shooting variables failed evaluation: {_failure_text(exc)}"
        ) from exc
    current_residuals = residuals
    progress = _Progress({"residual_norm": float(np.linalg.norm(residuals))}, "residual_norm", report=False)
    progress.record.limit = int(record_limit)
    if progress.record.limit < 1:
        raise ValueError("record_limit must be positive.")
    run_state = RunState.initial(current, metrics={"residual_norm": float(np.linalg.norm(residuals))}, optimizer_name=method)
    if previous_result is not None and previous_result.state is not None:
        import copy
        run_state.optimizer_state = copy.deepcopy(previous_result.state.optimizer_state)
        run_state.global_iteration = previous_result.state.global_iteration
    def advance(iteration):
        nonlocal current, current_residuals, residuals, jacobian, stop_reason, solver_converged, error
        if iteration > 0:
            try:
                residuals, jacobian = runtime.residuals_and_jacobian(current, options=options)
            except Exception as exc:
                stop_reason = 'jacobian_failed'
                error = _failure_text(exc)
                return stop_reason
            current_residuals = residuals
        norm = float(np.linalg.norm(residuals))
        max_abs = float(np.max(np.abs(residuals))) if residuals.size else 0.0
        row: dict[str, Any] = {'iteration': iteration_offset + iteration, 'residual_norm': norm, 'residual_max_abs': max_abs}
        history.append(row)
        if len(history) > record_limit:
            del history[0]
        if norm <= tolerance:
            solver_converged = True
            stop_reason = 'root_tolerance'
            emit('iteration', row)
            return stop_reason
        if iteration == int(maxiter):
            emit('iteration', row)
            return stop_reason
        from optimizer.jobs.boundary import _checkpoint
        _checkpoint('olss', current, iteration_offset + iteration)
        normal_gradient_norm = float(np.linalg.norm(current.scales * (jacobian.T @ residuals)))
        row['normal_gradient_norm'] = normal_gradient_norm
        try:
            if step is None:
                proposal = None
                delta = residual_step(jacobian, residuals, method=method, damping=damping, scales=current.scales, formulation='augmented')
            else:
                from optimizer.core.engine import StepContext, StepProposal, _proposal_state_update
                run_state.controls = current
                run_state.metrics = {"residual_norm": norm}
                context = StepContext(system, runtime, run_state, None, None, "minimize",
                                      iteration_offset + iteration, residuals=residuals.copy(), jacobian=jacobian.copy())
                raw = step(context)
                proposal = raw if isinstance(raw, StepProposal) else StepProposal(raw)
                from optimizer.system_olss.variables import _validate_shooting_spec_compatibility
                if not isinstance(proposal.controls, ShootingVariables):
                    raise TypeError("An OLSS step must propose ShootingVariables.")
                _validate_shooting_spec_compatibility(current.spec, proposal.controls.spec)
                delta = proposal.controls.flatten(copy=False) - current.flatten(copy=False)
        except Exception as exc:
            error = _failure_text(exc)
            stop_reason = 'singular_step' if step is None else 'proposal_failed'
            emit('iteration', row)
            return stop_reason
        raw_norm = float(np.linalg.norm(delta))
        if not np.all(np.isfinite(delta)):
            stop_reason = 'singular_step'
            emit('iteration', row)
            return stop_reason
        scaled_step_norm = float(np.linalg.norm(delta / current.spec.scales))
        row['scaled_step_norm'] = scaled_step_norm
        if scaled_step_norm <= np.finfo(float).eps:
            stop_reason = (str(proposal.stop_reason) if proposal is not None and proposal.stop_reason is not None
                           else 'least_squares_stationary')
            emit('iteration', row)
            return stop_reason
        if max_step_norm is not None and scaled_step_norm > float(max_step_norm):
            delta *= float(max_step_norm) / scaled_step_norm
        row['scaled_step_norm'] = float(np.linalg.norm(delta / current.scales))
        attempts = int(max_backtracks) if line_search else 1
        selected = backtrack(current, residuals, delta,
                             evaluate=lambda trial: runtime.residuals(trial, options=options),
                             merit=np.linalg.norm, attempts=attempts, shrink=shrink,
                             bounds=bounds, name="shooting_trial")
        accepted = selected.accepted
        candidate, candidate_residuals, candidate_norm = selected.value, selected.evidence, selected.merit
        scale, tried, failed_trials, trial_error = selected.scale, selected.attempted, selected.failed, selected.error
        if selected.bound_stalled:
            stop_reason = "bound_stalled"
        if proposal is not None:
            run_state.optimizer_state.update(_proposal_state_update(proposal, accepted=accepted))
        row.update({'jacobian_shape': list(jacobian.shape), 'raw_step_norm': raw_norm, 'accepted': accepted, 'accepted_scale': scale if accepted else 0.0, 'trial_residual_norm': candidate_norm, 'failed_trial_evaluations': failed_trials})
        if trial_error is not None:
            row['trial_error'] = trial_error
        emit('iteration', row)
        if not accepted:
            if stop_reason != 'bound_stalled':
                if tried and failed_trials == tried:
                    stop_reason = 'nonfinite_trial'
                    error = trial_error
                elif normal_gradient_norm <= max(tolerance, np.sqrt(np.finfo(float).eps)):
                    stop_reason = 'least_squares_stationary'
                else:
                    stop_reason = 'line_search_failed'
            return stop_reason
        current = candidate
        current_residuals = candidate_residuals
        run_state.update_current(current, {"residual_norm": candidate_norm}, iteration_increment=1)
        run_state.mark_best()
        if proposal is not None and proposal.stop_reason is not None:
            return str(proposal.stop_reason)

    stop_reason = drive(advance, maxiter=maxiter, terminal=True)

    final_residuals = current_residuals
    try:
        metrics = runtime.metrics(current, residuals=final_residuals, options=options)
    except Exception as exc:  # noqa: BLE001 - declared metrics are required for qualification.
        metrics = ShootingEvaluator._base_metrics(final_residuals)
        error = _failure_text(exc)
        if solver_converged:
            stop_reason = "evaluation_failed"
    verification = None
    verified = False
    if solver_converged and stop_reason != "evaluation_failed":
        resolved_verification_options, independent, independence_source = (
            resolve_verification_request(
                runtime.system_spec,
                working_options=options,
                verification_options=verification_options,
            )
        )
        try:
            verification = runtime.verify(
                current,
                options=resolved_verification_options,
                independent=independent,
                independence_source=independence_source,
            )
        except Exception as exc:  # noqa: BLE001 - a failed verification cannot qualify.
            verification = None
            error = _failure_text(exc)
            stop_reason = "verification_failed"
        if verification is not None:
            named_checks_passed = all(check.passed for check in verification.checks)
            verified = bool(
                verification.independent
                and
                verification.numerical_success
                and named_checks_passed
                and verification.residual_norm <= verification_tolerance
            )
            if not verified:
                stop_reason = (
                    "verification_failed"
                    if verification.independent
                    else "verification_not_independent"
                )
    source = runtime.system_spec["derivatives"]["jacobian"]["source"]
    run_state.controls = current
    run_state.metrics = dict(metrics)
    result = ShootingResult(
        variables=current,
        residuals=final_residuals,
        metrics=metrics,
        solver_converged=solver_converged,
        verified=verified,
        qualified=bool(solver_converged and verified),
        iterations=max(0, int(history[-1]["iteration"])) if history else 0,
        method=method,
        stop_reason=stop_reason,
        jacobian_source=str(source),
        verification=verification,
        counters=runtime.counters(compact=True),
        history=history,
        error=error,
        verification_tolerance=verification_tolerance,
        verification_tolerance_source=tolerance_source,
        run_record=progress.record,
        state=run_state,
    )
    progress.finish(runtime.counters(compact=True))
    run_state.stop_reason = stop_reason
    emit(
        "terminal",
        {
            "method": method,
            "stop_reason": result.stop_reason,
            "solver_converged": result.solver_converged,
            "verified": result.verified,
            "qualified": result.qualified,
            "iterations": result.iterations,
            "residual_norm": result.residual_norm,
            "counters": dict(result.counters),
            "error": result.error,
        },
    )
    return result


def solve_direct_shooting(
    system: Any,
    initial: ShootingVariables,
    *,
    tolerance: float = 1.0e-10,
    verification_tolerance: float | None = None,
    maxiter: int = 30,
    step_size: float = 0.25,
    damping: float = 1.0e-10,
    correction_maxiter: int = 12,
    options: Mapping[str, Any] | None = None,
    verification_options: Mapping[str, Any] | None = None,
) -> ShootingResult:
    """Minimize the declared selection metric along the feasible shooting set."""

    runtime = ShootingEvaluator(system)
    selection = runtime.system_spec.get("selection")
    if selection is None:
        raise TypeError("Direct shooting requires a declared selection metric.")
    if not callable(getattr(system, "cost_gradient", None)):
        raise TypeError("Direct shooting requires cost_gradient().")
    direction_sign = 1.0 if selection["direction"] == "minimize" else -1.0
    metric_id = str(selection["metric_id"])
    feasible = solve_shooting(
        system,
        initial,
        method="newton",
        tolerance=tolerance,
        verification_tolerance=verification_tolerance,
        maxiter=correction_maxiter,
        options=options,
        verification_options=verification_options,
        evaluator=runtime,
    )
    if not feasible.qualified:
        feasible.method = "direct_shooting"
        feasible.stop_reason = "initial_feasibility_failed"
        return feasible

    current = feasible.variables
    metrics = runtime.metrics(current, options=options)
    current_cost = float(metrics[metric_id])
    progress = _Progress({metric_id: current_cost}, metric_id, report=False)
    direct_history: list[dict[str, Any]] = []
    stop_reason = "maxiter"
    error: str | None = None
    def advance_cost(iteration):
        nonlocal current, current_cost, metrics, stop_reason, error
        try:
            residuals, jacobian = runtime.residuals_and_jacobian(current, options=options)
        except Exception as exc:
            stop_reason, error = ('jacobian_failed', _failure_text(exc))
            return stop_reason
        try:
            gradient = direction_sign * runtime.cost_gradient(current, options=options)
        except Exception as exc:
            stop_reason, error = ('cost_gradient_failed', _failure_text(exc))
            return stop_reason
        tangent_gradient = tangent_projection(gradient, jacobian, damping=damping, scales=current.scales)
        tangent_norm = float(np.linalg.norm(tangent_gradient / current.scales))
        row = {'iteration': iteration, 'cost': current_cost, 'residual_norm': float(np.linalg.norm(residuals)), 'tangent_gradient_norm': tangent_norm}
        direct_history.append(row)
        if len(direct_history) > progress.record.limit:
            del direct_history[0]
        _report(kind='iteration', method='solve_direct_shooting', iteration=iteration + 1, total=int(maxiter), metric='cost', value=float(current_cost))
        if tangent_norm <= tolerance:
            stop_reason = 'cost_stationary'
            return stop_reason
        accepted = False
        scale = float(step_size)
        failed_trials = 0
        for _ in range(12):
            try:
                trial = current.with_flat(current.flatten(copy=False) - scale * tangent_gradient)
                corrected = solve_shooting(system, trial, method='newton', tolerance=tolerance, verification_tolerance=verification_tolerance, maxiter=correction_maxiter, options=options, verification_options=verification_options, evaluator=runtime)
            except ValueError as exc:
                failed_trials += 1
                error = _failure_text(exc)
                scale *= 0.5
                continue
            if corrected.qualified:
                trial_metrics = runtime.metrics(corrected.variables, options=options)
                trial_cost = float(trial_metrics[metric_id])
                improved = trial_cost < current_cost if direction_sign > 0 else trial_cost > current_cost
                if improved:
                    current = corrected.variables
                    current_cost = trial_cost
                    metrics = trial_metrics
                    accepted = True
                    break
            scale *= 0.5
        row.update({'accepted': accepted, 'accepted_step': scale if accepted else 0.0, 'failed_trial_evaluations': failed_trials})
        progress.observe({metric_id: current_cost}, accepted=accepted, derivative_norm=tangent_norm, step_norm=scale * tangent_norm if accepted else 0.0)
        if not accepted:
            stop_reason = 'cost_line_search_failed'
            return stop_reason

    stop_reason = drive(advance_cost, maxiter=maxiter)

    final = solve_shooting(
        system,
        current,
        method="newton",
        tolerance=tolerance,
        verification_tolerance=verification_tolerance,
        maxiter=correction_maxiter,
        options=options,
        verification_options=verification_options,
        evaluator=runtime,
    )
    final.method = "direct_shooting"
    final.stop_reason = stop_reason if final.qualified else final.stop_reason
    final.history = (direct_history + final.history)[-progress.record.limit:]
    final.run_record = progress.record
    progress.finish(runtime.counters(compact=True))
    final.iterations = int(direct_history[-1]["iteration"]) + 1 if direct_history else 0
    final.metrics = runtime.metrics(final.variables, options=options)
    final.counters = runtime.counters(compact=True)
    if final.error is None:
        final.error = error
    return final
