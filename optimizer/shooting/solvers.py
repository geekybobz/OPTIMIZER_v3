"""Root and constrained-cost solvers for OLSS systems."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from optimizer.core.progress import _report

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


def _solve_delta(jacobian: np.ndarray, residuals: np.ndarray, method: str, damping: float) -> np.ndarray:
    if method == "newton":
        return np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
    if method in {"lm", "damped"}:
        identity = np.eye(jacobian.shape[1])
        augmented = np.vstack((jacobian, np.sqrt(damping) * identity))
        target = np.concatenate((-residuals, np.zeros(jacobian.shape[1])))
        return np.linalg.lstsq(augmented, target, rcond=None)[0]
    raise ValueError("method must be 'newton', 'lm', or 'damped'.")


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
) -> ShootingResult:
    """Solve the OLSS residual equation and independently verify the result."""

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
    for iteration in range(int(maxiter) + 1):
        if iteration > 0:
            try:
                residuals, jacobian = runtime.residuals_and_jacobian(current, options=options)
            except Exception as exc:  # noqa: BLE001 - recorded as a clean stop.
                stop_reason = "jacobian_failed"
                error = _failure_text(exc)
                break
            current_residuals = residuals
        norm = float(np.linalg.norm(residuals))
        max_abs = float(np.max(np.abs(residuals))) if residuals.size else 0.0
        row: dict[str, Any] = {
            "iteration": iteration_offset + iteration,
            "residual_norm": norm,
            "residual_max_abs": max_abs,
        }
        history.append(row)
        if norm <= tolerance:
            solver_converged = True
            stop_reason = "root_tolerance"
            emit("iteration", row)
            break
        if iteration == int(maxiter):
            emit("iteration", row)
            break

        normal_gradient_norm = float(np.linalg.norm(jacobian.T @ residuals))
        row["normal_gradient_norm"] = normal_gradient_norm
        try:
            delta = _solve_delta(jacobian, residuals, method, damping)
        except np.linalg.LinAlgError as exc:
            delta = np.full(current.spec.size, np.nan)
            error = _failure_text(exc)
        raw_norm = float(np.linalg.norm(delta))
        if not np.all(np.isfinite(delta)):
            stop_reason = "singular_step"
            emit("iteration", row)
            break
        scaled_step_norm = float(np.linalg.norm(delta / current.spec.scales))
        row["scaled_step_norm"] = scaled_step_norm
        if scaled_step_norm <= np.finfo(float).eps:
            stop_reason = "least_squares_stationary"
            emit("iteration", row)
            break
        if max_step_norm is not None and raw_norm > float(max_step_norm):
            delta *= float(max_step_norm) / raw_norm
        attempts = int(max_backtracks) if line_search else 1
        accepted = False
        scale = 1.0
        candidate_norm = norm
        candidate = current
        candidate_residuals = residuals
        tried = 0
        failed_trials = 0
        trial_error: str | None = None
        for _ in range(attempts):
            trial_values = current.flatten(copy=False) + scale * delta
            if bounds is not None:
                trial_values = np.clip(trial_values, bounds[0], bounds[1])
                if np.array_equal(trial_values, current.flatten(copy=False)):
                    stop_reason = "bound_stalled"
                    break
            tried += 1
            try:
                trial = current.with_flat(trial_values, name="shooting_trial")
                trial_residuals = runtime.residuals(trial, options=options)
            except Exception as exc:  # noqa: BLE001 - a failed trial is a rejection, as in the OLGS engine.
                failed_trials += 1
                trial_error = _failure_text(exc)
                scale *= shrink
                continue
            trial_norm = float(np.linalg.norm(trial_residuals))
            if trial_norm < candidate_norm:
                candidate = trial
                candidate_norm = trial_norm
                candidate_residuals = trial_residuals
                accepted = True
                break
            scale *= shrink
        row.update(
            {
                "jacobian_shape": list(jacobian.shape),
                "raw_step_norm": raw_norm,
                "accepted": accepted,
                "accepted_scale": scale if accepted else 0.0,
                "trial_residual_norm": candidate_norm,
                "failed_trial_evaluations": failed_trials,
            }
        )
        if trial_error is not None:
            row["trial_error"] = trial_error
        emit("iteration", row)
        if not accepted:
            if stop_reason != "bound_stalled":
                if tried and failed_trials == tried:
                    stop_reason = "nonfinite_trial"
                    error = trial_error
                elif normal_gradient_norm <= max(tolerance, np.sqrt(np.finfo(float).eps)):
                    stop_reason = "least_squares_stationary"
                else:
                    stop_reason = "line_search_failed"
            break
        current = candidate
        current_residuals = candidate_residuals

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
    result = ShootingResult(
        variables=current,
        residuals=final_residuals,
        metrics=metrics,
        solver_converged=solver_converged,
        verified=verified,
        qualified=bool(solver_converged and verified),
        iterations=max(0, len(history) - 1),
        method=method,
        stop_reason=stop_reason,
        jacobian_source=str(source),
        verification=verification,
        counters=runtime.counters(compact=True),
        history=history,
        error=error,
        verification_tolerance=verification_tolerance,
        verification_tolerance_source=tolerance_source,
    )
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
    direct_history: list[dict[str, Any]] = []
    stop_reason = "maxiter"
    error: str | None = None
    for iteration in range(int(maxiter)):
        try:
            residuals, jacobian = runtime.residuals_and_jacobian(current, options=options)
        except Exception as exc:  # noqa: BLE001 - recorded as a clean stop.
            stop_reason, error = "jacobian_failed", _failure_text(exc)
            break
        try:
            gradient = direction_sign * runtime.cost_gradient(current, options=options)
        except Exception as exc:  # noqa: BLE001 - recorded as a clean stop.
            stop_reason, error = "cost_gradient_failed", _failure_text(exc)
            break
        gram = jacobian @ jacobian.T + float(damping) * np.eye(jacobian.shape[0])
        multiplier = np.linalg.lstsq(gram, jacobian @ gradient, rcond=None)[0]
        tangent_gradient = gradient - jacobian.T @ multiplier
        tangent_norm = float(np.linalg.norm(tangent_gradient))
        row = {
            "iteration": iteration,
            "cost": current_cost,
            "residual_norm": float(np.linalg.norm(residuals)),
            "tangent_gradient_norm": tangent_norm,
        }
        direct_history.append(row)
        _report(kind="iteration", method="solve_direct_shooting", iteration=iteration + 1,
                total=int(maxiter), metric="cost", value=float(current_cost))
        if tangent_norm <= tolerance:
            stop_reason = "cost_stationary"
            break
        accepted = False
        scale = float(step_size)
        failed_trials = 0
        for _ in range(12):
            try:
                trial = current.with_flat(current.flatten(copy=False) - scale * tangent_gradient)
                corrected = solve_shooting(
                    system,
                    trial,
                    method="newton",
                    tolerance=tolerance,
                    verification_tolerance=verification_tolerance,
                    maxiter=correction_maxiter,
                    options=options,
                    verification_options=verification_options,
                    evaluator=runtime,
                )
            except ValueError as exc:  # non-finite trial or a trial start that cannot be evaluated
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
        row.update(
            {
                "accepted": accepted,
                "accepted_step": scale if accepted else 0.0,
                "failed_trial_evaluations": failed_trials,
            }
        )
        if not accepted:
            stop_reason = "cost_line_search_failed"
            break

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
    final.history = direct_history + final.history
    final.metrics = runtime.metrics(final.variables, options=options)
    final.counters = runtime.counters(compact=True)
    if final.error is None:
        final.error = error
    return final
