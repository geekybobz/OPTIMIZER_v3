"""Pseudo-arclength continuation for OLSS systems."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.core.progress import _report
from optimizer.core.numerics import jacobian_decomposition
from optimizer.core.coordinator import ContinuationCoordinator, StepSizeController, correct_root

from optimizer.shooting.evaluator import (
    ShootingEvaluator,
    resolve_verification_request,
    resolve_verification_tolerance,
)
from optimizer.system_olss import (
    ContinuationPoint,
    ParameterContinuationPoint,
    ParameterContinuationResult,
    ShootingContinuationResult,
    ShootingVariables,
)


def continue_shooting(
    system: Any,
    previous: ShootingVariables,
    current: ShootingVariables,
    *,
    steps: int,
    step_size: float,
    tolerance: float = 1.0e-10,
    verification_tolerance: float | None = None,
    max_corrector_iterations: int = 15,
    options: Mapping[str, Any] | None = None,
    verification_options: Mapping[str, Any] | None = None,
    min_step: float | None = None,
    max_step: float | None = None,
    shrink: float = 0.5,
    grow: float = 1.5,
    fast_corrector_iterations: int = 4,
    target_block: str | None = None,
    target_value: float | None = None,
    target_tolerance: float = 1.0e-10,
) -> ShootingContinuationResult:
    """Follow an OLSS branch with a secant predictor and arclength corrector."""

    if previous.spec != current.spec:
        raise ValueError("Continuation seeds must use the same ShootingSpec.")
    if int(steps) < 0 or float(step_size) <= 0.0:
        raise ValueError("steps must be nonnegative and step_size must be positive.")
    if not 0.0 < float(shrink) < 1.0 or float(grow) < 1.0:
        raise ValueError("shrink must be in (0, 1) and grow must be >= 1.")
    if (target_block is None) != (target_value is None):
        raise ValueError("target_block and target_value must be supplied together.")
    if not np.isfinite(float(target_tolerance)) or float(target_tolerance) < 0.0:
        raise ValueError("target_tolerance must be finite and nonnegative.")
    target_index: int | None = None
    if target_block is not None:
        block = current.spec.block(str(target_block))
        if block.size != 1:
            raise ValueError("target_block must name a scalar shooting block.")
        target_index = current.spec.offsets[block.name][0]
        target_value = float(target_value)
        if not np.isfinite(target_value):
            raise ValueError("target_value must be finite.")
    active_step = float(step_size)
    minimum_step = active_step * 1.0e-4 if min_step is None else float(min_step)
    maximum_step = active_step if max_step is None else float(max_step)
    if minimum_step <= 0.0 or maximum_step < minimum_step:
        raise ValueError("Continuation step limits must satisfy 0 < min_step <= max_step.")
    active_step = min(active_step, maximum_step)
    controller = StepSizeController(active_step, minimum_step, maximum_step, float(shrink), float(grow))
    coordinator = ContinuationCoordinator()
    runtime = ShootingEvaluator(system)
    verify_tol, tolerance_source = resolve_verification_tolerance(
        runtime.system_spec,
        working_tolerance=float(tolerance),
        verification_tolerance=verification_tolerance,
    )
    seed_jacobian = runtime.jacobian(current, options=options)
    _singular, _right, branch_rank, _threshold, _condition = jacobian_decomposition(seed_jacobian * current.scales)
    required_rank = current.spec.size - 1
    if branch_rank != required_rank:
        raise ValueError(
            "Pseudo-arclength continuation requires a one-dimensional root branch "
            f"(Jacobian rank {required_rank}); observed rank {branch_rank}."
        )

    resolved_verification_options, verification_independent, independence_source = (
        resolve_verification_request(
            runtime.system_spec,
            working_options=options,
            verification_options=verification_options,
        )
    )

    def point(value: ShootingVariables, tangent: np.ndarray, iterations: int) -> ContinuationPoint:
        residuals = runtime.residuals(value, options=options)
        try:
            verification = runtime.verify(
                value,
                options=resolved_verification_options,
                independent=verification_independent,
                independence_source=independence_source,
            )
        except Exception:  # noqa: BLE001 - a failed verification leaves the point unverified.
            verification = None
        return ContinuationPoint(
            variables=value.copy(),
            residual_norm=float(np.linalg.norm(residuals)),
            tangent=(value.scales * tangent).copy(),
            corrector_iterations=iterations,
            verified=bool(
                verification is not None
                and verification.independent
                and verification.numerical_success
                and all(check.passed for check in verification.checks)
                and verification.residual_norm <= verify_tol
            ),
            verification_independent=bool(verification is not None and verification.independent),
        )

    secant = (current.flatten(copy=False) - previous.flatten(copy=False)) / current.scales
    secant_norm = float(np.linalg.norm(secant))
    if secant_norm == 0.0:
        raise ValueError("Continuation seeds must be distinct.")
    tangent = secant / secant_norm
    points = [point(previous, tangent, 0), point(current, tangent, 0)]
    if not all(item.verified for item in points):
        return ShootingContinuationResult(
            points, 0, int(steps), "seed_verification_failed", active_step,
            verification_tolerance=verify_tol, verification_tolerance_source=tolerance_source,
        )
    if target_index is not None and abs(current.flatten(copy=False)[target_index] - target_value) <= float(target_tolerance):
        return ShootingContinuationResult(
            points, 0, int(steps), "target_reached", active_step,
            verification_tolerance=verify_tol, verification_tolerance_source=tolerance_source,
        )

    completed = 0
    stop_reason = "completed"
    prev = previous
    curr = current
    step_history = coordinator.attempts
    def advance_branch(iteration):
        nonlocal completed, stop_reason, prev, curr, active_step
        while True:
            secant = (curr.flatten(copy=False) - prev.flatten(copy=False)) / curr.scales
            tangent = secant / np.linalg.norm(secant)
            predictor = curr.flatten(copy=False) + active_step * curr.scales * tangent

            def arclength_equation(values):
                variables = curr.with_flat(values, name='continuation_corrector')
                residuals, jacobian = runtime.residuals_and_jacobian(variables, options=options)
                arclength = float(tangent @ ((values - predictor) / curr.scales))
                return (np.concatenate((residuals, [arclength])), np.vstack((jacobian, (tangent / curr.scales).reshape(1, -1))))
            corrected, iterations, corrector_error, converged = correct_root(predictor, arclength_equation, maxiter=int(max_corrector_iterations), tolerance=float(tolerance), scales=curr.scales)
            step_history.append({'step': completed + 1, 'attempted_step_size': active_step, 'corrector_iterations': iterations, 'accepted': converged, **({'error': corrector_error} if corrector_error is not None else {})})
            if converged:
                break
            if not controller.reject():
                stop_reason = 'minimum_step_reached'
                break
            active_step = controller.current
        if not converged:
            return stop_reason
        candidate = curr.with_flat(corrected, name='continuation_point')
        new_tangent = (candidate.flatten(copy=False) - curr.flatten(copy=False)) / curr.scales
        new_tangent /= np.linalg.norm(new_tangent)
        if float(new_tangent @ tangent) < 0.0:
            new_tangent *= -1.0
        candidate_point = point(candidate, new_tangent, iterations)
        if not candidate_point.verified:
            step_history[-1].update(accepted=False, reason='verification_failed')
            stop_reason = 'verification_failed'
            return stop_reason
        if target_index is not None:
            before_target = float(curr.flatten(copy=False)[target_index])
            after_target = float(candidate.flatten(copy=False)[target_index])
            crossed_target = (before_target - target_value) * (after_target - target_value) <= 0.0
            if crossed_target:
                landing = candidate
                landing_iterations = 0
                if abs(after_target - target_value) > float(target_tolerance):
                    denominator = after_target - before_target
                    fraction = 1.0 if denominator == 0.0 else (target_value - before_target) / denominator
                    landing_values = curr.flatten(copy=False) + fraction * (candidate.flatten(copy=False) - curr.flatten(copy=False))
                    selector = np.zeros(current.spec.size)
                    selector[target_index] = 1.0

                    def landing_equation(values):
                        trial = curr.with_flat(values, name='continuation_target_landing')
                        residuals, jacobian = runtime.residuals_and_jacobian(trial, options=options)
                        return (np.concatenate((residuals, [values[target_index] - target_value])), np.vstack((jacobian, selector.reshape(1, -1))))
                    landing_values, landing_iterations, landing_error, landed = correct_root(landing_values, landing_equation, maxiter=int(max_corrector_iterations), tolerance=float(tolerance), scales=curr.scales)
                    if landed:
                        landing = curr.with_flat(landing_values, name='continuation_target_landing')
                    if landing_error is not None:
                        step_history[-1]['error'] = landing_error
                    if not landed:
                        step_history[-1].update(accepted=False, reason='target_landing_failed')
                        stop_reason = 'target_landing_failed'
                        return stop_reason
                landing_tangent = (landing.flatten(copy=False) - curr.flatten(copy=False)) / curr.scales
                landing_norm = float(np.linalg.norm(landing_tangent))
                if landing_norm > 0.0:
                    landing_tangent /= landing_norm
                else:
                    landing_tangent = new_tangent
                landing_point = point(landing, landing_tangent, landing_iterations)
                if not landing_point.verified:
                    step_history[-1].update(accepted=False, reason='verification_failed')
                    stop_reason = 'verification_failed'
                    return stop_reason
                points.append(landing_point)
                completed += 1
                step_history[-1]['target_landing'] = True
                step_history[-1]['target_value'] = target_value
                stop_reason = 'target_reached'
                return stop_reason
        points.append(candidate_point)
        prev, curr = (curr, candidate)
        completed += 1
        _report(kind='item', method='continue_shooting', index=completed, total=int(steps), unit='step')
        controller.accept(fast=iterations <= int(fast_corrector_iterations))
        active_step = controller.current
        if stop_reason == 'target_reached':
            return stop_reason

    stop_reason = coordinator.run(advance_branch, limit=steps)

    return ShootingContinuationResult(
        points=points,
        completed_steps=completed,
        requested_steps=int(steps),
        stop_reason=stop_reason,
        step_size=active_step,
        step_history=step_history,
        verification_tolerance=verify_tol,
        verification_tolerance_source=tolerance_source,
    )


def continue_parameter(
    system: Any,
    initial: ShootingVariables,
    *,
    parameter: str,
    start: float,
    target: float,
    step_size: float,
    min_step: float | None = None,
    max_step: float | None = None,
    shrink: float = 0.5,
    grow: float = 1.5,
    max_steps: int = 100,
    solver_options: Mapping[str, Any] | None = None,
) -> ParameterContinuationResult:
    """Continue an OLSS root over one secondary parameter with exact landing."""

    from optimizer.shooting.solvers import _solve_or_record

    start = float(start)
    target = float(target)
    direction = 1.0 if target >= start else -1.0
    active_step = abs(float(step_size))
    minimum_step = active_step * 1.0e-4 if min_step is None else abs(float(min_step))
    maximum_step = active_step if max_step is None else abs(float(max_step))
    if active_step <= 0.0 or minimum_step <= 0.0 or maximum_step < minimum_step:
        raise ValueError("Parameter continuation requires 0 < min_step <= step_size <= max_step.")
    active_step = min(active_step, maximum_step)
    if not 0.0 < float(shrink) < 1.0 or float(grow) < 1.0:
        raise ValueError("shrink must be in (0, 1) and grow must be >= 1.")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1.")

    if not np.isfinite(start) or not np.isfinite(target):
        raise ValueError("Continuation endpoints must be finite.")
    controller = StepSizeController(active_step, minimum_step, maximum_step, float(shrink), float(grow))
    coordinator = ContinuationCoordinator()
    options = dict(solver_options or {})
    current_parameter = start
    current = initial
    points: list[ParameterContinuationPoint] = []
    step_history = coordinator.attempts
    stop_reason = "max_steps"
    def advance_parameter(iteration):
        nonlocal current_parameter, current, active_step, stop_reason
        remaining = abs(target - current_parameter)
        if remaining <= np.finfo(float).eps * max(1.0, abs(target)):
            stop_reason = 'target_reached'
            return stop_reason
        next_parameter, attempted = controller.landing(current_parameter, target)
        stage_system = system.with_secondary(**{str(parameter): next_parameter})
        result = _solve_or_record(stage_system, current, options)
        step_history.append({'parameter': next_parameter, 'attempted_step': attempted, 'qualified': result.qualified, 'solver_converged': result.solver_converged, 'stop_reason': result.stop_reason, 'residual_norm': result.residual_norm, 'verification_residual_norm': None if result.verification is None else result.verification.residual_norm, 'verification_tolerance': result.verification_tolerance, 'error': result.error})
        _report(kind='item', method='continue_parameter', index=len(step_history), total=int(max_steps), unit='step', metric=str(parameter), value=float(next_parameter))
        if result.qualified:
            points.append(ParameterContinuationPoint(next_parameter, result, attempted))
            current_parameter = next_parameter
            current = result.variables
            controller.accept()
            active_step = controller.current
            return None
        if not controller.reject():
            stop_reason = 'minimum_step_reached'
            return stop_reason
        active_step = controller.current

    stop_reason = coordinator.run(advance_parameter, limit=max_steps, budget_reason="max_steps")
    reached = bool(abs(target - current_parameter) <= np.finfo(float).eps * max(1.0, abs(target)))
    if reached:
        stop_reason = "target_reached"
    return ParameterContinuationResult(
        parameter=str(parameter),
        target=target,
        points=points,
        stop_reason=stop_reason,
        reached_target=reached,
        step_history=step_history,
    )
