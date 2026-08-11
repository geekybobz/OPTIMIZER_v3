"""Projected equality- and bound-constrained first-order optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.core.engine import AcceptanceDecision, StepContext, StepProposal, run_chunk
from optimizer.optimizers._common import clip_vector, coerce_warmstart, method_api_context, require_finite
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState
from optimizer.system_olgs import ControlBounds, get_system_spec, probe_system


@dataclass(frozen=True)
class ProjectedConfig:
    step_size: float = 0.1
    residual_name: str = "hard"
    feasibility_tolerance: float = 1.0e-8
    stationarity_tolerance: float = 1.0e-8
    damping: float = 1.0e-10
    restoration_maxiter: int = 5
    max_backtracks: int = 8
    shrink: float = 0.5
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        require_finite("step_size", self.step_size, positive=True)
        require_finite("feasibility_tolerance", self.feasibility_tolerance, nonnegative=True)
        require_finite("stationarity_tolerance", self.stationarity_tolerance, nonnegative=True)
        require_finite("damping", self.damping, nonnegative=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)
        if int(self.restoration_maxiter) < 0:
            raise ValueError("restoration_maxiter must be >= 0.")
        if int(self.max_backtracks) < 1:
            raise ValueError("max_backtracks must be >= 1.")
        if not 0.0 < float(self.shrink) < 1.0:
            raise ValueError("shrink must satisfy 0 < shrink < 1.")
        if not str(self.residual_name):
            raise ValueError("residual_name must be non-empty.")


def _constraint_scales(system: Any, name: str, dimension: int) -> np.ndarray:
    hook = getattr(system, "constraint_spec", None)
    if not callable(hook):
        return np.ones(int(dimension), dtype=float)
    payload = hook()
    if not isinstance(payload, Mapping):
        raise TypeError("system.constraint_spec() must return a mapping.")
    sets = payload.get("sets", payload)
    entry = sets.get(name, {}) if isinstance(sets, Mapping) else {}
    raw = entry.get("scales") if isinstance(entry, Mapping) else None
    if raw is None:
        return np.ones(int(dimension), dtype=float)
    scales = np.asarray(raw, dtype=float).reshape(-1)
    if scales.shape != (int(dimension),):
        raise ValueError(f"constraint scales must have shape ({dimension},), got {scales.shape}.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("constraint scales must be finite and positive.")
    return scales


def _scaled_geometry(
    context: StepContext,
    config: ProjectedConfig,
    *,
    has_equalities: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not has_equalities:
        size = context.state.controls.spec.size
        return np.zeros(0, dtype=float), np.zeros((0, size), dtype=float), np.ones(0, dtype=float)
    residual = context.evaluator.residuals(context.state.controls, name=config.residual_name)
    jacobian = context.evaluator.jacobian(context.state.controls, name=config.residual_name)
    scales = _constraint_scales(context.system, config.residual_name, residual.size)
    return residual / scales, jacobian / scales[:, None], scales


def _clip_bounds(flat: np.ndarray, bounds: ControlBounds | None) -> np.ndarray:
    if bounds is None:
        return np.asarray(flat, dtype=float).copy()
    lower = bounds.lower.flatten(copy=False).astype(float, copy=False)
    upper = bounds.upper.flatten(copy=False).astype(float, copy=False)
    return np.clip(np.asarray(flat, dtype=float), lower, upper)


def _controls(spec: ControlSpec, flat: np.ndarray, name: str) -> Controls:
    return Controls.from_flat(spec, np.asarray(flat, dtype=spec.dtype), name=name)


def _projected_direction(
    gradient: np.ndarray,
    jacobian: np.ndarray,
    controls: Controls,
    bounds: ControlBounds | None,
    *,
    damping: float,
) -> tuple[np.ndarray, int]:
    """Project descent into the equality and active-bound tangent space."""

    size = gradient.size
    active: set[int] = set()
    flat = controls.flatten(copy=False).astype(float, copy=False)
    lower = None if bounds is None else bounds.lower.flatten(copy=False).astype(float, copy=False)
    upper = None if bounds is None else bounds.upper.flatten(copy=False).astype(float, copy=False)
    tolerance = 10.0 * np.finfo(float).eps
    direction = -gradient.copy()
    for _ in range(size + 1):
        if active:
            rows = np.eye(size, dtype=float)[sorted(active)]
            combined = np.vstack((jacobian, rows)) if jacobian.shape[0] else rows
        else:
            combined = jacobian
        if combined.shape[0]:
            gram = combined @ combined.T + float(damping) * np.eye(combined.shape[0])
            multiplier = np.linalg.lstsq(gram, combined @ gradient, rcond=None)[0]
            direction = -(gradient - combined.T @ multiplier)
        else:
            direction = -gradient.copy()
        if bounds is None:
            break
        assert lower is not None and upper is not None
        blocked = set(
            np.flatnonzero(
                ((flat <= lower + tolerance) & (direction < 0.0))
                | ((flat >= upper - tolerance) & (direction > 0.0))
            ).tolist()
        )
        additions = blocked - active
        if not additions:
            break
        active.update(additions)
    return direction, len(active)


def _restore(
    context: StepContext,
    start: Controls,
    *,
    scales: np.ndarray,
    bounds: ControlBounds | None,
    config: ProjectedConfig,
) -> tuple[Controls, float, int, bool]:
    if scales.size == 0:
        return start, 0.0, 0, True
    current = start
    residual = context.evaluator.residuals(current, name=config.residual_name) / scales
    current_norm = float(np.linalg.norm(residual))
    total_trials = 0
    improved_any = False
    for _ in range(int(config.restoration_maxiter)):
        if current_norm <= float(config.feasibility_tolerance):
            break
        jacobian = context.evaluator.jacobian(current, name=config.residual_name) / scales[:, None]
        gram = jacobian @ jacobian.T + float(config.damping) * np.eye(jacobian.shape[0])
        multiplier = np.linalg.lstsq(gram, -residual, rcond=None)[0]
        delta = jacobian.T @ multiplier
        delta, _raw_norm, _clipped = clip_vector(delta, config.max_step_norm)
        base = current.flatten(copy=False).astype(float, copy=False)
        accepted = False
        scale = 1.0
        for _attempt in range(int(config.max_backtracks)):
            total_trials += 1
            trial_flat = _clip_bounds(base + scale * delta, bounds)
            trial = _controls(current.spec, trial_flat, "projected_restoration_trial")
            trial_residual = context.evaluator.residuals(trial, name=config.residual_name) / scales
            trial_norm = float(np.linalg.norm(trial_residual))
            if trial_norm < current_norm:
                current = trial
                residual = trial_residual
                current_norm = trial_norm
                improved_any = True
                accepted = True
                break
            scale *= float(config.shrink)
        if not accepted:
            break
    return current, current_norm, total_trials, improved_any


def _build_projected_step(config: ProjectedConfig):
    bounds_cache: list[ControlBounds | None] = []
    probe_cache = probe_system

    def step(context: StepContext) -> StepProposal:
        probe = probe_cache(context.system)
        if not bounds_cache:
            bounds_cache.append(context.evaluator.control_bounds() if probe.has_control_bounds else None)
        bounds = bounds_cache[0]
        residual, jacobian, scales = _scaled_geometry(
            context,
            config,
            has_equalities=probe.has_residuals and probe.has_jacobian,
        )
        residual_norm = float(np.linalg.norm(residual))
        gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
        step_size = float(context.state.step_size or config.step_size)

        if residual_norm > float(config.feasibility_tolerance):
            candidate, candidate_norm, trials, improved = _restore(
                context,
                context.state.controls,
                scales=scales,
                bounds=bounds,
                config=config,
            )
            return StepProposal(
                controls=candidate,
                step_size_on_accept=step_size,
                step_size_on_reject=step_size * float(config.shrink),
                optimizer_state_on_accept={
                    "phase": "restoration",
                    "residual_norm": candidate_norm,
                    "projected_kkt_norm": None,
                },
                optimizer_state_on_reject={
                    "phase": "restoration",
                    "residual_norm": residual_norm,
                    "projected_kkt_norm": None,
                },
                technical={
                    "phase": "restoration",
                    "residual_norm_before": residual_norm,
                    "residual_norm_after": candidate_norm,
                    "restoration_trials": trials,
                },
                stop_reason=None if improved else "restoration_failed",
            )

        direction, active_count = _projected_direction(
            gradient,
            jacobian,
            context.state.controls,
            bounds,
            damping=config.damping,
        )
        projected_kkt_norm = float(np.linalg.norm(direction))
        if projected_kkt_norm <= float(config.stationarity_tolerance):
            return StepProposal(
                controls=context.state.controls.copy(name="projected_converged"),
                step_size=step_size,
                optimizer_state={
                    "phase": "feasible",
                    "residual_norm": residual_norm,
                    "projected_kkt_norm": projected_kkt_norm,
                    "active_bound_count": active_count,
                },
                technical={
                    "phase": "feasible",
                    "residual_norm_before": residual_norm,
                    "residual_norm_after": residual_norm,
                    "projected_kkt_norm": projected_kkt_norm,
                    "active_bound_count": active_count,
                },
                stop_reason="projected_kkt_converged",
            )

        raw_update = step_size * direction
        update, raw_step_norm, step_clipped = clip_vector(raw_update, config.max_step_norm)
        base = context.state.controls.flatten(copy=False).astype(float, copy=False)
        trial_flat = _clip_bounds(base + update, bounds)
        trial = _controls(context.state.controls.spec, trial_flat, "projected_objective_trial")
        candidate, candidate_norm, repair_trials, _improved = _restore(
            context,
            trial,
            scales=scales,
            bounds=bounds,
            config=config,
        )
        accepted_state = {
            "phase": "feasible",
            "residual_norm": candidate_norm,
            "projected_kkt_norm": projected_kkt_norm,
            "active_bound_count": active_count,
        }
        rejected_state = {
            "phase": "feasible",
            "residual_norm": residual_norm,
            "projected_kkt_norm": projected_kkt_norm,
            "active_bound_count": active_count,
        }
        return StepProposal(
            controls=candidate,
            step_size_on_accept=step_size,
            step_size_on_reject=step_size * float(config.shrink),
            optimizer_state_on_accept=accepted_state,
            optimizer_state_on_reject=rejected_state,
            technical={
                "phase": "feasible",
                "residual_norm_before": residual_norm,
                "residual_norm_after": candidate_norm,
                "projected_kkt_norm": projected_kkt_norm,
                "active_bound_count": active_count,
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "step_clipped": step_clipped,
                "repair_trials": repair_trials,
            },
        )

    return step


def _projected_accept(config: ProjectedConfig, *, metric: str, mode: str, tolerance: float):
    def accept(
        current: Evaluation,
        trial: Evaluation,
        proposal: StepProposal,
        state: RunState,
    ) -> AcceptanceDecision:
        del state
        technical = dict(proposal.technical)
        before_residual = float(technical.get("residual_norm_before", np.inf))
        after_residual = float(technical.get("residual_norm_after", np.inf))
        if technical.get("phase") == "restoration":
            accepted = after_residual < before_residual
            return AcceptanceDecision(
                accepted=accepted,
                reason="restoration_reduced_residual" if accepted else "restoration_rejected",
                technical={"residual_reduction": before_residual - after_residual},
            )
        feasible = after_residual <= float(config.feasibility_tolerance)
        current_value = float(current.metrics[metric])
        trial_value = float(trial.metrics[metric])
        improved = (
            trial_value <= current_value + float(tolerance)
            if mode == "min"
            else trial_value >= current_value - float(tolerance)
        )
        accepted = feasible and improved
        return AcceptanceDecision(
            accepted=accepted,
            reason=(
                "feasible_objective_accepted"
                if accepted
                else ("infeasible_trial" if not feasible else "objective_rejected")
            ),
            technical={
                "feasible": feasible,
                "objective_improved": improved,
                "residual_norm": after_residual,
            },
        )

    return accept


def projected_gradient(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    step_size: float = 0.1,
    residual_name: str = "hard",
    feasibility_tolerance: float = 1.0e-8,
    stationarity_tolerance: float = 1.0e-8,
    damping: float = 1.0e-10,
    restoration_maxiter: int = 5,
    max_backtracks: int = 8,
    shrink: float = 0.5,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str | None = None,
    accept_mode: str | None = None,
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Minimize an objective on hard equality constraints and optional bounds."""

    probe = probe_system(system)
    partial_equality = probe.has_residuals != probe.has_jacobian
    if partial_equality:
        missing = "jacobian" if probe.has_residuals else "residuals"
        raise TypeError(f"projected_gradient equality constraints require system.{missing}(...).")
    if not (probe.has_residuals and probe.has_jacobian) and not probe.has_control_bounds:
        raise TypeError(
            "projected_gradient requires residuals+jacobian equality hooks, control_bounds(), or both."
        )
    objective = get_system_spec(system)["objective"]
    resolved_metric = str(objective["metric_id"]) if accept_metric is None else str(accept_metric)
    resolved_mode = "min" if objective["direction"] == "minimize" else "max"
    if accept_mode is not None:
        resolved_mode = str(accept_mode)
    config = ProjectedConfig(
        step_size=step_size,
        residual_name=residual_name,
        feasibility_tolerance=feasibility_tolerance,
        stationarity_tolerance=stationarity_tolerance,
        damping=damping,
        restoration_maxiter=restoration_maxiter,
        max_backtracks=max_backtracks,
        shrink=shrink,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_projected_step(config),
        optimizer_name="projected_gradient",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="projected_gradient"),
        step_size=step_size,
        accept=_projected_accept(
            config,
            metric=resolved_metric,
            mode=resolved_mode,
            tolerance=accept_tolerance,
        ),
        accept_metric=resolved_metric,
        accept_mode=resolved_mode,
        accept_tolerance=accept_tolerance,
        **method_api_context(engine_kwargs, method="projected_gradient", config=config),
    )
