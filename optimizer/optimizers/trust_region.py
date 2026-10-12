"""Matrix-free Steihaug trust-region optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.engine import AcceptanceDecision, StepContext, StepProposal, run_chunk
from optimizer.optimizers._common import coerce_warmstart, controls_from_flat_like, require_finite
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState
from optimizer.system_olgs import get_system_spec, probe_system


@dataclass(frozen=True)
class TrustRegionConfig:
    initial_radius: float = 1.0
    max_radius: float = 100.0
    gradient_tolerance: float = 1.0e-8
    cg_tolerance: float = 1.0e-3
    cg_maxiter: int = 50
    damping: float = 0.0
    curvature_tolerance: float = 1.0e-12
    acceptance_threshold: float = 0.1
    shrink_threshold: float = 0.25
    grow_threshold: float = 0.75
    shrink_factor: float = 0.25
    grow_factor: float = 2.0
    objective_metric: str = "J"
    objective_direction: str = "minimize"

    def __post_init__(self) -> None:
        require_finite("initial_radius", self.initial_radius, positive=True)
        require_finite("max_radius", self.max_radius, positive=True)
        require_finite("gradient_tolerance", self.gradient_tolerance, nonnegative=True)
        require_finite("cg_tolerance", self.cg_tolerance, positive=True)
        require_finite("damping", self.damping, nonnegative=True)
        require_finite("curvature_tolerance", self.curvature_tolerance, nonnegative=True)
        if int(self.cg_maxiter) < 1:
            raise ValueError("cg_maxiter must be >= 1.")
        if float(self.initial_radius) > float(self.max_radius):
            raise ValueError("initial_radius must be <= max_radius.")
        if not (
            0.0 <= float(self.acceptance_threshold)
            <= float(self.shrink_threshold)
            < float(self.grow_threshold)
            < 1.0
        ):
            raise ValueError(
                "thresholds must satisfy 0 <= acceptance <= shrink < grow < 1."
            )
        if not 0.0 < float(self.shrink_factor) < 1.0:
            raise ValueError("shrink_factor must satisfy 0 < value < 1.")
        if float(self.grow_factor) <= 1.0:
            raise ValueError("grow_factor must be > 1.")
        if self.objective_direction not in {"minimize", "maximize"}:
            raise ValueError("objective_direction must be 'minimize' or 'maximize'.")


def _boundary_tau(point: np.ndarray, direction: np.ndarray, radius: float) -> float:
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(point, direction))
    c = float(np.dot(point, point)) - float(radius) ** 2
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = [(-b + np.sqrt(discriminant)) / (2.0 * a), (-b - np.sqrt(discriminant)) / (2.0 * a)]
    positive = [root for root in roots if root >= 0.0]
    return max(positive) if positive else 0.0


def _hvp_flat(context: StepContext, vector: np.ndarray, damping: float, *, direction: str) -> np.ndarray:
    control_vector = controls_from_flat_like(context.state.controls, vector, name="trust_region_vector")
    product = context.evaluator.hvp(context.state.controls, control_vector)
    native = product.flatten(copy=False).astype(float, copy=False)
    descent_hessian = native if direction == "minimize" else -native
    return descent_hessian + float(damping) * vector


def _steihaug(context: StepContext, radius: float, config: TrustRegionConfig) -> tuple[np.ndarray, dict[str, Any]]:
    gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
    point = np.zeros_like(gradient)
    residual = gradient.copy()
    direction = -residual
    residual_sq = float(np.dot(residual, residual))
    target = float(config.cg_tolerance) * max(1.0, float(np.linalg.norm(gradient)))
    boundary_hit = False
    negative_curvature = False
    iterations = 0
    for iteration in range(int(config.cg_maxiter)):
        product = _hvp_flat(context, direction, config.damping, direction=config.objective_direction)
        curvature = float(np.dot(direction, product))
        if curvature <= float(config.curvature_tolerance):
            tau = _boundary_tau(point, direction, radius)
            point = point + tau * direction
            boundary_hit = True
            negative_curvature = True
            break
        alpha = residual_sq / curvature
        candidate = point + alpha * direction
        if float(np.linalg.norm(candidate)) >= radius:
            tau = _boundary_tau(point, direction, radius)
            point = point + tau * direction
            boundary_hit = True
            break
        point = candidate
        residual_next = residual + alpha * product
        iterations = iteration + 1
        next_sq = float(np.dot(residual_next, residual_next))
        if np.sqrt(next_sq) <= target:
            residual = residual_next
            residual_sq = next_sq
            break
        beta = next_sq / residual_sq
        direction = -residual_next + beta * direction
        residual = residual_next
        residual_sq = next_sq
    h_point = _hvp_flat(context, point, config.damping, direction=config.objective_direction)
    predicted = -float(np.dot(gradient, point)) - 0.5 * float(np.dot(point, h_point))
    return point, {
        "cg_iterations": iterations,
        "cg_residual_norm": float(np.sqrt(residual_sq)),
        "boundary_hit": boundary_hit,
        "negative_curvature": negative_curvature,
        "predicted_reduction": predicted,
    }


def _build_step(config: TrustRegionConfig):
    def step(context: StepContext) -> StepProposal:
        gradient_norm = context.gradient.norm()
        radius = float(context.state.step_size or config.initial_radius)
        step_vector, info = _steihaug(context, radius, config)
        candidate = controls_from_flat_like(
            context.state.controls,
            context.state.controls.flatten(copy=False) + step_vector,
            name="trust_region_trial",
        )
        return StepProposal(
            controls=candidate,
            step_size=radius,
            optimizer_state={"gradient_norm": gradient_norm, "trust_radius": radius, **info},
            technical={
                "gradient_norm": gradient_norm,
                "trust_radius": radius,
                "step_norm": float(np.linalg.norm(step_vector)),
                **info,
            },
        )

    return step


def _accept(config: TrustRegionConfig):
    def accept(current: Evaluation, trial: Evaluation, proposal: StepProposal, state: RunState) -> AcceptanceDecision:
        del state
        predicted = float(proposal.technical.get("predicted_reduction", 0.0))
        current_value = current.objective_value
        trial_value = trial.objective_value
        actual = (
            current_value - trial_value
            if config.objective_direction == "minimize"
            else trial_value - current_value
        )
        rho = actual / predicted if predicted > np.finfo(float).tiny else float("-inf")
        radius = float(proposal.technical["trust_radius"])
        boundary_hit = bool(proposal.technical.get("boundary_hit", False))
        if not np.isfinite(rho) or rho < float(config.shrink_threshold):
            next_radius = radius * float(config.shrink_factor)
        elif rho > float(config.grow_threshold) and boundary_hit:
            next_radius = min(float(config.max_radius), radius * float(config.grow_factor))
        else:
            next_radius = radius
        accepted = predicted > 0.0 and rho >= float(config.acceptance_threshold)
        return AcceptanceDecision(
            accepted=accepted,
            reason="trust_region_accepted" if accepted else "trust_region_rejected",
            technical={
                "actual_reduction": actual,
                "objective_metric": config.objective_metric,
                "objective_direction": config.objective_direction,
                "predicted_reduction": predicted,
                "rho": rho,
                "trust_radius_before": radius,
                "trust_radius_after": next_radius,
            },
            optimizer_state_update={
                "actual_reduction": actual,
                "rho": rho,
                "trust_radius": next_radius,
            },
            next_step_size=next_radius,
        )

    return accept


def trust_region(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    initial_radius: float = 1.0,
    max_radius: float = 100.0,
    gradient_tolerance: float = 1.0e-8,
    cg_tolerance: float = 1.0e-3,
    cg_maxiter: int = 50,
    damping: float = 0.0,
    curvature_tolerance: float = 1.0e-12,
    acceptance_threshold: float = 0.1,
    shrink_threshold: float = 0.25,
    grow_threshold: float = 0.75,
    shrink_factor: float = 0.25,
    grow_factor: float = 2.0,
    state: RunState | None = None,
    warmstart: Any | None = None,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run a Steihaug trust-region method for the declared system objective."""

    if not probe_system(system).has_hvp:
        raise TypeError("trust_region requires system.hvp(controls, vector).")
    objective = get_system_spec(system)["objective"]
    config = TrustRegionConfig(
        initial_radius=initial_radius,
        max_radius=max_radius,
        gradient_tolerance=gradient_tolerance,
        cg_tolerance=cg_tolerance,
        cg_maxiter=cg_maxiter,
        damping=damping,
        curvature_tolerance=curvature_tolerance,
        acceptance_threshold=acceptance_threshold,
        shrink_threshold=shrink_threshold,
        grow_threshold=grow_threshold,
        shrink_factor=shrink_factor,
        grow_factor=grow_factor,
        objective_metric=str(objective["metric_id"]),
        objective_direction=str(objective["direction"]),
    )
    return run_chunk(
        system,
        controls,
        step=_build_step(config),
        optimizer_name="trust_region",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="trust_region"),
        step_size=initial_radius,
        gradient_tolerance=gradient_tolerance,
        accept=_accept(config),
        accept_metric=config.objective_metric,
        accept_mode="min" if config.objective_direction == "minimize" else "max",
        **engine_kwargs,
    )
