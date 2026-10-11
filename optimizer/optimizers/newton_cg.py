"""Matrix-free Newton-CG optimization using analytical Hessian-vector products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.engine import StepContext, StepProposal, run_chunk
from optimizer.optimizers._common import clip_vector, coerce_warmstart, controls_from_flat_like, method_api_context, require_finite
from optimizer.result import OptimizerResult
from optimizer.state import RunState
from optimizer.system_olgs import probe_system


@dataclass(frozen=True)
class NewtonCGConfig:
    step_size: float = 1.0
    gradient_tolerance: float = 1.0e-8
    cg_tolerance: float = 1.0e-3
    cg_maxiter: int = 50
    damping: float = 0.0
    curvature_tolerance: float = 1.0e-12
    max_step_norm: float | None = None
    reject_shrink: float = 0.5

    def __post_init__(self) -> None:
        require_finite("step_size", self.step_size, positive=True)
        require_finite("gradient_tolerance", self.gradient_tolerance, nonnegative=True)
        require_finite("cg_tolerance", self.cg_tolerance, positive=True)
        require_finite("damping", self.damping, nonnegative=True)
        require_finite("curvature_tolerance", self.curvature_tolerance, nonnegative=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)
        if int(self.cg_maxiter) < 1:
            raise ValueError("cg_maxiter must be >= 1.")
        if not 0.0 < float(self.reject_shrink) < 1.0:
            raise ValueError("reject_shrink must satisfy 0 < reject_shrink < 1.")


def _hvp_flat(context: StepContext, vector: np.ndarray, damping: float) -> np.ndarray:
    vector_controls = controls_from_flat_like(context.state.controls, vector, name="newton_cg_vector")
    product = context.evaluator.hvp(context.state.controls, vector_controls)
    native = product.flatten(copy=False).astype(float, copy=False)
    # The engine exposes ``context.gradient`` in descent convention.  HVP hooks,
    # however, are system-native curvature, so maximizing systems need the same
    # sign conversion used by trust_region before CG inspects curvature.
    descent = native if context.objective_direction == "minimize" else -native
    return descent + float(damping) * vector


def _newton_direction(context: StepContext, config: NewtonCGConfig) -> tuple[np.ndarray, dict[str, Any]]:
    gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
    rhs = -gradient
    direction = np.zeros_like(rhs)
    residual = rhs.copy()
    search = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    target = float(config.cg_tolerance) * max(1.0, float(np.linalg.norm(gradient)))
    negative_curvature = False
    iterations = 0
    for iteration in range(int(config.cg_maxiter)):
        product = _hvp_flat(context, search, config.damping)
        curvature = float(np.dot(search, product))
        if curvature <= float(config.curvature_tolerance):
            negative_curvature = True
            if iteration == 0:
                direction = rhs.copy()
            break
        alpha = residual_sq / curvature
        direction = direction + alpha * search
        residual_next = residual - alpha * product
        iterations = iteration + 1
        next_sq = float(np.dot(residual_next, residual_next))
        if np.sqrt(next_sq) <= target:
            residual = residual_next
            residual_sq = next_sq
            break
        beta = next_sq / residual_sq
        search = residual_next + beta * search
        residual = residual_next
        residual_sq = next_sq
    directional_derivative = float(np.dot(gradient, direction))
    restarted = False
    if directional_derivative >= 0.0 or not np.all(np.isfinite(direction)):
        direction = -gradient
        directional_derivative = -float(np.dot(gradient, gradient))
        restarted = True
    return direction, {
        "cg_iterations": iterations,
        "cg_residual_norm": float(np.sqrt(residual_sq)),
        "negative_curvature": negative_curvature,
        "directional_derivative": directional_derivative,
        "restarted_to_gradient": restarted,
    }


def _build_step(config: NewtonCGConfig):
    def step(context: StepContext) -> StepProposal:
        gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
        gradient_norm = float(np.linalg.norm(gradient))
        step_size = float(context.state.step_size or config.step_size)
        direction, info = _newton_direction(context, config)
        update, raw_norm, clipped = clip_vector(step_size * direction, config.max_step_norm)
        candidate = controls_from_flat_like(
            context.state.controls,
            context.state.controls.flatten(copy=False) + update,
            name="newton_cg_trial",
        )
        memory = {"gradient_norm": gradient_norm, **info}
        return StepProposal(
            controls=candidate,
            step_size_on_accept=float(config.step_size),
            step_size_on_reject=step_size * float(config.reject_shrink),
            optimizer_state_on_accept=memory,
            optimizer_state_on_reject=memory,
            technical={
                "gradient_norm": gradient_norm,
                **info,
                "raw_step_norm": raw_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "step_clipped": clipped,
            },
        )

    return step


def newton_cg(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    step_size: float = 1.0,
    gradient_tolerance: float = 1.0e-8,
    cg_tolerance: float = 1.0e-3,
    cg_maxiter: int = 50,
    damping: float = 0.0,
    curvature_tolerance: float = 1.0e-12,
    max_step_norm: float | None = None,
    reject_shrink: float = 0.5,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str | None = None,
    accept_mode: str | None = None,
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run a line-search-style Newton-CG method using system.hvp."""

    if not probe_system(system).has_hvp:
        raise TypeError("newton_cg requires system.hvp(controls, vector).")
    config = NewtonCGConfig(
        step_size=step_size,
        gradient_tolerance=gradient_tolerance,
        cg_tolerance=cg_tolerance,
        cg_maxiter=cg_maxiter,
        damping=damping,
        curvature_tolerance=curvature_tolerance,
        max_step_norm=max_step_norm,
        reject_shrink=reject_shrink,
    )
    return run_chunk(
        system,
        controls,
        step=_build_step(config),
        optimizer_name="newton_cg",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="newton_cg"),
        step_size=step_size,
        gradient_tolerance=gradient_tolerance,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **method_api_context(engine_kwargs, method="newton_cg", config=config),
    )
