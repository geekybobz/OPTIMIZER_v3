"""Momentum optimizer family.

Why this file exists
--------------------
Plain gradient descent can waste iterations by zigzagging through narrow valleys.
Momentum adds a velocity buffer so repeated gradient directions accumulate and
oscillatory components are smoothed.  This gives v3 a low-memory optimizer that is
still easy to inspect before moving to Adam's adaptive moment machinery.

How it fits the architecture
----------------------------
- ``system.gradient`` supplies the analytical gradient signal.
- this module turns that signal into a velocity-based control proposal.
- the shared engine decides whether the proposal is accepted and records the run.
- velocity is stored in ``RunState.optimizer_state`` and can transfer by warmstart.

What this file deliberately does not do
---------------------------------------
It does not do residual repair, constraint projection, finite-difference gradients, or
line-search backtracking.  If a trial is rejected by the engine, branch-specific state
keeps the optimizer memory consistent.

Reviewer invariants
-------------------
- velocity has exactly the flattened control size.
- accepted trials advance velocity; rejected trials do not accidentally advance it.
- Nesterov evaluates the gradient at a lookahead control using the shared evaluator.
- optional clipping limits update norm without changing update direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.engine import StepContext, StepProposal, run_chunk
from optimizer.optimizers._common import (
    clip_vector,
    coerce_warmstart,
    controls_from_flat_like,
    flat_state_array,
    require_finite,
    require_probability_like,
    require_variant,
)
from optimizer.result import OptimizerResult
from optimizer.state import RunState


VALID_VARIANTS = ("heavy_ball", "nesterov", "restart", "clipped")


@dataclass(frozen=True)
class MomentumConfig:
    """Validated options for ``opt.momentum``."""

    variant: str = "heavy_ball"
    step_size: float | None = None
    momentum: float = 0.9
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family="momentum")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_probability_like("momentum", self.momentum)
        require_finite("max_step_norm", self.effective_max_step_norm, positive=True)

    @property
    def initial_step_size(self) -> float:
        """Default step used when no state/warmstart value exists."""

        return 1.0e-3 if self.step_size is None else float(self.step_size)

    @property
    def effective_max_step_norm(self) -> float | None:
        """Default clipping limit for the explicit ``clipped`` variant."""

        if self.max_step_norm is not None:
            return float(self.max_step_norm)
        if self.variant == "clipped":
            return 1.0
        return None


def _state_payload(
    *,
    config: MomentumConfig,
    velocity: np.ndarray,
    accepted: int,
    rejected: int,
    step_size: float,
    gradient_norm: float,
    raw_step_norm: float,
    clipped: bool,
) -> dict[str, Any]:
    """Build the optimizer-state payload stored after a branch decision."""

    return {
        "variant": config.variant,
        "velocity": velocity.copy(),
        "momentum": float(config.momentum),
        "step_size": float(step_size),
        "accept_count": int(accepted),
        "reject_count": int(rejected),
        "last_gradient_norm": float(gradient_norm),
        "last_raw_step_norm": float(raw_step_norm),
        "last_step_clipped": bool(clipped),
    }


def _build_momentum_step(config: MomentumConfig):
    """Create the engine step function for momentum variants."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        controls = state.controls
        current_flat = controls.flatten(copy=False)
        current_memory = dict(state.optimizer_state)
        velocity = flat_state_array(current_memory, "velocity", size=controls.spec.size)
        step_size = float(state.step_size or config.initial_step_size)
        beta = float(config.momentum)
        accepted = int(current_memory.get("accept_count", 0))
        rejected = int(current_memory.get("reject_count", 0))

        if config.variant == "nesterov":
            lookahead_flat = current_flat + beta * velocity
            lookahead = controls_from_flat_like(
                controls,
                lookahead_flat,
                name="momentum_lookahead",
            )
            outcome = context.evaluator.try_gradient(lookahead)
            if not outcome.ok or outcome.gradient is None:
                raise ValueError(f"Nesterov lookahead gradient failed: {outcome.error}")
            gradient = outcome.gradient
        else:
            gradient = context.gradient

        gradient_flat = gradient.flatten(copy=False)
        gradient_norm = float(np.linalg.norm(gradient_flat))
        next_velocity = beta * velocity - step_size * gradient_flat
        update, raw_step_norm, clipped = clip_vector(
            next_velocity,
            config.effective_max_step_norm,
        )
        candidate = controls_from_flat_like(
            controls,
            current_flat + update,
            name=f"momentum_{config.variant}_trial",
        )

        # On rejection, preserve the previous velocity for ordinary variants.  The
        # restart variant intentionally drops it so the next proposal is a fresh
        # gradient step instead of repeating a direction that just failed.
        reject_velocity = np.zeros_like(velocity) if config.variant == "restart" else velocity
        accept_state = _state_payload(
            config=config,
            velocity=next_velocity,
            accepted=accepted + 1,
            rejected=rejected,
            step_size=step_size,
            gradient_norm=gradient_norm,
            raw_step_norm=raw_step_norm,
            clipped=clipped,
        )
        reject_state = _state_payload(
            config=config,
            velocity=reject_velocity,
            accepted=accepted,
            rejected=rejected + 1,
            step_size=step_size,
            gradient_norm=gradient_norm,
            raw_step_norm=raw_step_norm,
            clipped=clipped,
        )
        return StepProposal(
            controls=candidate,
            step_size=step_size,
            optimizer_state_on_accept=accept_state,
            optimizer_state_on_reject=reject_state,
            technical={
                "variant": config.variant,
                "momentum": beta,
                "step_size": step_size,
                "gradient_norm": gradient_norm,
                "velocity_norm_before": float(np.linalg.norm(velocity)),
                "velocity_norm_after": float(np.linalg.norm(next_velocity)),
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "clipped": clipped,
            },
        )

    return step


def momentum(
    system: Any,
    controls: Controls | None = None,
    *,
    variant: str = "heavy_ball",
    maxiter: int = 100,
    step_size: float | None = None,
    momentum: float = 0.9,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run the momentum optimizer family."""

    config = MomentumConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="momentum"),
        step_size=step_size,
        momentum=momentum,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_momentum_step(config),
        optimizer_name="momentum",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="momentum"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )
