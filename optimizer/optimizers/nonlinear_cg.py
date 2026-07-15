"""Nonlinear conjugate-gradient optimizer.

Why this file exists
--------------------
Nonlinear conjugate gradient is a low-memory optimizer for large deterministic
control vectors.  It improves over steepest descent by combining the current gradient
with the previous accepted search direction.  This makes it a useful middle ground
between basic line search and L-BFGS.

How it fits the architecture
----------------------------
- the system supplies analytical gradients.
- this module builds one conjugate-gradient proposal per engine iteration.
- line-search-like step-size control is left to the caller through step size and the
  engine's accept/reject behavior.
- previous gradient/direction memory is stored in ``RunState.optimizer_state``.

What this file deliberately does not do
---------------------------------------
It does not compute Hessians, project constraints, or repair residuals.  It also does
not perform a full Wolfe line search; that can be layered later if needed.

Reviewer invariants
-------------------
- beta variants are explicit and logged.
- non-descent directions restart to steepest descent.
- accepted moves advance previous gradient/direction; rejected moves preserve memory.
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
    require_variant,
)
from optimizer.result import OptimizerResult
from optimizer.state import RunState


VALID_VARIANTS = ("fletcher_reeves", "polak_ribiere", "polak_ribiere_plus", "hestenes_stiefel")


@dataclass(frozen=True)
class NonlinearCGConfig:
    """Validated options for nonlinear conjugate gradient."""

    variant: str = "polak_ribiere_plus"
    step_size: float | None = None
    max_step_norm: float | None = None
    restart_on_nondescent: bool = True

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family="nonlinear_cg")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)

    @property
    def initial_step_size(self) -> float:
        return 1.0e-2 if self.step_size is None else float(self.step_size)


def _beta(config: NonlinearCGConfig, gradient: np.ndarray, previous_gradient: np.ndarray, previous_direction: np.ndarray) -> float:
    """Compute the conjugacy coefficient for the selected variant."""

    denom_eps = np.finfo(float).tiny
    if config.variant == "fletcher_reeves":
        denominator = max(float(np.dot(previous_gradient, previous_gradient)), denom_eps)
        return float(np.dot(gradient, gradient) / denominator)

    y = gradient - previous_gradient
    if config.variant in {"polak_ribiere", "polak_ribiere_plus"}:
        denominator = max(float(np.dot(previous_gradient, previous_gradient)), denom_eps)
        beta = float(np.dot(gradient, y) / denominator)
        return max(0.0, beta) if config.variant == "polak_ribiere_plus" else beta

    denominator = float(np.dot(previous_direction, y))
    if abs(denominator) <= denom_eps:
        return 0.0
    return float(np.dot(gradient, y) / denominator)


def _state_payload(
    *,
    config: NonlinearCGConfig,
    gradient: np.ndarray,
    direction: np.ndarray,
    accepted: int,
    rejected: int,
    step_size: float,
    beta: float,
    restarted: bool,
    raw_step_norm: float,
    clipped: bool,
) -> dict[str, Any]:
    """Build optimizer-private state for one branch."""

    return {
        "variant": config.variant,
        "previous_gradient": gradient.copy(),
        "direction": direction.copy(),
        "step_size": float(step_size),
        "beta": float(beta),
        "restarted": bool(restarted),
        "accept_count": int(accepted),
        "reject_count": int(rejected),
        "last_gradient_norm": float(np.linalg.norm(gradient)),
        "last_direction_norm": float(np.linalg.norm(direction)),
        "last_raw_step_norm": float(raw_step_norm),
        "last_step_clipped": bool(clipped),
    }


def _build_ncg_step(config: NonlinearCGConfig):
    """Create the engine step function for nonlinear CG."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        controls = state.controls
        memory = dict(state.optimizer_state)
        current_flat = controls.flatten(copy=False)
        gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
        previous_gradient = flat_state_array(memory, "previous_gradient", size=controls.spec.size)
        previous_direction = flat_state_array(memory, "direction", size=controls.spec.size)
        accepted = int(memory.get("accept_count", 0))
        rejected = int(memory.get("reject_count", 0))
        step_size = float(state.step_size or config.initial_step_size)

        if accepted == 0 and "previous_gradient" not in memory:
            beta = 0.0
            direction = -gradient
        else:
            beta = _beta(config, gradient, previous_gradient, previous_direction)
            direction = -gradient + beta * previous_direction

        directional_derivative = float(np.dot(gradient, direction))
        restarted = False
        if config.restart_on_nondescent and directional_derivative >= 0.0:
            direction = -gradient
            beta = 0.0
            directional_derivative = float(np.dot(gradient, direction))
            restarted = True

        raw_update = step_size * direction
        update, raw_step_norm, clipped = clip_vector(raw_update, config.max_step_norm)
        candidate = controls_from_flat_like(
            controls,
            current_flat + update,
            name=f"nonlinear_cg_{config.variant}_trial",
        )

        accept_state = _state_payload(
            config=config,
            gradient=gradient,
            direction=direction,
            accepted=accepted + 1,
            rejected=rejected,
            step_size=step_size,
            beta=beta,
            restarted=restarted,
            raw_step_norm=raw_step_norm,
            clipped=clipped,
        )
        reject_state = _state_payload(
            config=config,
            gradient=previous_gradient,
            direction=previous_direction,
            accepted=accepted,
            rejected=rejected + 1,
            step_size=step_size,
            beta=beta,
            restarted=restarted,
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
                "step_size": step_size,
                "beta": beta,
                "restarted": restarted,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "direction_norm": float(np.linalg.norm(direction)),
                "directional_derivative": directional_derivative,
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "clipped": clipped,
            },
        )

    return step


def nonlinear_cg(
    system: Any,
    controls: Controls | None = None,
    *,
    variant: str = "polak_ribiere_plus",
    maxiter: int = 100,
    step_size: float | None = None,
    max_step_norm: float | None = None,
    restart_on_nondescent: bool = True,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run nonlinear conjugate gradient."""

    config = NonlinearCGConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="nonlinear_cg"),
        step_size=step_size,
        max_step_norm=max_step_norm,
        restart_on_nondescent=restart_on_nondescent,
    )
    return run_chunk(
        system,
        controls,
        step=_build_ncg_step(config),
        optimizer_name="nonlinear_cg",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="nonlinear_cg"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )


def ncg(*args: Any, **kwargs: Any) -> OptimizerResult:
    """Short alias for ``nonlinear_cg``."""

    return nonlinear_cg(*args, **kwargs)
