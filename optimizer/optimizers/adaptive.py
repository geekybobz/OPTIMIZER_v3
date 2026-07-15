"""AdaGrad and RMSProp optimizer families.

Why this file exists
--------------------
Adam is not always the right adaptive method.  Sometimes the first-moment memory in
Adam pushes too hard, while per-coordinate gradient scaling is still useful.  AdaGrad
and RMSProp provide that smaller tool: both scale each control coordinate by a
history of squared gradients, but they avoid Adam's momentum-style first moment.

How it fits the architecture
----------------------------
- systems provide the analytical gradient.
- this module owns only adaptive first-order proposal logic.
- the shared engine handles evaluation, acceptance, tracing, checkpoints, and
  warmstart.
- accumulators live in ``RunState.optimizer_state`` as flat arrays.

What this file deliberately does not do
---------------------------------------
It does not repair constraints, project gradients, or choose physical metrics beyond
the engine's normal accept/reject options.  It is an optimizer phase module only.

Reviewer invariants
-------------------
- accumulator shape always equals flattened control size.
- accepted trials advance accumulators; rejected trials keep previous accumulators.
- RMSProp uses an exponential moving average, while AdaGrad uses a cumulative sum.
- optional clipping limits the actual update length without changing direction.
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
)
from optimizer.result import OptimizerResult
from optimizer.state import RunState


@dataclass(frozen=True)
class AdaptiveConfig:
    """Validated options shared by AdaGrad and RMSProp."""

    method: str
    step_size: float | None = None
    decay: float = 0.9
    eps: float = 1.0e-8
    initial_accumulator: float = 0.0
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"adagrad", "rmsprop"}:
            raise ValueError("method must be 'adagrad' or 'rmsprop'.")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_finite("eps", self.eps, positive=True)
        require_finite("initial_accumulator", self.initial_accumulator, nonnegative=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)
        if self.method == "rmsprop":
            require_probability_like("decay", self.decay)

    @property
    def initial_step_size(self) -> float:
        """Default step size used when state/warmstart does not provide one."""

        return 1.0e-2 if self.step_size is None else float(self.step_size)


def _state_payload(
    *,
    config: AdaptiveConfig,
    accumulator: np.ndarray,
    accepted: int,
    rejected: int,
    step_size: float,
    gradient_norm: float,
    raw_step_norm: float,
    clipped: bool,
) -> dict[str, Any]:
    """Build optimizer-private state for an accept/reject branch."""

    return {
        "method": config.method,
        "accumulator": accumulator.copy(),
        "step_size": float(step_size),
        "decay": float(config.decay),
        "eps": float(config.eps),
        "accept_count": int(accepted),
        "reject_count": int(rejected),
        "last_gradient_norm": float(gradient_norm),
        "last_raw_step_norm": float(raw_step_norm),
        "last_step_clipped": bool(clipped),
    }


def _build_adaptive_step(config: AdaptiveConfig):
    """Create an engine step function for AdaGrad/RMSProp."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        controls = state.controls
        memory = dict(state.optimizer_state)
        current_flat = controls.flatten(copy=False)
        gradient_flat = context.gradient.flatten(copy=False).astype(float, copy=False)
        gradient_norm = float(np.linalg.norm(gradient_flat))
        accumulator = flat_state_array(
            memory,
            "accumulator",
            size=controls.spec.size,
            default=float(config.initial_accumulator),
        )
        accepted = int(memory.get("accept_count", 0))
        rejected = int(memory.get("reject_count", 0))
        step_size = float(state.step_size or config.initial_step_size)

        if config.method == "adagrad":
            next_accumulator = accumulator + gradient_flat * gradient_flat
        else:
            next_accumulator = float(config.decay) * accumulator + (1.0 - float(config.decay)) * (
                gradient_flat * gradient_flat
            )

        raw_update = step_size * gradient_flat / (np.sqrt(next_accumulator) + float(config.eps))
        update, raw_step_norm, clipped = clip_vector(raw_update, config.max_step_norm)
        candidate = controls_from_flat_like(
            controls,
            current_flat - update,
            name=f"{config.method}_trial",
        )

        accept_state = _state_payload(
            config=config,
            accumulator=next_accumulator,
            accepted=accepted + 1,
            rejected=rejected,
            step_size=step_size,
            gradient_norm=gradient_norm,
            raw_step_norm=raw_step_norm,
            clipped=clipped,
        )
        reject_state = _state_payload(
            config=config,
            accumulator=accumulator,
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
                "method": config.method,
                "step_size": step_size,
                "gradient_norm": gradient_norm,
                "accumulator_norm_before": float(np.linalg.norm(accumulator)),
                "accumulator_norm_after": float(np.linalg.norm(next_accumulator)),
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "clipped": clipped,
            },
        )

    return step


def adagrad(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    step_size: float | None = None,
    eps: float = 1.0e-8,
    initial_accumulator: float = 0.0,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run AdaGrad with cumulative per-coordinate squared-gradient scaling."""

    config = AdaptiveConfig(
        method="adagrad",
        step_size=step_size,
        eps=eps,
        initial_accumulator=initial_accumulator,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_adaptive_step(config),
        optimizer_name="adagrad",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="adagrad"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )


def rmsprop(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    step_size: float | None = None,
    decay: float = 0.9,
    eps: float = 1.0e-8,
    initial_accumulator: float = 0.0,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run RMSProp with moving-average squared-gradient scaling."""

    config = AdaptiveConfig(
        method="rmsprop",
        step_size=step_size,
        decay=decay,
        eps=eps,
        initial_accumulator=initial_accumulator,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_adaptive_step(config),
        optimizer_name="rmsprop",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="rmsprop"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )
