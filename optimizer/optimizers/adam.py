"""Adam-family adaptive optimizer.

Why this file exists
--------------------
The downstream control problems can have very uneven gradient scales across channels
and time samples.  Adam-style optimizers are useful in that regime because they keep
first- and second-moment estimates and scale each coordinate's step accordingly.  This
module implements Adam as one optimizer family with explicit variants instead of a
collection of wrapper-like public classes.

How it fits the architecture
----------------------------
- the physical system provides the analytical gradient.
- this module owns Adam-family moment updates and trial control proposals.
- ``core.engine.run_chunk`` owns the loop, logging, checkpoints, and acceptance.
- moment buffers live in ``RunState.optimizer_state`` and transfer by warmstart.

What this file deliberately does not do
---------------------------------------
It does not hide energy penalties as optimizer weight decay by default, build physical
objectives, repair residuals, or project gradients.  If energy should be part of the
physics objective, it belongs in ``system.evaluate`` and ``system.gradient``.  AdamW's
decoupled weight decay is available only as an explicit optimizer option.

Reviewer invariants
-------------------
- ``m``, ``v``, and optional ``v_max`` are flat vectors matching the control size.
- accepted steps advance moments and timestep; rejected steps keep previous moments.
- AMSGrad uses the running maximum of bias-corrected second moments.
- RAdam falls back to momentum-like behavior until the variance rectifier is valid.
- optional step clipping limits the actual update norm without changing direction.
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


VALID_VARIANTS = ("adam", "amsgrad", "adamw", "radam", "adabelief")


@dataclass(frozen=True)
class AdamConfig:
    """Validated options for ``opt.adam``."""

    variant: str = "adam"
    step_size: float | None = None
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family="adam")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_probability_like("beta1", self.beta1)
        require_probability_like("beta2", self.beta2)
        require_finite("eps", self.eps, positive=True)
        require_finite("weight_decay", self.weight_decay, nonnegative=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)

    @property
    def initial_step_size(self) -> float:
        """Default Adam learning rate used without explicit state."""

        return 1.0e-3 if self.step_size is None else float(self.step_size)


def _radam_rectifier(beta2: float, t: int) -> tuple[float, float]:
    """Return ``(rho_t, rectifier)`` for RAdam.

    When ``rho_t`` is too small, the adaptive denominator is not reliable.  RAdam then
    behaves like a momentum method by using a rectifier of zero in the caller.
    """

    beta2_t = float(beta2) ** int(t)
    rho_inf = 2.0 / (1.0 - float(beta2)) - 1.0
    rho_t = rho_inf - (2.0 * int(t) * beta2_t) / (1.0 - beta2_t)
    if rho_t <= 4.0:
        return rho_t, 0.0
    numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
    denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
    return rho_t, float(np.sqrt(numerator / denominator))


def _state_payload(
    *,
    config: AdamConfig,
    t: int,
    m: np.ndarray,
    v: np.ndarray,
    v_max: np.ndarray,
    accepted: int,
    rejected: int,
    step_size: float,
    gradient_norm: float,
    raw_step_norm: float,
    clipped: bool,
) -> dict[str, Any]:
    """Build the optimizer-state payload for Adam branch updates."""

    return {
        "variant": config.variant,
        "t": int(t),
        "m": m.copy(),
        "v": v.copy(),
        "v_max": v_max.copy(),
        "beta1": float(config.beta1),
        "beta2": float(config.beta2),
        "eps": float(config.eps),
        "weight_decay": float(config.weight_decay),
        "step_size": float(step_size),
        "accept_count": int(accepted),
        "reject_count": int(rejected),
        "last_gradient_norm": float(gradient_norm),
        "last_raw_step_norm": float(raw_step_norm),
        "last_step_clipped": bool(clipped),
    }


def _adaptive_direction(
    *,
    config: AdamConfig,
    current_flat: np.ndarray,
    gradient_flat: np.ndarray,
    m_next: np.ndarray,
    v_next: np.ndarray,
    v_max_previous: np.ndarray,
    t_next: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute the Adam-family coordinate-scaled update direction."""

    beta1 = float(config.beta1)
    beta2 = float(config.beta2)
    m_hat = m_next / (1.0 - beta1**t_next)
    v_hat = v_next / (1.0 - beta2**t_next)
    v_max_next = v_max_previous.copy()
    technical: dict[str, Any] = {
        "bias_correction1": 1.0 - beta1**t_next,
        "bias_correction2": 1.0 - beta2**t_next,
    }

    if config.variant == "amsgrad":
        v_max_next = np.maximum(v_max_previous, v_hat)
        direction = m_hat / (np.sqrt(v_max_next) + float(config.eps))
        technical["amsgrad_v_max_norm"] = float(np.linalg.norm(v_max_next))
    elif config.variant == "radam":
        rho_t, rectifier = _radam_rectifier(beta2, t_next)
        technical["radam_rho_t"] = rho_t
        technical["radam_rectifier"] = rectifier
        if rectifier > 0.0:
            direction = rectifier * m_hat / (np.sqrt(v_hat) + float(config.eps))
        else:
            direction = m_hat
    else:
        direction = m_hat / (np.sqrt(v_hat) + float(config.eps))

    if config.variant == "adamw" or float(config.weight_decay) > 0.0:
        direction = direction + float(config.weight_decay) * current_flat
        technical["decoupled_weight_decay"] = float(config.weight_decay)

    return direction, v_max_next, technical


def _build_adam_step(config: AdamConfig):
    """Create the engine step function for Adam-family variants."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        controls = state.controls
        current_flat = controls.flatten(copy=False)
        gradient_flat = context.gradient.flatten(copy=False).astype(float, copy=False)
        gradient_norm = float(np.linalg.norm(gradient_flat))
        memory = dict(state.optimizer_state)
        size = controls.spec.size

        m = flat_state_array(memory, "m", size=size)
        v = flat_state_array(memory, "v", size=size)
        v_max = flat_state_array(memory, "v_max", size=size)
        t = int(memory.get("t", 0))
        accepted = int(memory.get("accept_count", 0))
        rejected = int(memory.get("reject_count", 0))
        step_size = float(state.step_size or config.initial_step_size)

        beta1 = float(config.beta1)
        beta2 = float(config.beta2)
        t_next = t + 1
        m_next = beta1 * m + (1.0 - beta1) * gradient_flat
        if config.variant == "adabelief":
            surprise = gradient_flat - m_next
            v_next = beta2 * v + (1.0 - beta2) * (surprise * surprise)
        else:
            v_next = beta2 * v + (1.0 - beta2) * (gradient_flat * gradient_flat)

        direction, v_max_next, variant_technical = _adaptive_direction(
            config=config,
            current_flat=current_flat,
            gradient_flat=gradient_flat,
            m_next=m_next,
            v_next=v_next,
            v_max_previous=v_max,
            t_next=t_next,
        )
        raw_update = step_size * direction
        update, raw_step_norm, clipped = clip_vector(raw_update, config.max_step_norm)
        candidate = controls_from_flat_like(
            controls,
            current_flat - update,
            name=f"adam_{config.variant}_trial",
        )

        accept_state = _state_payload(
            config=config,
            t=t_next,
            m=m_next,
            v=v_next,
            v_max=v_max_next,
            accepted=accepted + 1,
            rejected=rejected,
            step_size=step_size,
            gradient_norm=gradient_norm,
            raw_step_norm=raw_step_norm,
            clipped=clipped,
        )
        reject_state = _state_payload(
            config=config,
            t=t,
            m=m,
            v=v,
            v_max=v_max,
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
                "step_size": step_size,
                "beta1": beta1,
                "beta2": beta2,
                "eps": float(config.eps),
                "gradient_norm": gradient_norm,
                "m_norm_before": float(np.linalg.norm(m)),
                "m_norm_after": float(np.linalg.norm(m_next)),
                "v_norm_before": float(np.linalg.norm(v)),
                "v_norm_after": float(np.linalg.norm(v_next)),
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "clipped": clipped,
                **variant_technical,
            },
        )

    return step


def adam(
    system: Any,
    controls: Controls | None = None,
    *,
    variant: str = "adam",
    maxiter: int = 100,
    step_size: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run Adam or one of its explicit variants."""

    config = AdamConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="adam"),
        step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_adam_step(config),
        optimizer_name="adam",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="adam"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )
