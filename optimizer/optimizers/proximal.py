"""Proximal-gradient optimization for composite objectives."""

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
class ProximalGradientConfig:
    step_size: float = 0.1
    mapping_tolerance: float = 1.0e-8
    max_step_norm: float | None = None
    reject_shrink: float = 0.5

    def __post_init__(self) -> None:
        require_finite("step_size", self.step_size, positive=True)
        require_finite("mapping_tolerance", self.mapping_tolerance, nonnegative=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)
        if not 0.0 < float(self.reject_shrink) < 1.0:
            raise ValueError("reject_shrink must satisfy 0 < reject_shrink < 1.")


def _build_step(config: ProximalGradientConfig):
    def step(context: StepContext) -> StepProposal:
        controls = context.state.controls
        flat = controls.flatten(copy=False).astype(float, copy=False)
        gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
        step_size = float(context.state.step_size or config.step_size)
        smooth_trial = controls_from_flat_like(
            controls,
            flat - step_size * gradient,
            name="proximal_smooth_trial",
        )
        prox_trial = context.evaluator.prox(smooth_trial, step_size)
        mapping = (flat - prox_trial.flatten(copy=False).astype(float, copy=False)) / step_size
        mapping_norm = float(np.linalg.norm(mapping))
        if mapping_norm <= float(config.mapping_tolerance):
            return StepProposal(
                controls=controls.copy(name="proximal_converged"),
                step_size=step_size,
                optimizer_state={"proximal_gradient_mapping_norm": mapping_norm},
                technical={
                    "smooth_gradient_norm": float(np.linalg.norm(gradient)),
                    "proximal_gradient_mapping_norm": mapping_norm,
                    "step_size": step_size,
                },
                stop_reason="proximal_mapping_converged",
            )
        raw_update = prox_trial.flatten(copy=False).astype(float, copy=False) - flat
        update, raw_norm, clipped = clip_vector(raw_update, config.max_step_norm)
        candidate = controls_from_flat_like(controls, flat + update, name="proximal_trial")
        memory = {
            "proximal_gradient_mapping_norm": mapping_norm,
            "smooth_gradient_norm": float(np.linalg.norm(gradient)),
        }
        return StepProposal(
            controls=candidate,
            step_size_on_accept=step_size,
            step_size_on_reject=step_size * float(config.reject_shrink),
            optimizer_state_on_accept=memory,
            optimizer_state_on_reject=memory,
            technical={
                **memory,
                "raw_step_norm": raw_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "step_clipped": clipped,
                "step_size": step_size,
            },
        )

    return step


def proximal_gradient(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    step_size: float = 0.1,
    mapping_tolerance: float = 1.0e-8,
    max_step_norm: float | None = None,
    reject_shrink: float = 0.5,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str | None = None,
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Minimize a composite objective using smooth_gradient and prox hooks."""

    probe = probe_system(system)
    missing = [
        name
        for name, ok in (("smooth_gradient", probe.has_smooth_gradient), ("prox", probe.has_prox))
        if not ok
    ]
    if missing:
        raise TypeError(f"proximal_gradient requires system hooks: {', '.join(missing)}.")
    config = ProximalGradientConfig(
        step_size=step_size,
        mapping_tolerance=mapping_tolerance,
        max_step_norm=max_step_norm,
        reject_shrink=reject_shrink,
    )
    return run_chunk(
        system,
        controls,
        step=_build_step(config),
        optimizer_name="proximal_gradient",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="proximal_gradient"),
        step_size=step_size,
        gradient_source="smooth_gradient",
        prefer_value_and_gradient=False,
        accept_metric=accept_metric,
        accept_mode="min",
        accept_tolerance=accept_tolerance,
        **method_api_context(engine_kwargs, method="proximal_gradient", config=config),
    )
