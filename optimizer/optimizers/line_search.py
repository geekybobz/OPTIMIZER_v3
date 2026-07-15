"""Gradient-descent and line-search optimizer family.

Why this file exists
--------------------
The first optimizer in v3 should be the easiest one to audit.  Line search is that
method: compute the analytical gradient, try a step in the descent direction, shrink
the step when the trial is not acceptable, and let the shared engine record the run.
It gives the project a deterministic baseline before adding momentum and Adam memory.

How it fits the architecture
----------------------------
- the system owns ``evaluate`` and ``gradient``.
- this module owns only the line-search proposal logic.
- ``core.engine.run_chunk`` owns iteration counting, acceptance recording, tracing,
  checkpoints, best-so-far tracking, and warmstart handoff.
- ``optimizer.library`` exposes this as ``opt.line_search(...)`` and
  ``ctx.line_search(...)``.

What this file deliberately does not do
---------------------------------------
It does not implement Newton repair, residual projection, Fourier guesses, or
constraint restoration.  Those are later utility/advanced phases.  This optimizer is
only a first-order control mover.

Reviewer invariants
-------------------
- all control updates are vectorized over the dense ``Controls`` matrix.
- backtracking uses the engine evaluator, so repeated trial evaluations are cached.
- optimizer state records step-size adaptation and accept/reject counters.
- branch-specific proposal state prevents failed steps from being counted as accepted
  optimizer progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.core.engine import (
    AcceptanceDecision,
    StepContext,
    StepProposal,
    default_accept,
    run_chunk,
)
from optimizer.optimizers._common import (
    coerce_warmstart,
    controls_from_flat_like,
    require_finite,
    require_variant,
)
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState


VALID_VARIANTS = ("fixed", "backtracking", "normalized", "armijo")


@dataclass(frozen=True)
class LineSearchConfig:
    """Validated user options for ``opt.line_search``."""

    variant: str = "backtracking"
    step_size: float | None = None
    min_step: float = 1.0e-12
    max_step: float | None = None
    shrink: float = 0.5
    grow: float = 1.0
    max_backtracks: int = 12
    normalize: bool | None = None
    armijo_c1: float = 1.0e-4
    accept_metric: str = "J"
    accept_mode: str = "min"
    accept_tolerance: float = 0.0

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family="line_search")
        if self.accept_mode not in {"min", "max"}:
            raise ValueError("accept_mode must be 'min' or 'max'.")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_finite("min_step", self.min_step, positive=True)
        require_finite("max_step", self.max_step, positive=True)
        require_finite("shrink", self.shrink, positive=True)
        require_finite("grow", self.grow, positive=True)
        require_finite("armijo_c1", self.armijo_c1, positive=True)
        require_finite("accept_tolerance", self.accept_tolerance, nonnegative=True)
        if not 0.0 < float(self.shrink) < 1.0:
            raise ValueError("shrink must satisfy 0 < shrink < 1.")
        if int(self.max_backtracks) < 1:
            raise ValueError("max_backtracks must be >= 1.")
        if self.max_step is not None and float(self.max_step) < float(self.min_step):
            raise ValueError("max_step must be >= min_step when provided.")

    @property
    def initial_step_size(self) -> float:
        """Default step used when no state/warmstart step is available."""

        return 1.0e-2 if self.step_size is None else float(self.step_size)

    @property
    def uses_normalized_direction(self) -> bool:
        """Whether the gradient direction should be unit length."""

        if self.normalize is not None:
            return bool(self.normalize)
        return self.variant == "normalized"

    @property
    def attempts_per_iteration(self) -> int:
        """Number of trial step sizes the step function may inspect."""

        if self.variant in {"fixed", "normalized"}:
            return 1
        return int(self.max_backtracks)


def _clamp_step(step_size: float, config: LineSearchConfig) -> float:
    """Apply min/max step-size limits."""

    out = max(float(config.min_step), float(step_size))
    if config.max_step is not None:
        out = min(out, float(config.max_step))
    return out


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    """Read a scalar metric for line-search checks."""

    if key not in metrics:
        raise KeyError(f"metrics do not include {key!r}.")
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f"metric {key!r} must be scalar for line search.")
    return float(value)


def _armijo_accepts(
    *,
    current: Evaluation,
    trial: Evaluation,
    step_size: float,
    directional_derivative: float,
    config: LineSearchConfig,
) -> tuple[bool, float, float]:
    """Return Armijo decision plus current/trial thresholds.

    For minimization, a descent direction has a negative directional derivative and
    the Armijo threshold is below the current metric.  For maximization the signs are
    reversed; the same formula still gives the required threshold.
    """

    current_value = _metric_value(current.metrics, config.accept_metric)
    trial_value = _metric_value(trial.metrics, config.accept_metric)
    threshold = current_value + float(config.armijo_c1) * float(step_size) * float(
        directional_derivative
    )
    if config.accept_mode == "min":
        return trial_value <= threshold + float(config.accept_tolerance), trial_value, threshold
    return trial_value >= threshold - float(config.accept_tolerance), trial_value, threshold


def _line_search_accept(config: LineSearchConfig, custom_accept: Any | None = None):
    """Build the engine accept function for this line-search config."""

    def accept(
        current: Evaluation,
        trial: Evaluation,
        proposal: StepProposal,
        state: RunState,
    ) -> AcceptanceDecision:
        technical = dict(proposal.technical)
        if not bool(technical.get("line_search_success", True)):
            return AcceptanceDecision(
                accepted=False,
                reason=str(technical.get("line_search_reason", "line_search_failed")),
                technical={"line_search": technical},
            )

        if custom_accept is not None:
            decision = custom_accept(current, trial, proposal, state)
            if isinstance(decision, AcceptanceDecision):
                decision.technical.setdefault("line_search", technical)
                return decision
            if isinstance(decision, bool):
                return AcceptanceDecision(
                    accepted=decision,
                    reason="accepted" if decision else "rejected",
                    technical={"line_search": technical},
                )
            raise TypeError("custom accept must return AcceptanceDecision or bool.")

        if config.variant == "armijo":
            accepted, trial_value, threshold = _armijo_accepts(
                current=current,
                trial=trial,
                step_size=float(technical["accepted_step_size"]),
                directional_derivative=float(technical["directional_derivative"]),
                config=config,
            )
            return AcceptanceDecision(
                accepted=accepted,
                reason="accepted" if accepted else "rejected_armijo",
                technical={
                    "accept_metric": config.accept_metric,
                    "accept_mode": config.accept_mode,
                    "trial_value": trial_value,
                    "armijo_threshold": threshold,
                    "line_search": technical,
                },
            )

        decision = default_accept(
            current,
            trial,
            proposal,
            state,
            metric=config.accept_metric,
            mode=config.accept_mode,
            tolerance=config.accept_tolerance,
        )
        decision.technical["line_search"] = technical
        return decision

    return accept


def _build_line_search_step(config: LineSearchConfig):
    """Create a step function consumed by ``run_chunk``."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        current_controls = state.controls
        current_flat = current_controls.flatten(copy=False)
        gradient_flat = context.gradient.flatten(copy=False)
        gradient_norm = float(np.linalg.norm(gradient_flat))
        accept_count = int(state.optimizer_state.get("accept_count", 0))
        reject_count = int(state.optimizer_state.get("reject_count", 0))

        if gradient_norm == 0.0:
            next_step = _clamp_step(state.step_size or config.initial_step_size, config)
            reject_state = {
                "variant": config.variant,
                "step_size": next_step,
                "accept_count": accept_count,
                "reject_count": reject_count + 1,
                "last_gradient_norm": gradient_norm,
            }
            return StepProposal(
                controls=current_controls.copy(name="line_search_noop"),
                step_size_on_reject=next_step,
                optimizer_state_on_reject=reject_state,
                technical={
                    "line_search_success": False,
                    "line_search_reason": "zero_gradient",
                    "gradient_norm": gradient_norm,
                },
            )

        direction_sign = -1.0 if config.accept_mode == "min" else 1.0
        direction = direction_sign * gradient_flat.astype(float, copy=True)
        if config.uses_normalized_direction:
            direction /= gradient_norm
        direction_norm = float(np.linalg.norm(direction))
        directional_derivative = float(np.dot(gradient_flat, direction))

        base_step = _clamp_step(state.step_size or config.initial_step_size, config)
        trial_step = base_step
        attempts: list[dict[str, Any]] = []
        best_candidate: Controls | None = None
        accepted_step_size: float | None = None

        for attempt in range(config.attempts_per_iteration):
            candidate_flat = current_flat + trial_step * direction
            candidate = controls_from_flat_like(
                current_controls,
                candidate_flat,
                name=f"line_search_trial_{attempt}",
            )

            # Fixed and normalized variants let the engine perform the only trial
            # evaluation.  Backtracking/Armijo inspect candidates here so the step
            # function can shrink before handing a proposal to the engine.
            if config.variant in {"fixed", "normalized"}:
                best_candidate = candidate
                accepted_step_size = trial_step
                break

            outcome = context.evaluator.try_evaluate(candidate)
            attempt_record = {
                "attempt": attempt,
                "step_size": trial_step,
                "ok": outcome.ok,
                "error": outcome.error,
            }
            if outcome.ok and outcome.evaluation is not None:
                if config.variant == "armijo":
                    accepted, trial_value, threshold = _armijo_accepts(
                        current=context.evaluation,
                        trial=outcome.evaluation,
                        step_size=trial_step,
                        directional_derivative=directional_derivative,
                        config=config,
                    )
                    attempt_record.update(
                        {
                            "trial_value": trial_value,
                            "armijo_threshold": threshold,
                            "accepted": accepted,
                        }
                    )
                else:
                    decision = default_accept(
                        context.evaluation,
                        outcome.evaluation,
                        StepProposal(candidate, step_size=trial_step),
                        state,
                        metric=config.accept_metric,
                        mode=config.accept_mode,
                        tolerance=config.accept_tolerance,
                    )
                    accepted = decision.accepted
                    attempt_record.update(
                        {
                            "trial_value": _metric_value(
                                outcome.evaluation.metrics,
                                config.accept_metric,
                            ),
                            "accepted": accepted,
                        }
                    )
                if accepted:
                    best_candidate = candidate
                    accepted_step_size = trial_step
                    attempts.append(attempt_record)
                    break

            attempts.append(attempt_record)
            trial_step = _clamp_step(trial_step * float(config.shrink), config)

        if best_candidate is None or accepted_step_size is None:
            rejected_step = _clamp_step(trial_step, config)
            reject_state = {
                "variant": config.variant,
                "step_size": rejected_step,
                "accept_count": accept_count,
                "reject_count": reject_count + 1,
                "last_gradient_norm": gradient_norm,
                "last_direction_norm": direction_norm,
            }
            return StepProposal(
                controls=current_controls.copy(name="line_search_failed"),
                step_size_on_reject=rejected_step,
                optimizer_state_on_reject=reject_state,
                technical={
                    "variant": config.variant,
                    "line_search_success": False,
                    "line_search_reason": "no_acceptable_step",
                    "attempts": attempts,
                    "gradient_norm": gradient_norm,
                    "direction_norm": direction_norm,
                    "directional_derivative": directional_derivative,
                },
            )

        next_step = _clamp_step(accepted_step_size * float(config.grow), config)
        accept_state = {
            "variant": config.variant,
            "step_size": next_step,
            "accept_count": accept_count + 1,
            "reject_count": reject_count,
            "last_gradient_norm": gradient_norm,
            "last_direction_norm": direction_norm,
            "last_accepted_step_size": accepted_step_size,
        }
        reject_state = {
            "variant": config.variant,
            "step_size": _clamp_step(accepted_step_size * float(config.shrink), config),
            "accept_count": accept_count,
            "reject_count": reject_count + 1,
            "last_gradient_norm": gradient_norm,
            "last_direction_norm": direction_norm,
        }
        return StepProposal(
            controls=best_candidate,
            step_size=accepted_step_size,
            step_size_on_accept=next_step,
            step_size_on_reject=reject_state["step_size"],
            optimizer_state_on_accept=accept_state,
            optimizer_state_on_reject=reject_state,
            technical={
                "variant": config.variant,
                "line_search_success": True,
                "accepted_step_size": accepted_step_size,
                "next_step_size_on_accept": next_step,
                "gradient_norm": gradient_norm,
                "direction_norm": direction_norm,
                "directional_derivative": directional_derivative,
                "attempts": attempts,
                "normalized_direction": config.uses_normalized_direction,
            },
        )

    return step


def line_search(
    system: Any,
    controls: Controls | None = None,
    *,
    variant: str = "backtracking",
    maxiter: int = 100,
    step_size: float | None = None,
    min_step: float = 1.0e-12,
    max_step: float | None = None,
    shrink: float = 0.5,
    grow: float = 1.0,
    max_backtracks: int = 12,
    normalize: bool | None = None,
    armijo_c1: float = 1.0e-4,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    accept: Any | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run gradient descent with one of the line-search variants."""

    config = LineSearchConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="line_search"),
        step_size=step_size,
        min_step=min_step,
        max_step=max_step,
        shrink=shrink,
        grow=grow,
        max_backtracks=max_backtracks,
        normalize=normalize,
        armijo_c1=armijo_c1,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
    )
    return run_chunk(
        system,
        controls,
        step=_build_line_search_step(config),
        optimizer_name="line_search",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="line_search"),
        step_size=step_size,
        accept=_line_search_accept(config, custom_accept=accept),
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )
