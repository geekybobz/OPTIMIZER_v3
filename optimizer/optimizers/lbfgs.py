"""Limited-memory BFGS optimizer.

Why this file exists
--------------------
Adam and momentum are useful for rough starts, but smooth deterministic systems often
benefit from curvature information near a good solution.  L-BFGS approximates inverse
Hessian action using only a short history of accepted control differences and gradient
differences, making it practical for large control vectors.

How it fits the architecture
----------------------------
- systems provide analytical gradients.
- this module computes a quasi-Newton search direction with two-loop recursion.
- the shared engine owns acceptance, iteration records, checkpoints, and warmstart.
- history lists live in ``RunState.optimizer_state`` and update only on accepted
  proposals.

What this file deliberately does not do
---------------------------------------
It does not do bound constraints, projected constraints, or full trust-region logic.
It also does not implement a strong-Wolfe line search yet; users can keep step sizes
small or polish after line-search stages.

Reviewer invariants
-------------------
- history length never exceeds ``history_size``.
- curvature pairs with nonpositive ``s dot y`` are skipped.
- rejected trials preserve the previous history.
- the first iteration falls back to steepest descent.
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
)
from optimizer.result import OptimizerResult
from optimizer.state import RunState


@dataclass(frozen=True)
class LBFGSConfig:
    """Validated options for ``opt.lbfgs``."""

    history_size: int = 10
    step_size: float | None = None
    curvature_eps: float = 1.0e-12
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        if int(self.history_size) < 1:
            raise ValueError("history_size must be >= 1.")
        require_finite("step_size", self.initial_step_size, positive=True)
        require_finite("curvature_eps", self.curvature_eps, positive=True)
        require_finite("max_step_norm", self.max_step_norm, positive=True)

    @property
    def initial_step_size(self) -> float:
        return 1.0e-1 if self.step_size is None else float(self.step_size)


def _history_list(memory: dict[str, Any], key: str, *, size: int) -> list[np.ndarray]:
    """Return a validated list of flat history vectors."""

    raw_items = memory.get(key, [])
    out = []
    for item in raw_items:
        arr = np.asarray(item, dtype=float).reshape(-1)
        if arr.shape != (int(size),):
            raise ValueError(f"optimizer_state[{key!r}] contains vector with shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"optimizer_state[{key!r}] contains non-finite values.")
        out.append(arr.copy())
    return out


def _two_loop_direction(gradient: np.ndarray, s_history: list[np.ndarray], y_history: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the L-BFGS inverse-Hessian direction and technical details."""

    if not s_history:
        return -gradient.copy(), {"history_used": 0, "initial_gamma": 1.0}

    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s, y in zip(reversed(s_history), reversed(y_history), strict=True):
        sy = float(np.dot(s, y))
        rho = 1.0 / sy
        alpha = rho * float(np.dot(s, q))
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    last_s = s_history[-1]
    last_y = y_history[-1]
    yy = float(np.dot(last_y, last_y))
    gamma = float(np.dot(last_s, last_y) / yy) if yy > np.finfo(float).tiny else 1.0
    r = gamma * q

    for s, y, alpha, rho in zip(s_history, y_history, reversed(alphas), reversed(rhos), strict=True):
        beta = rho * float(np.dot(y, r))
        r = r + s * (alpha - beta)

    return -r, {"history_used": len(s_history), "initial_gamma": gamma}


def _trim_history(items: list[np.ndarray], *, history_size: int) -> list[np.ndarray]:
    """Keep only the newest ``history_size`` history vectors."""

    if len(items) <= int(history_size):
        return items
    return items[-int(history_size) :]


def _build_lbfgs_step(config: LBFGSConfig):
    """Create the engine step function for L-BFGS."""

    def step(context: StepContext) -> StepProposal:
        state = context.state
        controls = state.controls
        memory = dict(state.optimizer_state)
        current_flat = controls.flatten(copy=False)
        gradient = context.gradient.flatten(copy=False).astype(float, copy=False)
        size = controls.spec.size
        s_history = _history_list(memory, "s_history", size=size)
        y_history = _history_list(memory, "y_history", size=size)
        accepted = int(memory.get("accept_count", 0))
        rejected = int(memory.get("reject_count", 0))
        step_size = float(state.step_size or config.initial_step_size)

        direction, direction_info = _two_loop_direction(gradient, s_history, y_history)
        directional_derivative = float(np.dot(gradient, direction))
        restarted = False
        if directional_derivative >= 0.0:
            direction = -gradient
            directional_derivative = float(np.dot(gradient, direction))
            restarted = True

        raw_update = step_size * direction
        update, raw_step_norm, clipped = clip_vector(raw_update, config.max_step_norm)
        candidate = controls_from_flat_like(controls, current_flat + update, name="lbfgs_trial")

        outcome = context.evaluator.try_gradient(candidate)
        if not outcome.ok or outcome.gradient is None:
            raise ValueError(f"L-BFGS candidate gradient failed: {outcome.error}")
        next_gradient = outcome.gradient.flatten(copy=False).astype(float, copy=False)
        s = candidate.flatten(copy=False) - current_flat
        y = next_gradient - gradient
        curvature = float(np.dot(s, y))
        pair_accepted = curvature > float(config.curvature_eps)

        next_s_history = list(s_history)
        next_y_history = list(y_history)
        if pair_accepted:
            next_s_history.append(s.copy())
            next_y_history.append(y.copy())
            next_s_history = _trim_history(next_s_history, history_size=config.history_size)
            next_y_history = _trim_history(next_y_history, history_size=config.history_size)

        accept_state = {
            "history_size": int(config.history_size),
            "s_history": next_s_history,
            "y_history": next_y_history,
            "step_size": step_size,
            "accept_count": accepted + 1,
            "reject_count": rejected,
            "last_gradient": next_gradient.copy(),
            "last_gradient_norm": float(np.linalg.norm(next_gradient)),
            "last_curvature": curvature,
            "last_pair_accepted": bool(pair_accepted),
            "last_step_clipped": bool(clipped),
        }
        reject_state = {
            "history_size": int(config.history_size),
            "s_history": s_history,
            "y_history": y_history,
            "step_size": step_size,
            "accept_count": accepted,
            "reject_count": rejected + 1,
            "last_gradient": gradient.copy(),
            "last_gradient_norm": float(np.linalg.norm(gradient)),
            "last_curvature": curvature,
            "last_pair_accepted": False,
            "last_step_clipped": bool(clipped),
        }
        return StepProposal(
            controls=candidate,
            step_size=step_size,
            optimizer_state_on_accept=accept_state,
            optimizer_state_on_reject=reject_state,
            technical={
                "history_size": int(config.history_size),
                "history_used": direction_info["history_used"],
                "initial_gamma": direction_info["initial_gamma"],
                "step_size": step_size,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "direction_norm": float(np.linalg.norm(direction)),
                "directional_derivative": directional_derivative,
                "restarted": restarted,
                "curvature": curvature,
                "curvature_pair_accepted": pair_accepted,
                "raw_step_norm": raw_step_norm,
                "applied_step_norm": float(np.linalg.norm(update)),
                "clipped": clipped,
            },
        )

    return step


def lbfgs(
    system: Any,
    controls: Controls | None = None,
    *,
    maxiter: int = 100,
    history_size: int = 10,
    step_size: float | None = None,
    curvature_eps: float = 1.0e-12,
    max_step_norm: float | None = None,
    state: RunState | None = None,
    warmstart: Any | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    **engine_kwargs: Any,
) -> OptimizerResult:
    """Run limited-memory BFGS."""

    config = LBFGSConfig(
        history_size=history_size,
        step_size=step_size,
        curvature_eps=curvature_eps,
        max_step_norm=max_step_norm,
    )
    return run_chunk(
        system,
        controls,
        step=_build_lbfgs_step(config),
        optimizer_name="lbfgs",
        maxiter=maxiter,
        state=state,
        warmstart=coerce_warmstart(warmstart, target_optimizer="lbfgs"),
        step_size=step_size,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
        accept_tolerance=accept_tolerance,
        **engine_kwargs,
    )
