"""Residual repair utilities.

Why this file exists
--------------------
Some optimization stages produce controls that are close to useful but violate hard
analytical residuals.  A repair tool has a different job from Adam or momentum: it
does not follow the objective gradient; it solves a local residual equation and tries
to move controls back toward feasibility.

How it fits the architecture
----------------------------
- systems provide ``residuals`` and optional ``jacobian`` hooks.
- ``derivatives.get_jacobian`` supplies analytical Jacobians or explicit finite-
  difference fallbacks.
- ``repair_newton`` returns a rich ``RepairResult`` for logs and review.
- the projected optimizer implements counted restoration through ``SystemEvaluator``;
  this standalone repair remains an explicit workflow/diagnostic tool.

What this file deliberately does not do
---------------------------------------
It does not optimize energy, fidelity, or a curriculum objective.  It only reduces a
named residual vector such as ``hard`` or ``residual_polish``.

Reviewer invariants
-------------------
- accepted repair steps must reduce residual norm unless the caller disables line
  search by using a single full step.
- LM/Newton solves are minimum-norm style, suitable for many-control/few-residual
  problems.
- every returned result includes final residuals and iteration history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.iteration import drive, backtrack
from optimizer.core.numerics import residual_step, difference_jacobian
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.progress import _Progress
from optimizer.core.record import RunRecord
from optimizer.system_olgs import optional_residuals, validate_controls_for_system
from optimizer.serialization import to_plain


@dataclass
class RepairResult:
    """Public result returned by ``opt.repair_newton``."""

    controls: Controls
    residuals: np.ndarray
    residual_norm: float
    residual_max_abs: float
    converged: bool
    iterations: int
    method: str
    residual_name: str
    jacobian_source: str
    history: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "unknown"
    run_record: RunRecord | None = None
    counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self, *, include_controls: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly repair summary."""

        payload: dict[str, Any] = {
            "residual_norm": to_plain(self.residual_norm),
            "residual_max_abs": to_plain(self.residual_max_abs),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "method": self.method,
            "residual_name": self.residual_name,
            "jacobian_source": self.jacobian_source,
            "stop_reason": self.stop_reason,
            "history": to_plain(self.history),
            "run_record": None if self.run_record is None else self.run_record.to_dict(),
            "counters": dict(self.counters),
        }
        if include_controls:
            payload["controls"] = {
                "spec": self.controls.spec.to_dict(),
                "matrix": self.controls.as_matrix(copy=True).tolist(),
            }
        return payload


def _clip_delta(delta: np.ndarray, max_step_norm: float | None, scales=None) -> tuple[np.ndarray, float, bool]:
    """Clip repair step length if requested."""

    raw = np.asarray(delta, dtype=float).reshape(-1)
    physical_norm = float(np.linalg.norm(raw))
    norm = float(np.linalg.norm(raw if scales is None else raw / scales))
    if max_step_norm is None:
        return raw.copy(), physical_norm, False
    limit = float(max_step_norm)
    if limit <= 0.0 or not np.isfinite(limit):
        raise ValueError("max_step_norm must be finite and positive.")
    if norm <= limit or norm == 0.0:
        return raw.copy(), physical_norm, False
    return raw * (limit / norm), physical_norm, True




def repair_newton(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    method: str = "lm",
    maxiter: int = 10,
    tolerance: float = 1.0e-10,
    damping: float = 1.0e-8,
    fallback: bool = True,
    eps: float = 1.0e-6,
    rcond: float | None = None,
    max_step_norm: float | None = None,
    line_search: bool = True,
    shrink: float = 0.5,
    max_backtracks: int = 8,
) -> RepairResult:
    """Repair controls by reducing a named residual vector."""

    validate_controls_for_system(system, controls)
    method = str(method).lower()
    if method not in {"newton", "lm", "damped"}:
        raise ValueError("method must be 'newton', 'lm', or 'damped'.")
    if int(maxiter) < 0:
        raise ValueError("maxiter must be >= 0.")
    tolerance = float(tolerance)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative.")
    damping = float(damping)
    if damping < 0.0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and nonnegative.")
    shrink = float(shrink)
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must satisfy 0 < shrink < 1.")
    if int(max_backtracks) < 1:
        raise ValueError("max_backtracks must be >= 1.")

    runtime = SystemEvaluator(system)
    declared_bounds = runtime.control_bounds() if runtime.capabilities.get("has_control_bounds", False) else None
    bounds = None if declared_bounds is None else (declared_bounds.lower.flatten(copy=False),
                                                  declared_bounds.upper.flatten(copy=False))
    current = controls.copy(name=controls.name)
    final_residual = runtime.residuals(current, name=residuals)
    progress = _Progress({"residual_norm": float(np.linalg.norm(final_residual))}, "residual_norm")
    history: list[dict[str, Any]] = []
    jacobian_source = "uncomputed"
    stop_reason = "maxiter"

    def advance(iteration):
        nonlocal current, jacobian_source, stop_reason, final_residual
        residual_vec = final_residual
        residual_norm = float(np.linalg.norm(residual_vec))
        residual_max_abs = float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0
        history.append({'iteration': int(iteration), 'residual_norm': residual_norm, 'residual_max_abs': residual_max_abs})
        if len(history) > progress.record.limit:
            del history[0]
        if residual_norm <= tolerance:
            stop_reason = 'converged'
            return stop_reason
        if iteration == int(maxiter):
            return stop_reason
        try:
            if callable(getattr(system, "jacobian", None)):
                jacobian = runtime.jacobian(current, name=residuals)
                jacobian_source = "analytical"
            elif fallback:
                jacobian = difference_jacobian(lambda flat: runtime.residuals(current.with_flat(flat), name=residuals),
                                              current.flatten(copy=False), eps=eps, scales=current.scales)
                jacobian_source = "finite_difference"
            else:
                raise TypeError("Repair requires jacobian() when fallback=False.")
        except Exception as exc:
            history[-1]["error"] = f"{type(exc).__name__}: {exc}"
            return "jacobian_failed"
        try:
            delta = residual_step(jacobian, residual_vec, method=method, damping=damping, rcond=rcond, scales=current.scales, formulation='residual')
        except Exception as exc:
            history[-1]["error"] = f"{type(exc).__name__}: {exc}"
            return "singular_step"
        delta, raw_delta_norm, clipped = _clip_delta(delta, max_step_norm, current.scales)
        selected = backtrack(current, residual_vec, delta,
                             evaluate=lambda trial: runtime.residuals(trial, name=residuals),
                             merit=np.linalg.norm, attempts=int(max_backtracks) if line_search else 1,
                             shrink=shrink, bounds=bounds, name=f"repair_{method}_trial")
        accepted, best_candidate, best_norm, step_scale = selected.accepted, selected.value, selected.merit, selected.scale
        progress.observe({"residual_norm": best_norm}, accepted=accepted,
                         derivative_norm=float(np.linalg.norm(current.scales * (jacobian.T @ residual_vec))),
                         step_norm=float(np.linalg.norm((best_candidate.flatten(copy=False) - current.flatten(copy=False)) / current.scales)))
        history[-1].update({'jacobian_source': jacobian_source, 'jacobian_shape': list(jacobian.shape), 'raw_delta_norm': raw_delta_norm, 'applied_delta_norm': float(np.linalg.norm(delta)), 'step_clipped': bool(clipped), 'accepted': bool(accepted), 'accepted_scale': float(step_scale if accepted else 0.0), 'trial_residual_norm': float(best_norm)})
        if not accepted:
            stop_reason = ('bound_stalled' if selected.bound_stalled else
                           'nonfinite_trial' if selected.attempted and selected.failed == selected.attempted else
                           'line_search_failed')
            return stop_reason
        current = best_candidate
        final_residual = selected.evidence

    stop_reason = drive(advance, maxiter=maxiter, terminal=True)

    final_norm = float(np.linalg.norm(final_residual))
    final_max_abs = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
    result = RepairResult(
        controls=current,
        residuals=final_residual,
        residual_norm=final_norm,
        residual_max_abs=final_max_abs,
        converged=bool(final_norm <= tolerance),
        iterations=int(history[-1]["iteration"]) if history else 0,
        method=method,
        residual_name=residuals,
        jacobian_source=jacobian_source,
        history=history[-progress.record.limit:],
        run_record=progress.record,
        counters=runtime.counters(compact=True),
        stop_reason="converged" if final_norm <= tolerance else stop_reason,
    )
    progress.finish(runtime.counters(compact=True))
    return result
