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
- future projected optimizers can call repair after tentative energy steps.

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
from optimizer.blackbox.run import BlackBoxRun, ensure_run
from optimizer.system_olgs import optional_residuals, validate_controls_for_system
from optimizer.utils.derivatives import get_jacobian


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

    def to_dict(self, *, include_controls: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly repair summary."""

        payload: dict[str, Any] = {
            "residual_norm": float(self.residual_norm),
            "residual_max_abs": float(self.residual_max_abs),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "method": self.method,
            "residual_name": self.residual_name,
            "jacobian_source": self.jacobian_source,
            "stop_reason": self.stop_reason,
            "history": list(self.history),
        }
        if include_controls:
            payload["controls"] = {
                "spec": self.controls.spec.to_dict(),
                "matrix": self.controls.as_matrix(copy=True).tolist(),
            }
        return payload


def _controls_from_delta(controls: Controls, delta: np.ndarray, *, scale: float, name: str) -> Controls:
    """Apply a flat repair delta to controls."""

    flat = controls.flatten(copy=False).astype(float, copy=False)
    return Controls.from_flat(controls.spec, flat + float(scale) * delta, name=name)


def _clip_delta(delta: np.ndarray, max_step_norm: float | None) -> tuple[np.ndarray, float, bool]:
    """Clip repair step length if requested."""

    raw = np.asarray(delta, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(raw))
    if max_step_norm is None:
        return raw.copy(), norm, False
    limit = float(max_step_norm)
    if limit <= 0.0 or not np.isfinite(limit):
        raise ValueError("max_step_norm must be finite and positive.")
    if norm <= limit or norm == 0.0:
        return raw.copy(), norm, False
    return raw * (limit / norm), norm, True


def _solve_repair_delta(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    method: str,
    damping: float,
    rcond: float | None,
) -> np.ndarray:
    """Solve for a flat control step that reduces residuals."""

    if method == "newton":
        # ``lstsq(J, -r)`` returns the least-squares/minimum-norm update when the
        # system is underdetermined, which is common for time-discretized controls.
        return np.linalg.lstsq(jacobian, -residual, rcond=rcond)[0]

    if method in {"lm", "damped"}:
        gram = jacobian @ jacobian.T
        gram = gram + float(damping) * np.eye(gram.shape[0])
        multiplier = np.linalg.lstsq(gram, -residual, rcond=rcond)[0]
        return jacobian.T @ multiplier

    raise ValueError("method must be 'newton', 'lm', or 'damped'.")


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
    blackbox: BlackBoxRun | str | bool | None = None,
    blackbox_policy: Any | None = None,
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

    current = controls.copy(name=controls.name)
    active_blackbox = ensure_run(blackbox, policy=blackbox_policy)
    if active_blackbox is not None:
        active_blackbox.record_start(
            system=system,
            controls=controls,
            optimizer="repair_newton",
            stage="repair",
            objective={"residuals": residuals, "mode": "repair"},
            config={
                "method": method,
                "maxiter": int(maxiter),
                "tolerance": tolerance,
                "damping": damping,
                "fallback": bool(fallback),
                "eps": float(eps),
                "max_step_norm": max_step_norm,
                "line_search": bool(line_search),
                "shrink": shrink,
                "max_backtracks": int(max_backtracks),
            },
        )
    history: list[dict[str, Any]] = []
    jacobian_source = "uncomputed"
    stop_reason = "maxiter"

    for iteration in range(int(maxiter) + 1):
        residual_vec = optional_residuals(system, current, name=residuals)
        residual_norm = float(np.linalg.norm(residual_vec))
        residual_max_abs = float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0
        history.append(
            {
                "iteration": int(iteration),
                "residual_norm": residual_norm,
                "residual_max_abs": residual_max_abs,
            }
        )
        if residual_norm <= tolerance:
            stop_reason = "converged"
            result = RepairResult(
                controls=current,
                residuals=residual_vec,
                residual_norm=residual_norm,
                residual_max_abs=residual_max_abs,
                converged=True,
                iterations=iteration,
                method=method,
                residual_name=residuals,
                jacobian_source=jacobian_source,
                history=history,
                stop_reason=stop_reason,
            )
            if active_blackbox is not None:
                active_blackbox.record_repair(
                    method=method,
                    residual_name=residuals,
                    before_controls=controls,
                    after_controls=result.controls,
                    result=result,
                    stage="repair",
                )
            return result
        if iteration == int(maxiter):
            break

        jacobian, jacobian_source = get_jacobian(
            system,
            current,
            residuals=residuals,
            fallback=fallback,
            eps=eps,
        )
        delta = _solve_repair_delta(
            jacobian,
            residual_vec,
            method=method,
            damping=damping,
            rcond=rcond,
        )
        delta, raw_delta_norm, clipped = _clip_delta(delta, max_step_norm)

        accepted = False
        best_candidate = current
        best_norm = residual_norm
        attempts = int(max_backtracks) if line_search else 1
        step_scale = 1.0
        for attempt in range(attempts):
            candidate = _controls_from_delta(
                current,
                delta,
                scale=step_scale,
                name=f"repair_{method}_trial",
            )
            candidate_residual = optional_residuals(system, candidate, name=residuals)
            candidate_norm = float(np.linalg.norm(candidate_residual))
            if candidate_norm < best_norm:
                accepted = True
                best_candidate = candidate
                best_norm = candidate_norm
                break
            step_scale *= shrink

        history[-1].update(
            {
                "jacobian_source": jacobian_source,
                "jacobian_shape": list(jacobian.shape),
                "raw_delta_norm": raw_delta_norm,
                "applied_delta_norm": float(np.linalg.norm(delta)),
                "step_clipped": bool(clipped),
                "accepted": bool(accepted),
                "accepted_scale": float(step_scale if accepted else 0.0),
                "trial_residual_norm": float(best_norm),
            }
        )
        if not accepted:
            stop_reason = "line_search_failed"
            break
        current = best_candidate

    final_residual = optional_residuals(system, current, name=residuals)
    final_norm = float(np.linalg.norm(final_residual))
    final_max_abs = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
    result = RepairResult(
        controls=current,
        residuals=final_residual,
        residual_norm=final_norm,
        residual_max_abs=final_max_abs,
        converged=bool(final_norm <= tolerance),
        iterations=max(0, len(history) - 1),
        method=method,
        residual_name=residuals,
        jacobian_source=jacobian_source,
        history=history,
        stop_reason="converged" if final_norm <= tolerance else stop_reason,
    )
    if active_blackbox is not None:
        active_blackbox.record_repair(
            method=method,
            residual_name=residuals,
            before_controls=controls,
            after_controls=result.controls,
            result=result,
            stage="repair",
        )
    return result
