"""Short-run interpretation and in-memory method comparison."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.numerics import jacobian_decomposition
from optimizer.core.curvature import curvature_evidence
from optimizer.core.stop_reasons import StopReason
from optimizer.system_olgs import get_system_spec


def _trend(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "start": None, "end": None, "change": None, "relative_change": None}
    change = values[-1] - values[0]
    scale = max(abs(values[0]), np.finfo(float).eps)
    return {
        "count": len(values),
        "start": values[0],
        "end": values[-1],
        "change": change,
        "relative_change": change / scale,
    }


# Classification thresholds, relative to the run's starting value so that the
# same dynamics get the same label at any objective scale (for example an
# infidelity of 1e-6 as much as an energy of 1e3).
_DIAGNOSIS_THRESHOLDS = {
    "stalled_relative_change": 1.0e-6,
    "slow_relative_improvement": 1.0e-2,
    "diverging_relative_increase": 1.0e-3,
    "oscillation_reversal_fraction": 1.0 / 3.0,
    "repeating_rejection_ratio": 0.8,
}


def _reversals(values: list[float]) -> int:
    """Count direction changes along a sequence, ignoring exact ties."""

    steps = [step for step in np.diff(values) if step != 0.0]
    return int(sum(1 for left, right in zip(steps, steps[1:]) if left * right < 0.0))


def _classify_summary(relative, reversals, rejection_ratio, attempts):
    limits = _DIAGNOSIS_THRESHOLDS
    if attempts >= 3 and rejection_ratio >= limits["repeating_rejection_ratio"]:
        return "repeating_rejections"
    if attempts >= 3 and reversals >= max(2, int(np.ceil(attempts * limits["oscillation_reversal_fraction"]))):
        return "oscillating"
    if relative > limits["diverging_relative_increase"]:
        return "diverging"
    if relative >= -limits["stalled_relative_change"]:
        return "stalled"
    if -relative < limits["slow_relative_improvement"]:
        return "slow"
    return "converging"


def _classify(values, *, rejection_ratio, attempts):
    """Sequence adapter for the same aggregate classifier used by results."""
    reversals = _reversals(values)
    if len(values) < 2:
        relative = 0.0
    else:
        scale = abs(values[0]) or max(max(abs(v) for v in values), np.finfo(float).tiny)
        relative = (values[-1] - values[0]) / scale
    label = _classify_summary(relative, reversals, rejection_ratio, attempts)
    return label, (None if label == "repeating_rejections" else relative), reversals


def _stop_facts(reason: str | None) -> dict[str, Any]:
    """Outcome class and continuability of a stop reason from the shared registry."""

    entry = StopReason.lookup(reason)
    return {
        "stop_outcome": None if entry is None else entry.outcome,
        "stop_continuable": None if entry is None else entry.continuable,
    }


def diagnose_run(result: Any, *, mode: str | None = None) -> dict[str, Any]:
    """Classify the common numerical record without evaluating the system."""
    from optimizer.result import OptimizerResult
    from optimizer.system_olss import ShootingResult
    if not isinstance(result, (OptimizerResult, ShootingResult)):
        raise TypeError("diagnose_run expects an OptimizerResult or a ShootingResult.")
    shooting = isinstance(result, ShootingResult)
    resolved_mode = "min" if mode is None else str(mode)
    if resolved_mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    base = dict(kind="run_diagnosis", family="olss" if shooting else "olgs",
                stop_reason=result.stop_reason, **_stop_facts(result.stop_reason),
                thresholds=dict(_DIAGNOSIS_THRESHOLDS))
    if shooting:
        base.update(method=result.method, solver_converged=bool(result.solver_converged), qualified=bool(result.qualified))
        data = {} if result.run_record is None else result.run_record.summary
        if not data:
            values = [float(row["residual_norm"]) for row in result.history if "residual_norm" in row]
            accepted = [bool(row["accepted"]) for row in result.history if "accepted" in row]
            data = _legacy_aggregates(values, accepted)
    else:
        base.update(optimizer=result.optimizer, objective_metric=result.objective_metric, objective_mode=resolved_mode)
        # Keep explicit diagnostics removal as the existing way to suppress evidence.
        data = result.diagnostics
    if not data:
        return dict(base, classification="insufficient_evidence",
                    final_value=None if shooting and not np.isfinite(result.residual_norm) else (result.residual_norm if shooting else result.objective_value),
                    iterations=result.iterations)
    attempts, reversals = data["attempts"], data["reversals"]
    rejection_ratio = data["rejected"] / attempts if attempts else 0.0
    scale = abs(data["start"]) or max(abs(data["minimum"]), abs(data["maximum"]), np.finfo(float).tiny)
    relative = (data["end"] - data["start"]) / scale * (1 if resolved_mode == "min" else -1)
    label = ("converging" if shooting and result.solver_converged else
             _classify_summary(relative, reversals, rejection_ratio, attempts))
    trend = _trend([data["start"], data["end"]])
    trend["count"] = attempts + 1
    key = "residual_trend" if shooting else "objective_trend"
    return dict(base, classification=label, **{key: trend}, relative_change=relative,
                direction_reversals=reversals, rejection_ratio=rejection_ratio, iterations=attempts,
                evidence="constant_size_numerical_aggregates",
                samples_truncated=bool(result.run_record is not None and result.run_record.truncated))


def _legacy_aggregates(values, accepted):
    if not values:
        return {}
    return dict(start=values[0], end=values[-1], minimum=min(values), maximum=max(values),
                attempts=len(accepted), rejected=sum(not value for value in accepted),
                reversals=_reversals(values))


def curvature_probe(
    system: Any,
    controls: Controls,
    *,
    directions: int = 8,
    eps: float = 1.0e-5,
    seed: int = 0,
) -> dict[str, Any]:
    """Report local curvature, a negative-curvature flag, and a step estimate."""

    return curvature_evidence(system, controls, directions=directions, eps=eps, seed=seed)


def compare_methods(
    system: Any,
    controls: Controls,
    methods: Iterable[str | Callable[..., Any]] | Mapping[str, Callable[..., Any]],
    *,
    iterations: int = 10,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run several methods briefly from the same controls and rank in memory."""

    if int(iterations) < 1:
        raise ValueError("iterations must be >= 1.")
    from optimizer import optimizers

    entries: list[tuple[str, Callable[..., Any]]] = []
    if isinstance(methods, Mapping):
        entries = [(str(name), method) for name, method in methods.items()]
    else:
        for item in methods:
            if isinstance(item, str):
                method = getattr(optimizers, item, None)
                if not callable(method):
                    raise KeyError(f"Unknown optimizer method {item!r}.")
                entries.append((item, method))
            elif callable(item):
                entries.append((getattr(item, "__name__", type(item).__name__), item))
            else:
                raise TypeError("methods entries must be names or callables.")
    if not entries:
        raise ValueError("methods must contain at least one optimizer.")
    method_options = {str(key): dict(value) for key, value in dict(options or {}).items()}
    objective = get_system_spec(system)["objective"]
    metric = str(objective["metric_id"])
    direction = str(objective["direction"])
    rows = []
    for name, method in entries:
        kwargs = dict(method_options.get(name, {}))
        kwargs.setdefault("maxiter", int(iterations))
        result = method(system, controls.copy(name=controls.name), **kwargs)
        rows.append(
            {
                "method": name,
                "objective": float(result.metrics[metric]),
                "stop_reason": result.stop_reason,
                "iterations": result.iterations,
                "diagnosis": diagnose_run(
                    result,
                    mode="min" if direction == "minimize" else "max",
                ),
                "result": result,
            }
        )
    rows.sort(key=lambda row: row["objective"], reverse=direction == "maximize")
    return {
        "kind": "method_comparison",
        "objective_metric": metric,
        "objective_direction": direction,
        "iterations": int(iterations),
        "ranking": rows,
    }


def shooting_geometry_probe(
    system: Any,
    variables: Any,
    *,
    options: Mapping[str, Any] | None = None,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Report OLSS Jacobian rank, conditioning, weak blocks, and residual groups."""

    from optimizer.shooting import ShootingEvaluator

    runtime = ShootingEvaluator(system)
    residuals, jacobian = runtime.residuals_and_jacobian(variables, options=options)
    singular_values, right_t, rank, threshold, condition = jacobian_decomposition(jacobian * variables.scales, rcond=rcond)
    weak = right_t[-1] if right_t.size else np.zeros(variables.spec.size)
    weak_blocks = {
        block.name: float(np.linalg.norm(weak[slice(*variables.spec.offsets[block.name])]))
        for block in variables.spec.blocks
    }
    groups = []
    cursor = 0
    for group in runtime.system_spec["residuals"].get("groups", ()):
        size = int(np.prod(group.get("shape", (1,))))
        values = residuals[cursor : cursor + size]
        groups.append(
            {
                "name": str(group.get("name", f"group_{len(groups)}")),
                "size": size,
                "norm": float(np.linalg.norm(values)),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
            }
        )
        cursor += size
    if cursor < residuals.size:
        values = residuals[cursor:]
        groups.append(
            {
                "name": "undeclared",
                "size": int(values.size),
                "norm": float(np.linalg.norm(values)),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
            }
        )
    return {
        "kind": "shooting_geometry_probe",
        "coordinates": "scaled",
        "jacobian_shape": list(jacobian.shape),
        "singular_values": singular_values.tolist(),
        "rank": rank,
        "rank_tolerance": threshold,
        "condition": condition,
        "weak_direction_by_block": weak_blocks,
        "residual_norm": float(np.linalg.norm(residuals)),
        "residual_groups": groups,
        "counters": runtime.counters(compact=True),
    }
