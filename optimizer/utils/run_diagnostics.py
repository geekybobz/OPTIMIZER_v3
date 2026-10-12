"""Short-run interpretation and in-memory method comparison."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from optimizer.controls import Controls
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


def _classify(
    values: list[float],
    *,
    rejection_ratio: float,
    attempts: int,
) -> tuple[str, float | None, int]:
    """Classify a lower-is-better sequence whose first entry is the starting value."""

    thresholds = _DIAGNOSIS_THRESHOLDS
    if attempts >= 3 and rejection_ratio >= thresholds["repeating_rejection_ratio"]:
        return "repeating_rejections", None, 0
    if len(values) < 2:
        return "stalled", 0.0, 0
    start, end = values[0], values[-1]
    scale = abs(start) if start != 0.0 else max(max(abs(value) for value in values), np.finfo(float).tiny)
    relative = (end - start) / scale
    reversals = _reversals(values)
    if len(values) >= 4 and reversals >= max(2, int(np.ceil((len(values) - 1) * thresholds["oscillation_reversal_fraction"]))):
        return "oscillating", relative, reversals
    if relative > thresholds["diverging_relative_increase"]:
        return "diverging", relative, reversals
    if relative >= -thresholds["stalled_relative_change"]:
        return "stalled", relative, reversals
    if -relative < thresholds["slow_relative_improvement"]:
        return "slow", relative, reversals
    return "converging", relative, reversals


def _stop_facts(reason: str | None) -> dict[str, Any]:
    """Outcome class and continuability of a stop reason from the shared registry."""

    entry = StopReason.lookup(reason)
    return {
        "stop_outcome": None if entry is None else entry.outcome,
        "stop_continuable": None if entry is None else entry.continuable,
    }


def _optimizer_series(result: Any) -> dict[str, Any] | None:
    """Extract the run's own samples from its trace; ``None`` without a trace.

    The start value comes from the run's chunk record, so the trend starts
    before the first step rather than after it.
    """

    trace = getattr(result, "trace", None)
    if trace is None:
        return None
    metric = str(getattr(result, "objective_metric", "J"))
    records = list(trace.iteration_records)
    chunks = list(getattr(trace, "chunk_records", ()))
    start_value = None
    if chunks:
        last = chunks[-1]
        records = [
            record for record in records if last.start_iteration < record.iteration <= last.end_iteration
        ] or records
        if metric in last.start_metrics:
            start_value = float(last.start_metrics[metric])
    return {
        "metric": metric,
        "start": start_value,
        "values": [float(record.metrics[metric]) for record in records if metric in record.metrics],
        "accepted": [record.accepted for record in records if record.accepted is not None],
        "gradient_norms": [
            float(record.technical["gradient_norm"])
            for record in records
            if record.technical.get("gradient_norm") is not None
        ],
        "step_norms": [
            float(record.technical["proposal"]["applied_step_norm"])
            for record in records
            if (record.technical.get("proposal") or {}).get("applied_step_norm") is not None
        ],
        "iterations": len(records),
    }


def diagnose_run(result: Any, *, mode: str | None = None) -> dict[str, Any]:
    """Classify a short OLGS or OLSS run from already-recorded evidence.

    Labels: ``converging``, ``slow``, ``stalled``, ``oscillating``,
    ``diverging``, ``repeating_rejections``, and ``no_trace`` when an OLGS run
    kept no iteration samples.  Thresholds are relative to the starting value
    and are returned with every diagnosis.
    """

    from optimizer.result import OptimizerResult
    from optimizer.system_olss import ShootingResult

    if isinstance(result, ShootingResult):
        return _diagnose_shooting(result)
    if not isinstance(result, OptimizerResult):
        raise TypeError(
            "diagnose_run expects an OptimizerResult or a ShootingResult, "
            f"got {type(result).__name__}."
        )
    resolved_mode = "min" if mode is None else str(mode)
    if resolved_mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    series = _optimizer_series(result)
    base = {
        "kind": "run_diagnosis",
        "family": "olgs",
        "optimizer": result.optimizer,
        "stop_reason": result.stop_reason,
        **_stop_facts(result.stop_reason),
        "objective_metric": result.objective_metric,
        "objective_mode": resolved_mode,
        "thresholds": dict(_DIAGNOSIS_THRESHOLDS),
    }
    if series is None:
        return {
            **base,
            "classification": "no_trace",
            "message": "The run kept no iteration samples (create_trace=False); only its final metrics are known.",
            "final_value": float(result.metrics[result.objective_metric]),
            "iterations": int(result.iterations),
        }
    values = series["values"] if series["start"] is None else [series["start"], *series["values"]]
    signed = values if resolved_mode == "min" else [-value for value in values]
    accepted = series["accepted"]
    rejection_ratio = sum(not value for value in accepted) / len(accepted) if accepted else 0.0
    classification, relative, reversals = _classify(
        signed, rejection_ratio=rejection_ratio, attempts=len(accepted)
    )
    return {
        **base,
        "classification": classification,
        "objective_trend": _trend(values),
        "relative_change": relative,
        "direction_reversals": reversals,
        "gradient_trend": _trend(series["gradient_norms"]),
        "step_trend": _trend(series["step_norms"]),
        "rejection_ratio": rejection_ratio,
        "iterations": series["iterations"],
    }


def _diagnose_shooting(result: Any) -> dict[str, Any]:
    history = list(result.history)
    residuals = [float(row["residual_norm"]) for row in history if "residual_norm" in row]
    accepted = [bool(row["accepted"]) for row in history if "accepted" in row]
    rejection_ratio = sum(not value for value in accepted) / len(accepted) if accepted else 0.0
    if result.solver_converged:
        classification, relative, reversals = "converging", None, _reversals(residuals)
        if len(residuals) >= 2:
            relative = (residuals[-1] - residuals[0]) / max(residuals[0], np.finfo(float).tiny)
    else:
        classification, relative, reversals = _classify(
            residuals, rejection_ratio=rejection_ratio, attempts=len(accepted)
        )
    return {
        "kind": "run_diagnosis",
        "family": "olss",
        "method": result.method,
        "stop_reason": result.stop_reason,
        **_stop_facts(result.stop_reason),
        "solver_converged": bool(result.solver_converged),
        "qualified": bool(result.qualified),
        "classification": classification,
        "residual_trend": _trend(residuals),
        "relative_change": relative,
        "direction_reversals": reversals,
        "rejection_ratio": rejection_ratio,
        "iterations": len(history),
        "thresholds": dict(_DIAGNOSIS_THRESHOLDS),
    }


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
    left, singular_values, right_t = np.linalg.svd(jacobian, full_matrices=False)
    del left
    threshold = (
        float(rcond) * singular_values[0]
        if rcond is not None and singular_values.size
        else max(jacobian.shape) * np.finfo(float).eps * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.sum(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > threshold
        else float("inf")
    )
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
