"""Short-run interpretation and in-memory method comparison."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.curvature import curvature_evidence
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


def diagnose_run(result: Any, *, mode: str | None = None) -> dict[str, Any]:
    """Classify a short OLGS or OLSS run from already-recorded evidence."""

    if hasattr(result, "trace") and result.trace is not None:
        records = list(result.trace.iteration_records)
        metric = str(getattr(result, "objective_metric", "J"))
        values = [float(record.metrics[metric]) for record in records if metric in record.metrics]
        accepted = [record.accepted for record in records if record.accepted is not None]
        gradient_norms = [
            float(record.technical.get("gradient_norm"))
            for record in records
            if record.technical.get("gradient_norm") is not None
        ]
        step_norms = [
            float(record.technical.get("proposal", {}).get("applied_step_norm"))
            for record in records
            if record.technical.get("proposal", {}).get("applied_step_norm") is not None
        ]
        rejection_ratio = (
            sum(not value for value in accepted) / len(accepted) if accepted else 0.0
        )
        resolved_mode = "min" if mode is None else str(mode)
        if resolved_mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        signed = values if resolved_mode == "min" else [-value for value in values]
        signed_change = signed[-1] - signed[0] if len(signed) >= 2 else 0.0
        signed_scale = max(1.0, abs(signed[0])) if signed else 1.0
        if rejection_ratio >= 0.8 and len(accepted) >= 3:
            classification = "repeating_rejections"
        elif len(signed) >= 2 and signed_change > 1.0e-6 * signed_scale:
            classification = "diverging"
        elif len(signed) >= 4 and np.sum(np.diff(signed)[:-1] * np.diff(signed)[1:] < 0.0) >= 2:
            classification = "oscillating"
        elif len(signed) >= 2 and abs(signed[-1] - signed[0]) <= 1.0e-6 * max(1.0, abs(signed[0])):
            classification = "stalled"
        elif len(signed) >= 2 and abs(signed[-1] - signed[0]) <= 1.0e-2 * max(1.0, abs(signed[0])):
            classification = "slow"
        else:
            classification = "converging"
        return {
            "kind": "run_diagnosis",
            "family": "olgs",
            "optimizer": getattr(result, "optimizer", None),
            "stop_reason": getattr(result, "stop_reason", None),
            "classification": classification,
            "objective_metric": metric,
            "objective_mode": resolved_mode,
            "objective_trend": _trend(values),
            "gradient_trend": _trend(gradient_norms),
            "step_trend": _trend(step_norms),
            "rejection_ratio": rejection_ratio,
            "iterations": len(records),
        }

    history = list(getattr(result, "history", ()))
    residuals = [float(row["residual_norm"]) for row in history if "residual_norm" in row]
    accepted = [bool(row["accepted"]) for row in history if "accepted" in row]
    if len(residuals) >= 2 and residuals[-1] < residuals[0] * 0.1:
        classification = "converging"
    elif len(residuals) >= 2 and residuals[-1] < residuals[0]:
        classification = "slow"
    elif accepted and not any(accepted):
        classification = "repeating_rejections"
    elif len(residuals) >= 2 and residuals[-1] > residuals[0] * (1.0 + 1.0e-6):
        classification = "diverging"
    else:
        classification = "stalled"
    return {
        "kind": "run_diagnosis",
        "family": "olss",
        "method": getattr(result, "method", None),
        "stop_reason": getattr(result, "stop_reason", None),
        "classification": classification,
        "residual_trend": _trend(residuals),
        "rejection_ratio": sum(not value for value in accepted) / len(accepted) if accepted else 0.0,
        "iterations": len(history),
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
