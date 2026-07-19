"""Structured numeric analysis over blackbox ledgers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from optimizer.blackbox.policy import BlackBoxPolicy
from optimizer.blackbox.reader import get_path, read_manifest, read_records
from optimizer.blackbox.records import json_safe


def _finite_values(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            item = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(item):
            out.append(item)
    return out


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _ratio(last: float | None, first: float | None) -> float | None:
    if last is None or first is None:
        return None
    denom = max(abs(float(first)), float(np.finfo(float).tiny))
    return float(last) / denom


def analyze_records(
    records: list[dict[str, Any]],
    *,
    window: int = 10,
    policy: BlackBoxPolicy | str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return structured numeric signals from recent iteration records."""

    active_policy = BlackBoxPolicy.from_value(policy)
    iterations = [record for record in records if record.get("kind") == "iteration"]
    selected = iterations[-int(window) :] if int(window) > 0 else iterations
    if not selected:
        return {
            "kind": "window_analysis",
            "window": {"size": 0, "start_i": None, "end_i": None},
            "cost": {"extra_evaluations": 0, "extra_gradients": 0, "extra_jacobians": 0},
            "signals": [],
            "suggested_actions": [],
        }

    j_values = _finite_values(get_path(record, "metrics.J") for record in selected)
    grad_norms = _finite_values(get_path(record, "gradient.norm") for record in selected)
    rel_grad_deltas = _finite_values(get_path(record, "gradient.rel_delta_norm") for record in selected)
    step_sizes = _finite_values(get_path(record, "step.step_size") for record in selected)
    accepted = [bool(get_path(record, "decision.accepted", False)) for record in selected]
    acceptance_rate = float(sum(accepted) / len(accepted)) if accepted else 0.0
    reject_rate = 1.0 - acceptance_rate
    j_slope = _linear_slope(j_values)
    step_ratio = _ratio(step_sizes[-1], step_sizes[0]) if step_sizes else None

    signals: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    if reject_rate >= float(active_policy.reject_rate_window_threshold):
        signals.append(
            {
                "code": "high_reject_rate",
                "confidence": min(1.0, reject_rate),
                "evidence": {
                    "reject_rate": reject_rate,
                    "threshold": float(active_policy.reject_rate_window_threshold),
                    "window_size": int(len(selected)),
                },
            }
        )
        actions.append({"code": "reduce_step_size", "factor": 0.5, "reason_code": "high_reject_rate"})

    grad_median = float(np.median(grad_norms)) if grad_norms else None
    if j_slope is not None and abs(j_slope) <= float(active_policy.stall_slope_abs) and grad_median not in {None, 0.0}:
        signals.append(
            {
                "code": "stalling",
                "confidence": 0.8,
                "evidence": {
                    "J_slope": j_slope,
                    "stall_slope_abs": float(active_policy.stall_slope_abs),
                    "gradient_norm_median": grad_median,
                    "acceptance_rate": acceptance_rate,
                },
            }
        )
        actions.append({"code": "change_step_or_method", "reason_code": "stalling"})

    max_rel_grad_delta = max(rel_grad_deltas) if rel_grad_deltas else None
    if max_rel_grad_delta is not None and max_rel_grad_delta >= float(active_policy.gradient_rel_delta_threshold):
        signals.append(
            {
                "code": "gradient_spike",
                "confidence": min(1.0, max_rel_grad_delta / max(float(active_policy.gradient_rel_delta_threshold), 1.0)),
                "evidence": {
                    "max_gradient_rel_delta_norm": max_rel_grad_delta,
                    "threshold": float(active_policy.gradient_rel_delta_threshold),
                },
            }
        )
        actions.append({"code": "inspect_saved_gradient_or_reduce_step", "reason_code": "gradient_spike"})

    if step_ratio is not None and step_ratio <= float(active_policy.step_collapse_ratio):
        signals.append(
            {
                "code": "step_collapse",
                "confidence": 0.85,
                "evidence": {
                    "step_size_first": step_sizes[0],
                    "step_size_last": step_sizes[-1],
                    "ratio": step_ratio,
                    "threshold": float(active_policy.step_collapse_ratio),
                },
            }
        )
        actions.append({"code": "restart_or_change_schedule", "reason_code": "step_collapse"})

    return json_safe(
        {
            "kind": "window_analysis",
            "window": {
                "size": int(len(selected)),
                "start_i": selected[0].get("i", selected[0].get("iteration")),
                "end_i": selected[-1].get("i", selected[-1].get("iteration")),
                "start_seq": selected[0].get("seq"),
                "end_seq": selected[-1].get("seq"),
            },
            "cost": {"extra_evaluations": 0, "extra_gradients": 0, "extra_jacobians": 0},
            "series": {
                "J_first": j_values[0] if j_values else None,
                "J_last": j_values[-1] if j_values else None,
                "J_slope": j_slope,
                "gradient_norm_median": grad_median,
                "acceptance_rate": acceptance_rate,
                "reject_rate": reject_rate,
                "step_size_first": step_sizes[0] if step_sizes else None,
                "step_size_last": step_sizes[-1] if step_sizes else None,
                "step_size_ratio": step_ratio,
            },
            "signals": signals,
            "suggested_actions": actions,
        }
    )


def analyze_path(
    run_dir: Path | str,
    *,
    window: int = 10,
    policy: BlackBoxPolicy | str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a blackbox run folder."""

    if policy is None:
        try:
            policy = read_manifest(run_dir).get("policy")
        except FileNotFoundError:
            policy = None
    return analyze_records(read_records(run_dir), window=window, policy=policy)


def diagnostics(
    run_dir: Path | str,
    *,
    details: str = "summary",
    window: int = 10,
) -> dict[str, Any]:
    """Return structured historical diagnostics from a blackbox run."""

    manifest = read_manifest(run_dir)
    records = read_records(run_dir)
    analysis = analyze_records(records, window=window, policy=manifest.get("policy"))
    latest_iteration = next((record for record in reversed(records) if record.get("kind") == "iteration"), None)
    payload: dict[str, Any] = {
        "kind": "blackbox_diagnostics",
        "details": str(details),
        "manifest": manifest,
        "latest_iteration": latest_iteration,
        "analysis": analysis,
    }
    if details == "gradient":
        payload["gradient"] = [
            {
                "seq": record.get("seq"),
                "i": record.get("i"),
                "norm": get_path(record, "gradient.norm"),
                "rel_delta_norm": get_path(record, "gradient.rel_delta_norm"),
                "cos_prev": get_path(record, "gradient.cos_prev"),
            }
            for record in records
            if record.get("kind") == "iteration" and "gradient" in record
        ]
    elif details == "decisions":
        payload["decisions"] = [
            {"seq": record.get("seq"), "i": record.get("i"), **dict(record.get("decision", {}))}
            for record in records
            if record.get("kind") == "iteration" and "decision" in record
        ]
    elif details == "repairs":
        payload["repairs"] = [record for record in records if record.get("kind") == "repair"]
    elif details == "thresholds":
        payload["thresholds"] = [record for record in records if record.get("kind") == "threshold"]
    return json_safe(payload)
