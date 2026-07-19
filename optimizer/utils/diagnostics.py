"""Metric, residual, and geometry diagnostics.

Why this file exists
--------------------
Training without diagnostics becomes guesswork.  Before changing optimizer weights,
switching algorithms, or attempting repair, users need a compact technical picture:
what are the current metrics, how large are the controls, how large are the hard
residuals, is the gradient finite, and is the residual Jacobian locally well behaved.

How it fits the architecture
----------------------------
- ``diagnostic_report`` is the high-level summary for notebooks and logs.
- ``metric_report`` is a lightweight alias when only current scalar/vector metrics are
  needed.
- ``geometry_probe`` adds residual/Jacobian rank and conditioning information.
- repair and projected methods can use the same report shape for trace records.

What this file deliberately does not do
---------------------------------------
It does not modify controls.  If a diagnostic discovers a problem, the caller decides
whether to run an optimizer, repair tool, or curriculum adjustment.

Reviewer invariants
-------------------
- reports are JSON-friendly for common NumPy scalar/array values.
- missing optional hooks are reported as unavailable instead of hidden.
- gradient evaluation is optional because it can be expensive for some systems.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls
from optimizer.system_olgs import (
    evaluate_system,
    gradient_system,
    optional_residuals,
    probe_system,
    validate_controls_for_system,
)
from optimizer.utils.derivatives import get_jacobian
from optimizer.utils.geometry import jacobian_geometry


def _json_safe(value: Any) -> Any:
    """Convert common NumPy values into JSON-friendly payloads."""

    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize metric values without discarding vector/complex diagnostics."""

    summary: dict[str, Any] = {}
    for key, value in metrics.items():
        arr = np.asarray(value)
        if arr.shape == ():
            summary[key] = _json_safe(arr.item())
        else:
            summary[key] = {
                "shape": list(arr.shape),
                "norm": float(np.linalg.norm(arr.reshape(-1))),
                "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                "value": _json_safe(arr),
            }
    return summary


def metric_report(system: Any, controls: Controls) -> dict[str, Any]:
    """Return a JSON-friendly report of current system metrics."""

    validate_controls_for_system(system, controls)
    metrics = evaluate_system(system, controls)
    return {
        "kind": "metric_report",
        "metrics": _metric_summary(metrics),
        "raw_metric_keys": list(metrics.keys()),
    }


def diagnostic_report(
    system: Any,
    controls: Controls,
    *,
    residuals: str | None = "hard",
    include_gradient: bool = True,
) -> dict[str, Any]:
    """Return metrics plus control, residual, and optional gradient summaries."""

    validate_controls_for_system(system, controls)
    metrics = evaluate_system(system, controls)
    matrix = controls.as_matrix(copy=False)
    report: dict[str, Any] = {
        "kind": "diagnostic_report",
        "system_hooks": probe_system(system).to_dict(),
        "control_spec": controls.spec.to_dict(),
        "controls": {
            "name": controls.name,
            "shape": list(controls.shape),
            "norm": controls.norm(),
            "max_abs": controls.max_abs(),
            "channel_norms": controls.channel_norms(),
            "finite": bool(np.all(np.isfinite(matrix))),
        },
        "metrics": _metric_summary(metrics),
    }

    if include_gradient:
        gradient = gradient_system(system, controls)
        report["gradient"] = {
            "norm": gradient.norm(),
            "max_abs": gradient.max_abs(),
            "channel_norms": gradient.channel_norms(),
            "finite": bool(np.all(np.isfinite(gradient.as_matrix(copy=False)))),
        }

    if residuals is not None:
        try:
            residual_vec = optional_residuals(system, controls, name=residuals)
            report["residuals"] = {
                "name": residuals,
                "available": True,
                "count": int(residual_vec.size),
                "norm": float(np.linalg.norm(residual_vec)),
                "max_abs": float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0,
                "values": residual_vec.tolist(),
            }
        except AttributeError as exc:
            report["residuals"] = {
                "name": residuals,
                "available": False,
                "error": str(exc),
            }

    return report


def geometry_probe(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    fallback: bool = True,
    eps: float = 1.0e-6,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Return local residual/Jacobian geometry for controls."""

    validate_controls_for_system(system, controls)
    residual_vec = optional_residuals(system, controls, name=residuals)
    jacobian, source = get_jacobian(
        system,
        controls,
        residuals=residuals,
        fallback=fallback,
        eps=eps,
    )
    geometry = jacobian_geometry(jacobian, rcond=rcond)
    return {
        "kind": "geometry_probe",
        "residuals": residuals,
        "residual_count": int(residual_vec.size),
        "residual_norm": float(np.linalg.norm(residual_vec)),
        "residual_max_abs": float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0,
        "jacobian_source": source,
        **geometry,
        "repair_locally_possible": bool(geometry["rank"] > 0 and geometry["rank"] <= residual_vec.size),
    }
