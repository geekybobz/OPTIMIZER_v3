"""Numeric record helpers for blackbox run ledgers.

The blackbox ledger stores compact machine-readable facts.  These helpers keep JSON
conversion, metric deltas, and array summaries consistent across optimizers, repairs,
and offline diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from optimizer.controls import Controls


SCHEMA_VERSION = "optimizer.blackbox.v1"
TINY = float(np.finfo(float).tiny)


def utc_now() -> str:
    """Return a compact UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-friendly values."""

    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Controls):
        return controls_summary(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def scalar_float(value: Any) -> float | None:
    """Return a finite scalar float when ``value`` is scalar, otherwise ``None``."""

    arr = np.asarray(value)
    if arr.shape != ():
        return None
    out = float(np.real(arr.item()))
    return out if np.isfinite(out) else None


def array_summary(value: Any, *, previous: Any | None = None) -> dict[str, Any]:
    """Return norm/max/RMS details for an array-like value."""

    arr = np.asarray(value)
    flat = arr.reshape(-1)
    abs_flat = np.abs(flat)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "size": int(flat.size),
        "norm": float(np.linalg.norm(flat)),
        "max_abs": float(np.max(abs_flat)) if flat.size else 0.0,
        "rms": float(np.sqrt(np.mean(abs_flat * abs_flat))) if flat.size else 0.0,
        "finite": bool(np.all(np.isfinite(flat))),
    }
    if previous is not None:
        prev = np.asarray(previous).reshape(-1)
        if prev.shape == flat.shape:
            delta = flat - prev
            prev_norm = float(np.linalg.norm(prev))
            delta_norm = float(np.linalg.norm(delta))
            out.update(
                {
                    "delta_norm": delta_norm,
                    "rel_delta_norm": delta_norm / max(prev_norm, TINY),
                    "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
                }
            )
            denom = float(np.linalg.norm(flat) * prev_norm)
            if denom > TINY:
                out["cos_prev"] = float(np.real(np.vdot(prev, flat)) / denom)
    return out


def controls_summary(controls: Controls | None, *, previous: Controls | np.ndarray | None = None) -> dict[str, Any]:
    """Return compact numeric facts for controls."""

    if controls is None:
        return {}
    previous_matrix: np.ndarray | None
    if isinstance(previous, Controls):
        previous_matrix = previous.as_matrix(copy=False)
    elif previous is None:
        previous_matrix = None
    else:
        previous_matrix = np.asarray(previous)
    matrix = controls.as_matrix(copy=False)
    out = array_summary(matrix, previous=previous_matrix)
    out.update(
        {
            "name": controls.name,
            "keys": list(controls.keys),
            "channel_norms": controls.channel_norms(),
        }
    )
    return out


def gradient_summary(gradient: Controls | None, *, previous_flat: np.ndarray | None = None) -> dict[str, Any]:
    """Return compact numeric facts for a gradient Controls object."""

    if gradient is None:
        return {}
    flat = gradient.flatten(copy=False)
    out = array_summary(flat, previous=previous_flat)
    out["channel_norms"] = gradient.channel_norms()
    return out


def metrics_summary(
    metrics: Mapping[str, Any] | None,
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return scalar metrics and structured array metric summaries with deltas."""

    out: dict[str, Any] = {}
    if not metrics:
        return out
    previous = previous or {}
    for key, value in metrics.items():
        scalar = scalar_float(value)
        prev_scalar = scalar_float(previous[key]) if key in previous else None
        if scalar is not None:
            out[str(key)] = scalar
            if prev_scalar is not None:
                delta = scalar - prev_scalar
                out[f"d{key}"] = delta
                out[f"rel_d{key}"] = delta / max(abs(prev_scalar), TINY)
            continue
        out[str(key)] = array_summary(value, previous=previous.get(key))
    return out


def system_summary(system: Any | None) -> dict[str, Any]:
    """Return metadata about a system without evaluating it."""

    if system is None:
        return {}
    params = getattr(system, "params", None)
    if isinstance(params, Mapping):
        params_payload = dict(params)
    elif params is not None and hasattr(params, "__dict__"):
        params_payload = dict(vars(params))
    else:
        params_payload = {}
    return {
        "type": type(system).__name__,
        "module": type(system).__module__,
        "params": json_safe(params_payload),
    }


def record_base(seq: int, kind: str, *, run_id: str) -> dict[str, Any]:
    """Return the common fields for one ledger record."""

    return {
        "v": 1,
        "seq": int(seq),
        "kind": str(kind),
        "run_id": str(run_id),
        "t_utc": utc_now(),
    }
