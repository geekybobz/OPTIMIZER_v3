"""Static, machine-readable OLGS system specifications.

The specification is the semantic half of system discovery.  Structural hook
availability is still derived by :func:`probe_system`; systems declare what
their controls, metrics, derivatives, constraints, and explicit data products
mean here without running numerical work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from optimizer.controls import ControlSpec


SYSTEM_SPEC_SCHEMA = "optimizer.system-spec/v1"
_DIRECTIONS = {"minimize", "maximize"}


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _control_payload(value: ControlSpec | Mapping[str, Any]) -> dict[str, Any]:
    payload = value.to_dict() if isinstance(value, ControlSpec) else _plain_mapping(value, name="controls")
    keys = tuple(str(key) for key in payload.get("keys", ()))
    control_dim = int(payload.get("control_dim", 0))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("system_spec controls require unique, non-empty keys.")
    if control_dim < 1:
        raise ValueError("system_spec controls require control_dim >= 1.")
    payload["keys"] = list(keys)
    payload["control_dim"] = control_dim
    payload.setdefault("n_controls", len(keys))
    payload.setdefault("shape", [len(keys), control_dim])
    return payload


def metric_spec(
    label: str,
    *,
    description: str | None = None,
    role: str = "diagnostic",
    dtype: str = "float",
    shape: Iterable[int] = (),
    unit: str | None = None,
    required: bool = False,
    allow_nonfinite: bool = False,
) -> dict[str, Any]:
    """Build one normalized metric declaration."""

    normalized_shape = [int(item) for item in shape]
    if any(item < 0 for item in normalized_shape):
        raise ValueError("metric shape entries must be nonnegative.")
    return {
        "label": str(label),
        "description": str(description or label),
        "role": str(role),
        "dtype": str(dtype),
        "shape": normalized_shape,
        "unit": None if unit is None else str(unit),
        "required": bool(required),
        "allow_nonfinite": bool(allow_nonfinite),
    }


def _normalize_metric(metric_id: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"label": value}
    payload = _plain_mapping(value, name=f"metric {metric_id!r}")
    label = str(payload.get("label", metric_id))
    normalized = metric_spec(
        label,
        description=payload.get("description", label),
        role=payload.get("role", "diagnostic"),
        dtype=payload.get("dtype", "float"),
        shape=payload.get("shape", ()),
        unit=payload.get("unit"),
        required=payload.get("required", metric_id == "J"),
        allow_nonfinite=payload.get("allow_nonfinite", False),
    )
    for key, item in payload.items():
        if key not in normalized:
            normalized[key] = item
    return normalized


def basic_system_spec(
    system_id: str,
    *,
    name: str | None = None,
    system_type: str = "generic",
    description: str = "",
    control_spec: ControlSpec | Mapping[str, Any],
    objective_direction: str = "minimize",
    metrics: Mapping[str, Any] | Iterable[str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    derivatives: Mapping[str, Any] | None = None,
    constraints: Mapping[str, Any] | None = None,
    visualizations: Mapping[str, Any] | None = None,
    allow_additional_metrics: bool = False,
) -> dict[str, Any]:
    """Build the minimum complete specification for a new OLGS system."""

    metric_payload: dict[str, Any] = {
        "J": metric_spec(
            "Objective",
            description="Scalar objective optimized by the selected numerical method.",
            role="objective",
            required=True,
        )
    }
    if metrics is not None:
        if isinstance(metrics, Mapping):
            metric_payload.update({str(key): value for key, value in metrics.items()})
        else:
            metric_payload.update({str(key): metric_spec(str(key).replace("_", " ").title()) for key in metrics})
    derivative_payload = {
        "gradient": {
            "target": "J",
            "hook": "gradient",
            "source": "analytical",
            "convention": "dJ/du",
        },
        "fallback_policy": "explicit-verification-only",
    }
    if derivatives:
        derivative_payload.update(dict(derivatives))
    return validate_system_spec(
        {
            "schema_version": SYSTEM_SPEC_SCHEMA,
            "identity": {
                "id": str(system_id),
                "name": str(name or system_id),
                "system_type": str(system_type),
                "description": str(description),
            },
            "controls": _control_payload(control_spec),
            "parameters": dict(parameters or {}),
            "objective": {"metric_id": "J", "direction": str(objective_direction)},
            "metrics": metric_payload,
            "derivatives": derivative_payload,
            "constraints": dict(constraints or {}),
            "visualizations": dict(visualizations or {}),
            "allow_additional_metrics": bool(allow_additional_metrics),
        }
    )


def validate_system_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a static ``system_spec()`` mapping."""

    payload = _plain_mapping(value, name="system_spec()")
    if payload.get("schema_version") != SYSTEM_SPEC_SCHEMA:
        raise ValueError(f"system_spec() schema_version must be {SYSTEM_SPEC_SCHEMA!r}.")

    identity = _plain_mapping(payload.get("identity"), name="system_spec()['identity']")
    for field in ("id", "name", "system_type"):
        if not str(identity.get(field, "")).strip():
            raise ValueError(f"system_spec()['identity']['{field}'] must be non-empty.")
    identity.setdefault("description", "")

    controls = _control_payload(payload.get("controls", {}))
    objective = _plain_mapping(payload.get("objective"), name="system_spec()['objective']")
    metric_id = str(objective.get("metric_id", ""))
    direction = str(objective.get("direction", ""))
    if not metric_id:
        raise ValueError("system_spec objective requires metric_id.")
    if direction not in _DIRECTIONS:
        raise ValueError("system_spec objective direction must be 'minimize' or 'maximize'.")

    raw_metrics = _plain_mapping(payload.get("metrics"), name="system_spec()['metrics']")
    metrics = {metric: _normalize_metric(metric, item) for metric, item in raw_metrics.items()}
    if metric_id not in metrics:
        raise ValueError(f"system_spec objective metric {metric_id!r} is not declared in metrics.")
    metrics[metric_id]["role"] = "objective"
    metrics[metric_id]["required"] = True

    derivatives = _plain_mapping(payload.get("derivatives"), name="system_spec()['derivatives']")
    gradient = _plain_mapping(derivatives.get("gradient"), name="system_spec derivatives.gradient")
    if str(gradient.get("source", "")) != "analytical":
        raise ValueError("OLGS system_spec derivatives.gradient.source must be 'analytical'.")
    if str(gradient.get("target", "")) != metric_id:
        raise ValueError("system_spec analytical gradient target must match the objective metric.")
    gradient.setdefault("hook", "gradient")
    gradient.setdefault("convention", "dJ/du")
    derivatives["gradient"] = gradient
    derivatives.setdefault("fallback_policy", "explicit-verification-only")

    return {
        "schema_version": SYSTEM_SPEC_SCHEMA,
        "identity": identity,
        "controls": controls,
        "parameters": dict(payload.get("parameters", {}) or {}),
        "objective": {**objective, "metric_id": metric_id, "direction": direction},
        "metrics": metrics,
        "derivatives": derivatives,
        "constraints": dict(payload.get("constraints", {}) or {}),
        "visualizations": dict(payload.get("visualizations", {}) or {}),
        "allow_additional_metrics": bool(payload.get("allow_additional_metrics", False)),
    }


def validate_declared_metrics(metrics: Mapping[str, Any], system_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metric names and declared shapes without changing system values."""

    declared = _plain_mapping(system_spec.get("metrics"), name="system_spec metrics")
    unknown = sorted(set(str(key) for key in metrics) - set(declared))
    if unknown and not bool(system_spec.get("allow_additional_metrics", False)):
        raise KeyError(f"system.evaluate(...) returned undeclared metrics: {', '.join(unknown)}.")
    missing = sorted(
        metric_id
        for metric_id, spec in declared.items()
        if bool(spec.get("required", False)) and metric_id not in metrics
    )
    if missing:
        raise KeyError(f"system.evaluate(...) omitted required metrics: {', '.join(missing)}.")
    for metric_id, raw_value in metrics.items():
        if metric_id not in declared:
            continue
        spec = declared[metric_id]
        value = np.asarray(raw_value)
        expected = tuple(int(item) for item in spec.get("shape", ()))
        if value.shape != expected:
            raise ValueError(
                f"metric {metric_id!r} must have declared shape {expected}, got {value.shape}."
            )
        if not bool(spec.get("allow_nonfinite", False)):
            try:
                finite = bool(np.all(np.isfinite(value)))
            except TypeError:
                finite = True
            if not finite:
                raise ValueError(f"metric {metric_id!r} must be finite by its system specification.")
    return dict(metrics)
