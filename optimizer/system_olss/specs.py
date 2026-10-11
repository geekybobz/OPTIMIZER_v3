"""Static machine-readable declarations for OLSS systems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from optimizer.system_olss.variables import ShootingSpec


SHOOTING_SYSTEM_SPEC_SCHEMA = "optimizer.olss-system-spec/v1"
_DIRECTIONS = {"minimize", "maximize"}
_JACOBIAN_SOURCES = {"analytical", "variational", "automatic", "numerical-declared"}


def shooting_metric_spec(
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
    normalized_shape = [int(item) for item in shape]
    if any(item < 0 for item in normalized_shape):
        raise ValueError("Metric shape entries must be nonnegative.")
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
    if not isinstance(value, Mapping):
        raise TypeError(f"Metric {metric_id!r} declaration must be a mapping.")
    result = shooting_metric_spec(
        str(value.get("label", metric_id)),
        description=value.get("description"),
        role=value.get("role", "diagnostic"),
        dtype=value.get("dtype", "float"),
        shape=value.get("shape", ()),
        unit=value.get("unit"),
        required=value.get("required", False),
        allow_nonfinite=value.get("allow_nonfinite", False),
    )
    result.update({str(key): item for key, item in value.items() if key not in result})
    return result


def basic_shooting_system_spec(
    system_id: str,
    *,
    shooting_spec: ShootingSpec,
    residual_size: int,
    name: str | None = None,
    system_type: str = "shooting",
    description: str = "",
    residual_groups: Iterable[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | Iterable[str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    jacobian_source: str = "analytical",
    capabilities: Mapping[str, Any] | None = None,
    selection_metric: str | None = None,
    selection_direction: str = "minimize",
    reconstruction: Mapping[str, Any] | None = None,
    verification_options: Mapping[str, Any] | None = None,
    allow_additional_metrics: bool = False,
) -> dict[str, Any]:
    metric_payload: dict[str, Any] = {
        "residual_norm": shooting_metric_spec(
            "Residual 2-norm", role="feasibility", required=True
        ),
        "residual_max_abs": shooting_metric_spec(
            "Maximum absolute residual", role="feasibility", required=True
        ),
    }
    if metrics is not None:
        if isinstance(metrics, Mapping):
            metric_payload.update({str(key): value for key, value in metrics.items()})
        else:
            metric_payload.update(
                {str(key): shooting_metric_spec(str(key).replace("_", " ").title()) for key in metrics}
            )
    selection = None
    if selection_metric is not None:
        selection = {"metric_id": str(selection_metric), "direction": str(selection_direction)}
    return validate_shooting_system_spec(
        {
            "schema_version": SHOOTING_SYSTEM_SPEC_SCHEMA,
            "family": "olss",
            "identity": {
                "id": str(system_id),
                "name": str(name or system_id),
                "system_type": str(system_type),
                "description": str(description),
            },
            "variables": shooting_spec.to_dict(),
            "parameters": dict(parameters or {}),
            "residuals": {
                "size": int(residual_size),
                "groups": [dict(group) for group in (residual_groups or ())],
                "unscaled": True,
            },
            "metrics": metric_payload,
            "derivatives": {
                "jacobian": {
                    "target": "residuals",
                    "hook": "jacobian",
                    "source": str(jacobian_source),
                },
                "fallback_policy": "explicit-only",
            },
            "selection": selection,
            "capabilities": dict(capabilities or {}),
            "reconstruction": dict(reconstruction or {}),
            "verification": {
                "default_options": (
                    None if verification_options is None else dict(verification_options)
                )
            },
            "allow_additional_metrics": bool(allow_additional_metrics),
        }
    )


def validate_shooting_system_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("system_spec() must return a mapping.")
    payload = dict(value)
    if payload.get("schema_version") != SHOOTING_SYSTEM_SPEC_SCHEMA:
        raise ValueError(
            f"OLSS system_spec schema_version must be {SHOOTING_SYSTEM_SPEC_SCHEMA!r}."
        )
    if payload.get("family") != "olss":
        raise ValueError("OLSS system_spec family must be 'olss'.")

    identity = dict(payload.get("identity") or {})
    for field in ("id", "name", "system_type"):
        if not str(identity.get(field, "")).strip():
            raise ValueError(f"OLSS system_spec identity.{field} must be non-empty.")
    identity.setdefault("description", "")

    variables = dict(payload.get("variables") or {})
    variable_size = int(variables.get("size", 0))
    if variable_size < 1:
        raise ValueError("OLSS system_spec variables.size must be positive.")

    residuals = dict(payload.get("residuals") or {})
    residual_size = int(residuals.get("size", 0))
    if residual_size < 1:
        raise ValueError("OLSS system_spec residuals.size must be positive.")
    residuals["size"] = residual_size
    residuals["groups"] = [dict(group) for group in residuals.get("groups", ())]
    residuals["unscaled"] = bool(residuals.get("unscaled", True))
    if not residuals["unscaled"]:
        raise ValueError("OLSS mandatory residual vector must be declared unscaled.")

    metrics_raw = payload.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise TypeError("OLSS system_spec metrics must be a mapping.")
    metrics = {str(key): _normalize_metric(str(key), item) for key, item in metrics_raw.items()}
    for metric_id in ("residual_norm", "residual_max_abs"):
        if metric_id not in metrics:
            raise ValueError(f"OLSS system_spec must declare {metric_id!r}.")
        metrics[metric_id]["required"] = True
        metrics[metric_id]["role"] = "feasibility"

    derivatives = dict(payload.get("derivatives") or {})
    jacobian = dict(derivatives.get("jacobian") or {})
    source = str(jacobian.get("source", ""))
    if source not in _JACOBIAN_SOURCES:
        raise ValueError(f"Unsupported OLSS Jacobian source {source!r}.")
    if str(jacobian.get("target", "")) != "residuals":
        raise ValueError("OLSS Jacobian target must be 'residuals'.")
    jacobian.setdefault("hook", "jacobian")
    derivatives["jacobian"] = jacobian
    derivatives.setdefault("fallback_policy", "explicit-only")

    selection_raw = payload.get("selection")
    selection = None
    if selection_raw is not None:
        if not isinstance(selection_raw, Mapping):
            raise TypeError("OLSS selection must be a mapping or null.")
        metric_id = str(selection_raw.get("metric_id", ""))
        direction = str(selection_raw.get("direction", ""))
        if metric_id not in metrics:
            raise ValueError("OLSS selection metric must be declared in metrics.")
        if direction not in _DIRECTIONS:
            raise ValueError("OLSS selection direction must be 'minimize' or 'maximize'.")
        metrics[metric_id]["role"] = "selection"
        selection = {"metric_id": metric_id, "direction": direction}

    verification_raw = payload.get("verification")
    if verification_raw is None:
        verification_raw = {}
    if not isinstance(verification_raw, Mapping):
        raise TypeError("OLSS verification declaration must be a mapping.")
    default_options = verification_raw.get("default_options")
    if default_options is not None and not isinstance(default_options, Mapping):
        raise TypeError("OLSS verification.default_options must be a mapping or null.")
    verification = {
        "default_options": None if default_options is None else dict(default_options)
    }

    return {
        "schema_version": SHOOTING_SYSTEM_SPEC_SCHEMA,
        "family": "olss",
        "identity": identity,
        "variables": variables,
        "parameters": dict(payload.get("parameters") or {}),
        "residuals": residuals,
        "metrics": metrics,
        "derivatives": derivatives,
        "selection": selection,
        "capabilities": dict(payload.get("capabilities") or {}),
        "reconstruction": dict(payload.get("reconstruction") or {}),
        "verification": verification,
        "allow_additional_metrics": bool(payload.get("allow_additional_metrics", False)),
    }
