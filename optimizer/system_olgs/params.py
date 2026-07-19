"""Small parameter helpers for OLGS systems.

Project systems may define richer dataclasses.  These helpers cover the common
primary/secondary split and provide generic coercion utilities for templates and
simple systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimaryParams:
    """Common primary physical/grid parameters for simple OLGS systems."""

    tau: float | None = None
    T: float | None = None
    N: int | None = None
    dt: float | None = None
    state_dim: tuple[int, ...] | None = None
    control_channels: tuple[str, ...] | None = None
    meta: Mapping[str, Any] | None = None

    def resolved_tau(self) -> float | None:
        """Return ``tau`` using ``T`` as an accepted alias."""

        tau = self.tau if self.tau is not None else self.T
        return None if tau is None else float(tau)

    def resolved_dt(self) -> float | None:
        """Return validated ``dt`` or derive it from endpoint-sampled ``tau``/``N``."""

        if self.dt is not None:
            dt = float(self.dt)
        else:
            tau = self.resolved_tau()
            if tau is None or self.N is None:
                return None
            if int(self.N) < 2:
                raise ValueError("N must be >= 2 when deriving dt from tau.")
            dt = tau / float(int(self.N) - 1)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        return dt

    def validate(self) -> "PrimaryParams":
        """Validate common fields and return self."""

        tau = self.resolved_tau()
        if tau is not None and (not np.isfinite(tau) or tau <= 0.0):
            raise ValueError("tau/T must be finite and positive.")
        if self.N is not None and int(self.N) < 1:
            raise ValueError("N must be positive.")
        if tau is not None and self.N is not None and int(self.N) < 2:
            raise ValueError("N must be >= 2 when tau/T is provided.")
        if self.dt is not None:
            self.resolved_dt()
        if tau is not None and self.N is not None and self.dt is not None:
            expected = tau / float(int(self.N) - 1)
            if not np.isclose(float(self.dt), expected):
                raise ValueError(f"dt={self.dt} is inconsistent with tau/(N-1)={expected}.")
        if self.control_channels is not None:
            keys = tuple(str(key) for key in self.control_channels)
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("control_channels must be non-empty and unique.")
        return self


@dataclass(frozen=True)
class SecondaryParams:
    """Generic secondary objective/curriculum parameters."""

    values: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values or {})


def params_to_dict(params: Any) -> dict[str, Any]:
    """Return a best-effort dict from a params object."""

    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    if is_dataclass(params):
        return asdict(params)
    if hasattr(params, "__dict__"):
        return dict(vars(params))
    raise TypeError("params must be None, mapping, dataclass, or object with __dict__.")


def coerce_dataclass_params(dataclass_type: type[Any], params: Any = None, **overrides: Any) -> Any:
    """Coerce mapping/dataclass/object params into ``dataclass_type``."""

    payload = params_to_dict(params)
    payload.update(overrides)
    allowed = {field.name for field in fields(dataclass_type)}
    extra = set(payload) - allowed
    if extra:
        raise TypeError(f"Unknown params for {dataclass_type.__name__}: {sorted(extra)}.")
    return dataclass_type(**payload)


def replace_params(params: Any, **updates: Any) -> Any:
    """Return params with updates while preserving the original representation."""

    if params is None:
        return dict(updates)
    if isinstance(params, Mapping):
        payload = dict(params)
        payload.update(updates)
        return payload
    if is_dataclass(params):
        return replace(params, **updates)
    payload = params_to_dict(params)
    payload.update(updates)
    try:
        return params.__class__(**payload)
    except Exception:
        return payload


def normalize_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """Return validated control channel names."""

    keys = tuple(str(channel) for channel in channels)
    if not keys:
        raise ValueError("At least one control channel is required.")
    if len(set(keys)) != len(keys):
        raise ValueError(f"Control channel names must be unique, got {keys!r}.")
    if any(not key for key in keys):
        raise ValueError("Control channel names must be non-empty strings.")
    return keys
