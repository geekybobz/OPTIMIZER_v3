"""Vectorized control specifications and containers.

This module is the Phase 1 foundation for OPTIMIZER v3.  It deliberately knows
nothing about any physical system or optimizer.  It only defines a stable,
channel-aware representation for controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlSpec:
    """Description of a named control matrix layout.

    Controls are stored as a matrix with shape ``(n_controls, control_dim)``.  The
    row order is exactly the order of ``keys``.
    """

    keys: Sequence[str]
    control_dim: int
    dtype: Any = float
    dt: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = tuple(str(key) for key in self.keys)
        if not keys:
            raise ValueError("ControlSpec requires at least one control key.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"ControlSpec keys must be unique, got {keys!r}.")
        if any(not key for key in keys):
            raise ValueError("ControlSpec keys must be non-empty strings.")

        control_dim = int(self.control_dim)
        if control_dim < 1:
            raise ValueError("control_dim must be >= 1.")

        if self.dt is not None:
            dt = float(self.dt)
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be finite and positive when provided.")
        else:
            dt = None

        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "control_dim", control_dim)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    @property
    def n_controls(self) -> int:
        return len(self.keys)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_controls, self.control_dim)

    @property
    def size(self) -> int:
        return self.n_controls * self.control_dim

    def channel_index(self, key: str) -> int:
        try:
            return self.keys.index(str(key))
        except ValueError as exc:
            raise KeyError(f"Unknown control key {key!r}; valid keys are {self.keys!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "control_dim": int(self.control_dim),
            "n_controls": int(self.n_controls),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "dt": self.dt,
            "meta": dict(self.meta),
        }


class Controls:
    """Named control values backed by a dense NumPy matrix."""

    def __init__(
        self,
        spec: ControlSpec,
        values: np.ndarray | Sequence[Sequence[float]],
        *,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
        copy: bool = True,
    ) -> None:
        if not isinstance(spec, ControlSpec):
            raise TypeError("spec must be a ControlSpec.")
        matrix = self._coerce_matrix(spec, values, copy=copy)
        self.spec = spec
        self.u = matrix
        self.name = name
        self.meta = dict(meta or {})

    @staticmethod
    def _coerce_matrix(
        spec: ControlSpec,
        values: np.ndarray | Sequence[Sequence[float]],
        *,
        copy: bool,
    ) -> np.ndarray:
        arr = np.asarray(values)
        if arr.shape != spec.shape:
            raise ValueError(f"Controls matrix must have shape {spec.shape}, got {arr.shape}.")
        if arr.dtype == spec.dtype:
            matrix = arr.copy() if copy else arr
        else:
            matrix = arr.astype(spec.dtype, copy=True)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Controls contain non-finite values.")
        return matrix

    @classmethod
    def zeros(cls, spec: ControlSpec, *, name: str | None = None) -> "Controls":
        return cls(spec, np.zeros(spec.shape, dtype=spec.dtype), name=name or "zeros", copy=False)

    @classmethod
    def constant(
        cls,
        spec: ControlSpec,
        value: float | Mapping[str, float] | Sequence[float],
        *,
        name: str | None = None,
    ) -> "Controls":
        matrix = np.zeros(spec.shape, dtype=spec.dtype)
        if isinstance(value, Mapping):
            missing = set(spec.keys) - set(value)
            extra = set(value) - set(spec.keys)
            if missing or extra:
                raise KeyError(f"Constant values must match spec keys; missing={missing}, extra={extra}.")
            for key in spec.keys:
                matrix[spec.channel_index(key), :] = float(value[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != spec.n_controls:
                raise ValueError(f"Expected {spec.n_controls} constants, got {len(value)}.")
            for row, scalar in enumerate(value):
                matrix[row, :] = float(scalar)
        else:
            matrix[:, :] = float(value)
        return cls(spec, matrix, name=name or "constant", copy=False)

    @classmethod
    def from_matrix(
        cls,
        spec: ControlSpec,
        matrix: np.ndarray | Sequence[Sequence[float]],
        *,
        copy: bool = True,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> "Controls":
        return cls(spec, matrix, name=name, meta=meta, copy=copy)

    @classmethod
    def from_dict(
        cls,
        spec: ControlSpec,
        values: Mapping[str, Sequence[float] | np.ndarray],
        *,
        copy: bool = True,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> "Controls":
        missing = set(spec.keys) - set(values)
        extra = set(values) - set(spec.keys)
        if missing or extra:
            raise KeyError(f"Control dict must match spec keys; missing={missing}, extra={extra}.")
        matrix = np.stack([np.asarray(values[key], dtype=spec.dtype) for key in spec.keys], axis=0)
        return cls(spec, matrix, name=name, meta=meta, copy=copy)

    @classmethod
    def from_flat(
        cls,
        spec: ControlSpec,
        values: Sequence[float] | np.ndarray,
        *,
        copy: bool = True,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> "Controls":
        arr = np.asarray(values, dtype=spec.dtype)
        if arr.shape != (spec.size,):
            raise ValueError(f"Flat controls must have shape ({spec.size},), got {arr.shape}.")
        return cls(spec, arr.reshape(spec.shape), name=name, meta=meta, copy=copy)

    @property
    def keys(self) -> tuple[str, ...]:
        return self.spec.keys

    @property
    def shape(self) -> tuple[int, int]:
        return self.spec.shape

    def channel(self, key: str, *, copy: bool = False) -> np.ndarray:
        row = self.spec.channel_index(key)
        out = self.u[row]
        return out.copy() if copy else out

    def set_channel(self, key: str, values: Sequence[float] | np.ndarray) -> None:
        row = self.spec.channel_index(key)
        arr = np.asarray(values, dtype=self.spec.dtype)
        if arr.shape != (self.spec.control_dim,):
            raise ValueError(
                f"Control channel {key!r} must have shape ({self.spec.control_dim},), got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Control channel {key!r} contains non-finite values.")
        self.u[row, :] = arr

    def as_matrix(self, *, copy: bool = True) -> np.ndarray:
        return self.u.copy() if copy else self.u

    def as_dict(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        return {key: self.channel(key, copy=copy) for key in self.spec.keys}

    def flatten(self, *, copy: bool = True) -> np.ndarray:
        flat = self.u.reshape(-1)
        return flat.copy() if copy else flat

    def copy(
        self,
        *,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> "Controls":
        return Controls(
            self.spec,
            self.u,
            name=self.name if name is None else name,
            meta=self.meta if meta is None else meta,
            copy=True,
        )

    def norm(self, ord: int | float | str | None = None) -> float:
        return float(np.linalg.norm(self.u.reshape(-1), ord=ord))

    def channel_norms(self, ord: int | float | str | None = None) -> dict[str, float]:
        return {key: float(np.linalg.norm(self.channel(key), ord=ord)) for key in self.spec.keys}

    def max_abs(self) -> float:
        return float(np.max(np.abs(self.u)))

    def _compatible_matrix(self, other: Any) -> np.ndarray:
        if isinstance(other, Controls):
            if other.spec.keys != self.spec.keys or other.spec.control_dim != self.spec.control_dim:
                raise ValueError("Controls must have matching keys and control_dim.")
            return other.as_matrix(copy=False)
        arr = np.asarray(other, dtype=self.spec.dtype)
        if arr.shape == ():
            return arr
        if arr.shape != self.spec.shape:
            raise ValueError(f"Operand must be scalar or shape {self.spec.shape}, got {arr.shape}.")
        return arr

    def _binary(self, other: Any, op: Any, *, name: str | None = None) -> "Controls":
        matrix = op(self.u, self._compatible_matrix(other))
        return Controls(self.spec, matrix, name=name or self.name, meta=self.meta, copy=False)

    def __add__(self, other: Any) -> "Controls":
        return self._binary(other, np.add)

    def __radd__(self, other: Any) -> "Controls":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Controls":
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> "Controls":
        matrix = np.subtract(self._compatible_matrix(other), self.u)
        return Controls(self.spec, matrix, name=self.name, meta=self.meta, copy=False)

    def __mul__(self, other: Any) -> "Controls":
        return self._binary(other, np.multiply)

    def __rmul__(self, other: Any) -> "Controls":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Controls":
        return self._binary(other, np.divide)

    def __neg__(self) -> "Controls":
        return Controls(self.spec, -self.u, name=self.name, meta=self.meta, copy=False)

    def __repr__(self) -> str:
        label = "" if self.name is None else f", name={self.name!r}"
        return f"Controls(keys={self.spec.keys!r}, shape={self.spec.shape!r}{label})"

