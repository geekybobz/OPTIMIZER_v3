"""Typed shooting-variable layouts and values."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import prod
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ShootingBlock:
    """One named block in a flat shooting vector."""

    name: str
    shape: tuple[int, ...] = ()
    meaning: str = ""
    unit: str | None = None
    scale: float = 1.0
    role: str = "variable"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        shape = tuple(int(item) for item in self.shape)
        scale = float(self.scale)
        if not name:
            raise ValueError("ShootingBlock name must be non-empty.")
        if any(item < 1 for item in shape):
            raise ValueError("ShootingBlock shape entries must be positive.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("ShootingBlock scale must be finite and positive.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "meaning", str(self.meaning or name))
        object.__setattr__(self, "unit", None if self.unit is None else str(self.unit))
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def size(self) -> int:
        return int(prod(self.shape)) if self.shape else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "size": self.size,
            "meaning": self.meaning,
            "unit": self.unit,
            "scale": self.scale,
            "role": self.role,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ShootingSpec:
    """Ordered layout for a real shooting vector."""

    blocks: tuple[ShootingBlock, ...]
    dtype: str = "float64"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("ShootingSpec requires at least one block.")
        if not all(isinstance(block, ShootingBlock) for block in blocks):
            raise TypeError("ShootingSpec blocks must be ShootingBlock instances.")
        names = tuple(block.name for block in blocks)
        if len(set(names)) != len(names):
            raise ValueError("ShootingSpec block names must be unique.")
        dtype = np.dtype(self.dtype)
        if dtype.kind != "f":
            raise ValueError("ShootingSpec dtype must be a real floating-point dtype.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    @cached_property
    def offsets(self) -> dict[str, tuple[int, int]]:
        cursor = 0
        result: dict[str, tuple[int, int]] = {}
        for block in self.blocks:
            result[block.name] = (cursor, cursor + block.size)
            cursor += block.size
        return result

    @property
    def size(self) -> int:
        return sum(block.size for block in self.blocks)

    @cached_property
    def scales(self) -> np.ndarray:
        """Return one positive characteristic scale per flat variable."""

        values = np.concatenate(
            [np.full(block.size, block.scale, dtype=float) for block in self.blocks]
        )
        values.setflags(write=False)
        return values

    def block(self, name: str) -> ShootingBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Unknown shooting block {name!r}.")

    def blocks_with_role(self, role: str) -> tuple[ShootingBlock, ...]:
        return tuple(block for block in self.blocks if block.role == role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "size": self.size,
            "dtype": self.dtype,
            "metadata": dict(self.metadata),
        }


def _validate_shooting_spec_compatibility(expected: ShootingSpec, actual: ShootingSpec) -> None:
    """Compare variable meaning and coordinates, excluding labels and metadata."""

    def identity(spec: ShootingSpec) -> tuple:
        return (
            spec.dtype,
            tuple((b.name, b.shape, b.unit, b.scale, b.role) for b in spec.blocks),
        )

    if identity(actual) != identity(expected):
        raise ValueError("ShootingVariables layout does not match system.shooting_spec().")


@dataclass(frozen=True)
class ShootingVariables:
    """Finite real values attached to a :class:`ShootingSpec`."""

    spec: ShootingSpec
    values: np.ndarray
    name: str = "shooting"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ShootingSpec):
            raise TypeError("ShootingVariables spec must be a ShootingSpec.")
        values = np.asarray(self.values, dtype=self.spec.dtype).reshape(-1)
        if values.shape != (self.spec.size,):
            raise ValueError(
                f"ShootingVariables values must have shape ({self.spec.size},), got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("ShootingVariables values must be finite.")
        owned = values.copy()
        owned.setflags(write=False)
        object.__setattr__(self, "values", owned)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def zeros(cls, spec: ShootingSpec, *, name: str = "zero") -> "ShootingVariables":
        return cls(spec, np.zeros(spec.size, dtype=spec.dtype), name=name)

    @classmethod
    def from_flat(
        cls,
        spec: ShootingSpec,
        values: Iterable[float] | np.ndarray,
        *,
        name: str = "shooting",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ShootingVariables":
        return cls(spec, np.asarray(values), name=name, metadata=dict(metadata or {}))

    @classmethod
    def from_blocks(
        cls,
        spec: ShootingSpec,
        values: Mapping[str, Any],
        *,
        name: str = "shooting",
    ) -> "ShootingVariables":
        unknown = sorted(set(values) - set(spec.names))
        missing = sorted(set(spec.names) - set(values))
        if unknown or missing:
            raise ValueError(f"Shooting block mismatch: missing={missing}, unknown={unknown}.")
        flat: list[np.ndarray] = []
        for block in spec.blocks:
            value = np.asarray(values[block.name], dtype=spec.dtype)
            expected = block.shape
            if value.shape != expected:
                if not expected and value.shape in {(), (1,)}:
                    value = value.reshape(())
                else:
                    raise ValueError(
                        f"Block {block.name!r} must have shape {expected}, got {value.shape}."
                    )
            flat.append(value.reshape(-1))
        return cls(spec, np.concatenate(flat), name=name)

    def flatten(self, *, copy: bool = True) -> np.ndarray:
        return self.values.copy() if copy else self.values

    def block(self, name: str, *, copy: bool = True) -> np.ndarray | float:
        block = self.spec.block(name)
        start, stop = self.spec.offsets[name]
        value = self.values[start:stop].reshape(block.shape)
        if not block.shape:
            return float(value)
        return value.copy() if copy else value

    def unpack(self, *, copy: bool = True) -> dict[str, Any]:
        return {name: self.block(name, copy=copy) for name in self.spec.names}

    def copy(self, *, name: str | None = None) -> "ShootingVariables":
        return ShootingVariables(
            self.spec,
            self.values,
            name=self.name if name is None else name,
            metadata=self.metadata,
        )

    def with_flat(self, values: Any, *, name: str | None = None) -> "ShootingVariables":
        return ShootingVariables.from_flat(
            self.spec,
            values,
            name=self.name if name is None else name,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "values": self.values.tolist(),
            "metadata": dict(self.metadata),
        }
