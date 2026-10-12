"""Structural numerical coordinates; systems retain their own physics contracts."""
from __future__ import annotations
from typing import Any, Mapping, Protocol, Self, runtime_checkable
import numpy as np


@runtime_checkable
class Variables(Protocol):
    spec: Any

    @property
    def scales(self) -> np.ndarray: ...
    def flatten(self, *, copy: bool = True) -> np.ndarray: ...
    def with_flat(self, values: Any, *, name: str | None = None) -> Self: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, spec: Any, payload: Mapping[str, Any]) -> Self: ...
