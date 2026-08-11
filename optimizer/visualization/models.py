"""Generic, renderer-independent visualization data models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _coordinate_array(value: Any, *, layer: str, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"plot layer {layer!r} coordinate {name!r} must be an array.")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"plot layer {layer!r} coordinate {name!r} contains non-finite values.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlotLayer:
    """One named data layer with generic coordinates and presentation hints."""

    name: str
    coordinates: Mapping[str, Any]
    label: str | None = None
    style: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("PlotLayer name must be non-empty.")
        if not isinstance(self.coordinates, Mapping) or not self.coordinates:
            raise ValueError("PlotLayer coordinates must be a non-empty mapping.")
        coordinates = {
            str(key): _coordinate_array(value, layer=name, name=str(key))
            for key, value in self.coordinates.items()
        }
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "label", None if self.label is None else str(self.label))
        object.__setattr__(self, "style", dict(self.style or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "coordinates": {key: value.tolist() for key, value in self.coordinates.items()},
            "style": dict(self.style),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PlotData:
    """A named visualization data product composed of independent layers."""

    kind: str
    layers: Sequence[PlotLayer]
    annotations: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("PlotData kind must be non-empty.")
        layers = tuple(self.layers)
        if not layers or not all(isinstance(layer, PlotLayer) for layer in layers):
            raise ValueError("PlotData layers must contain at least one PlotLayer.")
        names = [layer.name for layer in layers]
        if len(set(names)) != len(names):
            raise ValueError("PlotData layer names must be unique.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "annotations", dict(self.annotations or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def layer(self, name: str) -> PlotLayer:
        for layer in self.layers:
            if layer.name == str(name):
                return layer
        raise KeyError(f"Unknown plot layer {name!r}; valid layers are {[item.name for item in self.layers]!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layers": [layer.to_dict() for layer in self.layers],
            "annotations": dict(self.annotations),
            "metadata": dict(self.metadata),
        }


def coerce_plot_data(value: PlotData | Mapping[str, Any]) -> PlotData:
    """Normalize a ``PlotData`` instance or equivalent mapping."""

    if isinstance(value, PlotData):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("system.plot_data(...) must return PlotData or a mapping.")
    raw_layers = value.get("layers", ())
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)):
        raise TypeError("plot-data layers must be a sequence.")
    layers = []
    for raw in raw_layers:
        if isinstance(raw, PlotLayer):
            layers.append(raw)
        elif isinstance(raw, Mapping):
            layers.append(PlotLayer(**dict(raw)))
        else:
            raise TypeError("each plot-data layer must be PlotLayer or a mapping.")
    return PlotData(
        kind=str(value.get("kind", "")),
        layers=tuple(layers),
        annotations=dict(value.get("annotations", {}) or {}),
        metadata=dict(value.get("metadata", {}) or {}),
    )
