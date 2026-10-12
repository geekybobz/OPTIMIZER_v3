"""One JSON conversion policy for results, reports, and transaction responses.

``to_plain`` turns library values into plain JSON data, so every result,
diagnostic report, campaign record, and transaction response serializes the
same way and stays valid strict JSON:

- non-finite floats become the strings ``"nan"``, ``"inf"``, and ``"-inf"``;
- complex numbers become ``{"real": ..., "imag": ...}``;
- NumPy scalars and arrays become Python values and lists;
- objects with ``to_dict()`` use it, ``Controls`` use their serialized form,
  dataclasses become field mappings, and systems become their type and
  ``system_spec()``;
- callables become ``"module:qualname"`` references and paths become strings.

Anything else raises ``TypeError``: durable output never falls back to ``repr()``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import numpy as np


def _float(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0.0 else "-inf"


def _reference(value: Any) -> str:
    if isinstance(value, functools.partial):
        return f"{_reference(value.func)} (partial)"
    module = getattr(value, "__module__", None) or type(value).__module__
    qualname = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"{module}:{qualname}"


def to_plain(value: Any) -> Any:
    """Return ``value`` as plain JSON data under the library's single policy."""

    from optimizer.controls import Controls

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return to_plain(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, complex):
        return {"real": _float(value.real), "imag": _float(value.imag)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": to_plain(np.real(value)), "imag": to_plain(np.imag(value))}
        return to_plain(value.tolist())
    if isinstance(value, Controls):
        from optimizer.result import controls_to_dict

        return to_plain(controls_to_dict(value))
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [to_plain(item) for item in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, PurePath):
        return str(value)
    if not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_plain(to_dict())
        system_spec = getattr(value, "system_spec", None)
        if callable(system_spec):
            return {"system_type": type(value).__name__, "system_spec": to_plain(system_spec())}
        if dataclasses.is_dataclass(value):
            return {field.name: to_plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if callable(value):
        return _reference(value)
    raise TypeError(
        f"{type(value).__qualname__} is not JSON-serializable; give it to_dict() or convert it first."
    )
