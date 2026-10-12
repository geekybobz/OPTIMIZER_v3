"""Shared evaluator ownership, cache identity, counting and captured outcomes."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
import numpy as np
from optimizer.core._cache import BoundedCache, DEFAULT_CACHE_SIZE
from optimizer.serialization import to_plain

T = TypeVar("T")


@dataclass(frozen=True)
class CallOutcome(Generic[T]):
    value: T | None
    ok: bool
    reason: str = "ok"
    error: str | None = None


def candidate_key(variables, options=None):
    """Full coordinate identity, numeric content and numerical options."""
    layout = variables.spec.to_dict()
    # Descriptive metadata never affects numerical identity.
    layout.pop("meta", None)
    layout.pop("metadata", None)
    if "blocks" in layout:
        layout["blocks"] = [dict((k, v) for k, v in block.items()
                                  if k not in {"metadata", "meaning"}) for block in layout["blocks"]]
    return (json.dumps(to_plain(layout), sort_keys=True),
            np.ascontiguousarray(variables.flatten(copy=False)).tobytes(),
            json.dumps(to_plain(dict(options or {})), sort_keys=True))


class EvaluatorBase:
    """Common mechanics; adapters supply only validated project operations."""
    def _initialize_runtime(self, *, use_cache=True, cache_size=DEFAULT_CACHE_SIZE):
        self.use_cache = bool(use_cache)
        self._cache_size = cache_size
        self._caches = []

    def _new_cache(self):
        cache = BoundedCache(self._cache_size)
        self._caches.append(cache)
        return cache

    def clear_cache(self):
        for cache in self._caches:
            cache.clear()

    @staticmethod
    def attempt(function: Callable[..., T], *args, reason="evaluation_failed", **kwargs) -> CallOutcome[T]:
        try:
            return CallOutcome(function(*args, **kwargs), True)
        except Exception as exc:
            return CallOutcome(None, False, reason, f"{type(exc).__name__}: {exc}")

    def _counted(self, operation, function, *args, **kwargs):
        self._increment(operation + "_attempt")
        value = function(*args, **kwargs)
        self._increment(operation)
        return value

    def _increment(self, operation):
        if hasattr(self, "_counts"):
            self._counts[operation] += 1
        else:
            attribute = operation + "_count"
            setattr(self, attribute, getattr(self, attribute) + 1)
