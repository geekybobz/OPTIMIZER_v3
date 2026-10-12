"""Bounded least-recently-used store shared by the OLGS and OLSS evaluators.

An evaluator cache exists to avoid repeating a propagation for controls or
shooting variables that were just evaluated, for example a backtracking trial
or a verification of the point a corrector just reached. Old entries are never
needed again, so the store keeps only the most recent ones; ``max_entries=None``
keeps everything.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Iterator

DEFAULT_CACHE_SIZE = 32


class BoundedCache:
    """Dictionary-like store that evicts its least recently used entry."""

    def __init__(self, max_entries: int | None = DEFAULT_CACHE_SIZE) -> None:
        if max_entries is not None and int(max_entries) < 1:
            raise ValueError("cache size must be a positive integer or None.")
        self.max_entries = None if max_entries is None else int(max_entries)
        self._items: OrderedDict[Hashable, Any] = OrderedDict()

    def __contains__(self, key: Hashable) -> bool:
        return key in self._items

    def __getitem__(self, key: Hashable) -> Any:
        value = self._items[key]
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        if self.max_entries is not None:
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._items)

    def clear(self) -> None:
        self._items.clear()
