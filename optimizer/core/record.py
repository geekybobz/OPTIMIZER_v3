"""Bounded numerical evidence, independent of frontend command history."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from optimizer.serialization import to_plain


@dataclass
class RunRecord:
    """Retain the latest samples and exact whole-run scalar aggregates."""
    limit: int = 256
    rows: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    truncated: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.limit, int) or self.limit < 1:
            raise ValueError("RunRecord limit must be a positive integer.")
        self.rows = [dict(row) for row in self.rows[-self.limit:]]

    def append(self, row):
        self.rows.append(dict(row))
        self.total += 1
        if len(self.rows) > self.limit:
            del self.rows[0]
            self.truncated = True

    def to_dict(self):
        return to_plain(dict(rows=self.rows, total=self.total, truncated=self.truncated,
                             sampling="latest", limit=self.limit, summary=self.summary))
