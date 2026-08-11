"""Read-only utilities for durable OPTIMIZER tracker run directories.

This package intentionally has no dependency on :mod:`optimizer`.  A future
``optimizer_log`` command can therefore inspect a live run without importing
the numerical optimizer, NumPy, a system, or a journal writer.
"""

from optimizer_log.reader import (
    TimelineReadResult,
    follow_timeline,
    inspect_writer,
    metric_coverage,
    project_timeline,
    read_live,
    read_receipt,
    read_terminal,
    read_timeline,
    read_writer,
    validate_event,
    validate_receipt,
    validate_timeline,
)

__all__ = [
    "TimelineReadResult",
    "follow_timeline",
    "inspect_writer",
    "metric_coverage",
    "project_timeline",
    "read_live",
    "read_receipt",
    "read_terminal",
    "read_timeline",
    "read_writer",
    "validate_event",
    "validate_receipt",
    "validate_timeline",
]
