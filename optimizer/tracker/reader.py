"""Compatibility exports for the standalone read-only tracker reader.

New UI/read-only code should import :mod:`optimizer_log.reader`, whose package
does not initialize the numerical optimizer.  Existing optimizer users may
continue importing these names from ``optimizer.tracker``.
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
