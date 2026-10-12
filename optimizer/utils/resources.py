"""Conservative CPU admission shared by local executors and job workers."""
from __future__ import annotations

import os

RESERVED_CPUS = 3


def available_cpus() -> int:
    """Respect process affinity when available, otherwise use logical CPUs."""
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        count = process_count()
        if count is not None:
            return max(1, count)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def cpu_budget() -> int:
    """Leave three available logical CPUs free; small machines run serially."""
    return max(1, available_cpus() - RESERVED_CPUS)
