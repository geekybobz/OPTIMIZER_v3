"""Small parallel execution helpers for optimizer tools.

Why this file exists
--------------------
The library is expected to support vectorized and multicore workflows.  Batch
diagnostics, geometry probes, finite-difference experiments, restart scans, and guess
families will all need to map one function over many independent control objects.
If every module opens its own executor, resource use and error behavior will become
inconsistent.

This module provides a narrow ``parallel_map`` interface.  It is deliberately boring:
serial execution is the default, thread/process backends are opt-in, and result order
always matches input order.

How it fits the architecture
----------------------------
- this is generic execution plumbing, not optimizer policy, so it lives in ``utils``
  rather than ``core``: nothing in the chunk engine imports it.
- diagnostics and future batch modes can call ``parallel_map`` for independent work.
- public APIs can expose a small ``ParallelConfig`` instead of executor internals.

What this file deliberately does not do
---------------------------------------
It does not vectorize physics inside a system, share mutable run state across workers,
or manage distributed compute.  It only standardizes local map-style parallelism.

Reviewer invariants
-------------------
- serial execution has no executor overhead and is deterministic.
- thread/process outputs preserve the input order.
- backend names are explicit: ``serial``, ``thread``, or ``process``.
- invalid worker counts fail before work starts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import Callable, Iterable, Sequence, TypeVar


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for local map-style parallel execution."""

    backend: str = "serial"
    workers: int | None = None
    chunksize: int = 1

    def __post_init__(self) -> None:
        if self.backend not in {"serial", "thread", "process"}:
            raise ValueError("backend must be 'serial', 'thread', or 'process'.")
        if self.workers is not None and int(self.workers) < 1:
            raise ValueError("workers must be >= 1 when provided.")
        if int(self.chunksize) < 1:
            raise ValueError("chunksize must be >= 1.")
        object.__setattr__(self, "workers", None if self.workers is None else int(self.workers))
        object.__setattr__(self, "chunksize", int(self.chunksize))

    def resolved_workers(self) -> int:
        """Return the worker count this config implies for local execution."""

        if self.backend == "serial":
            return 1
        if self.workers is not None:
            return self.workers
        return max(1, cpu_count() or 1)


def parallel_map(
    function: Callable[[T], U],
    items: Iterable[T],
    *,
    config: ParallelConfig | None = None,
) -> list[U]:
    """Apply ``function`` to each item while preserving input order."""

    cfg = config or ParallelConfig()
    sequence: Sequence[T] = list(items)
    if cfg.backend == "serial" or cfg.resolved_workers() == 1:
        return [function(item) for item in sequence]

    executor_cls = ThreadPoolExecutor if cfg.backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=cfg.resolved_workers()) as executor:
        return list(executor.map(function, sequence, chunksize=cfg.chunksize))


def serial_config() -> ParallelConfig:
    """Return an explicit serial config for call sites that prefer clarity."""

    return ParallelConfig(backend="serial", workers=1)
