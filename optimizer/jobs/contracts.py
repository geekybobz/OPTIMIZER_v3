"""Data-only job identities and conservative local admission limits."""
from __future__ import annotations
from dataclasses import dataclass
import math
from optimizer.utils.resources import cpu_budget


@dataclass(frozen=True)
class JobHandle:
    """A retained identity, not a process handle or automatic state activation."""
    job_id: str
    runtime_id: str
    operation: str
    context: str
    configuration: int
    fingerprint: str
    label: str | None = None


@dataclass(frozen=True)
class JobsConfig:
    """Reservations bound admission; sampled RSS is not an OS allocation limit."""
    max_workers: int = min(2, cpu_budget())
    threads_per_worker: int = 1
    memory_bytes: int = 1024 * 1024 * 1024
    coordinator_bytes: int = 256 * 1024 * 1024
    max_queue: int = 16
    max_jobs: int = 64
    input_bytes: int = 8 * 1024 * 1024
    scratch_bytes: int = 128 * 1024 * 1024
    shutdown_seconds: float = 2.0

    def __post_init__(self):
        for name in ("max_workers", "threads_per_worker", "memory_bytes", "coordinator_bytes",
                     "max_queue", "max_jobs", "input_bytes", "scratch_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_queue" else 1):
                raise ValueError(f"{name} must be a {'nonnegative' if name == 'max_queue' else 'positive'} integer.")
        if self.max_workers * self.threads_per_worker > cpu_budget():
            raise ValueError("Worker/native-thread reservations exceed the CPU budget "
                             "(available logical CPUs minus three, minimum one).")
        if self.coordinator_bytes >= self.memory_bytes:
            raise ValueError("Reserve less coordinator memory than the total memory budget.")
        if not math.isfinite(self.shutdown_seconds) or not 0 < self.shutdown_seconds <= 30:
            raise ValueError("shutdown_seconds must be finite, > 0 and <= 30.")
