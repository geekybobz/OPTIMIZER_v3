"""Bounded service-owned process jobs. No ordinary-Python history writer."""
from optimizer.jobs.contracts import JobHandle, JobsConfig
from optimizer.jobs.api import configure, submit, status, wait, result, cancel, compare, release, shutdown
from optimizer.catalog import attach_namespace_helpers, exports

attach_namespace_helpers(globals(), "jobs")
__all__ = [*exports("jobs"), "info", "list"]
