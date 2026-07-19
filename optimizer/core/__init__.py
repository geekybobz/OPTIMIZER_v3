"""Shared chunk engine and its supporting boundaries.

Why this package exists
-----------------------
Every optimizer in the library needs the same outer mechanics: evaluate the current
controls, request an analytical gradient, build a trial, evaluate it, accept or reject
it, update run state, record logs, and stop consistently.  ``core`` owns that loop so
Adam, momentum, line search, L-BFGS, and nonlinear CG stay small proposal builders.

How it fits the architecture
----------------------------
- ``engine`` drives the chunk and defines the proposal contract.
- ``evaluate`` is the only validated path to ``system.evaluate`` / ``system.gradient``.
- ``stopping`` answers whether a chunk should end.
- ``guards`` turns metric rules into engine-compatible accept functions.

What this package deliberately does not do
------------------------------------------
It does not implement any specific method's mathematics, and it does not own generic
execution plumbing: map-style concurrency lives in ``optimizer.utils.parallel``.

Reviewer invariants
-------------------
- this module is the curated surface; callers should not deep-import submodules.
- names exported here match the ``core`` catalog group one-for-one.
"""

from optimizer.core.engine import (
    AcceptanceDecision,
    StepContext,
    StepProposal,
    default_accept,
    run_chunk,
)
from optimizer.core.evaluate import EvaluationOutcome, GradientOutcome, SystemEvaluator
from optimizer.core.guards import MetricGuard, metric_guard
from optimizer.core.stopping import StopDecision, StoppingConfig, StopTracker
from optimizer.catalog import attach_namespace_helpers


attach_namespace_helpers(globals(), "core")

__all__ = [
    "AcceptanceDecision",
    "EvaluationOutcome",
    "GradientOutcome",
    "MetricGuard",
    "StepContext",
    "StepProposal",
    "StopDecision",
    "StopTracker",
    "StoppingConfig",
    "SystemEvaluator",
    "default_accept",
    "info",
    "list",
    "metric_guard",
    "run_chunk",
]
