"""Bounded numerical samples, scalar aggregates and display-only progress.

Nothing here writes files or persistent logs. A frontend may install a reporter
for the length of one call with `_reporting`; library loops send it small
events through `_report`. Without a reporter, `_report` does nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
from optimizer.core.record import RunRecord

_REPORTER: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "optimizer_progress_reporter", default=None)


def _report(**event: Any) -> None:
    """Send one display-only event to the installed reporter, if any.

    A failing reporter must never change or stop the computation.
    """
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        reporter(event)
    except Exception:  # noqa: BLE001 - display problems are not numerical failures.
        pass


@contextmanager
def _reporting(reporter: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """Install `reporter` for the calls made inside this block."""
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


class _Progress:
    def __init__(self, metrics, metric, *, method=None, budget=None, report=True):
        self.report = bool(report)
        self.record = RunRecord()
        self.metric = metric
        self.method = method
        self.budget = budget
        self.start = self.end = float(metrics[metric])
        self.minimum = self.maximum = self.start
        self.attempts = self.accepted = self.reversals = 0
        self.last_sign = 0
        self.last_failure = None

    def observe(self, metrics, *, accepted, failure=None, derivative_norm=None, step_norm=None):
        value = float(metrics[self.metric])
        sign = (value > self.end) - (value < self.end)
        if sign:
            self.reversals += int(bool(self.last_sign) and sign != self.last_sign)
            self.last_sign = sign
        self.end = value
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)
        self.attempts += 1
        self.accepted += int(accepted)
        if not accepted and failure:
            self.last_failure = str(failure)
        self.record.append(dict(iteration=self.attempts, merit=value, derivative_norm=derivative_norm,
                                step_norm=step_norm, accepted=bool(accepted), reason=failure))
        if self.report:
            _report(kind="iteration", method=self.method, iteration=self.attempts, total=self.budget,
                metric=self.metric, value=value, accepted=bool(accepted))

    def finish(self, counters):
        summary = dict(objective_metric=self.metric, start=self.start, end=self.end,
                    minimum=self.minimum, maximum=self.maximum,
                    attempts=self.attempts, accepted=self.accepted,
                    rejected=self.attempts-self.accepted, reversals=self.reversals,
                    last_failure=self.last_failure, counters=dict(counters))

        self.record.summary = dict(summary)
        return summary
