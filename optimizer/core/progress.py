"""Constant-size numerical aggregates; no samples, events, IDs, or writer."""
from __future__ import annotations


class _Progress:
    def __init__(self, metrics, metric):
        self.metric = metric
        self.start = self.end = float(metrics[metric])
        self.minimum = self.maximum = self.start
        self.attempts = self.accepted = self.reversals = 0
        self.last_sign = 0
        self.last_failure = None

    def observe(self, metrics, *, accepted, failure=None):
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

    def finish(self, counters):
        return dict(objective_metric=self.metric, start=self.start, end=self.end,
                    minimum=self.minimum, maximum=self.maximum,
                    attempts=self.attempts, accepted=self.accepted,
                    rejected=self.attempts-self.accepted, reversals=self.reversals,
                    last_failure=self.last_failure, counters=dict(counters))
