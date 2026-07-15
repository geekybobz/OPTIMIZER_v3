# `schedules/`: Step And Trust Policies

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Schedules control how large an update is.

They should be reusable across optimizers instead of hard-coded inside each optimizer.

## Types

```text
constant step
adaptive shrink/grow
line-search backtracking
trust-region radius
LM damping
warmstart step estimate
```

## Basic Interface

```python
class Schedule:
    def initial(self, system, controls):
        ...

    def update(self, record):
        ...
```

## Why Separate From Optimizers

Adam, momentum, and line search all need step-size behavior. If schedules are separate,
we avoid duplicating shrink/grow and reset logic.

