---
title: Core Stopping
type: engine_policy
module: optimizer/core
tags:
  - optimizer
  - core
  - stopping
  - budget
---

# Core Stopping

This file documents the rules that end a chunk and the reasons they report.

All first-order optimizers need the same stopping behavior.  Keeping it in one place
is what stops the library drifting into several subtly different loops.

## StoppingConfig

```text
maxiter           attempted iteration budget, required, >= 0
target_value      stop once target_metric reaches it, default None
target_metric     metric watched by the target rule, default "J"
target_mode       "le" or "ge", default "le"
stall_patience    iterations without improvement before stopping, default None
stall_tolerance   improvement smaller than this does not count, default 0.0
stall_metric      metric watched by the stall rule, default "J"
stall_mode        "min" or "max", default "min"
check_finite      stop on non-finite metrics, default True
```

Validated on construction:

```text
maxiter >= 0
target_mode in {le, ge}
stall_mode in {min, max}
stall_patience >= 1 when provided
stall_tolerance finite and >= 0
target_value finite when provided
```

Invalid configuration fails immediately, before any system work happens.

Note that stopping has its own metric and mode fields, independent of the acceptance
metric and mode in [Core Engine](CORE_ENGINE.md#acceptance).  A run can accept on `J`
while watching a target on `fidelity`.

## Rule Order

Order matters, because the first rule that fires wins:

```text
finite  ->  target  ->  stall
```

Checking finiteness first means a run that goes numerically bad reports `nonfinite`
rather than a misleading `target` or `stall` derived from a corrupted value.

The budget rule is separate and runs before each iteration begins:

```text
check_before_iteration   stops when iteration >= maxiter
```

## Two Check Points

```text
check_initial_metrics    finite, then target
check_metrics            finite, then target, then stall
```

The initial check deliberately omits the stall rule so that starting metrics do not
consume a patience slot.  A chunk that starts at its target returns immediately with
zero iterations; a chunk that starts non-finite never requests a gradient.

## The Stall Rule

Stall is measured against the best value seen so far in the chunk, not the previous
iteration:

```text
min   improved when value <  best - stall_tolerance
max   improved when value >  best + stall_tolerance
```

```text
improved       best is updated, stall counter resets to zero
not improved   stall counter increments
counter >= stall_patience   stop with reason "stall"
```

Comparing against best-so-far rather than the previous value means an oscillating run
does not reset its own patience by bouncing upward and then partly back down.

## Finiteness

`check_finite` inspects numeric metric payloads only:

```text
numbers and numpy arrays    checked
lists and tuples            checked when they convert to float
anything else               ignored
```

Non-numeric entries are ignored on purpose, because systems legitimately put labels,
mode names, and other metadata in their metrics dict.

## Stop Reasons

```text
maxiter            iteration budget consumed
target             target_metric reached target_value
stall              no improvement for stall_patience iterations
nonfinite          metrics contained a non-finite numeric payload
gradient_failed    the system gradient call failed
proposal_failed    the step function raised or produced invalid controls
nonfinite_trial    the trial evaluation failed
stopped            fallback label
```

The first four come from this module.  The next three come from the loop's failure
branches, described in [Core Lifecycle](CORE_LIFECYCLE.md#failure-branches).

Every reason arrives as `result.stop_reason` on a normal `OptimizerResult`.

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Engine](CORE_ENGINE.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md)
- [Acceptance and Stopping](../optimizers/OPTIMIZER_ACCEPTANCE.md)
- [Core Stopping Source](stopping.py)
