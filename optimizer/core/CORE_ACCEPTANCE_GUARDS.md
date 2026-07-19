---
title: Core Acceptance and Guards
type: engine_policy
module: optimizer/core
tags:
  - optimizer
  - core
  - acceptance
  - guards
---

# Core Acceptance and Guards

This file documents how the engine decides whether a trial replaces the current
controls, and how to state a stricter rule.

[Acceptance and Stopping](../optimizers/OPTIMIZER_ACCEPTANCE.md) covers what this
means for each shipped method.  This note owns the rules themselves.

## Default Acceptance

With no `accept` function, the engine compares one scalar metric:

```text
min   accept when trial <= current + accept_tolerance
max   accept when trial >= current - accept_tolerance
```

Defaults:

```text
accept_metric = "J"
accept_mode = "min"
accept_tolerance = 0.0
```

At the default tolerance an equal metric is still accepted, so a flat step is a move
rather than a rejection.

Requirements:

```text
the metric must exist in the metrics dict
the metric must be scalar
accept_tolerance must be finite and >= 0
```

Recorded technical payload:

```text
accept_metric, accept_mode, current_value, trial_value, improvement, tolerance
```

Rejection uses `reason="rejected_worse_metric"`.

## Multi-Metric Guards

Single-metric acceptance is not enough when a candidate can improve `J` while
damaging fidelity, energy, or a hard-condition metric.  `metric_guard` builds an
accept function that states both concerns:

```python
guard = opt.core.metric_guard(
    improve="J",
    mode="min",
    require={"fidelity": (">=", 0.99), "energy": ("<=", 1.0e-3)},
)

result = opt.run_chunk(
    system, controls, step=my_step,
    optimizer_name="guarded", maxiter=50, accept=guard,
)
```

A trial is accepted only when the `improve` metric does not get worse **and** every
`require` rule passes.

## Rule Syntax

```text
(operator, threshold)              uses the guard's default tolerance
(operator, threshold, tolerance)   uses a per-rule tolerance
```

Supported operators:

```text
<   <=   >   >=   ==   !=
```

Tolerance is applied so that comparisons stay forgiving in the intuitive direction:

```text
<  and <=   pass when value is below threshold + tolerance
>  and >=   pass when value is above threshold - tolerance
==          passes when |value - threshold| <= tolerance
!=          passes when |value - threshold| >  tolerance
```

## Failure Reporting

A guard rejection reports every failed rule, not just the first:

```text
guard.improve, guard.mode
guard.current_value, guard.trial_value, guard.improvement
guard.requirements   every rule with metric, operator, threshold, tolerance, trial, passed
guard.failures       the subset that failed, plus an "improve" entry when relevant
```

Rejection uses `reason="guard_failed"`.  Because the full requirement list is
recorded, a trace shows which constraint was binding across a run, not merely that
something was.

## One Sharp Edge

Guards require finite metrics.  A non-finite value in the `improve` metric or in any
`require` metric raises `ValueError` rather than quietly rejecting.

That exception is **not** converted into a stop reason.  The engine wraps the `step`
function in a try/except and reports `proposal_failed`, but it does not wrap the
accept function, so a guard error propagates out of `run_chunk` to the caller:

```text
step raises      -> chunk ends cleanly with stop_reason="proposal_failed"
accept raises    -> exception leaves run_chunk
```

In practice the finiteness rule usually fires first.  Metrics that are already
non-finite at the start of the chunk stop it during setup, and metrics that go
non-finite after an accepted move are caught at the end of that iteration.  The gap
is a metric that is finite at the current controls and non-finite only in the trial:
there the guard sees it before any stopping rule does.

Default acceptance performs no finiteness check at all, so the same trial merely
compares `J` and lets [Core Stopping](CORE_STOPPING.md) end the chunk with
`nonfinite`.

If a system can emit `NaN` in a guarded metric, prefer stopping on finiteness over
relying on the guard to absorb it.

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Contract](CORE_CONTRACT.md)
- [Core Engine](CORE_ENGINE.md)
- [Core Stopping](CORE_STOPPING.md)
- [Acceptance and Stopping](../optimizers/OPTIMIZER_ACCEPTANCE.md)
- [Line Search](../optimizers/LINE_SEARCH.md)
- [Core Guards Source](guards.py)
