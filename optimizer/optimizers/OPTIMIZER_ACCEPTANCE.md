---
title: OPTIMIZER_ACCEPTANCE
type: engine_policy
module: optimizer/optimizers
tags:
  - optimizer
  - acceptance
  - stopping
---

# Acceptance and Stopping

The shared engine decides whether a trial replaces the current controls.

## Default Acceptance

Default acceptance compares one scalar metric:

```text
accept_metric = "J"
accept_mode = "min"
accept_tolerance = 0.0
```

For minimization:

```text
accept when trial_metric <= current_metric + accept_tolerance
```

For maximization:

```text
accept when trial_metric >= current_metric - accept_tolerance
```

The selected metric must exist and be scalar.

`accept_mode` changes only this test for most methods. Among the gradient methods,
only `line_search` flips its proposal direction for `accept_mode="max"`; the others
always propose steps along `-grad J`.

## Custom Acceptance

`line_search` accepts an optional `accept` function.

Custom accept functions should return either:

```text
AcceptanceDecision
bool
```

Use this for guarded workflows where one metric is optimized only if other metrics
remain acceptable.

## Line Search Specifics

`line_search` variants:

```text
fixed
normalized
backtracking
armijo
```

`fixed` and `normalized` build one trial and let the engine evaluate it.

`backtracking` and `armijo` evaluate trial candidates inside the proposal function so
the step can shrink before the engine receives a proposal.

## Stopping Rules

Common stopping inputs:

```text
maxiter
target_value
target_metric
stall_patience
stall_tolerance
```

Stop reasons may include:

```text
maxiter
target
stall
nonfinite
gradient_failed
proposal_failed
nonfinite_trial
```

`maxiter` is attempted iterations.

## Best-So-Far

The engine updates `state.best_controls` and `state.best_metrics` from accepted
finite metrics only.

Rejected trial controls are not marked as best.

## Related Notes

- [Optimizer Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [Line Search](LINE_SEARCH.md)
- [Core Stopping Source](../core/stopping.py)
- [Core Engine Source](../core/engine.py)
