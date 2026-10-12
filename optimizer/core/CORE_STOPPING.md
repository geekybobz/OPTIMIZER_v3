---
role: topic
parent: CORE_HUB.md
summary: "Stopping rules, stationarity, and every stop reason a chunk can report."
read_when: "Read when interpreting result.stop_reason or configuring stops."
status: active
reviewed: 2026-10-12
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

## Stationarity

`run_chunk(gradient_tolerance=...)` stops before a step when the norm of the
descent gradient is at or below the tolerance. The engine default is `None`
(off); Adam, momentum, AdaGrad, RMSProp, L-BFGS, and nonlinear CG pass `1e-8`.
Newton-CG and trust region keep their own `gradient_tolerance`.

When the system declares `hvp`, the engine samples curvature in up to 16
directions before naming the stop. A minimum curvature below
`-stationary_curvature_tolerance` (default `1e-10`) reports
`stationary_nonminimum`; otherwise the stop is `gradient_converged`. The
gradient norm and the curvature evidence are stored in
`result.state.optimizer_state["stationary_evidence"]`. Without `hvp` the stop is
`gradient_converged` with `curvature_checked: false`, so a saddle cannot be told
apart from a minimum. The tolerance is an absolute norm; it does not use control
scales.

## Stop Reasons

`opt.core.STOP_REASONS` is the one registry of stop reasons for both families.
Each entry gives the reason's meaning, its outcome class (`converged`,
`stopped`, `failed`, or `interrupted`), the families that use it, and whether
the final state is a valid start for another run. The generated
[stop-reason table](../../API_REFERENCE.md#stop-reasons) lists every entry, and
`opt.diagnose_run` reports the outcome and continuability of a run's reason.

A chunk ends with one of these reasons:

| Origin | Reasons |
|---|---|
| The rules of this module | `maxiter`, `target`, `stall`, `nonfinite` |
| The stationarity check | `gradient_converged`, `stationary_nonminimum` |
| The loop's failure branches, in [Core Lifecycle](CORE_LIFECYCLE.md#failure-branches) | `gradient_failed`, `smooth_gradient_failed`, `proposal_failed`, `nonfinite_trial`, `interrupted`, and the fallback `stopped` |
| A method's own `StepProposal.stop_reason` | for example `projected_kkt_converged` or `proximal_mapping_converged` |

A new reason enters the registry first: a test fails when the library assigns a
reason the registry does not list.

A trial that fails to evaluate is not a stop by itself: the iteration is
recorded with reason `trial_evaluation_rejected`, the step size shrinks by
`failed_trial_shrink` (default `0.5`), and the run continues until
`failed_trial_retries` (default `3`) consecutive trials have failed.

Every reason arrives as `result.stop_reason` on a normal `OptimizerResult`.

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Engine](CORE_ENGINE.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md)
- [Acceptance and Stopping](../optimizers/OPTIMIZER_ACCEPTANCE.md)
- [Core Stopping Source](stopping.py)

---

[⌂ Home](CORE_HUB.md)
