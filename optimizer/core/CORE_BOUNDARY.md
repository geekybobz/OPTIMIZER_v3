---
title: Core Boundary
type: module_boundary
module: optimizer/core
tags:
  - optimizer
  - core
  - boundary
  - architecture
---

# Core Boundary

This file defines what belongs in `optimizer/core`.

The test for core membership is narrow: a thing belongs here if every gradient method
needs it and no method may implement it differently.

## Core Owns

```text
run_chunk, the shared chunk loop
StepContext and StepProposal, the proposal contract
AcceptanceDecision and the acceptance policy
SystemEvaluator, the validated path to the system
StoppingConfig, StopTracker, and stop reasons
metric guards
trace and checkpoint plumbing for the loop
```

## Core Does Not Own

```text
method mathematics          -> optimizers
physics and gradients       -> system_olgs
initial controls            -> guesses
step-size policy            -> schedules
diagnostics and repair      -> utils
map-style parallelism       -> utils
record storage and analysis -> logs, blackbox
derivations and intuition   -> Theory
```

Optimizers call core; core stays method-agnostic.  Core calls the system through one
validated boundary and never reaches into physics itself.

## Two Settled Cases

Both were previously ambiguous and are now fixed.

### `metric_guard` belongs to core

It is implemented in `guards.py`, catalogued in the `core` group, and called as
`opt.core.metric_guard(...)`.  It reads like a utility, but it produces an engine
accept function and is meaningless outside the acceptance contract.

- [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md)

### Parallel execution belongs to utils

`parallel_map` is generic execution plumbing with no optimizer semantics, and nothing
in the chunk engine imports it.  It lives at `optimizer/utils/parallel.py`.

- [Util Boundary](../utils/UTIL_BOUNDARY.md)

## `core` and `base` Are Different Namespaces

The word `core` names this package only:

```text
opt.core    run_chunk, StepContext, StepProposal, AcceptanceDecision,
            SystemEvaluator, StoppingConfig, metric_guard

opt.base    Controls, ControlSpec, OptimizerResult, WarmStartState,
            Trace, BlackBoxRun, evaluate, gradient, context
```

`base` covers the top-level spine that everything shares: the data objects and the
entry points that live directly under `optimizer/`.  Those are not engine mechanics,
so they are not core.

Both namespaces are catalogued, and `opt.core.list()` and `opt.base.list()` describe
their own contents.

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Contract](CORE_CONTRACT.md)
- [Optimizer Boundary](../optimizers/OPTIMIZER_BOUNDARY.md)
- [Util Boundary](../utils/UTIL_BOUNDARY.md)
- [OLGS Logging Boundary](../system_olgs/LOGGING_BOUNDARY.md)
- [Logs Hub](../logs/LOGS_HUB.md)
- [Blackbox Hub](../blackbox/BLACKBOX_HUB.md)
