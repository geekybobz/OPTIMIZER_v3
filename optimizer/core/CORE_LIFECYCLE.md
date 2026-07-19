---
title: Core Lifecycle
type: computation_flow
module: optimizer/core
tags:
  - optimizer
  - core
  - engine
  - lifecycle
---

# Core Lifecycle

This file defines the exact order of one chunk and what each exit path leaves behind.

[Optimizer Lifecycle](../optimizers/OPTIMIZER_LIFECYCLE.md) describes the same loop
from a method author's point of view.  This note is the engine's own account: the
ordering guarantees, the counting rules, and the failure branches.

## Chunk Setup

Before the loop runs:

```text
validate that step is callable
build the SystemEvaluator
resolve trace and blackbox
choose starting controls
validate controls against system.control_spec()
evaluate starting controls
build RunState
record the chunk_start checkpoint
check initial stopping rules
```

Two setup outcomes are not failures of the loop:

```text
starting controls fail evaluation  -> raises ValueError, no result is returned
initial metrics already stop       -> returns a result with zero iterations
```

The first is an argument error and is loud on purpose.  The second is normal: a
target that is already met, or non-finite starting metrics, ends the chunk cleanly
before any gradient is requested.

## Iteration Order

Each pass through the loop runs in this order:

```text
check iteration budget
compute analytical gradient
build StepContext
call the method's step function
validate the proposed controls
evaluate the proposed controls
decide acceptance
apply optimizer-private state for the chosen branch
update controls and best-so-far, or count a rejection
record the iteration
check target, stall, and finiteness rules
```

The budget check happens *before* work, so `maxiter=0` returns immediately without
requesting a gradient.

## Accepted and Rejected

```text
accepted   controls and metrics are replaced
           step_size updated from the accept branch
           best-so-far updated when the metric improved
           latest / accepted / best checkpoints written
           iteration and global_iteration advance by one

rejected   controls and metrics are left untouched
           step_size updated from the reject branch
           no checkpoint written
           iteration and global_iteration still advance by one
```

Both branches advance the counters.  `maxiter` therefore means attempted iterations,
not accepted ones, and a chunk that rejects every trial still terminates.

Best-so-far is updated only from accepted, finite metrics.  A rejected trial is never
recorded as best even when its metric looks better under a different measure.

## Failure Branches

Three failures end the chunk immediately.  None of them retry, and all of them return
a normal `OptimizerResult` carrying the reason:

```text
gradient_failed    the system's gradient call raised or failed validation
                   controls and metrics unchanged
                   the failing iteration is counted
                   technical payload carries the error string

proposal_failed    the step function raised, or its controls failed validation
                   controls and metrics unchanged
                   the failing iteration is counted
                   technical payload carries "TypeName: message"

nonfinite_trial    the trial controls evaluated to a failure or non-finite metrics
                   controls and metrics unchanged
                   the failing iteration is counted
                   reject-branch optimizer_state and step_size are still applied
```

The distinction worth remembering: a failure inside the *loop* is data, recorded and
returned. A failure during *setup* is an exception.

## Chunk Exit

Every exit path converges on the same tail:

```text
set state.stop_reason
record the chunk record on the trace
record the chunk record on the blackbox
build OptimizerResult from state
close the blackbox when the engine created it
```

Because the tail is shared, a chunk that stopped at `maxiter`, `target`, `stall`, or
any failure branch returns the same result shape.  Callers branch on
`result.stop_reason`, never on result type.

## Chaining Chunks

The engine is deliberately short-horizon.  Long runs are chunks in sequence:

```python
first = opt.optimizers.adam(system, controls, maxiter=50)
second = opt.optimizers.lbfgs(system, warmstart=first.warmstart(target_optimizer="lbfgs"))
```

What carries across a chunk boundary is described in
[Core Engine](CORE_ENGINE.md#start-state-precedence).

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Contract](CORE_CONTRACT.md)
- [Core Engine](CORE_ENGINE.md)
- [Core Stopping](CORE_STOPPING.md)
- [Core Evaluation](CORE_EVALUATION.md)
- [Optimizer Lifecycle](../optimizers/OPTIMIZER_LIFECYCLE.md)
- [Optimizer State and Warmstart](../optimizers/OPTIMIZER_STATE_WARMSTART.md)
- [Core Engine Source](engine.py)
