---
title: Core Engine
type: engine_policy
module: optimizer/core
tags:
  - optimizer
  - core
  - engine
  - api
---

# Core Engine

This file documents `run_chunk`: its argument surface, how a chunk chooses its
starting point, and what it records.

`run_chunk` takes a wide argument list because it is the single driver for every
gradient method.  The arguments group into seven concerns, and most callers touch
only the first two.

## Method and Budget

```text
system            positional; satisfies the optimizer system contract
controls          positional; starting controls, optional when state/warmstart given
step              required; StepContext -> StepProposal | Controls
optimizer_name    required; recorded in the result and every log record
maxiter           required; attempted iteration budget
```

## Starting Point

```text
state             RunState from a previous chunk of the same method
warmstart         WarmStartState handed over from another method
step_size         explicit initial step size
```

### Start-State Precedence

Three independent precedence chains resolve the starting point:

```text
controls         state > warmstart > controls argument
step_size        step_size argument > state > warmstart > None
optimizer_state  state > warmstart > empty
```

If none of `controls`, `state`, or `warmstart` is supplied, `run_chunk` raises
`ValueError`.

Continuation history is carried separately:

```text
from a previous state     global_iteration, checkpoint_ids, best_controls, best_metrics
from a warmstart only     checkpoint_ids
```

This is why `state` continues a run and `warmstart` starts a fresh one that merely
inherits a position: a warmstart deliberately drops best-so-far and the global
iteration count.

## Stopping

```text
stopping          StoppingConfig; replaces every loose stopping argument
target_value      stop once target_metric reaches this value
target_metric     metric watched for the target rule, default "J"
stall_patience    iterations without improvement before stopping
stall_tolerance   improvement smaller than this does not count
```

Passing `stopping=` overrides the loose arguments **including `maxiter`**:

```python
opt.run_chunk(..., maxiter=100, stopping=opt.StoppingConfig(maxiter=5))
```

runs five iterations, not one hundred.  Use one style or the other, never both.
Rule semantics live in [Core Stopping](CORE_STOPPING.md).

## Acceptance

```text
accept            custom accept function; overrides the default entirely
accept_metric     metric compared by the default rule, default "J"
accept_mode       "min" or "max"
accept_tolerance  how much worse a trial may be and still be accepted
```

`accept_metric` and `accept_mode` are also used for best-so-far tracking and for the
`best_<metric>` checkpoint label, so they still matter when `accept` is supplied.
See [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md).

## Logging

```text
trace             existing Trace to append to
create_trace      create a Trace when none was passed, default True
blackbox          BlackBoxRun, str, bool, or None
blackbox_policy   policy used when the engine creates the run
```

The `blackbox` argument is deliberately permissive:

```text
BlackBoxRun   append to that run; the caller keeps ownership and closes it
str           treated as a run target and opened by the engine
bool          True enables a default run, False disables
None          no blackbox recording
```

When the engine creates the run it also closes it at the end of the chunk.  When the
caller passes a `BlackBoxRun`, the engine never closes it.

See [Logs Hub](../logs/LOGS_HUB.md) and [Blackbox Hub](../blackbox/BLACKBOX_HUB.md).

## Checkpoints

```text
checkpoint_start      write "chunk_start" before the loop, default True
checkpoint_latest     write "latest" after each accepted trial, default True
checkpoint_accepted   write "accepted" after each accepted trial, default True
checkpoint_best       write "best_<accept_metric>" when best-so-far improves
```

Labels written by a chunk:

```text
chunk_start
latest
accepted
best_J          (or best_<accept_metric> for another metric)
```

Rejected trials write no checkpoint.  Turning all four off leaves iteration and chunk
records intact; checkpoints carry control payloads, records do not.

## Context

```text
system_params   explicit params for state and log context
stage           curriculum label recorded on every record
chunk           chunk index recorded on the chunk record
use_cache       reuse evaluations for unchanged controls, default True
```

When `system_params` is omitted the engine reads `system.params` if present, accepting
either a mapping or an object with attributes.  It never requires a params class.

## What Gets Recorded

```text
trace       iteration records, chunk records, checkpoints
blackbox    the same, plus previous controls, proposal controls,
            gradient, and trial metrics
```

The blackbox payload is richer because it is the durable numeric ledger; the trace is
the in-memory view.  Either can be absent, and the loop behaves identically.

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Contract](CORE_CONTRACT.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Stopping](CORE_STOPPING.md)
- [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md)
- [Optimizer State and Warmstart](../optimizers/OPTIMIZER_STATE_WARMSTART.md)
- [Core Engine Source](engine.py)
