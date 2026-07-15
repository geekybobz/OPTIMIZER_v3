# `core/engine.py`: Engine

Status: review draft.
Last updated: 2026-07-15.

## Purpose

The engine owns the shared optimizer loop. Individual optimizers should not duplicate
driver logic.

## Engine Responsibilities

```text
initialize RunState
evaluate current controls
request gradient/residual/Jacobian as needed
ask optimizer for update
apply schedule
apply constraints
evaluate trial
accept or reject
update logs
save checkpoints
test stopping rules
return OptimizerResult
```

## Chunk Primitive

The native unit should be:

```python
engine.run_chunk(system, controls, optimizer, maxiter=10, state=None)
```

Full optimization is just repeated chunks until a stop rule fires.

This avoids the old duplicated scheduler and optimizer loops.

## Evaluation Cache

The engine should avoid rerunning propagation when the same controls are evaluated for
metrics and gradient in the same iteration.

The system can cache internally, but the engine should also pass around the latest
`Evaluation` where useful.

## Stopping

Basic stopping rules:

```text
maxiter
target metric reached
J improvement stalled
step too small
gradient norm small
budget exhausted
non-finite metric
guard failed
```

