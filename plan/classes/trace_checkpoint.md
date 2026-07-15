# `logs/`: Trace And Checkpoint

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Trace and checkpointing make curriculum safe. They let the user inspect what happened
and return to a useful state.

## Trace

`Trace` records lightweight events and iteration rows.

Example row:

```text
run_id
chunk
iter
global_iter
optimizer
stage
system_params
metrics
grad_norm
step_norm
control_norm
accepted
reason
```

Trace should be cheap enough to write often.

## Checkpoint

`Checkpoint` saves restorable state:

```text
controls
optimizer_state
system_params
metrics
iteration
stage label
random state
previous checkpoint id
```

Checkpoints are heavier than logs and should be saved by cadence or label.

## Labels

Important labels:

```text
latest
stage_start
accepted
best_J
best_safe
```

`stage_start` is the rollback anchor for curriculum.

## Restore

The restore API should be simple:

```python
controls, state = trace.restore("stage_start")
```

If multiple checkpoints share a label, restore the latest unless a specific checkpoint
id is passed.

