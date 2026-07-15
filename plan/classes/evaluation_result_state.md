# Evaluation, Result, And State

Status: review draft.
Last updated: 2026-07-15.

## Evaluation

`Evaluation` is the output of `system.evaluate(controls)`.

It should hold:

```text
metrics
state snapshot reference or selected state data
cost value J
optional cached trajectory id
```

Example:

```python
eval.metrics["J"]
eval.metrics["fidelity"]
eval.metrics["energy"]
```

## OptimizerResult

`OptimizerResult` is returned by public calls:

```python
result = opt.adam(system, controls)
```

It should hold:

```text
controls
metrics
state
trace
stop_reason
iterations
optimizer
system_params
history summary
checkpoint ids
```

It should support:

```python
result.controls
result.metrics
result.state
result.to_dict()
```

## RunState

`RunState` is internal mutable state during a run:

```text
current controls
current metrics
current gradient
step size
iteration
global iteration
best controls
best metrics
stall counters
optimizer-specific state
trace id
```

## WarmStartState

`WarmStartState` is the transferable part of a previous result:

```text
controls
metrics
step estimate
compatible optimizer state
trace/checkpoint context
```

If optimizer-specific state is incompatible, keep only controls and metrics.

