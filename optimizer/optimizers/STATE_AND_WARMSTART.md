---
title: State and Warmstart
type: state_contract
module: optimizer/optimizers
tags:
  - optimizer
  - state
  - warmstart
---

# State and Warmstart

Optimizer calls return an `OptimizerResult`. The final run memory is available at:

```python
result.state
result.state.optimizer_state
```

## RunState

`RunState` stores:

```text
controls
metrics
iteration
global_iteration
step_size
optimizer_name
optimizer_state
best_controls
best_metrics
stop_reason
trace_id
checkpoint_ids
system_params
```

The current controls and metrics must describe the same point.

## WarmStartState

Create warmstart state through:

```python
warm = result.warmstart(target_optimizer="adam")
```

or:

```python
warm = opt.warmstart(result, target_optimizer="adam")
```

Warmstart always carries:

```text
controls
metrics
step_size
trace/checkpoint context
system params
```

Optimizer-private state transfers only when compatible:

```text
source_optimizer == target_optimizer
```

## Method State Keys

Common method state payloads:

```text
line_search
  variant
  step_size
  accept_count
  reject_count
  last_gradient_norm
  last_direction_norm
  last_accepted_step_size

momentum
  variant
  velocity
  momentum
  step_size
  accept_count
  reject_count
  last_gradient_norm
  last_raw_step_norm
  last_step_clipped

adam
  variant
  t
  m
  v
  v_max
  beta1
  beta2
  eps
  weight_decay
  step_size
  accept_count
  reject_count

adagrad/rmsprop
  method
  accumulator
  step_size
  decay
  eps
  accept_count
  reject_count

lbfgs
  history_size
  s_history
  y_history
  step_size
  last_gradient
  last_curvature
  last_pair_accepted

nonlinear_cg
  variant
  previous_gradient
  direction
  step_size
  beta
  restarted

cma_es
  variant
  mean
  diagonal_sigma
  population_size
  elite_count
  generation
  seed
  best_population_value
  best_population_index
```

## Branch Behavior

Accepted steps update current controls and accepted optimizer memory.

Rejected steps leave current controls unchanged and apply rejected-branch memory.
This is important for:

```text
Adam moments
Momentum velocity
L-BFGS curvature history
Nonlinear-CG directions
Line-search step-size adaptation
```

## Related Notes

- [Optimizer Lifecycle](LIFECYCLE.md)
- [Adam](ADAM.md)
- [Momentum](MOMENTUM.md)
- [L-BFGS](LBFGS.md)
- [RunState Source](../state.py)
