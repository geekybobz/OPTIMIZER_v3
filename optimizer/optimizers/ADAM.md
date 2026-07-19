---
title: Adam Optimizer
type: method_reference
module: optimizer/optimizers
method: adam
source: optimizer/optimizers/adam.py
tags:
  - optimizer
  - adam
  - adaptive
  - first_order
  - warmstart
---

# Adam

`adam` is an adaptive first-order optimizer with first and second moment state.

It is usually the default rough optimizer when gradient coordinates have uneven
scales.

## Public Call

```python
result = opt.optimizers.adam(
    system,
    controls,
    variant="adam",
    step_size=0.05,
    maxiter=50,
)
```

## Requires

```text
system.evaluate(controls)
system.gradient(controls)
Controls
```

## Variants

```text
adam
  Standard Adam.

amsgrad
  Uses a running maximum of second-moment estimates.

adamw
  Adds decoupled weight decay.

radam
  Rectified Adam; uses momentum-like behavior before variance rectifier is reliable.

adabelief
  Tracks squared surprise relative to the first-moment estimate.
```

## Important Arguments

```text
variant
step_size
beta1
beta2
eps
weight_decay
max_step_norm
state
warmstart
accept_metric
accept_mode
accept_tolerance
```

## Optimizer State

Typical keys:

```text
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
last_gradient_norm
last_raw_step_norm
last_step_clipped
```

`m`, `v`, and `v_max` are flat arrays with shape:

```text
(controls.spec.size,)
```

Accepted steps advance moments and timestep. Rejected steps preserve previous
moments and timestep.

## Watch Out

```text
step_size dominates behavior
weight_decay is explicit optimizer decay, not a replacement for physical energy terms
warmstart transfers moment state only into another adam call
```

## Related Theory

- [Adam Family Theory](../../Theory/optimizers/adam_family.md)

## Related Notes

- [Methods](METHODS.md)
- [State and Warmstart](STATE_AND_WARMSTART.md)
- [AdaGrad and RMSProp](ADAGRAD_RMSPROP.md)
- [Source](./adam.py)
