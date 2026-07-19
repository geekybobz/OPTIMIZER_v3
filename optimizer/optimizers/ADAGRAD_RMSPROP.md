---
title: AdaGrad and RMSProp
type: method_reference
module: optimizer/optimizers
method: adagrad_rmsprop
source: optimizer/optimizers/adaptive.py
tags:
  - optimizer
  - adagrad
  - rmsprop
  - adaptive
  - first_order
---

# AdaGrad and RMSProp

`adagrad` and `rmsprop` are adaptive first-order methods without Adam's first-moment
buffer.

Both scale each coordinate by a history of squared gradients.

## Public Calls

```python
adagrad_result = opt.optimizers.adagrad(
    system,
    controls,
    step_size=0.05,
    maxiter=30,
)
```

```python
rmsprop_result = opt.optimizers.rmsprop(
    system,
    controls,
    step_size=0.02,
    decay=0.8,
    maxiter=30,
)
```

## Requires

```text
system.evaluate(controls)
system.gradient(controls)
Controls
```

## Method Difference

```text
adagrad
  accumulator = accumulator + gradient^2

rmsprop
  accumulator = decay * accumulator + (1 - decay) * gradient^2
```

AdaGrad's accumulator only grows. RMSProp's accumulator is a moving average.

## Important Arguments

```text
step_size
eps
initial_accumulator
max_step_norm
state
warmstart
accept_metric
accept_mode
accept_tolerance
```

RMSProp additionally accepts:

```text
decay
```

## Optimizer State

Typical keys:

```text
method
accumulator
step_size
decay
eps
accept_count
reject_count
last_gradient_norm
last_raw_step_norm
last_step_clipped
```

The accumulator is a flat array with shape:

```text
(controls.spec.size,)
```

## Watch Out

```text
AdaGrad can make later steps very small
RMSProp decay controls how fast scaling adapts
these methods do not carry first-moment momentum
```

## Related Theory

- [Adaptive Scaling Theory](../../Theory/optimizers/adaptive_scaling.md)

## Related Notes

- [Methods](METHODS.md)
- [Adam](ADAM.md)
- [Source](./adaptive.py)
