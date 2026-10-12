---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "AdaGrad and RMSProp: per-coordinate scaling by squared-gradient history, arguments, and state."
read_when: "Read when calling adagrad or rmsprop."
status: active
reviewed: 2026-10-12
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
step_size            float or schedule object (see the Schedules Hub)
eps
initial_accumulator
max_step_norm
gradient_tolerance   stationarity stop, default 1e-8
state
warmstart
accept               default: accept every trial that evaluates successfully
accept_metric        metric for best-so-far tracking
accept_mode          min or max for best-so-far tracking
accept_tolerance     used only by the engine default (accept=opt.core.default_accept)
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

- [Adaptive Scaling Theory](../../Theory/optimizers/THEORY_ADAPTIVE_SCALING.md)

## Related Notes

- [Methods](OPTIMIZER_METHODS.md)
- [Adam](ADAM.md)
- [Source](./adaptive.py)

---

[⌂ Home](OPTIMIZERS_HUB.md)
