---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Adam and its variants: arguments, standard acceptance, schedules, and moment state."
read_when: "Read when calling adam or continuing an Adam run."
status: active
reviewed: 2026-10-12
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
step_size            float or schedule object (see the Schedules Hub)
beta1
beta2
eps
weight_decay
max_step_norm
gradient_tolerance   stationarity stop, default 1e-8
state
warmstart
accept               default: accept every trial that evaluates successfully
accept_metric        metric for best-so-far tracking
accept_mode          min or max for best-so-far tracking
accept_tolerance     used only by the engine default (accept=opt.core.default_accept)
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

- [Adam Family Theory](../../Theory/optimizers/THEORY_ADAM_FAMILY.md)

## Related Notes

- [Methods](OPTIMIZER_METHODS.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [AdaGrad and RMSProp](ADAGRAD_RMSPROP.md)
- [Source](./adam.py)

---

[⌂ Home](OPTIMIZERS_HUB.md)
