---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Heavy-ball, Nesterov, restart, and clipped momentum: arguments and velocity state."
read_when: "Read when calling momentum."
status: active
reviewed: 2026-10-12
---
# Momentum

`momentum` is a low-memory gradient optimizer with a velocity buffer.

It is useful when plain gradient descent zigzags or makes inefficient progress along
consistent gradient directions.

## Public Call

```python
result = opt.optimizers.momentum(
    system,
    controls,
    variant="nesterov",
    step_size=0.03,
    momentum=0.9,
    maxiter=20,
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
heavy_ball
  Standard velocity update.

nesterov
  Computes gradient at a lookahead control.

restart
  Resets velocity after rejected proposals.

clipped
  Applies default update norm clipping when max_step_norm is not supplied.
```

## Important Arguments

```text
variant
step_size            float or schedule object (see the Schedules Hub)
momentum
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
velocity
momentum
step_size
accept_count
reject_count
last_gradient_norm
last_raw_step_norm
last_step_clipped
```

Velocity shape is always:

```text
(controls.spec.size,)
```

## Watch Out

```text
large step_size can cause repeated rejection
Nesterov performs an additional gradient attempt at lookahead controls
restart intentionally drops velocity after rejection
```

## Related Theory

- [Momentum Theory](../../Theory/optimizers/THEORY_MOMENTUM.md)

## Related Notes

- [Methods](OPTIMIZER_METHODS.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [Source](./momentum.py)

---

[⌂ Home](OPTIMIZERS_HUB.md)
