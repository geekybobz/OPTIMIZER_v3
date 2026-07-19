---
title: Adam Family Theory
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/ADAM.md
tags:
  - optimizer
  - theory
  - adam
  - adaptive
  - first_order
---

# Adam Family Theory

This note gives the mathematical context for:

- [Adam API](../../optimizer/optimizers/ADAM.md)

## Adam Moments

Adam keeps first and second moment estimates:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

Bias-corrected moments:

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

Update for minimization:

```text
u_next = u - alpha * m_hat / (sqrt(v_hat) + eps)
```

## Why It Helps

Adam adapts per coordinate.

Large historical squared gradients reduce future steps for that coordinate.
Small historical squared gradients allow larger coordinate steps.

This is useful for controls where channels or time samples have different gradient
scales.

## AMSGrad

AMSGrad stores a running maximum:

```text
v_max = max(v_max, v_hat)
```

Then uses:

```text
m_hat / (sqrt(v_max) + eps)
```

This prevents the denominator from decreasing coordinate-wise.

## AdamW

AdamW uses decoupled weight decay:

```text
direction = adaptive_direction + weight_decay * u
```

In physical-control problems, objective energy penalties usually belong in
`system.evaluate` and `system.gradient`. AdamW weight decay should be an explicit
optimizer choice, not a hidden replacement for physical objective terms.

## RAdam

RAdam rectifies the adaptive denominator only after the variance estimate has enough
effective history.

Before the rectifier is reliable, this implementation behaves more like a momentum
method using the first-moment direction.

## AdaBelief

AdaBelief changes the second moment from squared gradient to squared surprise:

```text
surprise = g_t - m_t
v_t = beta2 * v_{t-1} + (1 - beta2) * surprise^2
```

Coordinates with predictable gradients can receive different scaling than
coordinates with volatile gradients.

## Practical Use

Use Adam-family methods when:

```text
gradient coordinates have uneven scales
the initial guess is rough
you want a robust first optimizer before polishing
```

Watch for:

```text
step_size dominating performance
moment state advancing only on accepted proposals
incompatible optimizer_state not transferring to other optimizer families
```

## API Reference

- [Adam](../../optimizer/optimizers/ADAM.md)
- [AdaGrad and RMSProp](../../optimizer/optimizers/ADAGRAD_RMSPROP.md)
- [State and Warmstart](../../optimizer/optimizers/STATE_AND_WARMSTART.md)
