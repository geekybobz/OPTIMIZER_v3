---
title: THEORY_ADAM_FAMILY
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

## Worked Example

Two coordinates with a 1000x gradient scale difference, using the defaults
`beta1 = 0.9`, `beta2 = 0.999`:

```text
g = (0.01, 10) at every step

t = 1:
  m1 = 0.1 * g              = (0.001, 1.0)
  v1 = 0.001 * g^2          = (1e-7, 0.1)
  m_hat = m1 / (1 - 0.9)    = (0.01, 10)     (equals g)
  v_hat = v1 / (1 - 0.999)  = (1e-4, 100)    (equals g^2)

update = alpha * m_hat / (sqrt(v_hat) + eps) approx alpha * (1, 1)
```

Both coordinates move by about `alpha` even though the raw gradients differ by
a factor of 1000. A plain gradient step `alpha * g` would move the second
coordinate 1000 times farther. This per-coordinate normalization is the whole
point of the family.

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

## Convergence Notes

```text
Adam has no general convergence guarantee; Reddi et al. give convex
  counterexamples, and AMSGrad restores the guarantee
the practical strength is robustness to gradient scale, not asymptotic rate
in this engine, moments advance only on accepted steps, a safeguard the
  original method does not have
polish stages usually switch to nonlinear_cg or lbfgs after Adam stalls
```

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

## References

```text
Kingma & Ba (2015), Adam: A Method for Stochastic Optimization
Reddi, Kale, Kumar (2018), On the Convergence of Adam and Beyond    (AMSGrad)
Loshchilov & Hutter (2019), Decoupled Weight Decay Regularization   (AdamW)
Liu et al. (2020), On the Variance of the Adaptive Learning Rate
  and Beyond                                                        (RAdam)
Zhuang et al. (2020), AdaBelief Optimizer: Adapting Stepsizes by the
  Belief in Observed Gradients
```

## API Reference

- [Adam](../../optimizer/optimizers/ADAM.md)
- [AdaGrad and RMSProp](../../optimizer/optimizers/ADAGRAD_RMSPROP.md)
- [State and Warmstart](../../optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md)
