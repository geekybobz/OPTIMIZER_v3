---
title: THEORY_ADAPTIVE_SCALING
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/ADAGRAD_RMSPROP.md
tags:
  - optimizer
  - theory
  - adagrad
  - rmsprop
  - adaptive
---

# Adaptive Scaling Theory

This note gives the mathematical context for:

- [AdaGrad and RMSProp API](../../optimizer/optimizers/ADAGRAD_RMSPROP.md)

## Shared Idea

Both AdaGrad and RMSProp scale each coordinate by a squared-gradient accumulator:

```text
u_next = u - alpha * g / (sqrt(accumulator) + eps)
```

where `g` is the flattened analytical gradient.

## AdaGrad

AdaGrad accumulates squared gradients:

```text
accumulator_t = accumulator_{t-1} + g_t^2
```

The accumulator only grows. This can be useful for sparse or rarely active
coordinates, but later steps may become very small.

## RMSProp

RMSProp uses a moving average:

```text
accumulator_t = decay * accumulator_{t-1} + (1 - decay) * g_t^2
```

The method can adapt when the scale of gradients changes over time.

## Relationship To Adam

Adam combines:

```text
first-moment averaging like momentum
second-moment adaptive scaling like RMSProp
```

AdaGrad and RMSProp are smaller tools when adaptive scaling is useful but
first-moment memory is not wanted.

## Worked Example

The same uneven gradient as the Adam note, applied at every step:

```text
g = (0.01, 10)

AdaGrad after k identical steps:
  accumulator = k * g^2
  update = alpha * g / (sqrt(k) * |g|) approx alpha / sqrt(k) per coordinate
  k = 1: step approx alpha,  k = 4: alpha/2,  k = 100: alpha/10

RMSProp with decay = 0.9 reaches a steady state:
  accumulator -> g^2
  update -> approx alpha per coordinate, with no decay over time
```

Both methods equalize the two coordinates immediately. The difference is the
time axis: AdaGrad's effective step shrinks like `1/sqrt(k)` while RMSProp's
stays level once the accumulator saturates.

## Convergence Notes

```text
AdaGrad's cumulative accumulator gives the classic online-learning regret
  bound; the visible cost is the built-in 1/sqrt(k) step decay
RMSProp trades that guarantee for responsiveness to nonstationary scales
eps matters when accumulators start near zero: the first steps can be large
```

## Practical Use

Use AdaGrad when:

```text
some coordinates are sparse or rarely active
simple cumulative scaling is acceptable
```

Use RMSProp when:

```text
gradient scales are nonstationary
you want adaptive scaling without Adam momentum
```

Watch for:

```text
AdaGrad steps decaying too strongly
RMSProp decay reacting too slowly or too quickly
eps affecting very small accumulators
```

## References

```text
Duchi, Hazan, Singer (2011), Adaptive Subgradient Methods for Online
  Learning and Stochastic Optimization                              (AdaGrad)
Tieleman & Hinton (2012), COURSERA Neural Networks, lecture 6.5     (RMSProp)
```

## API Reference

- [AdaGrad and RMSProp](../../optimizer/optimizers/ADAGRAD_RMSPROP.md)
- [Adam](../../optimizer/optimizers/ADAM.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
