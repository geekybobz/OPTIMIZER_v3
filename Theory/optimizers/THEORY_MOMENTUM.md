---
title: THEORY_MOMENTUM
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/MOMENTUM.md
tags:
  - optimizer
  - theory
  - momentum
  - nesterov
  - first_order
---

# Momentum Theory

This note gives the mathematical context for:

- [Momentum API](../../optimizer/optimizers/MOMENTUM.md)

## Heavy-Ball Momentum

Heavy-ball momentum stores a velocity vector:

```text
v_next = beta * v - alpha * grad J(u)
u_next = u + v_next
```

where:

```text
alpha = step_size
beta  = momentum coefficient
```

Repeated gradient directions accumulate. Oscillatory components can cancel.

## Nesterov Lookahead

Nesterov-style momentum evaluates the gradient at a lookahead point:

```text
u_lookahead = u + beta * v
v_next = beta * v - alpha * grad J(u_lookahead)
u_next = u + v_next
```

The update uses information from where the velocity is already pointing.

## Restart Behavior

Restart logic is useful when velocity points into a region that fails acceptance.

In this implementation:

```text
restart variant resets velocity after rejected proposals
ordinary variants preserve old velocity on rejection
```

## Clipping

Velocity or update clipping limits the applied update norm:

```text
if ||v_next|| > max_step_norm:
    v_applied = v_next * max_step_norm / ||v_next||
```

The direction is preserved and only the length changes.

## Worked Example

One coordinate with a consistent downhill direction shows velocity buildup:

```text
J(u) = 0.5 * u^2, grad = u
alpha = 0.1, beta = 0.9, u0 = 1, v0 = 0

iter 1: v1 = -0.1 * 1.000               = -0.100, u1 = 0.900
iter 2: v2 = 0.9*(-0.100) - 0.1*0.900   = -0.180, u2 = 0.720
iter 3: v3 = 0.9*(-0.180) - 0.1*0.720   = -0.234, u3 = 0.486
```

Plain gradient descent from the same start takes steps 0.100, 0.090, 0.081 and
only reaches 0.729. When gradient and velocity keep agreeing, the momentum step
grows toward the `alpha * |grad| / (1 - beta)` limit a persistent direction
allows — a 10x amplification at `beta = 0.9`.

## Convergence Notes

```text
on quadratics with tuned alpha and beta, heavy ball contracts error like
  (sqrt(kappa) - 1) / (sqrt(kappa) + 1) per step, versus
  (kappa - 1) / (kappa + 1) for plain gradient descent
Nesterov's method achieves the O(1/k^2) rate on smooth convex objectives
beta too close to 1 can oscillate or diverge outside the quadratic picture
```

## Practical Use

Use momentum when:

```text
plain gradient descent zigzags
you want a simple low-memory method
you need easier behavior to inspect than Adam
```

Watch for:

```text
large alpha causing rejected proposals
beta too close to 1 causing stale direction memory
Nesterov requiring an extra gradient computation
```

## References

```text
Polyak (1964), Some methods of speeding up the convergence of iteration
  methods
Nesterov (1983), A method of solving a convex programming problem with
  convergence rate O(1/k^2)
Sutskever, Martens, Dahl, Hinton (2013), On the importance of initialization
  and momentum in deep learning
```

## API Reference

- [Momentum](../../optimizer/optimizers/MOMENTUM.md)
- [State and Warmstart](../../optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
