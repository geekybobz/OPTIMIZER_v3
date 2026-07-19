---
title: Momentum Theory
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

## API Reference

- [Momentum](../../optimizer/optimizers/MOMENTUM.md)
- [State and Warmstart](../../optimizer/optimizers/STATE_AND_WARMSTART.md)
- [Optimizer Methods](../../optimizer/optimizers/METHODS.md)
