---
title: Gradient Descent and Line Search Theory
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/LINE_SEARCH.md
tags:
  - optimizer
  - theory
  - gradient_descent
  - line_search
  - armijo
---

# Gradient Descent and Line Search Theory

This note gives the mathematical context for:

- [Line Search API](../../optimizer/optimizers/LINE_SEARCH.md)

## Basic Descent Step

For a minimization objective `J(u)`, gradient descent uses:

```text
u_next = u - alpha * grad J(u)
```

where:

```text
u      = flattened Controls vector
alpha  = step_size
grad J = analytical gradient returned by system.gradient(controls)
```

For maximization, the sign is reversed.

## Directional Derivative

For a direction `p`, the first-order model is:

```text
J(u + alpha p) approx J(u) + alpha * grad J(u)^T p
```

For minimization, `p` is a descent direction when:

```text
grad J(u)^T p < 0
```

Line search methods use this quantity to decide whether a step size is reasonable.

## Normalized Direction

A normalized gradient step uses:

```text
p = -grad J / ||grad J||
```

This makes `step_size` represent an update length rather than a raw gradient scale.
It can help when gradients are very large or have changing magnitude.

## Backtracking

Backtracking starts with a trial step and shrinks it:

```text
alpha, alpha * shrink, alpha * shrink^2, ...
```

until the selected accept metric improves enough for the engine's acceptance rule.

## Armijo Condition

For minimization, Armijo sufficient decrease accepts when:

```text
J(u + alpha p) <= J(u) + c1 * alpha * grad J(u)^T p
```

where `0 < c1` is small.

This requires the trial to beat a first-order decrease threshold rather than merely
not get worse.

## Practical Use

Use line search when:

```text
you need an auditable baseline
step size is uncertain
gradient correctness is being debugged
custom metric guards are needed
```

Avoid using it as the only method when:

```text
coordinates have very uneven scales
the landscape needs curvature information
the objective is extremely expensive and repeated trial evaluations are costly
```

## API Reference

- [Line Search](../../optimizer/optimizers/LINE_SEARCH.md)
- [Acceptance and Stopping](../../optimizer/optimizers/ACCEPTANCE_AND_STOPPING.md)
- [Optimizer Lifecycle](../../optimizer/optimizers/LIFECYCLE.md)
