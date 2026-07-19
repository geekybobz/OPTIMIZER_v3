---
title: THEORY_LINE_SEARCH
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

## Worked Example

Take an ill-conditioned quadratic with its minimum at the origin:

```text
J(u) = 0.5 * (u1^2 + 10 * u2^2)
grad J = (u1, 10 * u2)

start: u = (1, 1), J = 5.5, grad = (1, 10)
```

A fixed step with `alpha = 0.3` overshoots the steep coordinate:

```text
u_trial = (1, 1) - 0.3 * (1, 10) = (0.7, -2.0)
J_trial = 20.2   -> rejected (worse than 5.5)
```

Backtracking shrinks by `shrink = 0.5` and retries:

```text
alpha = 0.15
u_trial = (0.85, -0.5)
J_trial = 1.61   -> accepted
```

The Armijo variant additionally checks sufficient decrease at `alpha = 0.15`:

```text
grad^T p = -(1^2 + 10^2) = -101
threshold = 5.5 + 1e-4 * 0.15 * (-101) = 5.4985
J_trial = 1.61 <= 5.4985   -> sufficient decrease holds
```

The sign flip in `u2` is the classic zigzag signature of ill conditioning: the
step size that survives is dictated by the steepest coordinate.

## Convergence Notes

```text
a fixed step alpha <= 1/L converges at O(1/k) on L-smooth convex objectives
the rate degrades with the condition number; zigzag is the visible symptom
backtracking removes the need to know L in advance
engine acceptance keeps the run monotone but does not fix conditioning
```

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

## References

```text
Nocedal & Wright, Numerical Optimization, 2nd ed., ch. 3 (line search methods)
Armijo (1966), Minimization of functions having Lipschitz continuous first
  partial derivatives
Boyd & Vandenberghe, Convex Optimization, sec. 9.2-9.3 (descent, backtracking)
```

## API Reference

- [Line Search](../../optimizer/optimizers/LINE_SEARCH.md)
- [Acceptance and Stopping](../../optimizer/optimizers/OPTIMIZER_ACCEPTANCE.md)
- [Optimizer Lifecycle](../../optimizer/optimizers/OPTIMIZER_LIFECYCLE.md)
