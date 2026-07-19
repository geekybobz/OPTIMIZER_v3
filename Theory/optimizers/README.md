---
title: optimizer theory
type: theory_index
module: Theory/optimizers
tags:
  - optimizer
  - theory
  - research
---

# Optimizer Theory Notes

This folder contains longer mathematical and research-facing notes for optimizer
methods.

Runtime API documentation lives in:

```text
optimizer/optimizers/
```

Theory notes live here so derivations, equations, illustrations, and references do
not make the code-facing docs too large.

## Theory Map

```text
gradient_descent_and_line_search.md
  Gradient descent, normalized steps, backtracking, Armijo.

momentum.md
  Heavy-ball momentum, Nesterov lookahead, restart behavior.

adam_family.md
  Adam, AMSGrad, AdamW, RAdam, AdaBelief.

adaptive_scaling.md
  AdaGrad and RMSProp squared-gradient scaling.

nonlinear_cg.md
  Nonlinear conjugate-gradient beta formulas and restart logic.

lbfgs.md
  Limited-memory BFGS curvature pairs and two-loop recursion.

cma_es.md
  Population search, elite selection, diagonal/isotropic covariance adaptation.
```

## Reading Rule

Use API docs when coding:

- [Optimizer Methods](../../optimizer/optimizers/METHODS.md)
- [Optimizer Contract](../../optimizer/optimizers/CONTRACT.md)

Use theory notes when checking mathematical meaning:

```text
update equations
geometric intuition
assumptions
limitations
method choice
research references
```

## Graph Convention

Each theory note links back to the method API reference.

Each method API reference links forward to the relevant theory note.

This gives two-way traversal:

```text
optimizer/optimizers/ADAM.md
  -> Theory/optimizers/adam_family.md
      -> optimizer/optimizers/ADAM.md
```

## Related Notes

- [Optimizer Documentation Hub](../../optimizer/optimizers/README.md)
- [Optimizer Methods](../../optimizer/optimizers/METHODS.md)
- [Optimizer Boundary](../../optimizer/optimizers/BOUNDARY.md)
