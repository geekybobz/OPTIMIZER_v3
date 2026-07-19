---
title: OPTIMIZER_THEORY_HUB
type: theory_hub
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
THEORY_LINE_SEARCH.md
  Gradient descent, normalized steps, backtracking, Armijo.

THEORY_MOMENTUM.md
  Heavy-ball momentum, Nesterov lookahead, restart behavior.

THEORY_ADAM_FAMILY.md
  Adam, AMSGrad, AdamW, RAdam, AdaBelief.

THEORY_ADAPTIVE_SCALING.md
  AdaGrad and RMSProp squared-gradient scaling.

THEORY_NONLINEAR_CG.md
  Nonlinear conjugate-gradient beta formulas and restart logic.

THEORY_LBFGS.md
  Limited-memory BFGS curvature pairs and two-loop recursion.

THEORY_CMA_ES.md
  Population search, elite selection, diagonal/isotropic covariance adaptation.
```

## Reading Rule

Use API docs when coding:

- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
- [Optimizer Contract](../../optimizer/optimizers/OPTIMIZER_CONTRACT.md)

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
  -> Theory/optimizers/THEORY_ADAM_FAMILY.md
      -> optimizer/optimizers/ADAM.md
```

## Related Notes

- [Optimizer Documentation Hub](../../optimizer/optimizers/OPTIMIZERS_HUB.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
- [Optimizer Boundary](../../optimizer/optimizers/OPTIMIZER_BOUNDARY.md)
