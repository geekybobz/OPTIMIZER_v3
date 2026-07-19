---
title: CMA-ES Theory
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/CMA_ES.md
tags:
  - optimizer
  - theory
  - cma_es
  - derivative_free
  - population
---

# CMA-ES Theory

This note gives the mathematical context for:

- [CMA-ES API](../../optimizer/optimizers/CMA_ES.md)

## Population Search

CMA-ES-style methods sample candidate controls around a mean:

```text
x_i = mean + noise_i
```

Candidates are evaluated through `system.evaluate(controls)` and ranked by a scalar
metric.

## Isotropic Sampling

The isotropic variant uses one global sampling radius:

```text
x_i = mean + sigma * normal(0, I)
```

All flattened control coordinates share the same scale.

## Diagonal Sampling

The diagonal variant uses one radius per flattened coordinate:

```text
x_i = mean + diagonal_sigma * normal(0, I)
```

This adapts coordinate-wise spread without storing a full covariance matrix.

## Elite Mean

After ranking candidates, the best fraction is retained:

```text
elite_count = ceil(elite_fraction * population_size)
```

The next candidate mean is built from a weighted elite average.

## Sigma Adaptation

The implementation estimates elite spread and blends it into the current diagonal
variance:

```text
sigma_next^2 = (1 - lr) * sigma^2 + lr * elite_spread
```

The radius is bounded below by `min_sigma`.

## Compact Implementation Boundary

This package implementation is deliberately compact.

It does not include:

```text
full covariance matrix adaptation
evolution paths
all standard CMA-ES control parameters
research-grade restart schemes
```

It is intended as a practical derivative-free search option inside OPTIMIZER v3.

## Practical Use

Use CMA-ES when:

```text
gradients are unavailable or unreliable
the initial guess is poor
rough global search is needed before local methods
the control parameterization is low or moderate dimension
```

Watch for:

```text
many evaluations per generation
high-dimensional controls becoming expensive
sigma requiring tuning
local polish usually needing gradient methods later
```

## API Reference

- [CMA-ES](../../optimizer/optimizers/CMA_ES.md)
- [Optimizer Lifecycle](../../optimizer/optimizers/LIFECYCLE.md)
- [Optimizer Methods](../../optimizer/optimizers/METHODS.md)
