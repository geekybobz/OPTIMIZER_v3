---
title: THEORY_CMA_ES
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

Rejected generations also adapt: the radius shrinks by a fixed factor so that
repeated failures stop resampling the same large cloud:

```text
sigma_next = max(0.9 * sigma, min_sigma)   when no candidate improves
```

## Worked Example

One diagonal-variant generation in one dimension, minimizing
`J(u) = (u - 0.3)^2` with `mean = 0`, `sigma = 0.5`, `population_size = 4`,
`elite_fraction = 0.5`:

```text
candidates: 0.00 (the mean itself), 0.62, -0.41, 0.35
J values:   0.090, 0.102, 0.504, 0.003

ranking: 0.35, 0.00, 0.62, -0.41     elite_count = 2 -> elites {0.35, 0.00}

log weights for two elites: (0.804, 0.196)
next mean = 0.804 * 0.35 + 0.196 * 0.00 = 0.281
```

The best candidate improves the current value (0.003 < 0.090), so the
generation is accepted and sigma blends toward the elite spread:

```text
elite_spread = 0.042
sigma_next = sqrt(0.8 * 0.25 + 0.2 * 0.042) = 0.457
```

The mean moved most of the way to the optimum 0.3 in one generation, at the
cost of four evaluations.

## Convergence Notes

```text
selection is rank-based, so the search is invariant to monotone
  transformations of the metric
the diagonal variant adapts per-coordinate scale but stays axis-aligned; it
  cannot learn rotated correlations the way full CMA-ES does
cost is population_size evaluations per generation; the default population
  grows only logarithmically with dimension
```

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

## References

```text
Hansen & Ostermeier (2001), Completely derandomized self-adaptation in
  evolution strategies
Hansen (2016), The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772
```

## API Reference

- [CMA-ES](../../optimizer/optimizers/CMA_ES.md)
- [Optimizer Lifecycle](../../optimizer/optimizers/OPTIMIZER_LIFECYCLE.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
