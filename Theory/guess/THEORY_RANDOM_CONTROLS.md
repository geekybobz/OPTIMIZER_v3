---
title: THEORY_RANDOM_CONTROLS
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_RANDOM.md
tags:
  - optimizer
  - guesses
  - theory
  - random
---

# Random Controls Theory

This note gives the mathematical context for:

- [Random Guess API](../../optimizer/guesses/GUESS_RANDOM.md)

## Random Matrix View

A raw random guess samples a matrix:

```text
U in R^(n_controls x control_dim)
```

Each selected channel row receives independent samples before shared shaping and
scaling.

## Distributions

Supported raw distributions:

```text
uniform
  samples from [-1, 1]

normal
  samples from standard normal

rademacher
  samples from {-1, +1}
```

## Amplitude Range

For random families:

```python
amplitude = (low, high)
```

means each selected channel samples its own amplitude from that interval.

## Roughness

Independent sample-by-sample controls have high roughness and broad frequency
content.

This is useful for stress testing but often less useful as a physically smooth start.

## Practical Use

Use raw random controls when:

```text
you want a robustness stress test
you need a deliberately rough starting point
you are checking amplitude and channel handling
```

Use smoother random families for ordinary multi-start exploration.

## API Reference

- [Random Guess](../../optimizer/guesses/GUESS_RANDOM.md)
- [Random Smooth Guess](../../optimizer/guesses/GUESS_RANDOM_SMOOTH.md)
- [Random Fourier Guess](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)
