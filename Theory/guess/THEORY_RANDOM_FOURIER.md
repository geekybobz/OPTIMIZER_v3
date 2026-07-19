---
title: THEORY_RANDOM_FOURIER
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_RANDOM_FOURIER.md
tags:
  - optimizer
  - guesses
  - theory
  - random
  - fourier
---

# Random Fourier Theory

This note gives the mathematical context for:

- [Random Fourier Guess API](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)

## Random Basis Coefficients

Random Fourier controls sample coefficients and phases:

```text
a_k ~ Normal(0, coefficient_scale)
phi_k ~ Uniform(0, 2*pi)
```

Then they synthesize a finite Fourier series:

```text
u(t) = sum_k a_k * decay_k * sin(2*pi*frequency_base*k*t + phi_k)
```

## Smoothness Bias

Decay controls high-mode damping:

```text
flat
1/k
1/k2
```

Stronger decay gives smoother starts and less high-frequency content.

## Reproducibility

The seed controls:

```text
coefficients
phases
random amplitude ranges
```

Use a fixed seed when comparing optimizer behavior across method settings.

## Practical Use

Random Fourier is a strong default for multi-start search because it gives:

```text
random diversity
smooth waveforms
low-frequency bias
compact parameter intuition
```

## API Reference

- [Random Fourier Guess](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)
- [Fourier Guess](../../optimizer/guesses/GUESS_FOURIER.md)
- [Smooth Random Guess](../../optimizer/guesses/GUESS_RANDOM_SMOOTH.md)
