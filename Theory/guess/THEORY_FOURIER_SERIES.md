---
title: THEORY_FOURIER_SERIES
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_FOURIER.md
tags:
  - optimizer
  - guesses
  - theory
  - fourier
  - basis
---

# Fourier Series Theory

This note gives the mathematical context for:

- [Fourier Guess API](../../optimizer/guesses/GUESS_FOURIER.md)

## Finite Sine Series

The deterministic Fourier guess uses a finite sine series:

```text
u(t) = sum_{k=1}^{K} a_k * sin(2*pi*frequency_base*k*t + phi_k)
```

where:

```text
K = modes
a_k = coefficients
phi_k = phases
```

## Low-Frequency Bias

Coefficient decay controls the amount of high-frequency content:

```text
flat
  equal default mode weights

1/k
  moderate high-mode damping

1/k2
  stronger high-mode damping
```

This gives a smooth basis-style family without requiring a separate parameterized
control object.

## Channel-Wise Coefficients

Coefficients can be:

```text
shared across channels
one row per channel
mapping from channel name to mode vector
```

This lets related channels share structure or vary independently.

## Practical Use

Use deterministic Fourier guesses when:

```text
you want smooth low-frequency controls
you want repeatable basis experiments
you want a shaped start before Adam or line search
```

## API Reference

- [Fourier Guess](../../optimizer/guesses/GUESS_FOURIER.md)
- [Harmonic Guesses](../../optimizer/guesses/GUESS_HARMONIC.md)
- [Random Fourier Guess](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)
