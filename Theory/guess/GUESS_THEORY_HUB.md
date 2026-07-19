---
title: GUESS_THEORY_HUB
type: theory_hub
module: Theory/guess
tags:
  - optimizer
  - guesses
  - theory
  - hub
---

# Guess Theory Hub

This folder contains mathematical and research-facing notes for guess generators.

Runtime API documentation lives in:

```text
optimizer/guesses/
```

Theory notes live here so the method docs stay compact.

## Theory Map

```text
THEORY_CONSTANT_RAMP.md
  Constant controls and ramp profiles.

THEORY_HARMONIC_WAVES.md
  Sine and cosine controls on normalized time.

THEORY_LOCALIZED_PULSES.md
  Gaussian and sinc pulse shapes.

THEORY_FOURIER_SERIES.md
  Deterministic finite Fourier bases.

THEORY_RANDOM_CONTROLS.md
  Raw random controls and distributions.

THEORY_SMOOTH_RANDOM.md
  Gaussian-kernel smoothing and correlation length.

THEORY_RANDOM_FOURIER.md
  Random low-frequency Fourier starts.

THEORY_COMPOSITE_RESTARTS.md
  Scaling, mixing, perturbing, and restart strategy.
```

## Reading Rule

Use API docs when coding:

- [Guess Methods](../../optimizer/guesses/GUESS_METHODS.md)
- [Guess Common API](../../optimizer/guesses/GUESS_COMMON_API.md)

Use theory docs when reasoning about:

```text
shape families
frequency content
smoothness
randomness and reproducibility
restart geometry
method choice
```

## Graph Convention

Each theory note links back to the method-family API note.

Each method-family API note links forward to the relevant theory note.

Example:

```text
optimizer/guesses/GUESS_RANDOM_FOURIER.md
  -> Theory/guess/THEORY_RANDOM_FOURIER.md
      -> optimizer/guesses/GUESS_RANDOM_FOURIER.md
```

## Related Notes

- [Guess Documentation Hub](../../optimizer/guesses/GUESSES_HUB.md)
- [Guess Methods](../../optimizer/guesses/GUESS_METHODS.md)
- [Guess Boundary](../../optimizer/guesses/GUESS_BOUNDARY.md)
