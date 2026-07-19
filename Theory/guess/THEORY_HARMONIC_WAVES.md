---
title: THEORY_HARMONIC_WAVES
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_HARMONIC.md
tags:
  - optimizer
  - guesses
  - theory
  - harmonic
  - frequency
---

# Harmonic Wave Theory

This note gives the mathematical context for:

- [Harmonic Guess API](../../optimizer/guesses/GUESS_HARMONIC.md)

## Normalized Time

Harmonic guesses use:

```text
t in [0, 1]
```

The parameter `frequency` means cycles across this normalized interval.

## Sine Controls

```text
u(t) = A * sin(2*pi*f*t + phi)
```

where:

```text
A   amplitude after scale convention
f   frequency in cycles
phi phase in radians
```

## Cosine Controls

```text
u(t) = A * cos(2*pi*f*t + phi)
```

Cosine and sine differ only by phase, but both are useful because initial endpoint
values and symmetry can differ before endpoint policy is applied.

## Frequency Content

A single harmonic gives concentrated frequency content.

This is useful when:

```text
expected controls are smooth
one or two oscillatory modes are physically plausible
you want a deterministic phase/frequency sweep
```

## Envelope Interaction

Applying an envelope multiplies the harmonic:

```text
u_shaped(t) = envelope(t) * raw_wave(t)
```

This reduces endpoint energy or localizes the waveform, but it also broadens
frequency content.

## API Reference

- [Harmonic Guesses](../../optimizer/guesses/GUESS_HARMONIC.md)
- [Fourier Guess](../../optimizer/guesses/GUESS_FOURIER.md)
- [Guess Common API](../../optimizer/guesses/GUESS_COMMON_API.md)
