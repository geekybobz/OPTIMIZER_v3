---
title: GUESS_HARMONIC
type: method_reference
module: optimizer/guesses
methods:
  - sine_guess
  - cosine_guess
tags:
  - optimizer
  - guesses
  - harmonic
  - smooth
---

# Harmonic Guesses

Harmonic guesses create smooth sine or cosine controls on the normalized time grid.

## Public Calls

```python
controls = opt.guesses.sine_guess(system, amplitude=0.2, frequency=2, phase=0.0)
```

```python
controls = opt.guesses.cosine_guess(system, amplitude=0.2, frequency=1)
```

## Methods

```text
sine_guess
  sin(2*pi*frequency*t + phase)

cosine_guess
  cos(2*pi*frequency*t + phase)
```

Frequency is measured in cycles over `t in [0, 1]`.

## Important Arguments

```text
amplitude
frequency
phase
offset
channels
envelope
endpoint
scale
name
meta
```

## Returned Metadata

```text
{"guess": "sine", ...}
{"guess": "cosine", ...}
```

## Watch Out

```text
phase is in radians
frequency can be channel-wise
scale controls whether amplitude means max_abs, l2, energy, or raw multiplier
```

## Related Theory

- [Harmonic Wave Theory](../../Theory/guess/THEORY_HARMONIC_WAVES.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Fourier Guess](GUESS_FOURIER.md)
- [Source](./harmonic.py)
