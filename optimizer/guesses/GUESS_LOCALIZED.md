---
title: GUESS_LOCALIZED
type: method_reference
module: optimizer/guesses
methods:
  - gaussian_guess
  - sinc_guess
tags:
  - optimizer
  - guesses
  - localized
  - pulses
---

# Localized Guesses

Localized guesses create pulses concentrated around a center in normalized time.

## Public Calls

```python
controls = opt.guesses.gaussian_guess(system, amplitude=0.2, center=0.5, width=0.15)
```

```python
controls = opt.guesses.sinc_guess(system, amplitude=0.25, center=0.5, width=5.0)
```

## Methods

```text
gaussian_guess
  exp(-0.5 * ((t - center) / width)^2)

sinc_guess
  sinc(width * (t - center))
```

## Important Arguments

```text
amplitude
center
width
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
{"guess": "gaussian", ...}
{"guess": "sinc", ...}
```

## Watch Out

```text
width must be positive
Gaussian width is a normalized time width
sinc width controls side-lobe density
```

## Related Theory

- [Localized Pulse Theory](../../Theory/guess/THEORY_LOCALIZED_PULSES.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Guess Common API](GUESS_COMMON_API.md)
- [Source](./harmonic.py)
