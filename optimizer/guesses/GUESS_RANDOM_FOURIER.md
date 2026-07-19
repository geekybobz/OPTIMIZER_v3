---
title: GUESS_RANDOM_FOURIER
type: method_reference
module: optimizer/guesses
methods:
  - random_fourier_guess
tags:
  - optimizer
  - guesses
  - random
  - fourier
  - multi_start
---

# Random Fourier Guess

`random_fourier_guess` creates smooth random low-frequency controls by sampling
Fourier coefficients and phases, then delegating waveform synthesis to
`fourier_guess`.

## Public Call

```python
controls = opt.guesses.random_fourier_guess(
    system,
    amplitude=0.1,
    modes=5,
    seed=1,
)
```

## Important Arguments

```text
amplitude
modes
frequency_base
coefficient_scale
decay
offset
channels
seed
envelope
endpoint
scale
name
meta
```

Supported coefficient decay:

```text
flat
1/k
1/k2
```

## Returned Metadata

```text
guess
modes
frequency_base
coefficient_scale
decay
seed
```

## Watch Out

```text
seed controls coefficients, phases, and random amplitude ranges
coefficient_scale must be positive
modes must be >= 1
```

## Related Theory

- [Random Fourier Theory](../../Theory/guess/THEORY_RANDOM_FOURIER.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Fourier Guess](GUESS_FOURIER.md)
- [Random Smooth Guess](GUESS_RANDOM_SMOOTH.md)
- [Source](./random.py)
