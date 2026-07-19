---
title: GUESS_RANDOM_SMOOTH
type: method_reference
module: optimizer/guesses
methods:
  - random_smooth_guess
tags:
  - optimizer
  - guesses
  - random
  - smooth
---

# Random Smooth Guess

`random_smooth_guess` creates random controls and smooths each channel with a
Gaussian correlation kernel.

## Public Call

```python
controls = opt.guesses.random_smooth_guess(
    system,
    amplitude=0.2,
    correlation=0.2,
    seed=4,
)
```

## Important Arguments

```text
amplitude
correlation
distribution
offset
channels
seed
envelope
endpoint
scale
name
meta
```

`correlation <= 1` is interpreted as a fraction of `control_dim`.

Default envelope:

```text
hann
```

## Returned Metadata

```text
guess
distribution
correlation
seed
```

## Watch Out

```text
smoothing reduces roughness but still samples in the full control grid
large correlation can over-smooth and reduce diversity
distribution is uniform or normal
```

## Related Theory

- [Smooth Random Theory](../../Theory/guess/THEORY_SMOOTH_RANDOM.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Random Guess](GUESS_RANDOM.md)
- [Random Fourier Guess](GUESS_RANDOM_FOURIER.md)
- [Source](./random.py)
