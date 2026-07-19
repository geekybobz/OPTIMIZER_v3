---
title: GUESS_RANDOM
type: method_reference
module: optimizer/guesses
methods:
  - random_guess
tags:
  - optimizer
  - guesses
  - random
---

# Random Guess

`random_guess` creates raw random controls before optional envelope, endpoint, and
scale processing.

## Public Call

```python
controls = opt.guesses.random_guess(
    system,
    amplitude=0.2,
    distribution="normal",
    seed=4,
)
```

## Distributions

```text
uniform
normal
rademacher
```

## Important Arguments

```text
amplitude
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

For random families, a two-number amplitude can mean a sampled per-channel amplitude
range:

```python
amplitude = (0.05, 0.2)
```

## Returned Metadata

```text
guess
distribution
seed
```

## Watch Out

```text
raw random controls can have high-frequency roughness
use seed for reproducibility
use random_smooth_guess or random_fourier_guess for smoother starts
```

## Related Theory

- [Random Controls Theory](../../Theory/guess/THEORY_RANDOM_CONTROLS.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Random Smooth Guess](GUESS_RANDOM_SMOOTH.md)
- [Random Fourier Guess](GUESS_RANDOM_FOURIER.md)
- [Source](./random.py)
