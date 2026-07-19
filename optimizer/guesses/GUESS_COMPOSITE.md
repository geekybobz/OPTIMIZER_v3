---
title: GUESS_COMPOSITE
type: method_reference
module: optimizer/guesses
methods:
  - scale_guess
  - mix_guess
  - perturb_guess
tags:
  - optimizer
  - guesses
  - composite
  - restart
---

# Composite Guess Helpers

Composite helpers build new starting controls from existing `Controls`.

## Public Calls

```python
scaled = opt.guesses.scale_guess(controls, amplitude=0.1)
```

```python
mixed = opt.guesses.mix_guess([base, perturb], weights=[0.8, 0.2])
```

```python
trial = opt.guesses.perturb_guess(base, amplitude=0.01, kind="random_fourier", seed=2)
```

## Methods

```text
scale_guess
  rescale existing Controls through common amplitude semantics

mix_guess
  weighted matrix sum of compatible Controls

perturb_guess
  existing Controls plus generated perturbation
```

Perturbation kinds:

```text
random_fourier
random_smooth
random
fourier
```

## Important Arguments

```text
controls
guesses
weights
amplitude
offset
scale
channels
endpoint
kind
seed
kwargs
name
meta
```

## Returned Metadata

```text
scale_guess: {"guess": "scale", "source": ...}
mix_guess: {"guess": "mix", "sources": [...]}
perturb_guess: {"guess": "perturb", "source": ..., "perturbation_kind": ...}
```

## Watch Out

```text
mix_guess inputs must share keys and control_dim
scale_guess does not mutate the original controls
perturb_guess forwards kwargs to the selected perturbation generator
```

## Related Theory

- [Composite Restart Theory](../../Theory/guess/THEORY_COMPOSITE_RESTARTS.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Random Fourier Guess](GUESS_RANDOM_FOURIER.md)
- [Source](./composite.py)
