---
title: GUESS_LIFECYCLE
type: computation_flow
module: optimizer/guesses
tags:
  - optimizer
  - guesses
  - lifecycle
---

# Guess Lifecycle

Most fresh guess functions follow the same path.

## Fresh Guess Flow

```text
target
  -> resolve ControlSpec
  -> build normalized time grid
  -> build raw dimensionless waveform
  -> select channels
  -> apply envelope
  -> scale by amplitude convention
  -> add offset
  -> apply endpoint policy
  -> return Controls
```

The shared final step is `finalize_guess(...)`.

## Composite Flow

Composite helpers start from existing controls:

```text
scale_guess
  existing Controls -> rescale matrix -> Controls

mix_guess
  compatible Controls list -> weighted matrix sum -> Controls

perturb_guess
  existing Controls -> generated perturbation -> matrix sum -> Controls
```

## Metadata

Returned controls include family metadata such as:

```text
guess
kind
modes
frequency_base
distribution
correlation
seed
source
sources
perturbation_kind
```

This helps traces and notebooks remember where a starting pulse came from.

## Validation

Guess construction validates:

```text
ControlSpec shape
finite values
channel names
positive widths or frequency scales where required
compatible specs for mixing
```

## Related Notes

- [Guess Common API](GUESS_COMMON_API.md)
- [Guess Contract](GUESS_CONTRACT.md)
- [Guess Methods](GUESS_METHODS.md)
- [Base Source](./base.py)
