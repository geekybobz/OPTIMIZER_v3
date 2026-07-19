---
title: GUESS_METHODS
type: method_index
module: optimizer/guesses
tags:
  - optimizer
  - guesses
  - methods
---

# Guess Methods

This is the compact lookup table for implemented guess families.

## Method Table

| Method | Source | Family Doc | Best For |
|---|---|---|---|
| `constant_guess` | [simple.py](simple.py) | [GUESS_SIMPLE](GUESS_SIMPLE.md) | fixed channel values, simple offsets |
| `ramp_guess` | [simple.py](simple.py) | [GUESS_SIMPLE](GUESS_SIMPLE.md) | slow turn-on controls, endpoint tests |
| `sine_guess` | [harmonic.py](harmonic.py) | [GUESS_HARMONIC](GUESS_HARMONIC.md) | single-frequency smooth starts |
| `cosine_guess` | [harmonic.py](harmonic.py) | [GUESS_HARMONIC](GUESS_HARMONIC.md) | phase/frequency experiments |
| `gaussian_guess` | [harmonic.py](harmonic.py) | [GUESS_LOCALIZED](GUESS_LOCALIZED.md) | localized smooth pulses |
| `sinc_guess` | [harmonic.py](harmonic.py) | [GUESS_LOCALIZED](GUESS_LOCALIZED.md) | localized pulses with side lobes |
| `fourier_guess` | [harmonic.py](harmonic.py) | [GUESS_FOURIER](GUESS_FOURIER.md) | deterministic low-frequency bases |
| `random_guess` | [random.py](random.py) | [GUESS_RANDOM](GUESS_RANDOM.md) | stress tests, rough random starts |
| `random_smooth_guess` | [random.py](random.py) | [GUESS_RANDOM_SMOOTH](GUESS_RANDOM_SMOOTH.md) | smooth random multi-starts |
| `random_fourier_guess` | [random.py](random.py) | [GUESS_RANDOM_FOURIER](GUESS_RANDOM_FOURIER.md) | default smooth random multi-starts |
| `scale_guess` | [composite.py](composite.py) | [GUESS_COMPOSITE](GUESS_COMPOSITE.md) | rescaling previous controls |
| `mix_guess` | [composite.py](composite.py) | [GUESS_COMPOSITE](GUESS_COMPOSITE.md) | combining compatible starts |
| `perturb_guess` | [composite.py](composite.py) | [GUESS_COMPOSITE](GUESS_COMPOSITE.md) | local restarts around known controls |

## Suggested Workflow

Typical research flow:

```text
pick a deterministic or random guess
  -> inspect smoothness/spectrum if needed
  -> run optimizer
  -> rescale, mix, or perturb useful controls
  -> repeat multi-start or polish
```

## Method References

- [Simple Guesses](GUESS_SIMPLE.md)
- [Harmonic Guesses](GUESS_HARMONIC.md)
- [Localized Guesses](GUESS_LOCALIZED.md)
- [Fourier Guess](GUESS_FOURIER.md)
- [Random Guess](GUESS_RANDOM.md)
- [Random Smooth Guess](GUESS_RANDOM_SMOOTH.md)
- [Random Fourier Guess](GUESS_RANDOM_FOURIER.md)
- [Composite Guess Helpers](GUESS_COMPOSITE.md)

## Related Notes

- [Guess Common API](GUESS_COMMON_API.md)
- [Guess Lifecycle](GUESS_LIFECYCLE.md)
- [Guess Theory Hub](../../Theory/guess/GUESS_THEORY_HUB.md)
