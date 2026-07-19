---
title: UTIL_SPECTRUM_SMOOTHNESS
type: category_reference
module: optimizer/utils
source: optimizer/utils/spectrum.py
methods:
  - control_spectrum
  - smoothness_report
tags:
  - optimizer
  - utils
  - spectrum
  - smoothness
  - diagnostics
---

# Spectrum and Smoothness

Spectrum and smoothness utilities measure pulse shape without changing controls.

## Public Calls

```python
spectrum = opt.utils.control_spectrum(controls, high_frequency_cutoff=5.0)
smoothness = opt.utils.smoothness_report(controls)
```

## `control_spectrum`

Computes real FFT summaries by channel.

Returns:

```text
frequencies
dominant_frequency
total_power
high_frequency_fraction
amplitude
power
```

`dt` defaults to `controls.spec.dt` when available, otherwise `1.0`.

## `smoothness_report`

Computes finite-difference roughness summaries:

```text
first_difference_norm
second_difference_norm
total_variation
max_jump
global_first_difference_norm
global_second_difference_norm
global_total_variation
```

## Best For

```text
checking high-frequency artifacts
comparing raw random and smooth random guesses
spotting large jumps
reviewing pulse roughness after optimization
```

## Watch Out

```text
these tools only measure controls
they do not smooth, filter, clip, or project
frequency units depend on dt
roughness is measured along the time/control dimension
```

## Related Theory

- [Spectrum and Smoothness Theory](../../Theory/utils/THEORY_SPECTRUM_SMOOTHNESS.md)

## Related Notes

- [Diagnostics](UTIL_DIAGNOSTICS.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Fourier Guess](../guesses/GUESS_FOURIER.md)
- [Smooth Random Guess](../guesses/GUESS_RANDOM_SMOOTH.md)
- [Source](./spectrum.py)
