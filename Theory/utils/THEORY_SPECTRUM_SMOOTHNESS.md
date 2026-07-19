---
title: THEORY_SPECTRUM_SMOOTHNESS
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_SPECTRUM_SMOOTHNESS.md
tags:
  - optimizer
  - utils
  - theory
  - spectrum
  - smoothness
---

# Spectrum and Smoothness Theory

This note gives the mathematical context for:

- [Spectrum and Smoothness API](../../optimizer/utils/UTIL_SPECTRUM_SMOOTHNESS.md)

## Discrete Controls

For one channel:

```text
u = [u_0, u_1, ..., u_{N-1}]
```

the utilities inspect both frequency content and local roughness.

## Real FFT

The real FFT decomposes a real-valued pulse into frequency bins:

```text
u_n -> U_k
```

Amplitude:

```text
A_k = |U_k|
```

Power:

```text
P_k = |U_k|^2
```

Dominant frequency is the bin with maximum power.

## Frequency Grid

For sample spacing `dt`, FFT frequency bins are:

```text
f_k = k / (N dt)
```

for the real FFT nonnegative-frequency range.

If `dt` is wrong or missing, frequency units are only relative.

## High-Frequency Fraction

Given a cutoff `f_c`, high-frequency fraction is:

```text
sum_{f_k >= f_c} P_k
--------------------
sum_all_k P_k
```

This is useful for detecting optimization artifacts or overly rough guesses.

## First Difference Roughness

First differences:

```text
Delta u_n = u_{n+1} - u_n
```

measure jumps between adjacent samples.

Large first-difference norm or max jump means the pulse has sharp steps.

## Second Difference Roughness

Second differences:

```text
Delta^2 u_n = u_{n+2} - 2 u_{n+1} + u_n
```

measure curvature-like roughness.

Large second-difference norm means the pulse changes direction sharply.

## Total Variation

Total variation is:

```text
TV(u) = sum_n |u_{n+1} - u_n|
```

It is a compact roughness measure that grows with oscillation and jump size.

## Illustration

```text
time-domain controls
   |
   | FFT
   v
frequency power

time-domain controls
   |
   | finite differences
   v
jump and roughness summaries
```

## Practical Use

Use these reports to compare:

```text
raw random vs smooth random guesses
Fourier starts with different mode counts
pre-optimization vs post-optimization controls
repair output vs original controls
```

## API Reference

- [Spectrum and Smoothness](../../optimizer/utils/UTIL_SPECTRUM_SMOOTHNESS.md)
- [Fourier Guess](../../optimizer/guesses/GUESS_FOURIER.md)
- [Smooth Random Guess](../../optimizer/guesses/GUESS_RANDOM_SMOOTH.md)
