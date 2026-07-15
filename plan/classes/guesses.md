# `guesses/`: Initial Control Generation

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Guess functions generate initial controls from `system.control_spec()`.

They should work for any number of control channels.

## Public Functions

```python
opt.zero_guess(system)
opt.constant_guess(system, amplitude=0.1)
opt.sine_guess(system, amplitude=0.05, frequency=1)
opt.sinc_guess(system, amplitude=0.05)
opt.fourier_guess(system, n_terms=8, amplitude=0.03)
opt.random_guess(system, amplitude=0.01)
opt.mix_controls(system, a, b, ratio=0.2)
```

## Amplitude Control

Every guess should support simple amplitude control:

```text
low amplitude
medium amplitude
high amplitude
per-channel amplitude if needed
```

Example:

```python
controls = opt.fourier_guess(sys, n_terms=6, amplitude=0.03)
```

## Shape Rule

All guesses produce:

```text
(n_controls, control_dim)
```

based on `system.control_spec()`.

## Fourier Guess

Fourier guess should support:

```text
n_terms
amplitude
include_dc
seed
per-channel scaling
optional envelope
```

## Mix Guess

Useful for curriculum and restarts:

```python
mixed = opt.mix_controls(sys, best, random, ratio=0.05)
```

