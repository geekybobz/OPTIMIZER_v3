---
role: topic
parent: GUESSES_HUB.md
summary: "fourier_guess: deterministic low-frequency sine-series controls."
read_when: "Read when you want a smooth deterministic start."
status: active
reviewed: 2026-10-12
---
# Fourier Guess

`fourier_guess` creates deterministic low-frequency sine-series controls.

## Public Call

```python
controls = opt.guesses.fourier_guess(
    system,
    amplitude=0.2,
    modes=4,
    decay="1/k2",
)
```

## Template

```text
sum_k coefficients[k] * sin(2*pi*frequency_base*k*t + phase[k])
```

## Important Arguments

```text
amplitude
modes
frequency_base
coefficients
phases
decay
offset
channels
envelope
endpoint
scale
name
meta
```

Supported decay templates when coefficients are omitted:

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
decay
```

## Watch Out

```text
modes must be >= 1
frequency_base must be positive
coefficients and phases must match mode shape
```

## Related Theory

- [Fourier Series Theory](../../Theory/guess/THEORY_FOURIER_SERIES.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Harmonic Guesses](GUESS_HARMONIC.md)
- [Random Fourier Guess](GUESS_RANDOM_FOURIER.md)
- [Source](./harmonic.py)

---

[⌂ Home](GUESSES_HUB.md)
