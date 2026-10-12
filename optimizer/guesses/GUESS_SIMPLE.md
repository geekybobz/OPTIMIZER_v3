---
role: topic
parent: GUESSES_HUB.md
summary: "Constant and ramp controls."
read_when: "Read when you want an explicit deterministic start."
status: active
reviewed: 2026-10-12
---
# Simple Guesses

Simple guesses create explicit deterministic controls: constants and ramps.

## Public Calls

```python
controls = opt.guesses.constant_guess(system, value={"ux": 0.2, "uy": 0.0})
```

```python
controls = opt.guesses.ramp_guess(system, start=0.0, stop={"ux": 0.3}, kind="smoothstep")
```

## Methods

```text
constant_guess
  selected-channel constant controls

ramp_guess
  monotone transition between start and stop values
```

Ramp kinds:

```text
linear
quadratic
smoothstep
```

## Important Arguments

```text
value
start
stop
channels
kind
endpoint
name
meta
```

## Returned Metadata

```text
constant_guess: {"guess": "constant", ...}
ramp_guess: {"guess": "ramp", "kind": kind, ...}
```

## Watch Out

```text
start and stop are physical channel values, not normalized waveform amplitudes
endpoint policy is applied after ramp construction
unselected channels keep the neutral baseline
```

## Related Theory

- [Constant and Ramp Theory](../../Theory/guess/THEORY_CONSTANT_RAMP.md)

## Related Notes

- [Guess Methods](GUESS_METHODS.md)
- [Guess Common API](GUESS_COMMON_API.md)
- [Source](./simple.py)

---

[⌂ Home](GUESSES_HUB.md)
