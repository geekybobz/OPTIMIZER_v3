---
role: topic
parent: GUESS_THEORY_HUB.md
summary: "Mathematics of scaling, mixing, and perturbing controls for restarts."
read_when: "Read when combining or perturbing earlier controls to restart a run."
status: active
reviewed: 2026-10-12
---
# Composite Restart Theory

This note gives the mathematical context for:

- [Composite Guess API](../../optimizer/guesses/GUESS_COMPOSITE.md)

## Scaling

Scaling maps an existing control shape to a new amplitude convention:

```text
u_scaled = target_amplitude * u / scale_norm(u)
```

The shape is preserved, but the magnitude changes.

## Mixing

Mixing forms a weighted linear combination:

```text
u_mix = sum_i w_i * u_i
```

All controls must share the same control layout.

Weights are not required to sum to 1, but normalized weights are easier to reason
about.

## Perturbing

Perturbation adds generated noise or structure to an existing control:

```text
u_trial = u_base + delta
```

where `delta` comes from one of the supported generator families.

## Restart Strategy

Composite helpers are useful after a run finds a promising control:

```text
best controls
  -> lower amplitude
  -> mix with another useful pulse
  -> perturb locally
  -> rerun optimizer
```

This explores nearby basins without discarding a useful discovered shape.

## Practical Use

Use composite helpers when:

```text
you have a previous good pulse
you want local restarts
you want to average or interpolate starts
you want to test amplitude sensitivity
```

## API Reference

- [Composite Guess Helpers](../../optimizer/guesses/GUESS_COMPOSITE.md)
- [Random Fourier Guess](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)
- [Guess Methods](../../optimizer/guesses/GUESS_METHODS.md)

---

[⌂ Home](GUESS_THEORY_HUB.md)
