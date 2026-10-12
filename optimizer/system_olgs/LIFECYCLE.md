---
role: topic
parent: SYSTEM_OLGS_HUB.md
summary: "How an OLGS system goes from controls to metrics and gradients, with one propagation per iteration."
read_when: "Read when implementing evaluate and gradient for an OLGS system."
status: active
reviewed: 2026-10-12
---
# OLGS Computation Lifecycle

This file defines how an OLGS system should move from controls to metrics and
gradients.

## Main Flow

```text
Controls
  -> forward_prop
  -> metrics/evaluate
  -> back_prop
  -> gradient
```

The common analytical optimization path is:

```python
metrics = system.evaluate(controls)
grad = system.gradient(controls)
```

Evaluator code may use a combined call if available:

```text
value_and_gradient(controls)
```

but this is optional.

## One propagation per iteration

Each optimizer iteration evaluates a trial and, once it is accepted, asks for the
gradient at the same controls. Keep the last forward propagation, keyed by the
control values, and reuse it in `back_prop` and `gradient`:

```text
evaluate(trial)    -> forward_prop runs and is kept
gradient(trial)    -> forward_prop reused, back_prop runs, gradient assembled
```

An iteration then costs one forward and one backward propagation. Without the
cache the forward propagation runs again inside `gradient`. The scaffold from
`optimizer-system` already has this structure: `forward_prop` with the cache,
`_propagate_forward`, `back_prop`, `evaluate`, `gradient`, and `cache_reset`.
`with_secondary` returns a new system, so a parameter change never reuses a
stale propagation. A system that provides `value_and_gradient` instead is
evaluated through it under standard acceptance, with the same effect.

## `forward_prop`

Responsible for physical forward evolution.

Can compute:

```text
Hamiltonians
propagators
state trajectories
terminal overlaps
objective ingredients
Bloch coordinates
projection values
normalization checks
```

## `evaluate`

Responsible for fast optimizer metrics.

Should compute only what optimizer iterations need.

Must return:

```python
{"J": finite_scalar}
```

May also return:

```text
fidelity
infidelity
energy
constraint norms
objective component terms
```

## `back_prop`

Responsible for adjoint or costate propagation.

Allowed behavior:

```text
call forward_prop if the matching forward state is missing
or raise a clear error telling the user to run forward_prop first
```

## `gradient`

Responsible for analytical `dJ/du`.

Should usually:

```text
ensure forward quantities exist
ensure backward quantities exist
build gradient in control row order
return Controls
```

## `simulate`

`simulate` is optional and richer than `evaluate`.

It may run:

```text
full trajectory export
ensemble checks
finite perturbation robustness checks
plotting data
extra diagnostics
```

Optimizers should not depend on expensive simulation behavior.

## `plot_data`

`plot_data(controls, kind=...)` is an optional explicit data-product hook. It returns
renderer-independent named layers, coordinates, annotations, and metadata. It may
perform a full trajectory calculation, so optimizers and passive tracking never call
it automatically.

For a Bloch view, the qubit system owns propagation and state-to-Bloch conversion;
`optimizer.visualization.bloch` owns only validation and rendering.

## `metrics`

`metrics()` means latest valid metrics.

It should be clear whether metrics are available. If no valid computation has run,
raise a clear error.

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [API Contract](CONTRACT.md)
- [System Configuration](PARAMS.md)
- [Derivative Hooks](DERIVATIVES.md)
- [Logging Boundary](LOGGING_BOUNDARY.md)

---

[⌂ Home](SYSTEM_OLGS_HUB.md)
