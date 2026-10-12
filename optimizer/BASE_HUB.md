---
role: front-door
parent: ../README.md
summary: "Shared data types and top-level calls of the base catalog group: controls, results, state, and root calls."
read_when: "Read when working with Controls, results, run state, or a top-level opt call."
status: active
reviewed: 2026-10-12
---
# Base API Hub

The `base` catalog group: the shared data types and the top-level calls that
every other group builds on. They live in the root modules of `optimizer/`.

## In brief

`Controls` carry the numbers, `OptimizerResult` carries the outcome, and
`WarmStartState` carries a run into the next method. Every optimizer, guess,
utility, and checkpoint uses the same `Controls` matrix.

## Details

## Control layout

```text
ControlSpec(keys, control_dim, dtype=float, dt=None, meta={})
```

| Field | Meaning |
|---|---|
| `keys` | Unique, non-empty channel names; their order is the row order of the matrix |
| `control_dim` | Samples per channel; at least 1 |
| `dtype` | NumPy dtype of the values |
| `dt` | Optional time step; finite and positive when given |
| `meta` | Free-form metadata |

`ControlSpec` is frozen. It also offers `n_controls`, `shape`, `size`,
`channel_index(key)`, and `to_dict()`.

`Controls` hold finite values as a `(n_controls, control_dim)` matrix in `u`,
with `spec`, `name`, and `meta`. Build them with:

```text
Controls(spec, values, *, name=None, meta=None, copy=True)
Controls.zeros(spec)
Controls.constant(spec, value)       value: scalar, one per channel, or {key: value}
Controls.from_matrix(spec, matrix)
Controls.from_dict(spec, {key: values})
Controls.from_flat(spec, vector)
```

Read and change them with `channel(key)`, `set_channel(key, values)`,
`as_matrix()`, `as_dict()`, `flatten()`, `copy()`, `norm()`,
`channel_norms()`, and `max_abs()`.

## Results and state

| Type | Holds |
|---|---|
| `Evaluation` | `controls`, `metrics`, `system_state`, `objective_metric`, and the derived `objective_value` |
| `OptimizerResult` | `controls`, `metrics`, `stop_reason`, `iterations`, `optimizer`, `objective_metric`, `state`, `system_params`, `diagnostics` |
| `RunState` | The live run: current and best controls and metrics, iteration counters, step size, optimizer state, and stop reason |
| `WarmStartState` | A handoff: controls, metrics, step size, optimizer state, and the source and target optimizer |

`OptimizerResult.warmstart()` returns the `WarmStartState` for the next method.

## Top-level calls

| Call | Purpose |
|---|---|
| `opt.evaluate(system, controls)` | Evaluate the system and wrap its metrics in an `Evaluation` |
| `opt.gradient(system, controls)` | Call and validate the analytical gradient |
| `opt.value_and_gradient(system, controls)` | Use the system's combined objective and gradient, when it offers one |
| `opt.context(system, **defaults)` | Bind a system once for notebook or curriculum use |
| `opt.system_info(system)` | Report declared capabilities and metadata without evaluating controls |
| `opt.plot_data(system, controls, kind=...)` | Request one system-owned visualization data product |
| `opt.context_info(system=None)` | Report the registered optimizers, the OLGS contract, and native numerical capabilities |
| `opt.continuation(system, controls, stages=...)` | Run explicit parameter stages with safe control transfer |
| `opt.system_family(system)` | Return `"olgs"` or `"olss"` from the system's `system_spec()`; a missing family means OLGS |
| `opt.to_plain(value)` | Convert a result, report, or other library value to strict JSON data (see [Transactions](TRANSACTIONS.md#serialization)) |
| `opt.methods()` | Status of every public root function, built from the catalog declarations |

The first three accept `use_cache=True`, which reuses the cached evaluation for
identical controls.

## Module map

| Module | Owns |
|---|---|
| [controls.py](controls.py) | `ControlSpec` and `Controls` |
| [result.py](result.py) | `Evaluation` and `OptimizerResult` |
| [state.py](state.py) | `RunState` and `WarmStartState` |
| [library.py](library.py) | The root facade and the bound-system context |
| [continuation.py](continuation.py) | Parameter-stage continuation |
| [catalog.py](catalog.py) | One declaration per public export; the exports, `opt.list`, `opt.info`, `opt.search`, `opt.path`, and `opt.methods()` are built from it |
| [introspection.py](introspection.py) | Runtime self-description of registered APIs and signatures |
| [transactions.py](transactions.py) | Family-aware request and response envelopes for OLGS and OLSS |
| [serialization.py](serialization.py) | `to_plain`, the one JSON policy for results, reports, and responses |
| [families.py](families.py) | `system_family`, the one rule for telling OLGS and OLSS apart |

OLSS uses `ShootingSpec`, `ShootingVariables`, `ShootingEvaluation`, and
`ShootingResult` rather than `ControlSpec`, `Controls`, and `OptimizerResult`.
Their family contracts remain separate; the transaction envelope, runtime
discovery, serialization, family detection, and the
[stop-reason registry](core/CORE_STOPPING.md) are shared.

## Connections

- Related: [the V4 entry](../README.md)
- Related: [Core Hub](core/CORE_HUB.md), [Transactions](TRANSACTIONS.md), [Optimizer State and Warmstart](optimizers/OPTIMIZER_STATE_WARMSTART.md), [Command history](history/HISTORY.md)

---

[⌂ Home](../README.md)
