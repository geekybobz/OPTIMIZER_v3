---
role: deep
parent: CORE_HUB.md
summary: run_chunk arguments, start-state precedence and compact numerical results.
read_when: Read when calling run_chunk directly or interpreting native chunk state.
status: active
reviewed: 2026-10-12
---

# Core engine

`run_chunk(system, controls, step=..., optimizer_name=..., maxiter=...)` owns
validation, evaluation, bounds, acceptance and stopping. It creates no log,
trace, journal or checkpoint file. Numerical state and constant-size diagnostic
aggregates remain in `OptimizerResult`; frontend history and durable checkpoints
belong to the [interaction layer](../history/HISTORY.md).

## Start-state precedence

Controls resolve from `state > warmstart > controls`. Step size resolves from
an explicit argument, then state, then warmstart. Compatible method memory
comes from state or warmstart; otherwise it starts empty. With none of these
starting values, the call raises `ValueError`.

State retains global iteration and best-so-far controls/metrics across chunks.
Warmstart carries physical controls, metrics, step size and compatible private
memory; it starts a fresh iteration count and best-so-far calculation.

## Configuration

| Concern | Arguments and behavior |
|---|---|
| Method | `step`, `optimizer_name`, `maxiter`, `state`, `warmstart`, `step_size` |
| Stopping | `stopping` or target/stall/gradient/curvature options; a supplied StoppingConfig overrides loose options, including maxiter |
| Acceptance | `accept` or `accept_metric`, `accept_mode`, `accept_tolerance`; metric/mode also select best-so-far |
| Numerical guards | `enforce_bounds`, `failed_trial_retries`, `failed_trial_shrink` |
| Evaluation | `use_cache`, `gradient_source`, `prefer_value_and_gradient` |
| Numerical context | `system_params` and optional `stage` supplied to StepContext |

Use [the generated reference](../../API_REFERENCE.md) for the complete signature.
`system_params` defaults to a best-effort copy of system parameters; no particular
params class is required. `stage` is a proposal context label, not a log writer.

## Acceptance

A custom accept callable overrides the default scalar metric decision; best-so-far
still uses the selected acceptance metric and direction. See [guards](CORE_ACCEPTANCE_GUARDS.md).

## Return and diagnostics

Every normal exit returns the current controls, metrics, stop reason, attempted
iterations, native state and optimizer identity. `diagnostics` retains start/end
objective, extrema, attempts, accepted/rejected counts, direction reversals,
last failure and final evaluator counters in constant space. It has no sequence
of iteration records. [Run diagnosis](../utils/UTIL_DIAGNOSTICS.md) uses these
aggregates and does not fabricate gradient or step series.

Source: [engine.py](engine.py).

## Progress events

Each iteration also sends a small event to an optional display reporter: the
method, iteration count, budget, objective metric, value and acceptance. A
frontend installs the reporter for one call (`optimizer.core.progress._reporting`).
Without one, nothing happens. Events are not stored, written or recorded in
history, and a failing reporter never changes or stops the run. Shooting solvers,
multistart and continuation send the same kind of events.

## Connections

- Parent: [Core hub](CORE_HUB.md)
- Related: [Lifecycle](CORE_LIFECYCLE.md)
- Related: [Stopping](CORE_STOPPING.md)
- Related: [Acceptance](CORE_ACCEPTANCE_GUARDS.md)
- Related: [State and warmstart](../optimizers/OPTIMIZER_STATE_WARMSTART.md)

---

[⌂ Home](CORE_HUB.md)
