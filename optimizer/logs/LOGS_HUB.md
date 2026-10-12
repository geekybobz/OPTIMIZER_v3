---
role: front-door
parent: ../../README.md
summary: "The in-memory Trace that optimizers fill by default, and its checkpoints."
read_when: "Read when inspecting result.trace or its checkpoints."
status: active
reviewed: 2026-10-12
---
# Logs Hub

Lightweight, in-memory run records: the `Trace` that optimizers fill by default
and the checkpoints inside it.

## In brief

A `Trace` lives in memory for one session. For durable evidence on disk, use
the [Tracker Hub](../tracker/TRACKER_HUB.md); both receive the same engine
events.

## API

| Object | Holds |
|---|---|
| `Trace` | `run_id`, `iteration_records`, `chunk_records`, `event_records`, `checkpoints`, and checkpoint `labels` |
| `IterationRecord`, `ChunkRecord`, `EventRecord` | One iteration, one chunk, or one named event |
| `Checkpoint` | `id`, `label`, `controls`, `state`, `metrics`, `system_params`, `optimizer_state`, `iteration`, `stage`, `previous_id`, `random_state` |
| `clone_run_state` | A deep copy of a `RunState` for a checkpoint |

Useful `Trace` calls: `checkpoint(...)`, `latest_checkpoint_id(label)`,
`get_checkpoint(label_or_id)`, `restore(label_or_id)`, which returns the
controls and state, and `to_dict()`.

Optimizers create a `Trace` unless `create_trace=False`; see
[Optimizer Contract](../optimizers/OPTIMIZER_CONTRACT.md).

## Files

| File | Owns |
|---|---|
| [trace.py](trace.py) | `Trace` |
| [records.py](records.py) | The record types |
| [checkpoint.py](checkpoint.py) | `Checkpoint` and `clone_run_state` |

## Connections

- Related: [the V4 entry](../../README.md)
- Related: [Base API Hub](../BASE_HUB.md), [Optimizer State and Warmstart](../optimizers/OPTIMIZER_STATE_WARMSTART.md), [OLGS Logging Boundary](../system_olgs/LOGGING_BOUNDARY.md), [Core Engine](../core/CORE_ENGINE.md)

---

[⌂ Home](../../README.md)
