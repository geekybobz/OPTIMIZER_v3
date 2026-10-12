---
role: front-door
parent: ../../README.md
summary: "Durable run journals, records, and checkpoints written without calling the system."
read_when: "Read when a run needs a durable record on disk."
status: active
reviewed: 2026-10-12
---
# Tracker Hub

`optimizer.tracker` keeps durable evidence of a run: what the optimizer did,
iteration by iteration, without ever calling the system's numerics.

## In brief

A `RunJournal` owns one run folder. A `JournalObserver` passes facts the engine
has already computed into that journal. Pass the observer to any optimizer:

```python
import optimizer as opt
from optimizer.tracker import JournalObserver, RunJournal, RunReceipt, SourceReference

receipt = RunReceipt.create(
    run_id="run-001",
    objective_label="J",
    direction="minimize",
    references={
        "problem": SourceReference.create("my-problem"),
        "system": SourceReference.create("my-system"),
    },
)
journal = RunJournal.create(run_dir, receipt)
result = opt.adam(system, controls, maxiter=20, observers=[JournalObserver(journal)])
```

## Details

## Rules

- The tracker never calls `evaluate`, `gradient`, residuals, Jacobians, HVPs,
  or any propagation. Tracking a run adds no system calls.
- Every outer iteration is recorded, accepted or rejected. Full control arrays
  are written only as selected checkpoints, never inside iteration records.
- One `JournalObserver` records one optimizer chunk into a fresh journal. A
  campaign, not the journal, coordinates several chunks.
- A journal has one writer. It cannot be reopened by accident or appended to
  after its run ends.
- If the journal fails, the optimizer keeps running and reports the failure in
  `OptimizerResult.tracking_failures`.

## Run folder

```text
<run>/
  run_receipt.json          immutable run identity and configuration
  timeline.jsonl            append-only causal event stream
  live.json                 replaceable latest progress snapshot
  terminal.json             pointer to the selected final event
  writer.json               writer identity, heartbeat, and closed state
  .writer.lock              present only while the writer owns the run
  checkpoints/<id>.npz      selected control and resume artifacts
```

The timeline is the scientific record. A reader may skip one incomplete final
line; corruption earlier in the file is an error. `opt.tracker.info()` reports
this layout and the tracker's capabilities at runtime.

## Records

| Record | Holds |
|---|---|
| `RunReceipt` | Run ID, objective metric and direction, metric descriptors, API context, source references, method settings, and seed; it must reference the problem and the system |
| `SourceReference` | An ID with an optional path, hash, or version |
| `MetricDescriptor` | Metric ID, label, role, direction, unit, and source |
| `ScalarSnapshot` | Scalar metrics, their sources, omitted metrics, and cost counters |
| `IterationRecord` | Before, proposal, trial, and after states, the best so far, and the accept or reject reason |
| `ChunkRecord` | Method, phase, iteration window, system parameters, and entry and exit metrics |
| `AnalysisRecord` | A declared raw window, its inputs, and findings |
| `InterventionRecord` | An action outside normal stepping, with its reason and before and after states |
| `TerminalRecord` | The selected final state, checkpoint, and certificate status |
| `LiveSnapshot` | The latest compact progress state; a display convenience, not evidence |

Events follow this order, with optional events interleaved as they occur:

```text
run_start -> checkpoint? -> iteration* -> checkpoint? -> chunk
          -> analysis? -> intervention? -> run_end
```

## Checkpoints

`CheckpointPolicy` keeps durable checkpoints bounded: by default the chunk
start, the selected best, failures, and the chunk end. Every iteration still
appears in the timeline. `load_checkpoint` restores a SHA-256 verified control
artifact without a system object.

## Reading runs

The standalone [run log reader](../../optimizer_log/OPTIMIZER_LOG_READER.md)
reads, validates, and follows run folders without importing the optimizer or
NumPy.

## Not built yet

Run discovery and a command-line portal do not exist yet; `opt.tracker.info()`
reports both as unavailable.

## Connections

- Related: [the V4 entry](../../README.md)
- Related: [Optimizer Contract](../optimizers/OPTIMIZER_CONTRACT.md), [Campaign Modes](../campaigns/CAMPAIGNS.md), [Logs Hub](../logs/LOGS_HUB.md)

## Implementation details

### Claim boundary

The tracker shows the objective and metric trajectories, cost, accepted and
rejected steps, method handoffs, and the evidence that was actually requested.
It does not turn a loss curve into a theorem: a small raw gradient is not
constrained local stationarity, and one converged run is not a global optimum.

### Files

| File | Owns |
|---|---|
| [model.py](model.py) | Record types and the schema version |
| [journal.py](journal.py) | `RunJournal` |
| [bridge.py](bridge.py) | `JournalObserver` and `CheckpointPolicy` |
| [artifacts.py](artifacts.py) | Checkpoint save and load |
| [reader.py](reader.py) | Compatibility re-exports of the standalone reader |

---

[⌂ Home](../../README.md)
