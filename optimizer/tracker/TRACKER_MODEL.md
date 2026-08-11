# Canonical run model: technical draft and pilot findings

## Aim

`optimizer.tracker` is a passive, generic run journal.  It makes an
optimization reproducible as a numerical story:

```text
authoritative problem/system reference
  + initial controls
  + raw outer-iteration trajectory
  + occasional derived analysis
  + explicit intervention/checkpoint/terminal selection
```

It is not an optimizer, a policy, a convergence oracle, or a tracking service.

## Current status

The tracker schema remains `optimizer.tracker/v3` for reader compatibility, while
the V4 model, writer, checkpoint artifacts, and focused tests exist in this
worktree.  A generic core observer now writes a fresh journal live during
one `core.run_chunk` lifecycle through `JournalObserver`; it receives only facts
the engine already computed.  The standalone `optimizer_log.reader` package
reads tracker files without importing the numerical optimizer.

Receipt semantics and the one-run lifecycle are now enforced: the receipt
declares objective metric meaning and selection direction, the journal owns an
exclusive writer token and heartbeat, and important checkpoint artifacts save
controls, scalar evidence, optimizer state, API context, and integrity hashes.
The remaining staged plan is in
[`OPTIMIZATION_TRACKER_HANDOFF.md`](/Users/billabobz/OPTIMIZER/v4/OPTIMIZATION_TRACKER_HANDOFF.md).

## Source-of-truth rule

~~~text
problem/system files     remain physical and mathematical authority
run_receipt.json         identifies exactly which authority was used
timeline.jsonl           records what the optimizer actually did
live.json                is only a replaceable display convenience
writer.json              reports writer ownership, heartbeat, and closure
~~~

The receipt does not reproduce the full system in a competing specification.
It identifies system/problem inputs and records run-specific method settings,
seed, orientation, provenance, and, when no separate parameter artifact
exists, a compact resolved parameter snapshot.  The source system remains the
mathematical authority.

## File model

~~~text
<run>/
  run_receipt.json
  timeline.jsonl
  writer.json
  .writer.lock                  present only while the writer is live
  checkpoints/
    initial.npz                  optional selected artifact
    final.npz                    optional selected artifact
  live.json
  terminal.json
~~~

The timeline is append-only.  It is the durable record even if an interruption
leaves `live.json` stale.  The reader can ignore only one malformed final JSONL
line; corruption in the middle is an error because causal history would be
ambiguous.

## Record model

| Type | Holds | Does not do |
|---|---|---|
| `SourceReference` | ID and optional path/hash/version. | Copy/redefine a problem or system. |
| `MetricDescriptor` | Metric ID, label, role, direction, unit, and source. | Calculate or reinterpret a metric. |
| `RunReceipt` | Run ID, objective metric/direction, metric descriptors, API context, source references, method settings, seed. | Act as a second system specification. |
| `ScalarSnapshot` | Scalar metrics, metric sources, omitted-metric names, monotonic cost counters. | Store a pulse array or calculate a metric. |
| `IterationRecord` | Raw `before -> proposal -> trial -> after`, best-so-far, accepted/rejected, reason. | Interpret convergence. |
| `ChunkRecord` | Method/phase, iteration window, resolved system parameters, entry/exit metrics and reason. | Coordinate another chunk. |
| `AnalysisRecord` | Declared raw window, inputs, findings, extra counters. | Run the diagnostic itself. |
| `InterventionRecord` | Action/reason/parameters, before/after, optional analysis link. | Decide to alter the optimizer. |
| `TerminalRecord` | Selected terminal state/checkpoint and supplied certificate status. | Infer local/global optimality. |
| `LiveSnapshot` | Latest compact display state. | Replace the timeline as evidence. |

Large controls, populations, and resume state are checkpoint artifacts, not
iteration payloads.  This is what permits one raw record per outer iteration
without turning long optimization runs into large archives.

## Causal event order

~~~text
run_start
  -> checkpoint?                important initial reproducibility state
  -> iteration*                 every outer iteration, accepted or rejected
  -> checkpoint?                new best, failure, or explicit manual pin
  -> chunk                      resolved method/phase boundary
  -> analysis?                  scheduled/explicit analysis of a raw window
  -> intervention?              declared restart, repair, or method switch
  -> checkpoint?                selected sparse artifact
  -> observer_failure?          another passive sink was quarantined
  -> run_end
~~~

The marker `*` means repeated.  The other events are optional and may be
interleaved naturally.  Event sequence IDs, timestamps, elapsed time, and
monotonic counters make the record replayable.

If the initial system evaluation fails, the lifecycle is exactly
`run_start -> run_end`; there is no chunk boundary because no valid chunk began.
If the journal itself fails, observer isolation closes the writable prefix with
`tracking_failure` when possible and releases its ownership marker. The numerical
optimizer continues and reports the evidence failure on `OptimizerResult`.

## Computation boundary

`RunJournal` accepts values supplied by the engine/controller.  It has no
system object and cannot call:

```text
evaluate, gradient, residual, Jacobian, HVP, propagation, KKT routine
```

For the live path, the generic engine owns the cached initial
evaluation, gradient, proposal, trial evaluation, acceptance decision, and
counters.  It must hand those facts to the journal at the moment they occur.
This avoids the extra initial evaluation required by the temporary HO
post-run-projection pilot.

## HO pilot findings

The real HO v3 pilot showed that one compact timeline can carry:

- physical endpoint metrics such as fidelity, purity, entropy, energy, trace,
  and minimum eigenvalue when supplied by the system;
- exact-gradient and objective-call counters;
- Adam step/moment information;
- Armijo attempted step, threshold, trial value, and acceptance data;
- initial/final pulse checkpoint pointers; and
- explicit terminal statements that KKT and global optimality were not
  assessed.

The observed pilot sequence was:

```text
run_start -> initial checkpoint -> 3 iterations -> final checkpoint
          -> short trace analysis -> run_end
```

It did not contain an intervention because neither short test required a
restart, repair, or method change.  Events must reflect actual behavior, not
be invented merely to fill a schema.

## Current implementation boundary

`JournalObserver` remains deliberately one-shot: one fresh `RunJournal` records
one optimizer chunk so its counters remain monotone.  V4 adds the external
`Campaign` coordinator for multi-start, refinement, challenge, continuation,
reopen, and warm-start handoff.  A campaign must retain the exact static
`system_spec()` fingerprint that produced every candidate; reopening requires a
caller-provided resolver and rejects a mismatched system instead of silently
replaying it against a nearby configuration.

The durable default checkpoint policy retains the chunk start, the terminal
selected best candidate, terminal/failure evidence, and chunk end.  Every
iteration remains in the scalar timeline; ordinary accepted/latest checkpoint
arrays are intentionally not persisted.  This bounds long-run checkpoint
growth without inventing a destructive cleanup operation.  An explicit
reference-aware cleanup graph and dry-run deletion tool remain future work.

The tracker and campaign layer still do not infer physics, feasibility repairs,
or source paths.  Those facts must come from system APIs, declared control
transfer hooks, or caller-owned factories/resolvers.
