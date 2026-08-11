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

The v2 model, writer, reader, checkpoint artifacts, and focused tests exist in
the contained tree.  A temporary HO-purification v3 pilot has written real
Adam and Armijo line-search histories into this layout.

That pilot validates the data structure, not live integration: it projected the
already-completed in-memory Trace after each run.  The production plan is in
[`OPTIMIZATION_TRACKER_HANDOFF.md`](/Users/billabobz/OPTIMIZER/contained/OPTIMIZATION_TRACKER_HANDOFF.md).

## Source-of-truth rule

~~~text
problem/system files     remain physical and mathematical authority
run_receipt.json         identifies exactly which authority was used
timeline.jsonl           records what the optimizer actually did
live.json                is only a replaceable display convenience
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
| `RunReceipt` | Run ID, objective direction, source references, method settings, seed. | Act as a second system specification. |
| `ScalarSnapshot` | Scalar metrics, metric sources, monotonic cost counters. | Store a pulse array or calculate a metric. |
| `IterationRecord` | Raw `before -> proposal -> after`, best-so-far, accepted/rejected, reason. | Interpret convergence. |
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
  -> iteration*                 every outer iteration, accepted or rejected
  -> analysis?                  scheduled/explicit analysis of a raw window
  -> intervention?              declared restart, repair, or method switch
  -> checkpoint?                selected sparse artifact
  -> run_end
~~~

The marker `*` means repeated.  The other events are optional and may be
interleaved naturally.  Event sequence IDs, timestamps, elapsed time, and
monotonic counters make the record replayable.

## Computation boundary

`RunJournal` accepts values supplied by the engine/controller.  It has no
system object and cannot call:

```text
evaluate, gradient, residual, Jacobian, HVP, propagation, KKT routine
```

For the future live path, the generic engine owns the cached initial
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

## Next implementation target

The next approved code change should be a small optional journal hook at the
generic `core.run_chunk` boundary.  It must write the start/iteration/terminal
records live from existing engine data and must preserve tracker-on/off
numerical equivalence.  It is deliberately not an HO-specific adapter.

Before adding automatic analysis, interventions, a viewer, or campaigns, this
live core must pass the tests listed in the handoff plan.
