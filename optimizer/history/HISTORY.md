---
role: deep
parent: ../../README.md
summary: Frontend command history, durable statement commits, explicit migration and bounded replay.
read_when: Read before inspecting or replaying terminal history or integrating an MCP recorder.
---

# Command history

Named terminal experiments retain one append-only `history.jsonl`. It contains
submitted statements and outcomes: successful calls, errors, assignments,
inspections, undo, redo, restore and abandoned branches. Ordinary Python numerical
calls and `OperationService` calls create no automatic history or logging files.
Future MCP can use this same recorder; the MCP transport is not implemented.

## Calls

```python
opt.history.list(limit=50);
opt.history.show(command_id="...");
p = opt.history.replay_plan(target="baseline", mode="checkpoint");
opt.history.replay(plan=p, project="restored");
p = opt.history.replay_plan(target="baseline", mode="commands");
opt.history.replay(plan=p, project="recalculated");
```

`list` accepts 1–200 entries and a `before` command-ID cursor. `show` returns the
submission/outcome pair, compact results and committed head metadata. Replay
requires an explicit preview and a new experiment name; it never overwrites the
source. Targets are `latest`, an outcome command ID or a current bookmark. A
latest preview freezes its target before its own assignment is committed.

## Recording and ownership

The frontend records original text, a validated expression when available,
canonical operation IDs, effective arguments, immutable refs, compact results,
state transitions, structured errors, session identity and environment identity.
Nested calls belong to one parent statement and produce one submission/outcome
pair. Publication is atomic for that statement. Earlier successful statements
in a batch remain committed if a later statement fails. Whole-batch syntax
validation rejects invalid submissions before any numerical work.

A session without an experiment keeps at most 256 commands / 1 MiB in RAM.
Read-only experiment inspections use ephemeral session history and do not append
to the saved file. Durable history starts on initialization or writer attachment;
earlier ephemeral commands are not retroactively invented. EOF/script completion
records an empty-text lifecycle action with origin `frontend_close`. Incomplete
input cancellation records `input_cancelled`; no synthetic user command is added.
An explicitly started [persistent coordinator](../jobs/JOBS.md) keeps its writer
lease across client exit. Client EOF records detachment; system job transitions
carry `origin="coordinator"` and use the same history/checkpoint commit path.
Ordinary foreground lifetime retains the existing close behavior.

Existing index-owned stores require `project.open(path=..., migrate=True)` to
append an explicit baseline. Existing immutable IDs are preserved; old commands
are unknown. History-owned writers require the frontend recorder. Plain Python
may open them read-only or fork an independent index-owned experiment. This
prevents recovery from erasing unrecorded Python state changes.

## Durability and limits

The writer fsyncs a submitted record before computation. It stages all child
operations and immutable payloads, checks capacity/revision, then fsyncs the
successful outcome carrying the new index. This outcome is the durable commit
point. The replaceable `session.json` and RAM state follow it. Index-write failure
after this point reports a warning and preserves success for recovery.

Open validates the versioned hash chain, pairs and mandatory fields. It rebuilds
a missing/older index from committed outcomes, marks unmatched submissions
interrupted, and removes prepared orphans without numerical replay. Unknown
versions, truncated records and unrecorded index changes fail explicitly; the
reader does not silently repair or discard the original file. Hashes detect
corruption; they are not signatures against a malicious editor.

History counts against the experiment byte cap and is never pruned. Checkpoint
cleanup only removes eligible numerical payloads. Intent reserves 128 KiB for a
failure outcome; over-budget intent fails before computation. At total capacity
an attempted command can only be reported in ephemeral history. A standalone
literal `opt.storage.configure(max_bytes=...);` increase may fund its own
recording using that explicit new cap. The policy becomes active at its durable
commit. Closing still releases the lease if history writes fail.

Replay plans expose `executable`, `issues`, target state, statement count and
whether numerical work is required. Checkpoint mode copies retained typed data
without numerical work and requires exact native head state. Command mode
recalculates bounded successful numerical/session statements, including rollback,
with a new history. It requires an empty baseline, source/dependency/environment
compatibility, declared inputs and explicit stochastic seeds. Literal source
object IDs, omitted history-produced bindings and live registered callbacks are
not silently reconstructed. Changed source files, physical-only state, pruned
required payloads and unsupported recipes produce blocking issues. History alone
does not guarantee reproduction of external side effects or undeclared state.

## Removed logging API

The reviewed migration removes `optimizer.logs`, `optimizer.tracker`,
`optimizer_log`, `Trace`, observer/event exports and logger constructors. Numerical
calls no longer accept `trace`, `create_trace`, `observers`, `api_context`,
`random_state`, `chunk` or `checkpoint_*` logging options. Results/states no longer
carry trace IDs, trace objects, checkpoint ID lists or tracking failures.
`OptimizerResult.diagnostics` stores constant-size numerical aggregates; OLSS
retains native solver history and method state where numerically required.
Campaigns retain typed candidates in RAM and create no journals or directories;
`campaigns.open` is removed. Existing user archives are left untouched. The
checkpoint decoder strips retired legacy fields without changing saved IDs;
source/environment compatibility still governs restore authority.

## Source and verification

- [journal.py](journal.py): strict stream, quota accounting and recovery.
- [recorder.py](recorder.py): staged statements, frontend descriptors and replay.
- [history tests](../../tests/test_history.py): pairing, ownership, quotas,
  crash/fsync recovery, independent histories and replay parity.
- [Persistence](../persistence/PERSISTENCE.md): typed numerical data and retention.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Structured terminal](../cli/TERMINAL.md)
- Related: [Operation service](../operations/SERVICE.md)

---

[⌂ Home](../../README.md)
