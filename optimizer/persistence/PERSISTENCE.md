---
role: deep
parent: ../../README.md
summary: Independent experiments, typed checkpoints, exact continuation, rollback and bounded disk storage.
read_when: Read before saving, reopening, restoring or pruning an interactive experiment.
---

# Persistent experiments

The [structured terminal](../cli/TERMINAL.md) and [operation service](../operations/SERVICE.md)
can select an independent experiment under `.optimizer/<name>/`. Without
`project.init` or `project.open`, the session retains its existing RAM lifetime.
Closing an experiment preserves its committed data. Initialization never clears
an existing experiment; opening rejects a foreground session with unsaved names
or systems. Forking copies committed data into another independent store and
selects that store.

## Workflow

Launch `optimizer` once, or run `python -B -m optimizer.cli` from V4:

```python
opt.project.init(name="low-pulse", root=".");
opt.session.load(name="main", factory="./system.py", parameters={"N":1001},
    checkpoint={"stateless":True, "dependencies":[]});
u0 = opt.guesses.constant_guess(value=0.1);
opt.session.use(u0);
r = opt.adam(maxiter=20, step_size=0.01);
baseline = opt.session.save(name="baseline");
r = opt.adam(maxiter=5, step_size=0.01);
opt.session.undo();
opt.session.redo();
opt.session.restore(baseline);
opt.project.fork(name="another-guess");
opt.session.exit();
```

On another launch:

```python
opt.project.open(path=".optimizer/low-pulse");
opt.session.state();
opt.checkpoints.list();
opt.session.restore("baseline");
r = opt.adam(maxiter=5, step_size=0.01);
opt.project.close();
```

Calls retain the same spelling in Python:
`service.execute(opt.OperationRequest("project.init", {"name":"low-pulse"}))`.
Persistence belongs to service operations; directly mutating `service.session`
does not create a durable transaction. Ordinary numerical Python calls receive
no automatic history writer. Frontends add [command history](../history/HISTORY.md).
History-owned stores require the recorder for writes; ordinary Python can read
them read-only or fork an index-owned store. Existing index-owned stores require
explicit frontend migration with `project.open(..., migrate=True)`.

## Commands

Use `opt.info("project.open");` for exact signatures. The 19 lifecycle operations
are service descriptors, separate from numerical exports in the root catalog.

| Command | Behavior and main options |
|---|---|
| `project.init(name, root=".")` | Create a safe unique name; `max_bytes=536870912`, `keep_recent=20`, `undo_depth=10` |
| `project.open(path, read_only=False, mode="exact", migrate=False)` | Restore a committed store, without replaying optimization |
| `project.list(root=".")` | List at most 256 candidate directories without importing systems |
| `project.status()` / `storage.status()` | Identity, generation, bytes, policy, protection, resume levels and retained undo/redo availability |
| `project.fork(name, root=None)` | Copy the committed store, assign a new identity and origin, select the copy |
| `project.close()` | Release the lease and clear the foreground session; preserve files |
| `session.save(name, mode="exact", context=None)` | Pin the selected numerical snapshot; physical modes are `controls_only` or `variables_only` |
| `session.restore(checkpoint, mode="exact", context=None)` | Activate validated saved state; physical restore resets method memory through `session.use` |
| `session.undo(steps=1, context=None)` / `redo(...)` | Move numerical ancestry; reports, inspections and saves are not undo steps |
| `checkpoints.save(value, name=None, kind="report", pin=True)` | Explicit `full_state`, `controls`, `variables`, `report` or `input` payload |
| `checkpoints.load(checkpoint)` | Return owned typed data without activation |
| `checkpoints.list(limit=100)` | At most 256 metadata records, with available/pruned state and bookmarks |
| `checkpoints.pin(checkpoint)` / `unpin(...)` | Change payload protection without changing the byte cap |
| `storage.configure(max_bytes=None, keep_recent=None, undo_depth=None)` | Change explicit disk policy transactionally |
| `storage.plan(action="prune", ids=None)` | Describe eligible `prune`, explicit `delete`, or `purge` targets, hashes, shared blobs, bytes and affected references |
| `storage.apply(plan)` | Apply a matching project/generation plan; reject stale or protected targets |

Checkpoint arguments accept a bookmark, immutable ID, save-result mapping or
reference to that mapping. A saved result name and a checkpoint bookmark are
distinct: use `session.get("r")` for a retained result and `checkpoints.load("baseline")`
for a bookmarked checkpoint. Save/load does not evaluate or optimize the system.

## Continuation contract

| Resume level | Meaning |
|---|---|
| `exact` | Native method state and identity can continue under the recorded system/environment |
| `controls_only` | Compatible OLGS physical controls, with method memory reset |
| `variables_only` | Compatible OLSS physical variables, with method memory reset |
| `analysis_only` | Typed data for inspection, without numerical resume authority |

Exact continuation is limited to native Adam with `variant="adam"`, momentum with `variant="heavy_ball"`,
and `solve_shooting(method="lm")`. Neutral split/restored tests verify their controls/variables
against uninterrupted execution. Shooting preserves the native continuation
history, including its extra boundary evaluation; its history length need not
equal a single uninterrupted call. Other methods remain physical starts.

Exact systems require an explicit source file and
`checkpoint={"stateless":True, "dependencies":["relative-or-absolute-path", ...]}`.
This is the project owner's declaration that numerical behavior has no unrecorded
mutable state or random stream. Captured source/data bytes, effective factory
arguments/defaults, secondary updates and dependency hashes are retained.
Statically visible local imports must be declared too. Dynamic imports, external
services and hidden data reads are not automatically discovered; projects with
such undeclared numerical state cannot truthfully make this declaration. No
custom system-state or callback persistence codec is supplied.

Exact restore checks source/dependencies, Python, NumPy, optional SciPy,
platform/machine and optimizer source fingerprint. Cross-machine bitwise
reproduction is not promised. A physical restore requires a compatible loaded
context; a physical project open requires the saved family to match its mode.
Source dependency checks still apply when reconstructing a project factory.

The codec allowlists native dataclasses, schedules and numeric arrays, preserving
dtype, shape and complex values. It rejects arbitrary classes/callbacks, object
arrays, malformed containers, excessive nesting and decoded array budgets.
Arrays use content hashes and `allow_pickle=False`. Diagnostic infinities/NaNs
use explicit tags; active numerical controls/variables must remain finite.
Native state, metrics, numerical diagnostics and method memory are checkpoint
data. The decoder strips retired logging fields from legacy tagged checkpoints
without changing immutable IDs; exact restore still requires compatible source
and environment identity. Report saves
retain their recorded input references; cleanup can make those inputs unavailable.
Opening a store reconstructs trusted project Python, just as `session.load` does.

## Transactions, retention and recovery

Each store owns `project.json`, `session.json`, immutable
`checkpoints/manifests/<id>.json`, deduplicated `checkpoints/blobs/<digest>.npz`,
captured `inputs/<digest>/payload`, and `runtime/writer.lock`.
An OS lease permits one writer, or shared read-only opens while no writer owns
the store. Process exit releases the lease; recovery never kills a guessed PID.

Execution stages owned RAM state and payloads. The writer checks preparation
capacity, writes/fsyncs immutable data, rechecks the session revision, and
atomically replaces `session.json` before adopting the prepared RAM state.
Pre-commit failure leaves the old head/index intact. If replacement succeeds but
directory fsync fails, the response reports `durability_warning`; the committed
index and payloads are preserved. Reopening removes uncommitted preparation files
and unreferenced blobs, never reruns optimization. For index-owned Python stores, the atomic index is recovery authority. In
frontend history-owned stores, the fsynced successful command outcome is the
commit point; index/RAM publication follows it. [History](../history/HISTORY.md)
owns recorded-store recovery and failure behavior.

The disk cap includes permanent command history, source capsules, manifests, blobs, metadata and preparation
space. A full store needs explicit cleanup or a larger byte limit before another
write. Recent/undo counts are retention preferences; active heads and pins are
hard protection. Explicit cleanup may sacrifice preferred undo paths, as shown
in its plan. Undo after a new numerical mutation clears redo; later named results
retain their identities subject to payload retention. Pruned names remain
tombstones and return `payload_pruned` in RAM and after reopening.

Foreground retained-data limits remain separate: `Session(max_bytes=64*1024*1024,
max_objects=256)`. These limits and disk quotas do not bound process RSS,
factory/native-library memory, or numerical temporaries. Reopening named objects
must fit the foreground limit. No background workers or resource scheduler are
introduced here.

Structured failures include `project_exists`, `project_locked`, `unsaved_session`,
`read_only`, `storage_full`, `stale_plan`, `protected_payload`, `payload_pruned`,
`checkpoint_missing`, `checkpoint_corrupt`, `codec_limit`, `source_changed`,
`environment_changed`, `unsupported_persistence` and `unsupported_continuation`.

## Ownership and verification

- [controller.py](controller.py): lifecycle descriptors, staged service transactions, reconstruction and numerical ancestry.
- [store.py](store.py): leases, indexes, atomic writes, quotas, deduplication, fork and recovery.
- [codec.py](../checkpoints/codec.py): versioned tagged data and bounded numeric blob decoding.
- [persistence tests](../../tests/test_persistence.py): restored native parity, CLI exit/reopen, cleanup, corruption, leases and injected/process failures.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Structured terminal](../cli/TERMINAL.md)
- Related: [Operation service](../operations/SERVICE.md)
- Related: [Code map](../../CODE_MAP.md)

---

[⌂ Home](../../README.md)
