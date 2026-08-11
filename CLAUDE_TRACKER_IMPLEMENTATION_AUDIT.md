---
title: CLAUDE_TRACKER_IMPLEMENTATION_AUDIT
type: audit
scope: pre-v4-contained-baseline
status: historical-superseded
auditor: independent
updated: 2026-08-11
---

# Historical independent audit — pre-V4 tracker baseline

> This is the 2026-08-11 baseline audit, retained for provenance.  Its defect
> findings are not a statement about the V4 beta.  V4 validation and current
> release boundaries are recorded in
> [`OPTIMIZER_STATUS_AND_ROADMAP.md`](OPTIMIZER_STATUS_AND_ROADMAP.md).

## Executive summary

The tracker's *architecture* is sound: the record model is genuinely passive, the
observer contract hands over only facts the engine already owns, CMA-ES and the
gradient engine emit byte-identical event shapes, the standalone reader really does
import neither `optimizer` nor NumPy, and checkpoint artifacts are SHA-256 addressed
and pickle-free. All 96 tests pass and `git diff --check` is clean.

The *implementation* has one class of defect that invalidates the central invariant.
**The tracker is passive numerically but not behaviourally: four independent code
paths let a recording failure raise out of an observer callback and abort an
optimization run that succeeds with the tracker switched off.** No observer dispatch
site in `core/engine.py` or `optimizers/cma_es.py` contains any containment, so any
`ValueError`/`TypeError` raised while serialising a record propagates through
`run_chunk` to the caller. The most damaging instance triggers on a completely legal
OLGS system — one whose `evaluate()` returns a vector or non-finite *secondary*
metric, which `validate_metrics` explicitly permits — and it fires at the very end of
the run, in `on_chunk`, after all numerical work is done. The run directory is left
permanently in `status: "live"` with the writer lock held and no `terminal.json`.

Two further defects sit in the read path that Phase 2b will be built on: a
`read_timeline` → `follow_timeline` handoff that can silently skip events, and a
`follow_timeline` that stalls forever on mid-file corruption. Both are latent today
because nothing consumes them yet; both become data-loss bugs the moment a run index
exists.

Beyond that: the terminal record points at `chunk_end` while the receipt declares the
selection rule `best_finite_value`, and these disagree in practice; the gradient
engine never records random state even though Phase 1b's own completion condition
requires it; `inspect_writer` uses `os.kill(pid, 0)`, which on Windows terminates the
target process rather than probing it; the compact-payload guard bounds each node but
not the payload, so a single "compact" event can legally carry 8192 floats; and no
reader ever compares `schema_version` against anything.

The root Obsidian vault is well formed — one connected component of 202 notes, four
branch hubs, one broken link. Its graph configuration is the weak part: it disagrees
with the nested `contained/` vault on `showAttachments`/`hideUnresolved`, three colour
groups collide across the Contained/Stable/Legacy boundary they exist to separate, one
group can never colour a visible node, and the `optimizer_log` package — a stated
public entry point — has no note and no colour group at all.

**Counts (confirmed defects + design risks + missing tests):**

| Severity | Confirmed defects | Design risks | Missing tests | Total |
|---|---|---|---|---|
| critical | 2 | 0 | 0 | **2** |
| high | 5 | 0 | 1 | **6** |
| medium | 8 | 5 | 5 | **18** |
| low | 5 | 7 | 4 | **16** |
| **total** | **20** | **12** | **10** | **42** |

The single correct next phase is unchanged from the roadmap — **Phase 2b, the
read-only run index** — but D1–D7 must land first, because a run index whose inputs
can be truncated mid-run by the writer, or silently skipped by the reader, indexes
evidence that is not there.

---

## Scope, method, and boundaries honoured

Read in full: `optimizer/__init__.py`, `optimizer/core/{observer,engine,evaluate,stopping}.py`,
`optimizer/tracker/{__init__,model,journal,artifacts,bridge,reader}.py`,
`optimizer_log/{__init__,reader}.py`, `optimizer/optimizers/{cma_es,_common,adam}.py`,
`optimizer/system_olgs/{contract,validation}.py`, `optimizer/{state,logs/trace,logs/checkpoint}.py`,
`tests/{test_tracker,test_tracker_bridge,test_public_api}.py`, both design documents,
`.obsidian/graph.json`, `contained/.obsidian/graph.json`, and all four `graph/*.md` hubs.

Honoured boundaries: the deleted black-box feature is **not** reported as missing; UI,
modes, run discovery, retention, campaigns, and verification scans are listed only under
*verified intentional deferrals*; no implementation file was modified; the only write is
this report.

Five reproduction scripts and one link-graph analyser were written to a scratchpad
outside the project tree. Every finding below marked *confirmed* was executed.

---

## A. Confirmed defects

### D1 — critical — A legal secondary metric aborts a tracked run and strands the journal

**Files:** `optimizer/tracker/bridge.py:179-180` → `optimizer/tracker/model.py:106-115`
(`_scalar_metrics`) → `optimizer/tracker/model.py:39-49` (`json_safe`).

`JournalObserver.on_chunk` is the only bridge path that forwards engine metrics
**raw**. Every other path (`_snapshot`, `bridge.py:87-105`) filters each value through
`_scalar_or_skip`, which catches `TypeError`/`ValueError` and drops the value. `on_chunk`
passes `event.start_metrics` / `event.end_metrics` straight into `ChunkRecord.create`,
where `_scalar_metrics` raises `TypeError` for a non-scalar and `json_safe` raises
`ValueError` for a non-finite float.

This is reachable on a fully contract-compliant system. `validate_metrics`
(`optimizer/system_olgs/validation.py:37-51`) validates **only** that `J` exists and is
finite; secondary metrics are passed through untouched, and the fixture systems in this
very repository return vectors from `simulate()` and complex/array values in metric-like
mappings.

**Affected flow:** `run_chunk` → `_record_chunk` (`engine.py:1026`) → `observer.on_chunk`
(`engine.py:585-586`, no containment) → raise → out of `run_chunk` to the user. The
`run_end` event is never written, `terminal.json` never exists, `live.json` stays
`"running"`, and `.writer.lock` is never released, so `inspect_writer` reports `live`
for a dead process.

**Minimal reproduction** (`scratchpad/repro_a.py`, executed):

```python
class VectorMetricSystem:            # J finite, one vector secondary metric
    def evaluate(self, controls):
        u = controls.as_matrix(copy=False)
        return {"J": float(np.sum(u * u)), "populations": np.asarray([1.0, 2.0, 3.0])}
```

```text
--- vector-secondary-metric ---
  tracker OFF : ok=True
  tracker ON  : ok=False err=TypeError: Metric 'populations' must be a JSON scalar; ...
  timeline kinds : ['run_start','checkpoint','iteration','checkpoint','iteration',
                    'checkpoint','iteration','checkpoint','checkpoint']
  terminal.json  : False
  live status    : running
  writer state   : live
  lock still held: True
--- nan-secondary-metric ---
  tracker ON  : ok=False err=ValueError: Tracker records cannot contain non-finite float values.
--- inf-secondary-metric ---
  tracker ON  : ok=False err=ValueError: Tracker records cannot contain non-finite float values.
```

The NaN/inf variants abort even earlier: `StopTracker._check_finite` stops at the
initial metrics, and the initial-stop branch still calls `_record_chunk` with the
non-finite metrics attached.

**Recommended fix (do not implement here):** route `start_metrics`/`end_metrics` in
`bridge.py:179-180` through the same `_scalar_or_skip` filter used by `_snapshot`, and
record the dropped names under `_tracker_omitted_fields` (see D12). Independently apply
D2 so this class of failure can never abort a run again.

---

### D2 — critical — No observer dispatch site isolates observer failures

**Files:** `optimizer/core/engine.py:393-394, 427-428, 465-466, 499-500, 553-554, 585-586`;
`optimizer/optimizers/cma_es.py:291-293, 314-315, 342-343, 455-456, 488-490`.

Every dispatch is a bare `for observer in observers: observer.on_X(event)`. There is no
`try`, no per-observer quarantine, and no "recording failed" event. This is the root
cause that turns D1, D3, and D4 from *lost evidence* into *lost runs*, and it means any
third-party observer — which `normalize_observers` explicitly invites
(`observer.py:164-182`) — can kill a multi-hour optimization.

It also contradicts the stated contract. `OPTIMIZATION_TRACKER_HANDOFF.md` §12 Phase 1
item 1 requires tracker-enabled and tracker-disabled runs to give identical terminal
controls, metrics, stop reason, and evaluator counts. Under D1/D3/D4 the tracked run has
no terminal state at all.

**Affected flow:** all five lifecycle events, both drivers.

**Recommended fix:** wrap each dispatch in a helper that catches `Exception`, disables the
offending observer for the remainder of the chunk, and records the failure — into the
journal when the journal is still writable, and always onto the returned
`OptimizerResult` so the caller learns the evidence is incomplete. Never let a passive
sink change control flow. Keep `normalize_observers`' *pre-run* type validation strict
(that failure is a caller error and should still raise before any numerical work).

---

### D3 — high — A long exception message from a step function aborts a tracked run

**Files:** `optimizer/core/engine.py:858` (also `:813`, `:894-897`) →
`optimizer/tracker/bridge.py:162` (`_compact_mapping`, no string clamp) →
`optimizer/tracker/model.py:66-68` (`compact_json`, `_MAX_EVENT_STRING = 8192`).

`proposal_failed` and `gradient_failed` embed the full exception text as
`technical["error"]`. `_scalar_or_skip` accepts any string regardless of length; the
8192-byte limit is enforced later, inside `_append`, where it raises. NumPy routinely
produces multi-kilobyte exception messages when array shapes or dtypes appear in them.

**Minimal reproduction** (`scratchpad/repro_c.py`, executed):

```python
def loud_failure_step(context):
    raise ValueError("x" * 9000)
```

```text
tracked=False: OK stop_reason=proposal_failed
tracked=True : ABORTED ValueError: Tracker event string is too large at $.proposal.error.
```

**Recommended fix:** truncate in `_compact_mapping` rather than raising in `compact_json` —
clamp to `_MAX_EVENT_STRING`, append an explicit `…[truncated N chars]` marker, and keep
`compact_json`'s hard limit as a last-resort guard for the direct `RunJournal` API only.

---

### D4 — high — Checkpoint payloads bypass compaction, so a non-JSON `api_context` aborts the run

**File:** `optimizer/tracker/bridge.py:199-216`.

`on_checkpoint` forwards `event.api_context` (line 205), `event.optimizer_state` (203),
`event.system_params` (213) and `event.random_state` (214) **raw** into
`RunJournal.save_checkpoint`, which passes them to `_encode_json`
(`artifacts.py:30-31`) → `json_safe` → `TypeError` for any non-JSON-safe object. Every
other bridge path filters first. `resolved_api_context["caller"]` is verbatim
caller-supplied data (`engine.py:692`), so this is fully under user control.

**Minimal reproduction** (`scratchpad/repro_b.py`, executed):

```python
run_chunk(..., observers=[JournalObserver(journal)], api_context={"owner": object()})
```

```text
tracked run ABORTED: TypeError: Value of type object is not JSON-safe.
timeline kinds: ['run_start']          # dies at the chunk_start checkpoint
```

A NumPy array in `api_context` is fine (`json_safe` converts it), which makes the failure
mode inconsistent and hard to predict for a caller.

**Recommended fix:** apply `_compact_mapping` to `api_context`, `system_params` and
`metadata` at `bridge.py:199-216`; for `optimizer_state` — which legitimately contains
arrays — use `json_safe` inside a `try` and fall back to storing the offending keys as
native arrays in the `.npz` (or record them as omitted), never raising.

---

### D5 — high — `inspect_writer` terminates the writer process on Windows

**File:** `optimizer_log/reader.py:140-159`, specifically line 152 `os.kill(int(writer["pid"]), 0)`.

The docstring promises "Report local writer liveness **without taking ownership or
modifying files**." On POSIX, signal 0 is a permission/existence probe. On Windows,
CPython's `os.kill` calls `TerminateProcess` for every signal value other than
`CTRL_C_EVENT`/`CTRL_BREAK_EVENT` — so `os.kill(pid, 0)` **kills** the process. The guard
at line 150 restricts the call to `writer["host"] == socket.gethostname()`, i.e. exactly
the case where the live writer is the local optimization run.

`optimizer_log` declares no platform restriction and `pyproject.toml` sets only
`requires-python = ">=3.10"`, so this is in-scope portability, not a hypothetical.

Two secondary problems on the same line: PID reuse reports a recycled PID as `live`, and a
PID owned by another user raises `PermissionError` (an `OSError` subclass), which the
`except` at line 153 misclassifies as `abandoned`.

**Reproduction** (`scratchpad/repro_c.py`, POSIX behaviour verified):

```text
inspect_writer state          : live
after simulated crash (pid=999999) : abandoned
writer recorded on other host      : remote_or_unknown
```

**Recommended fix:** replace the probe with a platform-guarded helper — on Windows use
`OpenProcess(SYNCHRONIZE|PROCESS_QUERY_LIMITED_INFORMATION)` + `WaitForSingleObject`, or
simply return `unknown` and rely on heartbeat staleness. Add a `start_time`/boot-id field
to `writer.json` so PID reuse is detectable, and treat `PermissionError` as "alive but not
ours", not "abandoned".

---

### D6 — high — `read_timeline().next_offset` is taken from a later `stat()`, so a follower silently skips events

**File:** `optimizer_log/reader.py:173` (`path.read_text(...)`) vs `:201`
(`next_offset=path.stat().st_size`).

The file content is snapshotted first; the offset is measured afterwards. Any event
appended between those two statements is counted in `next_offset` but never returned, and
a subsequent `follow_timeline(offset=next_offset)` starts *after* it. The event is lost to
that reader permanently — there is no gap detection, because `validate_event`'s
`previous_sequence` check (line 84) only compares records **within one batch**.

**Minimal reproduction** (`scratchpad/repro_c.py`, executed — one event appended between
`read_text` and `stat`):

```text
records returned : ['run_start']
next_offset      : 857 (file size now 857)
follow from that offset returns: []      <- the iteration event is lost
```

This is exactly the read pattern a Phase 2b run index will use: full read to build the
index, then incremental follow for live updates.

**Recommended fix:** compute `next_offset` from the bytes actually consumed
(`len(text.encode("utf-8"))` for the parsed prefix, matching the tail-recovery arithmetic
already at lines 183-185), never from a second `stat()`. Additionally return the last
parsed `sequence` in `TimelineReadResult` and let `follow_timeline` accept it as
`previous_sequence`, so a gap becomes a detectable error rather than silent loss.

---

### D7 — high — `follow_timeline` stalls forever on mid-file corruption

**File:** `optimizer_log/reader.py:222-230`.

A JSON decode error returns `next_offset=line_start` — the correct behaviour for a
partially written tail, and the wrong behaviour for a permanently damaged interior line.
The follower re-reads the same bytes forever and never delivers any of the valid events
after it, including `run_end`. `read_timeline` treats the same file as a hard error
(line 191), so the two readers disagree about the same corruption.

**Minimal reproduction** (`scratchpad/repro_b.py`, executed — damaged line followed by a
valid `run_end`):

```text
first pass  : records=0 next_offset=360 tail=True
second pass : records=0 next_offset=360        (identical -> no progress)
a valid run_end after the damaged line is never delivered: True
```

**Recommended fix:** distinguish the two cases. If the writer is closed
(`inspect_writer` → `closed`/`abandoned`) or the damaged line is followed by a newline,
it is not a tail — surface a distinct terminal `corrupt_at_offset` result instead of
re-offering the same offset. Give `TimelineReadResult` an explicit
`stalled: bool` / `reason` field so a caller can tell "nothing new yet" from
"blocked forever".

---

### D8 — high — Terminal record points at `chunk_end` while the receipt declares `best_finite_value`

**Files:** `optimizer/tracker/model.py:377-381` (`"selection": {..., "rule": "best_finite_value"}`,
hardcoded and not configurable) vs `optimizer/tracker/bridge.py:228`
(`selected_checkpoint=self._checkpoint_references.get("chunk_end")`).

The receipt tells a reader that the run's selected result is the best finite value; the
terminal record points at the *final* state. These disagree whenever the run ends worse
than its best — which the engine permits routinely via `accept_tolerance > 0`, a stall
stop, a failure stop, or a `maxiter` stop after an overshoot.

**Minimal reproduction** (`scratchpad/repro_e.py`, executed; bowl objective, overshooting
fixed step, `accept_tolerance=10.0`):

```text
receipt selection rule : best_finite_value
final J   = 5.2488
best  J   = 0.5
selected_checkpoint    = chunk_end
durable checkpoints    = ['chunk_end', 'chunk_start']
  chunk_end    controls=[-1.12 -1.12] J=5.2488
  chunk_start  controls=[0. 0.]       J=0.5
terminal record points at the WORSE state: True
```

Note the best state survives only inside `chunk_start`, and nothing in `TerminalRecord`
references it. A consumer that trusts `selection.rule` and follows `selected_checkpoint`
retrieves controls 10× worse than the run's best.

Related: with `checkpoint_start=checkpoint_best=checkpoint_final=False` the terminal record
has `selected_checkpoint: None` and **no control vector survives the run at all**
(repro E2) — the journal keeps scalars only.

**Recommended fix:** make the selection rule explicit in `RunReceipt.create` rather than
hardcoded, and have `JournalObserver` track a `best_*` reference alongside `chunk_end`,
selecting according to the receipt's declared rule (falling back to `chunk_end` with an
explicit `selection_fallback` reason when no best checkpoint was written). Also emit a
`best_*` checkpoint when the initial state is never improved on, so "best" is always
addressable.

---

### D9 — medium — `nonfinite` stops are reported as `completed`; CMA-ES hardcodes `completed`

**Files:** `optimizer/core/engine.py:469-474` (`_terminal_status`);
`optimizer/optimizers/cma_es.py:476`.

```text
nonfinite          -> completed        # metrics went non-finite mid-run
nonfinite_trial    -> failed
gradient_failed    -> failed
stall              -> completed
target             -> completed
maxiter            -> completed
```

A run that diverged to NaN/inf is indistinguishable from a healthy budget exhaustion in
`terminal.json` and therefore in any future run index. CMA-ES never uses
`_terminal_status` at all — `status="completed"` is a literal, so a CMA-ES run in which
every candidate failed to evaluate, or `maxiter=0`, still reports success (repro B11:
`status=completed reason=maxiter iteration=0`).

**Recommended fix:** add `nonfinite` to the failed set in `_terminal_status`, and have
`cma_es` call `_terminal_status` (exported from the engine) on its own stop reason,
deriving a `failed`/`degenerate` status when no generation produced a usable evaluation.

---

### D10 — medium — `load_checkpoint` follows an unvalidated `relative_path`; integrity is optional

**File:** `optimizer/tracker/artifacts.py:128-146`, specifically line 139
`target = Path(run_dir) / reference.relative_path`.

The *write* side is guarded (`_safe_file_stem`, lines 23-27, correctly rejects `..` and
separators). The *read* side is not: a reference dict — which comes from a timeline event,
i.e. from a file the reader does not own — can carry `"relative_path": "../../secret.npz"`
or an absolute path, and `load_checkpoint` will read it. Combined with `"sha256": None`
(the dataclass default, line 49, and honoured at line 140), integrity checking is also
opt-out.

**Minimal reproduction** (`scratchpad/repro_c.py`, executed):

```text
loaded from outside the run dir: name='stolen' (no containment check)
save-side guard rejects traversal in checkpoint_id: yes
with sha256 dropped from the reference: loaded without any check
```

Mitigating: `np.load(..., allow_pickle=False)` (line 147) means this is arbitrary *read*,
not arbitrary code execution — a genuinely good decision that should be preserved.

**Recommended fix:** resolve `run_dir/relative_path` and require
`resolved.is_relative_to(run_dir.resolve())`; reject absolute paths outright. Treat a
missing `sha256` as an error unless the caller passes an explicit
`verify=False`, so integrity is opt-out by intent rather than by omission.

---

### D11 — medium — The gradient engine never records random state, so Phase 1b's completion condition is unmet

**Files:** `optimizer/core/engine.py:408` and `:425` (the `random_state` parameter exists and
is threaded into `CheckpointEvent`) — **no call site in `engine.py` supplies it**, verified
by source inspection. `optimizer/optimizers/cma_es.py:340` supplies it correctly.

`OPTIMIZATION_TRACKER_HANDOFF.md` §12, Phase 1b, item 5 requires that "Important checkpoint
artifacts reproduce exact controls, scalar evidence, optimizer **and random state**, API
context, system facts, trigger reason, and SHA-256 integrity evidence." Phase 1b is marked
*complete* in both design documents.

**Evidence** (`scratchpad/repro_d.py`, executed):

```text
--- engine-adam ---
  checkpoint labels: ['chunk_start','best_J','best_J','best_J','chunk_end']
  random_state set : [False, False, False, False, False]
--- cma-es ---
  checkpoint labels: ['chunk_start','best_J','best_J','best_J','chunk_end']
  random_state set : [True, True, True, True, True]
```

Deterministic Adam does not need it, but `run_chunk` accepts an arbitrary caller-supplied
`step` function, which may be stochastic (dropout-style perturbation, random restarts,
stochastic line search). Those runs are currently not reproducible from a checkpoint.

**Recommended fix:** add an optional `random_state: Callable[[], Mapping] | Mapping | None`
argument to `run_chunk` and pass its value through every `_record_checkpoint` call site
(lines 734, 751, 823, 868, 907, 996, 1014). Leave it `{}` when the caller supplies nothing,
and record `random_state_source` in the checkpoint metadata so absence is explicit rather
than ambiguous.

---

### D12 — medium — Metrics are silently dropped with no omission marker

**File:** `optimizer/tracker/bridge.py:93-96` — `if scalar is _SKIP: continue`.

`_compact_mapping` records what it dropped under `_tracker_omitted_fields`
(`bridge.py:82-83`). `_snapshot` does not. A system returning a per-timestep profile as a
metric therefore produces a journal in which that metric simply does not exist, with no
record that it was ever offered.

**Evidence** (`scratchpad/repro_b.py`, executed):

```text
system metrics keys : ['J', 'profile', 'label']
recorded metrics    : ['J', 'label']
omission marker     : False
```

This matters more than it looks: `RunReceipt.metric_descriptors` can declare a metric that
the timeline never contains, and nothing — writer or reader — cross-checks the two. A UI
would render an empty series and be unable to say whether the metric was never produced or
silently discarded.

**Recommended fix:** have `_snapshot` collect dropped names and pass them through as a
`ScalarSnapshot` field (e.g. `omitted_metrics: tuple[str, ...]`), and add a reader helper
that diffs declared descriptors against observed metric keys.

---

### D13 — medium — Compact-payload limits bound each node, not the payload

**File:** `optimizer/tracker/model.py:20-23` and `:62-88`.

`_MAX_EVENT_DEPTH=8`, `_MAX_EVENT_MAPPING=128`, `_MAX_EVENT_SEQUENCE=64`,
`_MAX_EVENT_STRING=8192` are all *per-node*. There is no aggregate byte or element cap, so
the product of the limits is astronomically larger than any of them.

**Evidence** (`scratchpad/repro_b.py`, executed):

```text
64x64 = 4096 floats accepted; encoded bytes = 29773
128 keys x 64 floats accepted; bytes = 42130
timeline.jsonl size after ONE 'compact' analysis event: 34477 bytes
```

The bridge is not exploitable — `_compact_value` (`bridge.py:48-68`) rejects any list that
is not entirely mappings, so arrays cannot reach a timeline event through the engine. The
bypass is on the **public** `RunJournal` API (`record_iteration`, `record_analysis`,
`record_intervention`), which the design documents present as a first-class interface for
controllers. `TRACKER_MODEL.md` states the invariant that "large controls, populations and
resume state are checkpoint artifacts, not iteration payloads"; the guard does not enforce
it.

**Recommended fix:** add a serialised-size cap in `compact_json` — measure
`len(json.dumps(safe, separators=(",",":")))` against a limit (a few KB is consistent with
"one compact scalar record per iteration") and raise with the offending path. Consider
lowering `_MAX_EVENT_SEQUENCE` for lists of scalars specifically, since the legitimate
use — line-search attempt records — is a list of small mappings.

---

### D14 — medium — `schema_version` is written on every record and validated by nothing

**File:** `optimizer_log/reader.py:47-64` (`validate_receipt`) and `:66-89` (`validate_event`).

Both require the key to be *present*; neither compares it to anything. `optimizer_log` has
no notion of which schema versions it understands.

**Evidence** (`scratchpad/repro_d.py`, executed):

```text
a v99 receipt and v99 event are accepted by a v3 reader: 1 record(s)
unknown future event kind is a hard error: Unknown timeline event kind: 'future_kind'.
```

The two halves fail in opposite directions: version skew is silently accepted, while an
added event kind is a hard error (`_EVENT_KINDS`, lines 13-21, is a closed set with no
"ignore unknown" policy). The roadmap's own Phase 3 adds a `decision` event kind — which
will make every existing reader raise on every new run.

**Recommended fix:** introduce `SUPPORTED_SCHEMA = {"optimizer.tracker/v3"}` in
`optimizer_log.reader`, reject a receipt outside it with a clear message, and change
`validate_event` to *skip and report* unknown kinds within a supported major version rather
than raising. Add the reverse test — a v99 fixture that must be rejected.

---

### D15 — medium — Silent behavioural divergence between the two initial-failure paths

**Files:** `optimizer/core/engine.py:518-554` vs `optimizer/optimizers/cma_es.py:258-294`.

The engine emits `on_start` to all observers, then `on_terminal` to all observers. CMA-ES
emits `on_start` *and* `on_terminal` per observer inside a single loop
(`cma_es.py:291-293`). With one observer the results agree; with two, observer A sees its
terminal before observer B sees its start. The two also differ in counter construction:
the engine reports real gradient counters via `_attempt_snapshot`
(`engine.py:503-515`), CMA-ES hardcodes `"gradient": 0, "gradient_attempt": 0`
(`cma_es.py:263, 266`).

Neither path emits `on_chunk`, so an initial failure produces a timeline with no chunk
boundary — correct and intentional, but undocumented in `TRACKER_MODEL.md`'s causal-order
section (lines 88-104).

**Recommended fix:** extract the engine's `_record_initial_failure` into a shared helper
used by both drivers, and document the "no chunk event on initial failure" rule in the
causal-order table.

---

### D16 — medium — CMA-ES updates its distribution from candidates that failed to evaluate

**File:** `optimizer/optimizers/cma_es.py:359, 372-376`.

Failed candidates keep the sentinel `values[row] = ±inf` (line 359) and remain in
`samples`. `_rank_indices` sorts the sentinel to the end, but `elites =
samples[order[:elite_count]]` (line 374) takes a fixed count regardless of how many
candidates actually succeeded. When more than `population - elite_count` candidates fail —
common early in a run with an aggressive `sigma` — the elite mean and `elite_spread` are
computed from points with no objective value at all.

This is a numerical-quality defect rather than a tracker defect, but it is visible in the
journal: `technical["failed_evaluations"]` (line 449) is recorded, so a reader can see the
condition without being able to tell that the distribution was corrupted by it.

**Recommended fix:** restrict `elites` to successfully evaluated rows
(`order[:min(elite_count, n_ok)]`), and if `n_ok == 0` skip the distribution update
entirely, shrinking `sigma` as the rejection branch already does.

---

### D17 — low — `elapsed_s` is measured from journal construction, not from run start

**Files:** `optimizer/tracker/journal.py:62` (`self._started_at = perf_counter()` in
`__init__`) and `:152`.

`RunJournal.create` is necessarily called before `run_chunk`, and nothing bounds the gap.
Every `elapsed_s` in the timeline therefore includes setup time that is not optimization
time, which invalidates the handoff's §8 "elapsed time and forward/gradient budget consumed"
window analysis.

**Evidence** (`scratchpad/repro_e.py`, executed): a 0.4 s gap before `record_start` yields
`run_start elapsed_s = 0.411`.

**Recommended fix:** reset `_started_at` in `record_start`, and record the
creation→start gap once as `setup_s` in the `run_start` payload.

---

### D18 — low — The writer lock is redundant with directory creation and has no reclaim path

**Files:** `optimizer/tracker/journal.py:72-83` (`mkdir(parents=True, exist_ok=False)`) and
`:103-112` (`_claim_writer`); advertised as `"writer_locking": True` at
`optimizer/tracker/__init__.py:69`.

`RunJournal.create` already fails on an existing directory, and `__init__` is
token-guarded against reopening, so `.writer.lock` can never actually be contended. What it
*does* do is persist after a crash with no takeover, recovery, or force API — an abandoned
run directory is permanently unwritable and only diagnosable through `inspect_writer`.

**Evidence** (`scratchpad/repro_c.py`, executed):

```text
second create blocked by: FileExistsError
inspect_writer state          : live
after simulated crash         : abandoned
writer recorded on other host : remote_or_unknown
no takeover / recovery API exists: True
```

**Recommended fix:** either document the lock as crash-forensics only (and downgrade the
`writer_locking` capability claim to something like `"writer_ownership_record"`), or give
Phase 2b an explicit `reclaim(run_dir, force=...)` that refuses while `inspect_writer`
returns `live`. Do not add silent takeover.

---

### D19 — low — Durability is per-file, not per-directory, and costs three fsyncs per event

**Files:** `optimizer/tracker/journal.py:134-142` (`_write_json_atomically`), `:144-162`
(`_append` → `_heartbeat`), `:199-224` (`_publish_live`), `optimizer/tracker/artifacts.py:103-117`.

Contents are fsynced and renames are atomic, but the *containing directory* is never
fsynced, so on a hard power loss a renamed file can still be absent. Every recorded event
performs three synchronous writes: `timeline.jsonl` append + fsync, `writer.json`
temp+fsync+rename, `live.json` temp+fsync+rename.

**Measured** (`scratchpad/repro_c.py`): 200 iteration events in 67.0 ms = **0.335 ms/event**
on APFS. Acceptable for expensive physics; material for the "cheap numerical problem" case
the handoff §10 item 8 explicitly anticipates.

Also here: `.tmp` files are left behind if the process dies mid-write, and readers do not
clean or ignore them explicitly.

**Recommended fix:** fsync the run directory after the first creation of each file; make the
`writer.json` heartbeat time-throttled (e.g. at most once per 500 ms, always on lifecycle
transitions and checkpoints) rather than per-event; keep `timeline.jsonl` strictly
per-event durable, as designed.

---

### D20 — low — Root/nested Obsidian graph configurations contradict each other

**Files:** `/Users/billabobz/OPTIMIZER/.obsidian/graph.json` vs
`/Users/billabobz/OPTIMIZER/contained/.obsidian/graph.json`.

| setting | root vault | contained vault |
|---|---|---|
| `showAttachments` | `false` | `true` |
| `hideUnresolved` | `false` | `true` |
| `optimizer_log` group | absent | `path:optimizer_log` |
| tests vs logs/schedules | merged into one group | split into two |
| rgb `9279918` means | Legacy branch | logs + schedules |

The same notes render as a fundamentally different graph depending on which folder is
opened as the vault. At the root, the 73 notes that link to `.py` files draw **no** edges at
all (measured), so the entire doc↔code structure is invisible; in `contained/` those edges
are the dominant structure. And rgb `9279918` means two unrelated things in the two configs.

**Recommended fix:** make the two configs agree on `showAttachments`/`hideUnresolved` (I
would set `showAttachments: true` in both — it is what makes the module notes meaningful),
add `path:contained/optimizer_log` to the root config, and keep one colour→meaning table
shared by both files.

---

## B. Design risks (not defects)

| # | Severity | Risk | Location | Note |
|---|---|---|---|---|
| R1 | medium | `path:README.md` is a bare substring matching **7** README files; it produces the intended result only if Obsidian resolves overlapping colour groups first-match-wins **and** every other README is caught by an earlier branch group. Under last-match-wins, every branch README turns registry-gold. | `.obsidian/graph.json` group #19 | Measured: `contained/README.md` matches groups #12 **and** #19; `stable/v3/README.md` matches #14 **and** #19; `graph/CONTAINED.md` matches #0 **and** #12 (`path:contained` matches `graph/CONTAINED.md` case-insensitively). Make the root group unambiguous, e.g. anchor it or rename the root note. |
| R2 | medium | Cross-branch colour collisions defeat the branch separation the groups exist to create. | `.obsidian/graph.json` | `contained/optimizer/core` **==** `stable/v3` (rgb 5011624); `contained/optimizer/system_olgs` **==** `legacy/unstable-v2/vault` (15896107); `contained/optimizer/optimizers` **==** `legacy/unstable-v2/problems` (14767961). Hub↔branch reuse (CONTAINED/LEGACY/REGISTRY) is fine and deliberate; these three are not. |
| R3 | medium | `JournalObserver` is one-shot and `RunJournal.create` requires a fresh directory, so a multi-chunk or warm-started sequence cannot share a journal. | `bridge.py:108-118`, `journal.py:72-83` | Deliberate and documented (`TRACKER_MODEL.md:145-151`), but it means the *common* curriculum workflow already exercised in `test_public_api.py:142-171` produces N unrelated journals with no linking identity. Phase 3's segment coordinator must supply `parent_run_id`. |
| R4 | medium | `RunReceipt.metric_descriptors` is never reconciled against the metrics actually recorded, and `system.tracking_spec()` has no receipt consumer. | `model.py:328-353`; `get_tracking_spec` used only by `olgs.py:57` and `describe_system` | The precedence chain declared in `OPTIMIZER_STATUS_AND_ROADMAP.md:77-83` is unimplemented. Verified by grep: nothing in `optimizer/tracker` or `optimizer_log` reads `tracking_spec`. |
| R5 | medium | `save_all_checkpoints=True` has no cap and no retention. | `bridge.py:125, 131-134` | Measured: a 5-iteration Adam run on a 6-element control wrote **17 `.npz` files / 53 274 bytes** (`latest`, `accepted`, `best_J` per accepted iteration). Retention is an acknowledged deferral, so this flag needs at least a docstring warning. |
| R6 | low | CMA-ES `random_state` embeds PCG64 128-bit integers, which no JSON consumer outside Python can represent exactly. | `cma_es.py:340` | Measured state ints of 127 and 123 bits; `142231634183583599241308901332361415022 > 2**53`. Python round-trips exactly; a JS/portal reader silently corrupts the RNG state. Store as a decimal **string** or hex. |
| R7 | low | `optimizer` hard-depends on `optimizer_log`. | `optimizer/tracker/__init__.py:19-32` → `optimizer/tracker/reader.py:8` | The "standalone, no optimizer dependency" contract is tested in one direction only (`test_tracker_bridge.py:331-346`). `import optimizer` fails outright if `optimizer_log` is absent. |
| R8 | low | `optimizer_log` is packaged only by the glob `include = ["optimizer*"]`. | `pyproject.toml` `[tool.setuptools.packages.find]` | It works today by coincidence of naming. Declare it explicitly, or split it into its own distribution, which is what the docstring's intent implies. |
| R9 | low | `IterationRecord.proposal` holds the whole engine `technical` dict, so the actual proposal data lives at `payload.proposal.proposal`. | `bridge.py:162`, exercised at `tests/test_tracker_bridge.py:210` | Harmless but confusing for a generic projection consumer; rename to `technical` before external readers depend on the path. |
| R10 | low | `RunJournal.update_live` is public and accepts lifecycle `"ended"`, with a caller-supplied unvalidated `last_event_id`. | `journal.py:182-197` | A caller can rewrite `live.json` with an arbitrary status after closure. Restrict to `"running"` or make the public entry point read-only. |
| R11 | low | `_snapshot_facts` labels every metric `"system.evaluate"` regardless of origin. | `engine.py:348-359` | `MetricDescriptor.source` exists precisely to carry real provenance; the bridge overrides it with a constant. |
| R12 | low | `save_checkpoint` stores `optimizer_state` as a JSON string inside the compressed `.npz` rather than as native arrays. | `artifacts.py:30-31, 103-114` | Float64 `repr` round-trips exactly, so no precision loss — but dtype is lost (float32 → float64 on reload) and Adam/L-BFGS/CMA-ES resume state is stored as text. |

---

## C. Missing and weak tests

| # | Severity | Gap | Where it belongs |
|---|---|---|---|
| T1 | high | **False-positive test.** `_CallSpy` is instantiated at `tests/test_tracker.py:97` and asserted at `:117-118`, but is never passed to the journal, the observer, or anything else — verified: the only three occurrences of `spy` in the file are the construction and the two assertions. `assertEqual(spy.objective_calls, 0)` cannot fail. The one test that claims to prove "the passive tracker must not call evaluate()" proves nothing. | Wire the spy in as the *system* of a real `run_chunk` with a `JournalObserver` attached, or delete it and rely on `test_tracker_bridge.py:106-160`, which does the check correctly by comparing evaluator counts. |
| T2 | medium | No test for a system with a **non-scalar or non-finite secondary metric** (D1). The `CountingQuadraticSystem` fixture only produces `J=inf` alongside `energy=inf`, and `validate_metrics` rejects the non-finite `J` first, so the metrics never reach the tracker. | `test_tracker_bridge.py` — assert tracker-on and tracker-off produce identical results for a system with a vector metric. |
| T3 | medium | No test that a **recording failure cannot abort a run** (D2/D3/D4). | A deliberately raising observer plus an oversize `technical["error"]` case. |
| T4 | medium | No **CMA-ES tracker-on/off equivalence** test. `test_cma_es_uses_common_observer_events_and_reproducible_checkpoints` (`test_tracker_bridge.py:300-329`) checks events and checkpoint restore, never numerical identity or system-call counts. | Mirror `test_tracker_on_off_have_identical_numerics_and_no_extra_system_calls` for `cma_es` with a fixed seed. |
| T5 | medium | No test for **reader race / offset continuity** (D6) or **mid-file corruption in `follow_timeline`** (D7). `test_incremental_reader_and_projection_keep_causal_fields` only follows a quiescent file. | `test_tracker.py`, alongside the existing tail-recovery test. |
| T6 | medium | No **negative tests** for the validators: a receipt missing `references`, a bad `direction`, a non-contiguous sequence, a non-object payload, `project_timeline(kind="nope")`, a v99 `schema_version`. Every validator branch in `optimizer_log/reader.py:41-109` is currently unexercised. | `test_tracker.py` — a small table-driven negative suite. |
| T7 | low | No test that a **tampered checkpoint is rejected**, and none that a reference with `sha256: None` skips verification. Confirmed working / confirmed unguarded respectively by repro D4. | `test_tracker.py` beside the existing checkpoint test. |
| T8 | low | Post-terminal rejection is tested only for `record_iteration` (`test_tracker.py:268-278`). `save_checkpoint`, `record_analysis`, `record_intervention`, and `record_chunk` after `run_end` are untested, as are `JournalObserver`'s own `_started`/`_ended` guards (`bridge.py:137-138, 150-151, 171-172, 189-190, 220-221`). | `test_tracker.py` / `test_tracker_bridge.py`. |
| T9 | low | No test that `checkpoint_final=False` leaves `selected_checkpoint: None` and no recoverable controls (D8/E2), nor that `best != final` is represented correctly. | `test_tracker_bridge.py`. |
| T10 | low | `test_standalone_reader_does_not_import_the_optimizer_or_numpy` (`test_tracker_bridge.py:331-346`) launches a subprocess with no `cwd=`, so it silently depends on the suite being run from `contained/`. | Pass `cwd=` explicitly and set `PYTHONPATH`. |

---

## D. Verified intentional deferrals (not defects)

Confirmed present and correctly declared; **none of these is reported as a defect**:

- **Black-box feature** — absent by intent. Not reported. (A `stable/v3/optimizer/blackbox` note still exists in the read-only stable worktree; that is the frozen branch and out of scope.)
- **UI / portal / CLI** — `optimizer_log` has no `__main__` and no entry point; `tracker.info()["capabilities"]["portal_cli"] = False`.
- **Run discovery / index** — `"run_discovery": False`; no run-root convention exists.
- **Retention** — `"retention_api": False`; no reference graph, no dry-run manifest.
- **Campaign / multistart** — `"campaign_model": False`.
- **Modes (manual / shadow / active policy)** — no decision event kind, no policy code.
- **Verification scans** — no KKT/feasibility certification writer; `TerminalRecord.certification` is a caller-supplied passthrough, correctly making no claim of its own.
- **Segment coordinator / real resume** — documented at `TRACKER_MODEL.md:145-151`.
- **`.obsidian/app.json` already excludes `runs/`** from indexing, which correctly anticipates tracker run directories before run discovery exists — a good pre-emptive decision worth keeping.

Also verified working as designed and worth protecting through any refactor:

- Tracker-on/off numerical equivalence for scalar-metric systems (identical controls, metrics, stop reason, and `evaluate`/`gradient` counts).
- CMA-ES ↔ engine event parity: identical event kinds, identical iteration payload fields, identical chunk payload fields, identical checkpoint labels (`repro_d.py`).
- `_compact_value` correctly refuses arrays and non-mapping sequences on the bridge path.
- `np.load(..., allow_pickle=False)` — no deserialisation RCE surface.
- Counter monotonicity enforcement (`journal.py:164-180`) and strict single-`run_start` / post-`run_end` rejection.
- Tail recovery that ignores exactly one damaged final line and refuses mid-stream corruption in `read_timeline`.
- The standalone reader importing neither `optimizer` nor NumPy.
- Root vault link health: **one connected component of 202 notes**, four branch hubs, all 15 hub link targets resolve.

---

## E. Obsidian root-graph findings (detail)

Measured with a resolver that reproduces Obsidian's rules (relative markdown links from
the containing folder, shortest-unique-basename wikilinks, non-`.md` targets treated as
attachments):

```text
notes=209  resolved edges=887  external=1
connected components among linked notes: 1   (size 202, spanning all five branches)
orphans (hidden by showOrphans:false) = 7    (3 are inside .claude/ or .pytest_cache/,
                                              already excluded by userIgnoreFilters)
unresolved links = 1
cross-branch edges: root->graph 4 | graph->contained 5 | graph->legacy 5
                    graph->stable 3 | graph->registry 2
hub degrees: README.md 4 | CONTAINED 6 | LEGACY 6 | STABLE 4 | REGISTRY 3
```

- **G1 (low, confirmed defect):** one broken link —
  `legacy/unstable-v2/README.md` → `v3_plans/README.md` does not exist. With
  `hideUnresolved: false` at the root it renders as a phantom node hanging off the Legacy
  branch. Fix the link or set `hideUnresolved: true` (see D20 — the nested vault already
  sets it `true`).
- **G2 (low, confirmed defect):** the colour group `path:legacy/unstable-v2/problems` can
  never colour a visible node. All three notes under that path
  (`problems/p1.md`, `problems/p1_three_level_delta1/README.md`,
  `problems/p1_three_level_delta1/results/global_comparison.md`) are orphans, and
  `showOrphans` is `false`. Either link them from `graph/LEGACY.md` or drop the group.
- **G3 (low, confirmed defect):** `contained/optimizer_log` contains **zero** markdown
  notes and has **no** colour group in the root config. The standalone read-only package —
  one of the four public import entry points named in the brief — has no representation in
  the root graph at all, while every other subpackage does. Add
  `contained/optimizer_log/OPTIMIZER_LOG_READER.md`, link it from `graph/CONTAINED.md`, and
  add the matching group.
- **G4 (low, observation):** `legacy/unstable-v2/AGENTS.md` is an orphan while
  `registry/AGENTS.md` is linked from `graph/REGISTRY.md` — an avoidable asymmetry between
  two files with the same role.
- The four hub notes carry consistent frontmatter (`title`, `type: workspace_branch`,
  `tags`), but `showTags: false` means the tags contribute nothing to the graph. Either
  enable tag nodes or drop the tags to avoid implying structure that is not rendered.
- No Dataview/`query` blocks exist anywhere in the vault (0 files), and no community
  plugins are installed, so nothing beyond the colour groups and link resolution shapes the
  graph. `bases: true` is enabled but no `.base` files exist.

Colour-group precedence (R1, R2, D20) is the substantive risk here and is documented above.

---

## F. Conclusion

### The single correct next phase

**Phase 2b — the read-only run index and a minimal `optimizer_log` detail/live API. No
graphical UI, no modes, no retention, no verification scans.**

This matches both design documents unchanged
(`OPTIMIZER_STATUS_AND_ROADMAP.md:108`, `OPTIMIZATION_TRACKER_HANDOFF.md` §11) and the
audit supports it: the write path and the event contract are structurally complete and at
parity across both drivers, so the missing capability really is discovery, not more
recording. Nothing found here argues for reordering the roadmap — only for repairing the
foundation Phase 2b will stand on.

### Fixes required before Phase 2b begins

Non-negotiable, because a run index that reads truncated or skipped evidence is worse than
no index:

1. **D2** — isolate observer failures at every dispatch site in `engine.py` and `cma_es.py`.
2. **D1** — filter `start_metrics`/`end_metrics` in `bridge.py:179-180`.
3. **D4** — compact `api_context` / `system_params` / `metadata` in `bridge.py:199-216`.
4. **D3** — truncate oversize strings in `_compact_mapping` instead of raising in `compact_json`.
5. **D6** — derive `next_offset` from bytes consumed, and carry `previous_sequence` across `follow_timeline` calls.
6. **D7** — distinguish a damaged tail from permanent mid-file corruption; never return an offset that cannot advance.
7. **D14** — gate on `schema_version`, and make an unknown event kind skippable rather than fatal. The index is the first consumer that must survive schema evolution, and Phase 3 already plans a new `decision` kind.

Strongly recommended before Phase 2b, because the index will surface these directly to a
user:

8. **D8** — reconcile the receipt's `selection.rule` with `TerminalRecord.selected_checkpoint`.
9. **D9** — classify `nonfinite` as failed, and derive CMA-ES's terminal status rather than hardcoding it.
10. **D5** — replace `os.kill` in the read-only reader; the index will call `inspect_writer` on every listed run.
11. **T1–T6** — the false-positive test and the five missing behavioural tests above.

Deferrable past Phase 2b: D10–D13, D15–D20, all of section B.

### Suggested implementation order

| Step | Work | Verified by |
|---|---|---|
| 1 | D2 observer isolation + T3 | A raising observer leaves `run_chunk` results byte-identical to tracker-off. |
| 2 | D1, D4, D3 payload filtering + T2 | Vector/NaN/inf secondary metrics, object `api_context`, and a 9 KB exception message all record cleanly and abort nothing. |
| 3 | D12 omission markers | Dropped metrics are named in the record; a reader can distinguish "absent" from "discarded". |
| 4 | D9, D8 terminal semantics + T9 | `nonfinite` reads as failed; `selected_checkpoint` obeys the declared rule or states its fallback. |
| 5 | D6, D7 reader offsets + T5 | A concurrent append is never skipped; mid-file corruption reports a distinct terminal state. |
| 6 | D14 schema gate + T6 | v99 is rejected; a synthetic `decision` event is skipped and reported, not fatal. |
| 7 | D5 writer-liveness probe | No signal is ever sent from `optimizer_log`; PID reuse is detectable. |
| 8 | T1 + T4 test repair | The passive-tracker assertion actually exercises a system; CMA-ES equivalence is guarded like the engine's. |
| 9 | **Phase 2b proper** — run-root convention, index from validated receipts + writer/terminal state, list/detail/follow API over the existing projection and offset reader. | Runs are discoverable and inspectable without changing any evidence. |
| 10 | D11 random-state plumbing | Closes the Phase 1b completion condition before Phase 3 depends on reproducible handoffs. |
| 11 | D20 + R1 + R2 + G1–G3 graph cleanup | Root and nested vaults agree; no cross-branch colour collision; `optimizer_log` is on the graph. |

D10 (checkpoint path containment), D13 (payload size cap), D15–D19 and section B fit
naturally alongside step 10 or immediately after Phase 2b.

### Commands and results used for verification

Note: the `conda` shell function was not executable in this session's sandbox
(`__conda_exe: permission denied`). The identical commands were run through the absolute
conda binary; the environment, interpreter, and results are the same.

| Command | Result |
|---|---|
| `conda run --no-capture-output -n optimizer_v3 python optimizer_context.py` | OK — 39 methods, 6 catalog groups, `live tracker bridge: yes`, run schema `optimizer.tracker/v3`, 15 tests, 5 documents, 56 worktree changes, head `main / 86f7bc4`. |
| `conda run --no-capture-output -n optimizer_v3 python -m unittest discover -s tests` | **96 tests, OK**, 0.259 s. |
| `git diff --check` | Clean, exit 0. |
| `python -c "import sys; import optimizer_log.reader; print('optimizer' in sys.modules, 'numpy' in sys.modules)"` | `False False` — standalone-reader claim holds. |
| `scratchpad/repro_a.py` | **D1 confirmed** — vector / NaN / inf secondary metric: tracker-off OK, tracker-on aborts; no `terminal.json`; `live=running`; writer `live`; lock held. |
| `scratchpad/repro_b.py` | **D4, D13, D7, D9, D12, D8** confirmed — object `api_context` aborts at first checkpoint; 64×64 and 128×64 float payloads accepted (29 773 / 42 130 bytes; a single analysis event grows `timeline.jsonl` to 34 477 bytes); `follow_timeline` returns the same offset twice and never delivers a valid `run_end` past a damaged line; `_terminal_status("nonfinite") == "completed"`; metrics dropped with no marker; PCG64 state ints of 127/123 bits. |
| `scratchpad/repro_c.py` | **D3, D6, D10, D18, D19** confirmed — 9 000-char exception aborts only the tracked run; `read_timeline` returns 1 record with `next_offset` covering 2, losing the second; `load_checkpoint` reads `../outside.npz`; second `RunJournal.create` blocked by `FileExistsError` (not the lock); crashed/remote writer states correct; 0.335 ms per recorded event. |
| `scratchpad/repro_d.py` | **D11, D14, R5, T1** confirmed — engine checkpoints `random_state` all `False`, CMA-ES all `True`; engine/CMA-ES event kinds and payload fields identical; `save_all_checkpoints` writes 17 `.npz` for 5 iterations; a v99 receipt+event is accepted while an unknown kind is fatal; tampered artifact correctly rejected, `sha256: None` skips the check. |
| `scratchpad/repro_e.py` | **D8, D17** confirmed — final J 5.2488 vs best J 0.5 with `selected_checkpoint = chunk_end`; disabling checkpoints leaves no recoverable controls; `elapsed_s = 0.411` after a 0.4 s pre-run gap. |
| `scratchpad/graph_audit.py` | Root vault: 209 notes, 887 resolved edges, 1 connected component of 202, 7 orphans, 1 unresolved link, 73 notes linking to hidden `.py` attachments; colour-group overlap and duplicate-rgb tables as reported in §E. |

No implementation file was modified. The only file written by this audit is this report.
