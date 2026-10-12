---
role: topic
parent: ../../README.md
summary: "Bounded process runs, persistent coordinators, safe live pause and explicit physical recovery."
read_when: "Read before running several optimization starts through the terminal or service."
status: active
reviewed: 2026-10-12
---
# Parallel optimization jobs

`opt.jobs` runs registered OLGS optimizers and OLSS shooting solvers in separate
processes. Each submission freezes the source recipe, declared dependencies,
system specification, initial controls/variables, configuration and options.
Workers reconstruct independent systems and caches. Results never activate the
foreground; choose their physical controls or variables explicitly.

Factories must be loaded from a source path with
`checkpoint={"stateless": True, "dependencies": [...]}`. Declare all local inputs;
the declaration promises no hidden state. Workers use the same Python environment
and installed dependencies. This is trusted project code, not a security sandbox.
Resolve declared local assets relative to the source file; absolute paths and
working-directory file access are not automatically rebased to the frozen copy.
Every job declares `maxiter`; CMA-ES also requires an explicit integer `seed`.
Live callbacks, method state, nested parallelism and arbitrary functions are rejected.

Jobs use client lifetime by default. Close/exit cancels and reaps those workers.
An explicitly started **persistent coordinator** owns one experiment's writer
lease and survives client EOF/exit. Clients share its named session and serialize
commands through it; ordinary `project.open` cannot compete for its lease.
Independent experiments have separate coordinators and separate resource budgets.

Adam, heavy-ball momentum and LM shooting support iteration-boundary pause.
`resume(mode="exact")` continues the **same live worker**, preserving numerical
locals, stop tracking, caches and method memory. Saved boundary snapshots declare
`physical_only`: a coordinator/worker crash marks live jobs `interrupted`, never
silently restarts them, and never claims an exact restart from disk. Use explicit
`resume(mode="physical", maxiter=...)` to create a new job from that snapshot.
Retrieved results do not certify foreground method-state continuation; select
their `.controls` or `.variables` explicitly. Other methods reject pause.

Defaults: at most **two workers**, one native-library thread each, 1 GiB total
reservation, 256 MiB coordinator reserve, 16 queued jobs, 64 retained jobs,
8 MiB per input/source bundle and 128 MiB aggregate temporary scratch.
Each job reserves 256 MiB worker RAM and 4 MiB serialized result by default.
Reservations reject work that cannot fit. POSIX RSS sampling requests cancellation
when a worker exceeds its estimate; this is **not a hard OS memory limit**.
Native thread environment variables are hints that project libraries must honor.
Paused workers retain their process slot and memory reservation. These limits
apply per coordinator/service, not as a machine-wide allocator across experiments.

## Commands

| Call | Behavior |
|---|---|
| `jobs.configure(max_workers=..., threads_per_worker=..., memory_bytes=..., coordinator_bytes=..., max_queue=..., max_jobs=..., input_bytes=..., scratch_bytes=..., shutdown_seconds=...)` | Configure while no jobs are retained. Worker count times thread count cannot exceed reported CPU capacity. |
| `jobs.submit("adam", controls=None, variables=None, options={"maxiter": ...}, label=None, worker_bytes=..., result_bytes=..., context=None)` | Freeze an explicit physical input or the selected head and queue one registered solver. |
| `jobs.status(job=None)` | One job or scheduler status, limits, reservations and enforcement scope. |
| `jobs.wait(job, timeout=10)` | Wait at most 30 seconds for terminal status or an acknowledged pause; cancellation waits for terminal status. A timeout does not cancel. |
| `jobs.result(job)` | Retain an owned native result. Not-ready jobs reject retrieval; failed numerical results may retain a candidate for inspection. |
| `jobs.compare([a,b], metric="J", direction="minimize")` | Rank finite metrics for successful compatible jobs. OLSS additionally requires verified, qualified results. Never selects a winner. |
| `jobs.cancel(job)` | Cancel queued work or request a project-call boundary in a running worker. |
| `jobs.release(job)` | Release a terminal job's frozen inputs/runtime result; independently retained candidates survive. |
| `jobs.pause(job)` | Request an iteration-boundary pause. Inspect `pausing`/`paused`; a long project call can delay acknowledgement. |
| `jobs.checkpoint(job)` | Retrieve the last acknowledged boundary: `physical`, `native_state`, `iteration`, `family` and `resume_level`. No implicit activation. |
| `jobs.resume(job, mode="exact", maxiter=None)` | Exact live continuation preserves all original options/budget. Physical mode requires an explicit budget, creates a new handle with parent provenance, and cancels a paused parent. |
| `jobs.detach()` | End this persistent terminal client, leaving its coordinator/jobs running. |
| `jobs.shutdown()` | Cancel/reap workers and stop the persistent coordinator, releasing its lease. Saved numerical artifacts remain. |

Every call above uses the `opt.` prefix in the terminal. Native Python uses the
same namespace with `service=opt.OperationService(...)`; close that service when
finished. Ordinary Python calls create no command history.

## Details

### Start, detach and reconnect

Initialize/load an ordinary named experiment, then exit that terminal. Start its
persistent owner and attach once:

```text
optimizer --start-coordinator /absolute/project/.optimizer/demo
```

Enter normal structured commands:

```python
j = opt.jobs.submit("adam", options={"maxiter": 100, "step_size": 0.02});
opt.jobs.pause(j);
opt.jobs.wait(j, timeout=10);
p = opt.jobs.checkpoint(j);
p["iteration"];
opt.jobs.detach();
```

Reconnect later, preserving all named bindings:

```text
optimizer --connect /absolute/project/.optimizer/demo
```

```python
opt.jobs.status(j);
opt.jobs.resume(j);
opt.jobs.wait(j, timeout=10);
r = opt.jobs.result(j);
r.metrics;
opt.session.use(r.controls);
opt.jobs.shutdown();
```

EOF, script completion and `opt.session.exit()` detach persistent clients.
`jobs.detach()`/`jobs.shutdown()` end the current command batch. Persistent
`project.close()` rejects with `coordinator_owned`; its lease belongs to the
owner. `project.fork()` creates an independent snapshot without switching that
owner's experiment or copying live jobs. Named numerical candidates remain.

`--connect` never starts or recovers computation. After owner loss, explicitly
use `--start-coordinator` to recover stored jobs as interrupted. Exact opening
requires compatible recorded source/environment; `--restore-mode controls_only`
or `variables_only` is an explicit physical opening choice when needed. A retained
boundary can then be forked with `jobs.resume(j, mode="physical", maxiter=...)`.
Reconnection alone preserves source/options; it does not perform migration.

The private Unix socket and credential file are POSIX local transport, with
bounded frames and authentication, not a network service. All commands still
use the established AST parser, operation service and recorder. Relative paths
resolve from the requesting client's directory. The reply timeout is 45 seconds;
split long sequences of waits into separate submissions. A lost reply is uncertain work:
the client does not automatically retry numerical commands. Inspect history.
Live client progress shows activity/elapsed time, not remote numerical counters.

### Terminal example

From the checkout root, launch `optimizer` once and enter:

```python
opt.session.load("main", "tests/fixtures/quadratic_system.py", parameters={"N": 1001}, checkpoint={"stateless": True, "dependencies": []});
low = opt.guesses.constant_guess(value=0.01);
high = opt.guesses.constant_guess(value=0.2);
a = opt.jobs.submit("adam", controls=low, options={"maxiter": 10, "step_size": 0.02}, label="low");
b = opt.jobs.submit("momentum", controls=high, options={"maxiter": 10, "step_size": 0.02}, label="high");
opt.jobs.wait(a, timeout=10);
opt.jobs.wait(b, timeout=10);
ranking = opt.jobs.compare([a, b], metric="J", direction="minimize");
ranking;
candidate = opt.jobs.result(a);
candidate.metrics;
opt.session.use(candidate.controls);
opt.jobs.release(a);
opt.jobs.release(b);
```

Inspect `ranking` and choose a job explicitly; result retrieval does not imply
that `a` is the winner. Retry `wait` if status is still queued/running.
For OLSS submit `"shooting.solve_shooting"` or
`"shooting.solve_direct_shooting"` with `variables=...`; inspect qualification
and select `.variables`. Independent verification remains system-owned.

### Ownership, cancellation and history

Workers read frozen code and allowlisted numeric codec payloads from private
temporary transport. Persistent coordinators also retain these inputs in the
existing checkpoint store. No pickle or worker log file is used. Completion reaps the owned
process and removes its directory. Runtime inputs/results remain bounded by
reservations until release. Foreground retained candidates and durable checkpoints
also obey their existing session/storage quotas.

Cancellation is checked before/after project numerical calls. A worker that fails
to acknowledge within `shutdown_seconds` (default two) is terminated and reported
`interrupted`, with no partial-state claim. This cannot interrupt arbitrary code
at a safe instruction boundary. EOF ownership is tracked by an inherited pipe;
an orphan worker exits itself. The manager never guesses or kills a reused PID.

The frontend commits submission/cancellation/release only after its statement
transaction and durable history outcome succeed. Nested statement failure never
starts or cancels a job. Workers do not write history or mutate foreground state.
CLI inspections and retrieval record the observed structured outcomes in the
existing [history](../history/HISTORY.md). Asynchronous timing prevents exact
command-only replay: restore saved physical checkpoints, or use checkpoint replay
when its existing exact-head requirements are met. Coordinator recovery marks
former live workers interrupted.

Persistent transitions are structured `jobs.transition` outcomes with
`origin="coordinator"`. They use the same history transaction, either as a system
event in a foreground outcome or as an empty-text coordinator lifecycle statement.
No user command is invented and no second log file exists. Submission and runtime
control effects follow durable acknowledgement; completion is committed before a
client receives its result. Failed outcome writes remain unacknowledged, pending
history intents recover as interrupted, and retries do not duplicate publication.

The optional job registry in the existing session index references typed,
immutable checkpoint payloads. Current job inputs, results, named handles and pause snapshots
are protected until release; previous versions follow existing preview-based
cleanup. Active/paused jobs additionally reserve result/preparation capacity.
When publication fails, result/submission requests report that failure; the
existing explicit `storage.configure(max_bytes=...)` increase can repair capacity.

Comparison requires the same frozen system/source fingerprint and an explicit
metric/direction. Changed foreground configuration is reported on retrieval.
Failed, cancelled, interrupted, missing-metric and unqualified shooting candidates
remain visible but cannot silently enter the ranking. Comparison never proves
global optimality.

### Source and verification

[manager.py](manager.py) owns admission, process lifetime and candidate publication;
[worker.py](worker.py) owns reconstruction and cancellation boundaries;
[contracts.py](contracts.py) owns resource limits and typed handles;
[api.py](api.py) owns registered calls. The existing service, persistence and
history owners serialize foreground transactions.
[coordinator.py](coordinator.py) owns local transport and the persistent writer;
[durable.py](durable.py) owns stored job records and history publication;
[boundary.py](boundary.py) supplies private synchronous numerical boundaries.
[test_jobs.py](../../tests/test_jobs.py) checks real process isolation, native
solver parity, seeded stochastic runs, queue and memory admission, cancellation,
worker failures, verification-aware comparison, durable commit ordering,
checkpoint reopening and the terminal entry point. The [MCP interface](../mcp/MCP.md)
shares these jobs through native requests. Scientific article benchmarks remain separate work.
[test_durable_jobs.py](../../tests/test_durable_jobs.py) checks live numerical
parity, detach/reconnect, coordinator loss, explicit physical restart,
authentication, single-writer exclusion, publication failure, retention and
concurrent client submission.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Command history](../history/HISTORY.md)
- Related: [Testing hub](../../tests/TESTING_HUB.md)

---

[⌂ Home](../../README.md)
