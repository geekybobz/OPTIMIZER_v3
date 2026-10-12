---
role: topic
parent: ../../README.md
summary: "Bounded independent process runs, explicit result selection and client-owned lifetime."
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

Jobs belong to the current client. Close/exit cancels and reaps owned workers;
a handle from a lost runtime reports `job_interrupted`. Save retrieved candidates
with the existing checkpoints to reuse them later. Durable jobs and pause/resume
are not implemented. Retrieved method state is uncertified: restart from
`.controls` or `.variables`, rather than assuming exact numerical continuation.

Defaults: at most **two workers**, one native-library thread each, 1 GiB total
reservation, 256 MiB coordinator reserve, 16 queued jobs, 64 retained jobs,
8 MiB per input/source bundle and 128 MiB aggregate temporary scratch.
Each job reserves 256 MiB worker RAM and 4 MiB serialized result by default.
Reservations reject work that cannot fit. POSIX RSS sampling requests cancellation
when a worker exceeds its estimate; this is **not a hard OS memory limit**.
Native thread environment variables are hints that project libraries must honor.

## Commands

| Call | Behavior |
|---|---|
| `jobs.configure(max_workers=..., threads_per_worker=..., memory_bytes=..., coordinator_bytes=..., max_queue=..., max_jobs=..., input_bytes=..., scratch_bytes=..., shutdown_seconds=...)` | Configure while no jobs are retained. Worker count times thread count cannot exceed reported CPU capacity. |
| `jobs.submit("adam", controls=None, variables=None, options={"maxiter": ...}, label=None, worker_bytes=..., result_bytes=..., context=None)` | Freeze an explicit physical input or the selected head and queue one registered solver. |
| `jobs.status(job=None)` | One job or scheduler status, limits, reservations and enforcement scope. |
| `jobs.wait(job, timeout=10)` | Wait at most 30 seconds; a timeout returns current status and does not cancel. |
| `jobs.result(job)` | Retain an owned native result. Not-ready jobs reject retrieval; failed numerical results may retain a candidate for inspection. |
| `jobs.compare([a,b], metric="J", direction="minimize")` | Rank finite metrics for successful compatible jobs. OLSS additionally requires verified, qualified results. Never selects a winner. |
| `jobs.cancel(job)` | Cancel queued work or request a project-call boundary in a running worker. |
| `jobs.release(job)` | Release a terminal job's frozen inputs/runtime result; independently retained candidates survive. |
| `jobs.shutdown()` | Cancel/reap all owned workers. Retained results remain inspectable until released. |

Every call above uses the `opt.` prefix in the terminal. Native Python uses the
same namespace with `service=opt.OperationService(...)`; close that service when
finished. Ordinary Python calls create no command history.

## Details

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

Only private temporary transport contains the frozen code and allowlisted numeric
codec payloads; no pickle or worker log file is used. Completion reaps the owned
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
when its existing exact-head requirements are met. No daemon
completion event or recovered live handle is invented in this phase.

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
[test_jobs.py](../../tests/test_jobs.py) checks real process isolation, native
solver parity, seeded stochastic runs, queue and memory admission, cancellation,
worker failures, verification-aware comparison, durable commit ordering,
checkpoint reopening and the terminal entry point. Scientific article benchmarks
and MCP transport remain separate work.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [Command history](../history/HISTORY.md)
- Related: [Testing hub](../../tests/TESTING_HUB.md)

---

[⌂ Home](../../README.md)
