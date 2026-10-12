---
role: topic
parent: ../../README.md
summary: "Local MCP tools for native optimizer operations, shared experiment ownership and bounded inspection."
read_when: "Read when connecting an MCP client or composing optimizer operations with object references."
status: active
reviewed: 2026-10-12
---
# MCP interface

`optimizer-mcp` exposes six local standard-I/O tools over the existing
[operation service](../operations/SERVICE.md). CLI and MCP clients attach to the
same [experiment coordinator](../jobs/JOBS.md): one writer, shared names,
numerical heads, jobs, checkpoints and command history. MCP executes native
operation requests directly. It never converts requests into shell or CLI text.

Install the optional `mcp` extra; ordinary Python/CLI use does not require it.
The adapter uses the official Python SDK's supported 1.x interface
(`mcp>=1.28,<2`). The transport is local standard I/O, with no HTTP listener.
Connecting a client does not authorize a numerical restart or silently start an
owner. Explicit startup recovers lost live jobs as interrupted.

Factories are explicit trusted source paths. Loading them executes project
Python; this is not a sandbox. Live Python systems, callbacks, arbitrary imports,
pickle and renderers have no transport representation. Discovery reports
unavailable service adapters and their reasons. Supported numerical operations
still require their declared system capabilities and compatible layouts.

Inputs are finite JSON values plus explicit reference/tuple envelopes. Numerical
arrays and results remain in the experiment; tools return compact summaries and
references. Operations inherit service activation defaults: ordinary solvers
activate valid results unless `activate=false`; job results never autoactivate.

Typed requests and outcomes enter the existing `history.jsonl` with `origin=mcp`,
empty raw text and a structured request expression. Rollback, errors and
inspections are retained. Protocol/schema rejections before dispatch have no
experiment transaction. Normal Python receives no automatic history writer.
Typed history blocks command-text replay explicitly; checkpoint replay follows
the existing source, retention and exact-head requirements. It copies numerical
artifacts into a separate experiment without live job ownership.

Transport messages are bounded to 128 KiB input / 256 KiB output; typed action
payloads and compact replies have a 64 KiB budget. Execute request identifiers,
operation names, context/binding names and experiment IDs are limited to 256 UTF-8
bytes each. Inputs also have depth/node
limits. Inspection pages contain at most 1024 elements, operation listings at
most 50 entries. Smaller pages recover output omitted by the byte budget.

The coordinator reply timeout is 45 seconds. Use `jobs.submit`, bounded
`jobs.wait` calls and inspection for long work. An MCP cancellation or lost reply
does not roll back already committed work or cancel a numerical job. Inspect
history/status before retrying; cancel jobs explicitly with `jobs.cancel`.
Disconnect/EOF detaches this client and leaves jobs running. `jobs.shutdown`
explicitly stops the selected owner. Live pause/physical restart guarantees and
per-experiment resource limits are owned by the [jobs reference](../jobs/JOBS.md).

## Tools

| Tool | Inputs | Result |
|---|---|---|
| `optimizer_attach` | `path`, optional `start_coordinator=false`, `restore_mode=exact` | Select one existing experiment; detach before switching. |
| `optimizer_detach` | none | Leave the owner and jobs running. |
| `optimizer_list` | optional `group`, `classification`, `offset=0`, `limit=25` | Paginated operation identities/classification. |
| `optimizer_info` | `operation` | Signature, defaults, injected context inputs, activation and transport limits. |
| `optimizer_execute` | `operation`, optional `arguments`, `context`, `bind`, `activate`, `request_id`, `expected_revision`, `experiment_id` | One transactional native operation response. |
| `optimizer_inspect` | `reference`, optional `detail=summary`, `offset=0`, `limit=128` | Compact summary or a bounded values page. |

## Details

### Launch

Use the approved wheel and global launcher described in
[Installation](../../docs/INSTALLATION.md):

```bash
optimizer-mcp --help
```

Initialize a named experiment in the ordinary CLI with `opt.project.init(...)`,
then exit that terminal. MCP can attach and explicitly start its owner through
`optimizer_attach`, or select it at server startup:

```bash
optimizer-mcp --experiment /absolute/project/.optimizer/demo --start-coordinator
```

For an already running owner, omit `--start-coordinator`. For explicit physical
opening after source/environment changes, select `--restore-mode controls_only`
or `variables_only`. This does not certify an exact numerical restart.
`python -B -m optimizer.mcp` provides the same server entry point.

A generic MCP client's server configuration uses its environment's absolute
executable and an explicit working directory:

```json
{
  "command": "/absolute/environment/bin/optimizer-mcp",
  "args": ["--experiment", "/absolute/project/.optimizer/demo"],
  "cwd": "/absolute/project"
}
```

The client owns the server's standard streams. Standard output contains only
protocol messages. No transcript or separate worker logging file is created.
This configuration is illustrative; the adapter does not modify client settings.

### Load, optimize and inspect

Call `optimizer_execute` with:

```json
{
  "operation": "session.load",
  "arguments": {
    "name": "main",
    "factory": "./system.py",
    "parameters": {"N": 1001},
    "checkpoint": {"stateless": true, "dependencies": []}
  }
}
```

Declare the factory's actual local dependencies. Relative paths resolve from the
MCP server's working directory. Then create and select an initial guess:

```json
{"operation":"guesses.constant_guess","arguments":{"value":0.1},"bind":"u"}
```

```json
{"operation":"session.use","arguments":{"value":{"$ref":{"object_id":"u"}}}}
```

Run a candidate without changing the numerical head:

```json
{"operation":"optimizers.adam","arguments":{"maxiter":10,"step_size":0.02},"bind":"candidate","activate":false}
```

Call `optimizer_inspect` on its metrics:

```json
{"reference":{"object_id":"candidate","path":["metrics"]},"detail":"values"}
```

After reviewing them, explicitly select its control through `optimizer_execute`:

```json
{"operation":"session.use","arguments":{"value":{"$ref":{"object_id":"candidate","path":["controls"]}}}}
```

### References, composition and conflicts

`$ref` resolves a retained name or immutable ID with optional public data-field
names and nonnegative sequence indices. Arbitrary attribute traversal and method
execution are rejected. Use `{"$tuple":[...]}` for constructors requiring tuples;
JSON arrays remain lists. Registered constructors create schedules, layouts,
shooting variables and other data objects that subsequent calls reference.

For control exchange, create explicit source/target grids and a target layout,
then call the existing `utils.resample_controls`, `smooth_controls`,
`filter_controls`, `scale_controls` or `normalize_controls`. Reference a
transform result's `controls` field in the next step. Channel names and counts
remain system-owned. The [control tools](../utils/UTIL_CONTROL_TRANSFORMS.md) own conversion,
extrapolation, endpoint and frequency conventions.

Responses include `state_after.revision`, `metadata.experiment_id`, command ID,
object references, status and summary/error. Include the observed revision and
experiment ID in a later execution request when the decision must reject changes
made by another CLI/MCP client. These guards are optional; they do not reserve a
session between calls. Ordinary sequential calls use the latest committed state.

Transport duplicate suppression is bounded to the live coordinator's response
cache. Reusing one transport ID with different content rejects the request.
`request_id` inside an operation is correlation data, not a durable idempotency
promise. Across lost replies/restarts, inspect history before repeating work.

### Implementation and verification

- [adapter.py](adapter.py): selected experiment, native coordinator calls and detach.
- [server.py](server.py): six schemas, official SDK server and bounded standard I/O.
- [frontend.py](../operations/frontend.py): typed references, inspection, revision
  guards and one recorded service transaction per request.
- [coordinator.py](../jobs/coordinator.py): serialized authenticated native/CLI paths.
- [test_mcp.py](../../tests/test_mcp.py): native OLGS/OLSS parity, control exchange,
  history, failed writes, references, quotas, shared jobs and real SDK clients.

Run protocol acceptance tests in an environment with the `mcp` extra. Without it,
the native adapter tests still run and SDK integration tests report skipped.
Scientific article benchmarks and portable exact job restart remain separate.

Protocol authority: [standard-I/O transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools),
and the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x).

---

[⌂ Home](../../README.md)
