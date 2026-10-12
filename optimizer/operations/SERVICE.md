---
role: deep
parent: ../../README.md
summary: Foreground operation requests, isolated sessions, activation, continuation and retained-data limits.
read_when: Read when integrating the operation service or preparing a terminal/MCP adapter.
---

# Foreground operation service

`OperationService` supplies a selected system and current numerical input to
existing library tools. `Session` owns independent named contexts, versioned
objects and active heads. Native OLGS and OLSS contracts remain separate.
This service has no terminal parser, disk store or automatic history writer.

## Calls and results

```python
import optimizer as opt

service = opt.OperationService()
loaded = service.execute(opt.OperationRequest(
    "session.load", {"name": "main", "factory": "./system.py",
                     "parameters": {"N": 1001}}))
initial = service.execute(opt.OperationRequest(
    "guesses.random_fourier_guess", {"seed": 7, "amplitude": 0.1},
    bind="u0", activate=True))
run = service.execute(opt.OperationRequest(
    "optimizers.adam", {"maxiter": 20, "step_size": 0.01}, bind="run1"))
trial = service.execute(opt.OperationRequest(
    "optimizers.lbfgs", {"maxiter": 5}, bind="trial", activate=False))
diagnosis = service.execute(opt.OperationRequest(
    "utils.diagnose_run", {"result": trial.object_refs[0]}))
chosen = service.execute(opt.OperationRequest(
    "session.use", {"value": trial.object_refs[0]}))
```

Check each response's `ok` and `error` before consuming its result. The Python
`result` retains its native type. `to_dict()` contains summaries, object refs,
state transitions and structured errors; it never exports full result arrays.
It is a transport summary, not a checkpoint or replay codec. Native convergence,
stop reasons and OLSS qualification are distinct from successful dispatch.

| Request field | Meaning |
|---|---|
| `operation` | Catalog name/alias or one of the session operations below |
| `arguments` | Native keyword arguments; unknown names fail before numerical calls |
| `context` | Explicit context, otherwise the selected one when context is needed |
| `bind` | Retain a numerical tool result under this name; only `session.use` accepts it among session mutations |
| `activate` | Override the numerical tool's declared activation default |
| `request_id` | Correlation ID, generated when omitted |

`ObjectRef(object_id, path=("controls",))` selects a declared dataclass field.
Mappings support key selection; tuple/list/array members use nonnegative integer
indices, such as `path=(0,)` for a transfer's control. Controls support declared
channel keys. References cannot invoke methods or traverse
private attributes. Names resolve through current bindings; stable IDs continue
to identify older versions while retained. Inspection returns defensive copies.

## Session operations

| Operation | Required arguments | Effect |
|---|---|---|
| `session.load` | `name`, `factory` | Construct, validate and select a new context; optional `parameters`, `factory_name`, `select` |
| `session.attach` | `name`, `system` | Attach an ephemeral Python system; optional independent zero-argument `clone`, `select` |
| `session.select` | `name` | Select an existing context without changing any head |
| `session.use` | `value` | Activate compatible controls, variables or same-configuration result refs |
| `session.update` | `updates` | Apply secondary updates to an isolated system; preserve physical input and discard metrics/private state |
| `session.get` | `reference` | Return an owned copy of a named/ID/field reference |
| `session.bind` | `name`, `reference` | Protect a stable version or retain an owned field with source context/configuration/method provenance |
| `session.release` | `name` | Remove a binding; discard its object immediately if no other binding/head protects it |
| `session.state` | none | Inspect contexts, selection, heads, bindings, revision and retained bytes |

Use the request's `context` for `session.use/update`. Existing context names are
never silently overwritten. A different layout requires a new named context and
explicit conversion/initialization. Factory parameters and defaults are captured
by value. File loading executes an explicit trusted Python module; it is not a
sandbox or automatic discovery/import of a repository tree.

File recipes record path, digest and factory name; reconstruction rejects a
changed source file. Callable recipes and attached systems are ephemeral. The
file digest does not capture imported dependencies or guarantee reproducibility
of hidden/global/project state. Factories/clones must return independent systems
with the same declared configuration. Normal runtime caches are not checkpoints.

## Activation and continuation

| Tool effect | Default |
|---|---|
| Evaluation, derivative, diagnostic, plot-data inspection | Preserve numerical head |
| Guess, transform, repair, seed or continuation report | Candidate only |
| Gradient optimizer or single shooting solver | Activate valid returned state |

`activate=False` retains a trial without changing the head. Failed native solver
results remain inspectable candidates and do not activate. `maxiter` with a valid
state is usable; an unqualified shooting result stays unqualified after activation.
Reports containing several candidates require explicit selection of a numerical
member before `session.use`. Transfer tuples likewise require their control member.

Compatible OLGS chunks retain copied `RunState` through native APIs. Compatibility
includes method, variant, effective configuration, schedule and layout; changing
only `maxiter` does not reset it. Explicit controls start fresh. Switching methods
or configuration restarts from physical input. Full explicit state needs a retained
reference from the same context/configuration with matching method settings.

`solve_shooting` continues variables and native cumulative history; it does not
claim a common OLGS/OLSS exact-resume engine. Direct shooting and CMA-ES restart
from physical input. The latter does not expose complete native RNG restoration.
`service.info(name)["continuation"]` reports the declared continuation level.

## Coverage and boundaries

`service.list()` classifies every existing catalog declaration. Public identities,
aliases and callable implementations remain catalog-owned. Dispatch policy lives
in [registry.py](registry.py); unsupported entries remain visible with reasons.

Available adapters cover gradient methods, guesses, schedules, numerical data
constructors, OLGS/OLSS validated helpers, diagnostics, repair, spectrum, plot-data
validation, shooting root/direct solvers, seeds, serial multistart/refinement,
shooting continuation and cross-family replay. `core.run_chunk` accepts explicit
Python/registered step callbacks. `register_reference(name, callback)` supplies
ephemeral callback references without adding arbitrary attribute execution.

Cross-family replay requires explicit `shooting_system` and `gradient_system`
context names and variables in the shooting context. It does not infer costates
from waveforms. Raw system transactions are available as `base.transact`; use
`session.update` for secondary changes. Plot options remain system-owned.

Live renderers, runtime/evaluator objects, durable campaign factories and the
OLGS coordinator returning live systems require lifecycle adapters. They remain
available through native Python and are discoverable with explicit reasons here.
Legacy logger entries are classified for later retirement. No generic serializer
or `deepcopy` is used as a physics-system reconstruction policy.

All state-changing numerical tools require reconstruction/clone isolation. A
Python-only attachment without that route supports read operations and explicit
physical activation. External project writes and callback side effects cannot
be rolled back by an in-memory session.

## Retention and transactions

`Session(max_bytes=64*1024*1024, max_objects=256)` bounds retained data. Recipe,
layout, result buffers, bindings and private method configuration contribute to byte
accounting. Context and binding counts are also bounded by `max_objects`. Current bindings and
all active heads are protected; unbound older results may be evicted. A quota
failure commits no head/binding. The budget is not a process-RAM limit: live
physics runtimes, libraries, callbacks and numerical temporaries need later job
resource policies.

The service serializes its requests. The session revision guards publication
against other services or reentrant callbacks: stale candidates fail with
`revision_conflict`. Inputs/output numerical buffers are owned, and publication
updates the binding and head together. System construction and computations occur
before publication; failed loads/updates preserve existing contexts.

Structured error codes include `invalid_request`, `invalid_arguments`,
`invalid_system`, `missing_context`, `missing_object`, `wrong_family`,
`incompatible_layout`, `capability_missing`, `adapter_unavailable`,
`unsupported_isolation`, `unsupported_continuation`, `source_changed`,
`computation_failed`, `invalid_result`, `object_quota` and `revision_conflict`.

## Verification and ownership

- [service.py](service.py): validation, native invocation and publication.
- [contracts.py](contracts.py): typed requests, responses and references.
- [session.py](../session.py): contexts, reconstruction recipes and retained versions.
- [service tests](../../tests/test_operation_service.py): native parity, continuation,
  candidates, layouts, ownership, quotas, revisions and Python history boundary.

The [structured terminal](../cli/TERMINAL.md) now calls this service. Named on-disk experiments/checkpoints, full
CLI/MCP history and old logger removal, new control conversion helpers and managed
parallel/pause/cancel workflows have separate delivery gates.

## Connections

- Parent: [V4 entry](../../README.md)
- Related: [System transactions](../TRANSACTIONS.md)
- Related: [Code map](../../CODE_MAP.md)
- Related: [Structured terminal](../cli/TERMINAL.md)

---

[⌂ Home](../../README.md)
