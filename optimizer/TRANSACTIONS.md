---
role: topic
parent: ../README.md
summary: "Family-aware request-response boundary shared by OLGS, OLSS, and future adapters."
read_when: "Read when integrating systems with shared core, terminal, or MCP-facing code."
status: active
reviewed: 2026-10-12
---
# System Transactions

The transaction layer converts a direct OLGS or OLSS method call into a stable,
structured request and response without changing either project contract. It
decodes the payload against the system's own layout and calls the system through
the same validated helpers the numerical tools use, so a request that passed
through JSON runs unchanged and a contract violation becomes a failed response.

## In brief

Project authors implement direct system methods. `SystemRequest`,
`SystemResponse`, `transact`, and `SystemTransactionAdapter` provide a common
boundary for generic consumers.

```text
request envelope -> family check -> capability check -> payload decoding
    -> validated system call -> response envelope
```

This layer covers system operations only. Running methods, sessions, and
persistence belong to the planned interactive operation service.

## Request

```python
SystemRequest(
    operation="residuals",
    family="olss",
    payload={"variables": variables},
    options={"accuracy": "working"},
)
```

| Field | Meaning |
|---|---|
| `request_id` | Non-empty string correlation identifier; generated when omitted |
| `family` | Optional assertion: `olgs` or `olss` |
| `operation` | Required non-empty string system operation |
| `payload` | Mapping with string keys for the operation's declared inputs |
| `options` | Mapping with string keys; accepted only by OLSS numerical calls |

Unknown envelope fields and malformed field types return `invalid_request`.
Unknown or missing operation payload fields return `invalid_payload`. OLSS
evaluation-option names and secondary-update names remain system-owned.
Constructing an invalid `SystemRequest` directly raises a normal Python exception;
passing a malformed mapping to `transact` returns a failed response.

### Payload forms

| Input | Accepted forms |
|---|---|
| `controls`, `vector` (OLGS) | `Controls`; the `controls_to_dict()` mapping with `matrix`; a channel mapping; `{"flat": [...]}`; a nested matrix list; a flat list |
| `variables` (OLSS) | `ShootingVariables`; its `to_dict()` mapping with `values`; a block mapping; a flat list |

Raw arrays and channel/block mappings use the target system's layout explicitly.
Serialized mappings may omit `spec` to use that layout; when `spec` is present it
must be complete and compatible. Both live objects and serialized objects use
the same numerical identity:

- Controls: ordered channel keys, control dimension, dtype and `dt`. Redundant
  serialized `shape` and `n_controls`, when present, must also agree.
- Shooting variables: ordered block names, shapes, sizes, units, scales and roles,
  plus total size and dtype. The system's static variable declaration must agree
  with `shooting_spec()` in these fields, even when total sizes happen to match.
- Descriptive names, block meaning text, and arbitrary metadata are not numerical
  identity. Richer grid representations need the later explicit transfer tools.

A declared control spec requires `keys`, `control_dim`, `dtype` and `dt`;
a shooting spec requires `blocks`, `size` and `dtype`. Each shooting block requires
`name`, `shape`, `size`, `unit`, `scale` and `role`. Units and `dt` can be null
where the target spec is also null. A grid or dtype mismatch is rejected; this
boundary does not resample or convert a declared object silently.

Complex controls accept `{real, imag}` scalar or array data under a complex
control layout. Real/imaginary shapes must agree without broadcasting. Complex
input under a real layout fails rather than discarding its imaginary part.
All numerical inputs must be finite, including values decoded from strings.
Decoding owns the numerical buffers before calling project code; an operation
cannot mutate the caller's submitted controls or arrays through that buffer.

## Response

| Field | Meaning |
|---|---|
| `request_id` | Matches a usable request ID; otherwise generated for the failed request |
| `family` | Detected family; null when request or family discovery failed |
| `operation` | Attempted string operation; null when it could not be identified |
| `ok` | Whether the validated call completed |
| `result` | Returned value on success |
| `error` | Structured `type` and `message` on failure |

Exceptions are converted into response evidence at this boundary. Direct calls
outside this adapter retain ordinary Python exception behavior.

## Operation matrix

| Family | Operation | Payload | Options | Direct call |
|---|---|---|---|---|
| Both | `system_spec` | none | none | `system.system_spec()` |
| OLGS | `control_spec` | none | none | `system.control_spec()` |
| OLSS | `shooting_spec` | none | none | `system.shooting_spec()` |
| Both | `with_secondary` | update fields | none | `system.with_secondary(**payload)` |
| OLGS | `evaluate` | `controls` | none | `system.evaluate(controls)` |
| OLGS | `gradient` | `controls` | none | `system.gradient(controls)` |
| OLGS | `value_and_gradient` | `controls` | none | optional hook; result `{metrics, gradient}` |
| OLGS | `residuals`, `jacobian` | `controls`, optional `name` | none | optional constraint hooks |
| OLGS | `hvp` | `controls`, `vector` | none | optional second-order hook |
| OLGS | `hessian` | `controls` | none | optional dense Hessian |
| OLGS | `smooth_gradient` | `controls` | none | optional proximal hook |
| OLGS | `prox` | `controls`, `step_size` | none | optional proximal hook |
| OLGS | `control_bounds` | none | none | optional hook; result `{lower, upper}` |
| OLSS | `residuals` | `variables` | evaluation options | `system.residuals(variables, options=...)` |
| OLSS | `jacobian` | `variables` | evaluation options | `system.jacobian(variables, options=...)` |
| OLSS | `residuals_and_jacobian` | `variables` | evaluation options | combined hook when present; result `{residuals, jacobian}` |
| OLSS | `evaluate` | `variables` | evaluation options | `system.evaluate(variables, options=...)` |
| OLSS | `verify` | `variables` | verification options | `system.verify(variables, options=...)` |
| OLSS | `cost_gradient` | `variables` | evaluation options | optional direct-shooting hook |
| OLSS | `shooting_bounds` | none | none | optional hook; result `{lower, upper}` |

Every numerical result passes the same validation as in the numerical tools:
shapes, finiteness, declared metrics, and layouts. Verification through this
layer reports `independence_source="transaction_request"`; whether a
verification is independent of a working solve is decided by the solvers.

## Capability map

OLGS requires `system_spec`, `control_spec`, `evaluate`, `gradient` and
`with_secondary`. OLSS requires `system_spec`, `shooting_spec`, `residuals`,
`jacobian`, `evaluate`, `verify` and `with_secondary`. Presence is discoverable
through `probe_system` / `probe_shooting_system`; `require_system` /
`require_shooting_system` check the complete contract when loading a system.
Individual transactions check the hooks their operation needs.

| Optional capability | Existing route | Transaction/service coverage |
|---|---|---|
| OLGS combined value/gradient | `value_and_gradient_system` | `value_and_gradient` transaction |
| OLGS hard rows and Jacobian | `optional_residuals`, `optional_jacobian` | `residuals`, `jacobian` transactions |
| Bounds | OLGS `optional_control_bounds`; OLSS `shooting_bounds_system` | `control_bounds`, `shooting_bounds` transactions |
| OLGS curvature | `optional_hessian`, `optional_hvp` | `hessian`, `hvp` transactions |
| OLGS composite objective | `optional_smooth_gradient`, `optional_prox` | `smooth_gradient`, `prox` transactions |
| OLSS combined residual/Jacobian | `residuals_and_jacobian_system` | Combined hook or separate required calls |
| OLSS objective gradient | `cost_gradient_system` | `cost_gradient` transaction |
| OLGS feasibility-preserving transfer | `optional_control_transfer` | Existing helper; service adapter still needed |
| OLSS control recovery | `recover_controls` used by `compare_shooting_replay` | Existing replay tool; service adapter still needed |
| Propagation and simulation | Project `forward_prop`, OLGS `back_prop`, `simulate` | Optional hooks; service adapters still needed |
| Plot data | OLGS `optional_plot_data`, project `plot_data` | Existing data tools; service adapters still needed |
| Cache inspection/reset | Project `cache_status`, `cache_reset` | Probe discovery; service adapters still needed |
| Sparse/operator Jacobians and root equivalence | OLSS probe and documented hooks | Optional capabilities; not a promise of solver/backend support |

An absent hook is different from a hook whose service adapter is not yet present.
Optional hooks never become mandatory just because the CLI exposes them. Numerical
methods, sessions, persistence and job control belong to the later operation
service, not this system transaction layer.

## Serialization

Responses, results, and reports share one conversion, `opt.to_plain`, so every
`to_dict()` output is strict JSON (`json.dumps(..., allow_nan=False)` succeeds):

- Non-finite numbers become the strings `"nan"`, `"inf"`, and `"-inf"`.
- NumPy arrays become lists and NumPy scalars become Python values; complex
  numbers and arrays become `{"real": ..., "imag": ...}`.
- `Controls`, shooting objects, and results use their `to_dict()` methods;
  other dataclasses become their fields.
- System objects are represented by their type and `system_spec()`.
- Callables become `"module:qualname"` references and paths become strings.
- Mappings, sequences, and sets are converted recursively.
- Any other object raises `TypeError`; durable output never falls back to
  `repr()`.

Tracker journals are stricter: their records reject non-finite values, and the
standalone reader validates them. Serialization is representation only; it does
not reconstruct arbitrary project systems or grant permission to import project
code.

## Failure rules

- Request parsing and family detection are inside the response boundary.
  Malformed envelopes return `invalid_request`; invalid family declarations or a
  missing/non-mapping/raising/invalid `system_spec` return `invalid_system`.
  The static declaration is validated before executing the operation and reused
  by transaction evaluation; this does not add numerical calls.
- A system must provide `system_spec()`.
- The declared family must be `olgs` or `olss`. A missing family means OLGS,
  by the library-wide rule of `opt.system_family`; specifications built with
  `basic_system_spec` declare `olgs`.
- A request family that disagrees with the detected family returns
  `family_mismatch` without invoking the requested operation.
- An optional operation whose hook the system lacks returns
  `capability_missing` before any computation.
- A missing, undecodable, or layout-mismatched payload, or options sent to an
  operation that takes none, returns `invalid_payload`.
- An unknown operation returns `unsupported_operation` with the supported list.
- A contract violation, such as a non-finite objective or an undeclared metric,
  returns the validator's exception type, for example `ValueError`.

## Connections

- Related: [V4 entry](../README.md)
- Related: [OLGS Contract](system_olgs/CONTRACT.md)
- Related: [OLSS Contract](system_olss/CONTRACT.md)
- Related: [Code Map](../CODE_MAP.md)
- Related: [Generated API Reference](../API_REFERENCE.md)

---

[⌂ Home](../README.md)
