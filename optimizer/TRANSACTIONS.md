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
| `request_id` | Correlation identifier; generated when omitted |
| `family` | Optional assertion: `olgs` or `olss` |
| `operation` | Required system operation |
| `payload` | Operation inputs such as controls, variables, or secondary updates |
| `options` | Per-call numerical options; accepted only by OLSS numerical calls |

### Payload forms

| Input | Accepted forms |
|---|---|
| `controls`, `vector` (OLGS) | `Controls`; the `controls_to_dict()` mapping with `matrix`; a channel mapping; `{"flat": [...]}`; a nested matrix list; a flat list |
| `variables` (OLSS) | `ShootingVariables`; its `to_dict()` mapping with `values`; a block mapping; a flat list |

Every form is rebuilt with the system's `control_spec()` or `shooting_spec()`. A
declared layout in the payload that disagrees with the system is rejected.

## Response

| Field | Meaning |
|---|---|
| `request_id` | Matches the request |
| `family` | Family detected from `system_spec()` |
| `operation` | Attempted operation |
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
