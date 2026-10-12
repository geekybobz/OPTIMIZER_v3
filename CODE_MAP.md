---
role: topic
parent: README.md
summary: "Runtime components, execution paths, source ownership, and change-impact guide."
read_when: "Read before changing architecture, dispatch, contracts, or shared execution behavior."
status: active
reviewed: 2026-10-12
---
# Code Map

OPTIMIZER separates project-owned numerical meaning from library-owned execution.

## In brief

```text
project system
    -> validated OLGS or OLSS boundary
    -> numerical method or coordinator
    -> structured result and evidence
    -> diagnostics, continuation, replay, or persistence
```

## Component responsibilities

| Component | Source owner | Responsibility | Primary checks |
|---|---|---|---|
| Public facade and catalog | `optimizer/__init__.py`, `library.py`, `catalog.py` | One catalog declaration per public export; namespace exports, `opt.methods()`, the optimizer-name lookup, discovery, and the API reference are built from it | `test_public_api.py`, `test_catalog.py`, `test_catalog_coverage.py` |
| Shared conventions | `serialization.py`, `families.py`, `core/stop_reasons.py` | One JSON policy (`to_plain`), one family rule (`system_family`), and one stop-reason registry (`STOP_REASONS`) for both families | `test_serialization.py`, `test_stop_reasons.py` |
| Shared data | `controls.py`, `result.py`, `state.py` | Controls, optimizer results, live state, and warm starts | `test_controls.py`, `test_result_state.py` |
| OLGS boundary | `optimizer/system_olgs/` | Gradient-system declarations, validation, and optional capabilities | `test_system_contract.py` |
| OLSS boundary | `optimizer/system_olss/` | Shooting declarations, variables, validation, and evidence types | `test_olss.py` |
| Transactions | `optimizer/transactions.py` | Family-aware request-response execution: payload decoding, validated calls, and serialization | `test_transactions.py`, `test_olgs_olss_integrations.py` |
| Foreground service | `optimizer/operations/`, `optimizer/session.py` | Catalog dispatch, contexts, owned candidates, atomic activation and retention limits | `test_operation_service.py` |
| Structured terminal | `optimizer/cli/` | Restricted AST commands, reference binding, prompt/scripts, colon commands, discovery, rich panels and the live progress line | `test_cli_language.py`, `test_cli.py`, `test_cli_ux.py` |
| MCP interface | `optimizer/mcp/`, `optimizer/operations/frontend.py` | Typed recorded requests, bounded standard I/O, shared experiment attachment and paginated inspection | `test_mcp.py` |
| Durable experiments | `optimizer/persistence/`, `optimizer/checkpoints/` | Independent stores, typed state, rollback, leases, quotas and atomic recovery | `test_persistence.py` |
| Shared numerical core | `optimizer/core/` | Evaluator mechanics, coordinate scaling, kernels, iteration driver, bounded records and continuation; see [numerical core](docs/SHARED_NUMERICAL_CORE.md) | `test_engine.py`, `test_shared_core.py` |
| Gradient methods | `optimizer/optimizers/` | Method-specific updates and private state | `test_optimizers.py`, `test_advanced_optimizers.py` |
| Shooting methods | `optimizer/shooting/` | Root solving, direct shooting, continuation, multistart, seeds, and replay | `test_olss.py`, `test_olss_scenarios.py` |
| Diagnostics and repair | `optimizer/utils/` | Short-run diagnosis, derivatives, geometry, repair, and control inspection | `test_utils_tools.py`, `test_run_diagnostics.py` |
| Campaign coordination | `campaigns/` | In-memory typed candidates and explicit validation | `test_campaigns.py` |
| Command history | `history/` | Frontend statement transactions, recorded outcomes, recovery and replay | `test_history.py` |
| [Parallel jobs](optimizer/jobs/JOBS.md) | `optimizer/jobs/` | Frozen process runs, bounded admission, persistent single-writer coordination, live pause and physical recovery | `test_jobs.py`, `test_durable_jobs.py` |

## OLGS execution path

| Step | Owner | Input | Output |
|---:|---|---|---|
| 1 | `system_olgs` validation | project system and `Controls` | validated declaration and layout |
| 2 | optimizer wrapper | controls, state, or warm start plus method parameters | method-specific proposal function and engine options |
| 3 | `core.run_chunk` | system, current state, proposal function | accepted/rejected iterations and named stop reason |
| 4 | `OptimizerResult` | terminal run state and numerical aggregates | controls, metrics, state, diagnostics and warm-start boundary |
| 5 | utils or another method | result controls/state | diagnosis, repair, continuation, or another chunk |

## OLSS execution path

| Step | Owner | Input | Output |
|---:|---|---|---|
| 1 | `system_olss` validation | project system and `ShootingVariables` | validated declaration, layout, and candidate |
| 2 | `ShootingEvaluator` | variables and working options | residuals, Jacobian, metrics, counters, and cache entries |
| 3 | shooting method | evaluator evidence and method parameters | candidate plus iteration history |
| 4 | independent verification | candidate and distinct verification options | verification residuals, metrics, and named checks |
| 5 | OLSS result/coordinator | solver and verification evidence | qualified result, branch, multistart ranking, or replay report |

## Transaction path

```text
SystemRequest
    -> family validation
    -> direct OLGS or OLSS method
    -> SystemResponse
    -> JSON-safe structured representation
```

The transaction layer invokes system operations. The
[foreground operation service](optimizer/operations/SERVICE.md) dispatches supported
numerical tools through the same native APIs and adds owned session state.
Terminal parsing and the MCP transport remain separate adapters.

## Change-impact guide

| Intended change | Start here | Also inspect | Verify with |
|---|---|---|---|
| Required OLGS method | `system_olgs/contract.py` | validation, catalog, scaffold, fixtures | contract and integration tests |
| Required OLSS method | `system_olss/contract.py` | validation, evaluator, catalog, scaffold | OLSS and integration tests |
| Transaction field or operation | `transactions.py` | catalog, serialization, future adapters | integration tests |
| Shared acceptance or stopping | `core/engine.py` | optimizer wrappers and numerical diagnostics | engine and optimizer tests |
| One optimizer's update rule | its module under `optimizers/` | state/warm-start documentation | optimizer tests |
| Shooting root behavior | `shooting/solvers.py` | evaluator, result evidence, campaigns | OLSS scenarios |
| Public API name or metadata | its declaration in `catalog.py`; exports follow | generated API reference | catalog coverage and reference-freshness tests |
| New stop reason | `core/stop_reasons.py` | the loop that assigns it; `diagnose_run` | `test_stop_reasons.py`, generated API reference |
| Result or report serialization | `serialization.py` | the result's `to_dict` | `test_serialization.py` |
| Documentation route | owning hub or topic | root map and index | Markdown Protocol verification |

## Connections

- Related: [V4 entry](README.md)
- Related: [System Transactions](optimizer/TRANSACTIONS.md)
- Related: [Testing Hub](tests/TESTING_HUB.md)
- Related: [API Reference](API_REFERENCE.md)

---

[← Previous](INDEX.md) · [⌂ Home](README.md)
