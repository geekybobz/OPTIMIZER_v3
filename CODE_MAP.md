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
| Public facade and catalog | `optimizer/__init__.py`, `library.py`, `catalog.py` | Namespaces, aliases, discovery, and structured API metadata | `test_public_api.py`, `test_catalog.py` |
| Shared data | `controls.py`, `result.py`, `state.py` | Controls, optimizer results, live state, and warm starts | `test_controls.py`, `test_result_state.py` |
| OLGS boundary | `optimizer/system_olgs/` | Gradient-system declarations, validation, and optional capabilities | `test_system_contract.py` |
| OLSS boundary | `optimizer/system_olss/` | Shooting declarations, variables, validation, and evidence types | `test_olss.py` |
| Transactions | `optimizer/transactions.py` | Family-aware request-response execution: payload decoding, validated calls, and serialization | `test_transactions.py`, `test_olgs_olss_integrations.py` |
| Shared gradient engine | `optimizer/core/` | Evaluation, proposal validation, bounds, acceptance, stopping, trace, and checkpoints | `test_engine.py` |
| Gradient methods | `optimizer/optimizers/` | Method-specific updates and private state | `test_optimizers.py`, `test_advanced_optimizers.py` |
| Shooting methods | `optimizer/shooting/` | Root solving, direct shooting, continuation, multistart, seeds, and replay | `test_olss.py`, `test_olss_scenarios.py` |
| Diagnostics and repair | `optimizer/utils/` | Short-run diagnosis, derivatives, geometry, repair, and control inspection | `test_utils_tools.py`, `test_run_diagnostics.py` |
| Coordination and persistence | `campaigns/`, `tracker/`, `logs/` | Multi-stage workflows, journals, observers, traces, and artifacts | tracker and campaign tests |

## OLGS execution path

| Step | Owner | Input | Output |
|---:|---|---|---|
| 1 | `system_olgs` validation | project system and `Controls` | validated declaration and layout |
| 2 | optimizer wrapper | controls, state, or warm start plus method parameters | method-specific proposal function and engine options |
| 3 | `core.run_chunk` | system, current state, proposal function | accepted/rejected iterations and named stop reason |
| 4 | `OptimizerResult` | terminal run state and trace | controls, metrics, state, history, and warm-start boundary |
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

The transaction layer invokes system operations. Optimizer and shooting-run
orchestration remain direct Python APIs until the terminal interface is designed.

## Change-impact guide

| Intended change | Start here | Also inspect | Verify with |
|---|---|---|---|
| Required OLGS method | `system_olgs/contract.py` | validation, catalog, scaffold, fixtures | contract and integration tests |
| Required OLSS method | `system_olss/contract.py` | validation, evaluator, catalog, scaffold | OLSS and integration tests |
| Transaction field or operation | `transactions.py` | catalog, serialization, future adapters | integration tests |
| Shared acceptance or stopping | `core/engine.py` | optimizer wrappers and trace evidence | engine and optimizer tests |
| One optimizer's update rule | its module under `optimizers/` | state/warm-start documentation | optimizer tests |
| Shooting root behavior | `shooting/solvers.py` | evaluator, result evidence, campaigns | OLSS scenarios |
| Public API name or metadata | `catalog.py` and exports | generated API reference | catalog and reference-freshness tests |
| Documentation route | owning hub or topic | root map and index | Markdown Protocol verification |

## Connections

- Related: [V4 entry](README.md)
- Related: [System Transactions](optimizer/TRANSACTIONS.md)
- Related: [Testing Hub](tests/TESTING_HUB.md)
- Related: [API Reference](API_REFERENCE.md)

---

[← Previous](INDEX.md) · [⌂ Home](README.md)
