---
role: front-door
parent: ../README.md
summary: "The test suite, its fixtures, contained scenario evidence, and contained numerical systems."
read_when: "Read when running tests or locating test evidence."
status: active
reviewed: 2026-10-12
---
# Testing Hub

The library test suite, its fixtures, and contained numerical systems.

## Run the suite

```bash
MPLCONFIGDIR=/tmp/optimizer-mpl python -B -m unittest discover -s tests
```

Run it from the checkout root inside the `Optimizer` environment. `-B` keeps
bytecode out of the checkout.

## Test files

| Area | Files |
|---|---|
| Controls, results, and state | `test_controls.py`, `test_result_state.py` |
| Public API and discovery | `test_public_api.py`, `test_namespaces.py`, `test_catalog.py`, `test_catalog_coverage.py`, `test_api_reference.py`, `test_optimizer_context.py` |
| Optimizers and engine | `test_optimizers.py`, `test_advanced_optimizers.py`, `test_engine.py`, `test_engine_trial_gradient.py`, `test_numerical_capabilities.py` |
| Integrated OLGS/OLSS changes and transactions | `test_olgs_olss_integrations.py`, `test_transactions.py` |
| Shared conventions: serialization, families, and stop reasons | `test_serialization.py`, `test_stop_reasons.py` |
| OLSS contract and scenarios | `test_olss.py`, `test_olss_scenarios.py`, `test_olss_regressions.py` |
| Guesses and utilities | `test_guesses.py`, `test_utils_tools.py`, `test_run_diagnostics.py`, `test_control_tools.py` |
| System contract, templates, and visualization | `test_system_contract.py`, `test_templates.py`, `test_system_spec_visualization.py` |
| Command history and campaigns | `test_history.py`, `test_campaigns.py` |
| Service, terminal and persistence | `test_operation_service.py`, `test_cli_language.py`, `test_cli.py`, `test_persistence.py` |
| Typed MCP and shared experiment ownership | `test_mcp.py` (official SDK protocol tests require the `mcp` extra) |
| Documentation | `test_document_links.py`: every relative link in the Markdown resolves |

## Details

### Worked examples

[Parallel job tests](test_jobs.py) use neutral OLGS/OLSS systems to check actual
worker processes, frozen inputs, bounded resources, cancellation and history
commit ordering. [The jobs reference](../optimizer/jobs/JOBS.md) owns their runtime
contracts; these tests do not replace scientific convergence benchmarks.
[Persistent job tests](test_durable_jobs.py) check actual local coordinators,
detachment, numerical pause/resume parity, crash recovery and history publication.

[fixtures/quadratic_system.py](fixtures/quadratic_system.py) is a complete,
minimal OLGS system used across the suite. Read it next to the
[OLGS Contract](../optimizer/system_olgs/CONTRACT.md).

The planned separation between fast deterministic regression coverage and longer
contained convergence studies is implemented under
`test_contained/olgs_convergence/` and defined in the
[OLGS Convergence Test Design](../test_contained/olgs_convergence/REPORT.md).

The shooting campaign findings, resolved issues, deferred numerical work, and
reproduction links are recorded in the
[OLSS Scenario Report](../test_contained/olss_scenarios/REPORT.md).

The 2026-10-12 review records five verified OLSS fixes, the open issues, and
the shared-core inspection in the
[2026-10-12 Review](../test_contained/review_2026_10_12/REPORT.md).

Known-answer pi-pulse tests run one workflow through the Python API, the
operation service and the terminal for gradient and shooting, soft and hard.
Their expected values, results and findings are in the
[Pi-Pulse Sample Tests](../test_contained/pi_pulse_samples/REPORT.md).

## Contained systems

These numerical definitions remain available. Retired tracker-specific trial and
live-journal scripts are preserved in workspace `temp/task5-retired-logging-docs/`;
they require their historical code revision. No new article benchmark ran in the
history migration. CLI recovery/replay coverage lives in [history tests](test_history.py).

- [HO purification trial](../test_contained/ho_purification/PROBLEM_CONTEXT.md)
- [Second-order PMP trial](../test_contained/second_order_pmp/PROBLEM_CONTEXT.md)

Local exploration notebooks may live under `tests/system_check/`, which Git
ignores.

## Connections

- Related: [the V4 entry](../README.md)
- Related: [OLGS Contract](../optimizer/system_olgs/CONTRACT.md), [Utils Documentation Hub](../optimizer/utils/UTILS_HUB.md)
- Deeper: [2026-10-12 Review](../test_contained/review_2026_10_12/REPORT.md)
- Deeper: [Pi-Pulse Sample Tests](../test_contained/pi_pulse_samples/REPORT.md)

---

[⌂ Home](../README.md)
