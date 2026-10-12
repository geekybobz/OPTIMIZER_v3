---
role: front-door
parent: ../README.md
summary: "The test suite, its fixtures, contained scenario evidence, and tracker trials."
read_when: "Read when running tests or locating test evidence."
status: active
reviewed: 2026-10-12
---
# Testing Hub

The library test suite, its fixtures, and the tracker trials.

## Run the suite

```bash
MPLCONFIGDIR=/tmp/optimizer-mpl python -B -m unittest discover -s tests
```

Run it from the checkout root inside the `optimizer_v4` environment. `-B` keeps
bytecode out of the checkout.

## Test files

| Area | Files |
|---|---|
| Controls, results, and state | `test_controls.py`, `test_result_state.py` |
| Public API and discovery | `test_public_api.py`, `test_namespaces.py`, `test_catalog.py`, `test_catalog_coverage.py`, `test_api_reference.py`, `test_optimizer_context.py` |
| Optimizers and engine | `test_optimizers.py`, `test_advanced_optimizers.py`, `test_engine.py`, `test_engine_trial_gradient.py`, `test_numerical_capabilities.py` |
| Integrated OLGS/OLSS changes and transactions | `test_olgs_olss_integrations.py`, `test_transactions.py` |
| OLSS contract and scenarios | `test_olss.py`, `test_olss_scenarios.py`, `test_olss_regressions.py` |
| Guesses and utilities | `test_guesses.py`, `test_utils_tools.py`, `test_run_diagnostics.py` |
| System contract, templates, and visualization | `test_system_contract.py`, `test_templates.py`, `test_system_spec_visualization.py` |
| Traces, tracker, and campaigns | `test_trace_checkpoint.py`, `test_tracker.py`, `test_tracker_bridge.py`, `test_campaigns.py` |
| Documentation | `test_document_links.py`: every relative link in the Markdown resolves |

## Worked example

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

## Tracker trials

Small contained systems that exercise the tracker on real physics:

- [HO purification trial](../test_contained/ho_purification/PROBLEM_CONTEXT.md)
- [Second-order PMP trial](../test_contained/second_order_pmp/PROBLEM_CONTEXT.md)

Local exploration notebooks may live under `tests/system_check/`, which Git
ignores.

## Connections

- Related: [the V4 entry](../README.md)
- Related: [OLGS Contract](../optimizer/system_olgs/CONTRACT.md), [Utils Documentation Hub](../optimizer/utils/UTILS_HUB.md)
- Deeper: [2026-10-12 Review](../test_contained/review_2026_10_12/REPORT.md)

---

[⌂ Home](../README.md)
